"""LangChain chat model backed by Apple's on-device Foundation Models framework."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Iterator, List, Optional, Sequence, Type, Union

import applefoundationmodels as afm
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field

_cached_framework_version: Optional[str] = None


def _framework_version() -> str:
    """Apple's FoundationModels framework version, e.g. for tracing metadata.

    `Session.get_version()` needs a live instance to call (it isn't a
    classmethod), so a throwaway Session gets created once and the result
    cached -- not re-created on every _get_ls_params() call.
    """
    global _cached_framework_version
    if _cached_framework_version is None:
        try:
            _cached_framework_version = afm.Session().get_version()
        except Exception:
            _cached_framework_version = "unknown"
    return _cached_framework_version


def _wrap_tool(tool: BaseTool) -> Callable[..., Any]:
    """Wrap a LangChain BaseTool as a plain Python callable.

    apple-foundation-models introspects a callable's signature and docstring
    to build the schema it hands to the on-device model, so the wrapper needs
    a real `__signature__` and a docstring with a Google-style Args section,
    not just a generic (*args, **kwargs) passthrough.
    """
    fields = tool.args_schema.model_fields if tool.args_schema else {}
    params = [
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=field.annotation,
            default=field.default if field.default is not None else inspect.Parameter.empty,
        )
        for name, field in fields.items()
    ]

    doc_lines = [tool.description or tool.name, "", "Args:"]
    for name, field in fields.items():
        desc = field.description or name
        doc_lines.append(f"    {name}: {desc}")

    def _call(**kwargs: Any) -> Any:
        return tool.run(kwargs)

    _call.__name__ = tool.name
    _call.__doc__ = "\n".join(doc_lines)
    _call.__signature__ = inspect.Signature(params)
    return _call


class ChatAppleFoundationModels(BaseChatModel):
    """Chat model wrapping Apple's on-device Foundation Models framework.

    Requires macOS 26+ with Apple Intelligence enabled. Runs entirely
    on-device -- no network calls, no API key.

    Example:
        .. code-block:: python

            from langchain_apple_foundation_models import ChatAppleFoundationModels

            llm = ChatAppleFoundationModels()
            llm.invoke("What is the capital of France?")
    """

    instructions: Optional[str] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)

    _session: Optional[afm.Session] = None
    _session_tools: Optional[List[Callable[..., Any]]] = None

    @property
    def _llm_type(self) -> str:
        return "apple-foundation-models"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    def _get_ls_params(self, stop: Optional[List[str]] = None, **kwargs: Any) -> dict:
        params = super()._get_ls_params(stop=stop, **kwargs)
        # There's only one on-device model -- no real "model override" concept
        # -- but honor an explicit `model=` kwarg if a caller passes one
        # anyway, so traces reflect what was actually asked for.
        params["ls_model_name"] = kwargs.get(
            "model", f"apple-foundation-models-{_framework_version()}"
        )
        return params

    def _get_session(
        self,
        system_instructions: Optional[str],
        tools: Optional[List[Callable[..., Any]]] = None,
    ) -> afm.Session:
        # `tools` arrives per-call via bind_tools' standard RunnableBinding
        # kwargs, not as instance state -- rebuild the session if the bound
        # tools changed (e.g. .bind_tools() was called again) rather than
        # silently keep serving the old tool set.
        if self._session is None or tools is not self._session_tools:
            self._session = afm.Session(
                instructions=system_instructions or self.instructions,
                tools=tools or None,
            )
            self._session_tools = tools
        return self._session

    @staticmethod
    def _extract_system_and_prompt(messages: Sequence[BaseMessage]) -> tuple[Optional[str], str]:
        system_instructions = None
        remaining = list(messages)
        if remaining and isinstance(remaining[0], SystemMessage):
            system_instructions = remaining[0].content
            remaining = remaining[1:]
        if not remaining or not isinstance(remaining[-1], HumanMessage):
            raise ValueError(
                "ChatAppleFoundationModels expects the last message to be a HumanMessage. "
                "Multi-turn history is tracked by the underlying Session, not replayed "
                "from the LangChain message list -- see README for this limitation."
            )
        return system_instructions, remaining[-1].content

    @staticmethod
    def _to_langchain_tool_calls(resp_tool_calls: Any) -> list[dict[str, Any]]:
        if not resp_tool_calls:
            return []
        import json

        out = []
        for tc in resp_tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (ValueError, AttributeError):
                args = {}
            out.append(
                {
                    "name": tc.function.name,
                    "args": args,
                    "id": tc.id,
                    "type": "tool_call",
                }
            )
        return out

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_instructions, prompt = self._extract_system_and_prompt(messages)
        session = self._get_session(system_instructions, kwargs.get("tools"))
        resp = session.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        message = AIMessage(
            content=resp.content if not resp.is_structured else "",
            tool_calls=self._to_langchain_tool_calls(resp.tool_calls),
            response_metadata={
                "finish_reason": resp.finish_reason,
                "is_structured": resp.is_structured,
            },
            additional_kwargs={"structured_content": resp.content} if resp.is_structured else {},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        system_instructions, prompt = self._extract_system_and_prompt(messages)
        session = self._get_session(system_instructions, kwargs.get("tools"))
        for chunk in session.generate(
            prompt,
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            text = chunk.content or ""
            gen_chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                run_manager.on_llm_new_token(text, chunk=gen_chunk)
            yield gen_chunk

    def bind_tools(
        self,
        tools: Sequence[Union[BaseTool, Callable[..., Any]]],
        **kwargs: Any,
    ) -> Any:
        # Standard LangChain convention: bind_tools returns a RunnableBinding
        # (via Runnable.bind), not a mutated copy of the model itself -- the
        # broader ecosystem (agents, tracing, serialization) expects this
        # exact shape. _generate/_stream read the tools back out of kwargs.
        wrapped = [_wrap_tool(t) if isinstance(t, BaseTool) else t for t in tools]
        return self.bind(tools=wrapped, **kwargs)

    def with_structured_output(
        self,
        schema: Union[dict, Type[Any]],
        **kwargs: Any,
    ) -> Any:
        from langchain_core.prompt_values import PromptValue
        from langchain_core.runnables import RunnableLambda

        def _invoke(value: Union[str, List[BaseMessage], PromptValue]) -> Any:
            if isinstance(value, str):
                messages: List[BaseMessage] = [HumanMessage(content=value)]
            elif isinstance(value, PromptValue):
                messages = value.to_messages()
            else:
                messages = list(value)
            system_instructions, prompt = self._extract_system_and_prompt(messages)
            session = self._get_session(system_instructions)
            # The on-device model's constrained decoding occasionally fails to
            # produce schema-conformant output on the first attempt (a known
            # characteristic of the smaller on-device model, not transient
            # infra flakiness) -- one retry clears the large majority of these.
            last_error: Optional[Exception] = None
            for _attempt in range(2):
                try:
                    resp = session.generate(prompt, schema=schema)
                    return resp.content
                except afm.DecodingFailureError as exc:
                    last_error = exc
            raise last_error

        return RunnableLambda(_invoke)
