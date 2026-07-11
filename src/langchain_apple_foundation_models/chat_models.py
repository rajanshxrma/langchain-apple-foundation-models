"""LangChain chat model backed by Apple's on-device Foundation Models framework."""

from __future__ import annotations

import inspect
import threading
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

# Serializes every call into the Foundation Models native layer, process-wide.
#
# Two hard constraints force this lock, both verified against
# apple-foundation-models 0.2.2:
#
# 1. Apple's framework allows at most ONE in-flight generation per session.
#    A concurrent call either trips a Swift runtime trap -- EXC_BREAKPOINT
#    (SIGTRAP) inside FoundationModels, killing the whole Python process --
#    or surfaces the framework's own error ("Attempted to call respond(to:)
#    a second time before the first invocation completed", error_code -18)
#    *as generated content*, i.e. silent output corruption.
# 2. The SDK's native layer keeps a single process-global session underneath:
#    its own `create_session` doc says "Session ID (always 0 for single global
#    session)", and `generate`/`get_transcript` take no session id at all. So
#    separate `afm.Session` objects -- even across different
#    ChatAppleFoundationModels instances -- still share one native session,
#    which is why this lock is module-level, not per-instance, and why a
#    session-per-call design would not help.
#
# LangChain's default `batch`/`abatch` fan out over threads (and the default
# async methods run the sync implementations on executor threads), so without
# this lock any batch of 2+ crashes. With it, concurrent calls complete
# correctly but sequentially -- an intentional trade dictated by the platform.
#
# This must stay a plain `threading.Lock`, never an `RLock`: langchain-core's
# default `_astream` steps a sync `_stream` generator via run_in_executor, so
# the thread that releases the lock (last `next()`) can differ from the one
# that acquired it (first `next()`). Lock allows that; RLock is owner-checked.
_SESSION_LOCK = threading.Lock()


def _framework_version() -> str:
    """Apple's FoundationModels framework version, e.g. for tracing metadata.

    `Session.get_version()` is a staticmethod wrapping a module-level FFI
    call, so no throwaway Session gets created here -- important because
    creating any Session resets the SDK's single process-global native
    session (see _SESSION_LOCK).
    """
    global _cached_framework_version
    if _cached_framework_version is None:
        try:
            _cached_framework_version = afm.Session.get_version()
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

    Thread safety / concurrency: the framework permits one in-flight
    generation per process (the SDK exposes a single global native session),
    so all generation calls -- across every instance in the process -- are
    serialized on an internal lock. `batch`/`abatch` and multi-threaded use
    are safe, but run sequentially under the hood; concurrency adds no
    throughput on this backend, and batch items on one instance join the
    same session history that sequential `invoke` calls share. See
    _SESSION_LOCK for the full rationale.

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
        # Callers must hold _SESSION_LOCK: the lazy check-then-create below is
        # itself a data race, and afm.Session() resets the SDK's process-global
        # native session, so it must never run while a generation is in flight.
        #
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
        # One critical section spanning session lookup AND generation: a gap
        # between them would let another thread swap the session mid-call.
        with _SESSION_LOCK:
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
        # The lock is held across the ENTIRE iteration, yields included: the
        # native generation stays in flight until the stream is exhausted (or
        # this generator is closed, which releases the lock via the `with`),
        # so releasing any earlier would re-expose the one-generation-at-a-time
        # constraint _SESSION_LOCK exists to enforce.
        with _SESSION_LOCK:
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
            # The on-device model's constrained decoding occasionally fails to
            # produce schema-conformant output on the first attempt (a known
            # characteristic of the smaller on-device model, not transient
            # infra flakiness) -- one retry clears the large majority of these.
            last_error: Optional[Exception] = None
            with _SESSION_LOCK:
                session = self._get_session(system_instructions)
                for _attempt in range(2):
                    try:
                        resp = session.generate(prompt, schema=schema)
                        return resp.content
                    except afm.DecodingFailureError as exc:
                        last_error = exc
            raise last_error

        return RunnableLambda(_invoke)
