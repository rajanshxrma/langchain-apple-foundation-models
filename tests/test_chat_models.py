"""Real, on-device tests -- no mocks. Requires macOS 26+ with Apple Intelligence enabled."""

import pytest
from langchain_core.tools import tool

from langchain_apple_foundation_models import ChatAppleFoundationModels


@pytest.fixture(autouse=True)
def _skip_if_unavailable():
    import applefoundationmodels as afm

    if not afm.apple_intelligence_available():
        pytest.skip("Apple Intelligence not available on this machine")


def test_basic_invoke():
    llm = ChatAppleFoundationModels()
    result = llm.invoke("Reply with exactly the word: pong")
    assert "pong" in result.content.lower()


def test_tool_calling():
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"The weather in {city} is sunny and 72F."

    llm = ChatAppleFoundationModels().bind_tools([get_weather])
    result = llm.invoke("What's the weather in Austin?")
    assert "austin" in result.content.lower()
    assert "sunny" in result.content.lower()


def test_structured_output():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "color": {"type": "string"},
        },
        "required": ["name", "color"],
    }
    chain = ChatAppleFoundationModels().with_structured_output(schema)
    result = chain.invoke("Give me a fruit and its typical color.")
    assert "name" in result
    assert "color" in result


def test_streaming():
    llm = ChatAppleFoundationModels()
    chunks = list(llm.stream("Count from 1 to 3."))
    assert len(chunks) > 1
    full = "".join(c.content for c in chunks)
    assert len(full) > 0
