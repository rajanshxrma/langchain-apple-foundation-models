"""LangChain's standard test suite wired against this provider.

Unit tests run anywhere; integration tests exercise the real on-device
model and require macOS 26+ with Apple Intelligence enabled.
"""

from typing import Type

from langchain_tests.integration_tests import ChatModelIntegrationTests
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_apple_foundation_models import ChatAppleFoundationModels


class TestChatAppleFoundationModelsUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatAppleFoundationModels]:
        return ChatAppleFoundationModels


class TestChatAppleFoundationModelsIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatAppleFoundationModels]:
        return ChatAppleFoundationModels
