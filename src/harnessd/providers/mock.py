from typing import Any
from ..types import Message, ProviderResponse


class MockProvider:
    """Returns scripted responses in order. Lets us test the harness without an LLM."""

    def __init__(self, script: list[ProviderResponse]) -> None:
        self.script = list(script)
        self.calls: list[list[Message]] = []

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ProviderResponse:
        self.calls.append(list(messages))
        if not self.script:
            return ProviderResponse(text="(mock script exhausted)")
        return self.script.pop(0)
