from typing import Any, Protocol
from ..types import Message, ProviderResponse


class Provider(Protocol):
    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ProviderResponse: ...
