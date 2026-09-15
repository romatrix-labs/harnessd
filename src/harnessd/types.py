from typing import Any, Literal
from pydantic import BaseModel, Field
import uuid


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    args: dict[str, Any] = {}


class ToolResult(BaseModel):
    call_id: str
    status: Literal["ok", "denied", "pending", "timeout", "error"]
    data: Any = None
    error: str | None = None


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = []
    tool_result: ToolResult | None = None


class ProviderResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = []
    input_tokens: int = 0
    output_tokens: int = 0
