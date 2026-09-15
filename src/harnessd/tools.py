import asyncio
import inspect
from typing import Any, Callable, Literal
from pydantic import BaseModel, ValidationError, create_model

from .types import ToolCall, ToolResult

Permission = Literal["auto", "confirm", "deny"]


class Tool(BaseModel):
    name: str
    description: str
    permission: Permission
    fn: Callable[..., Any]
    schema_model: type[BaseModel]

    model_config = {"arbitrary_types_allowed": True}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, permission: Permission = "confirm", description: str = ""):
        """Register a function as a tool. Default permission is 'confirm' (safe side)."""
        def decorator(fn: Callable[..., Any]):
            params = inspect.signature(fn).parameters
            fields = {
                n: (p.annotation, ... if p.default is inspect.Parameter.empty else p.default)
                for n, p in params.items()
            }
            schema = create_model(f"{fn.__name__}_args", **fields)
            self._tools[fn.__name__] = Tool(
                name=fn.__name__,
                description=description or (fn.__doc__ or "").strip(),
                permission=permission,
                fn=fn,
                schema_model=schema,
            )
            return fn
        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, Any]]:
        """Tool definitions to send to the LLM."""
        return [
            {"name": t.name, "description": t.description,
             "input_schema": t.schema_model.model_json_schema()}
            for t in self._tools.values()
        ]

    async def dispatch(self, call: ToolCall, timeout: float = 10.0) -> ToolResult:
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(call_id=call.id, status="error", error=f"unknown tool: {call.name}")

        try:
            args = tool.schema_model.model_validate(call.args)
        except ValidationError as e:
            return ToolResult(call_id=call.id, status="error", error=f"invalid args: {e.errors()}")

        # Phase 1: simple permission gate. Phase 2 replaces this with policy + approvals.
        if tool.permission == "deny":
            return ToolResult(call_id=call.id, status="denied", error="tool is denied by policy")
        if tool.permission == "confirm":
            return ToolResult(call_id=call.id, status="pending", error="requires human approval")

        try:
            kwargs = args.model_dump()
            if inspect.iscoroutinefunction(tool.fn):
                out = await asyncio.wait_for(tool.fn(**kwargs), timeout)
            else:
                out = await asyncio.wait_for(asyncio.to_thread(tool.fn, **kwargs), timeout)
            return ToolResult(call_id=call.id, status="ok", data=out)
        except asyncio.TimeoutError:
            return ToolResult(call_id=call.id, status="timeout", error=f"exceeded {timeout}s")
        except Exception as e:  # tool bugs never crash the loop
            return ToolResult(call_id=call.id, status="error", error=f"{type(e).__name__}: {e}")
