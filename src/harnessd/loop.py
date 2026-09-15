from pydantic import BaseModel

from .providers.base import Provider
from .tools import ToolRegistry
from .trace import TraceLogger
from .types import Message, ToolResult


class RunResult(BaseModel):
    status: str                 # "done" | "max_steps" | "pending_approval"
    final_text: str = ""
    steps: int = 0
    messages: list[Message] = []
    pending: list[ToolResult] = []


class Agent:
    def __init__(self, provider: Provider, registry: ToolRegistry,
                 trace: TraceLogger | None = None, max_steps: int = 10) -> None:
        self.provider = provider
        self.registry = registry
        self.trace = trace or TraceLogger()
        self.max_steps = max_steps

    async def run(self, task: str) -> RunResult:
        messages = [Message(role="user", content=task)]
        self.trace.log("run_start", task=task)

        for step in range(1, self.max_steps + 1):
            resp = await self.provider.complete(messages, self.registry.specs())
            self.trace.log("llm_response", step=step, text=resp.text,
                           tool_calls=[c.model_dump() for c in resp.tool_calls],
                           input_tokens=resp.input_tokens, output_tokens=resp.output_tokens)
            messages.append(Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls))

            if not resp.tool_calls:
                self.trace.log("run_end", status="done", steps=step)
                return RunResult(status="done", final_text=resp.text, steps=step, messages=messages)

            pending: list[ToolResult] = []
            for call in resp.tool_calls:
                result = await self.registry.dispatch(call)
                self.trace.log("tool_result", step=step, call=call.model_dump(), result=result.model_dump())
                messages.append(Message(role="tool", tool_result=result))
                if result.status == "pending":
                    pending.append(result)

            if pending:  # Phase 2 will resume after approval
                self.trace.log("run_end", status="pending_approval", steps=step)
                return RunResult(status="pending_approval", steps=step, messages=messages, pending=pending)

        self.trace.log("run_end", status="max_steps", steps=self.max_steps)
        return RunResult(status="max_steps", steps=self.max_steps, messages=messages)
