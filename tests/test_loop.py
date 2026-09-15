import asyncio

from harnessd import Agent, ProviderResponse, ToolCall, ToolRegistry, TraceLogger
from harnessd.providers import MockProvider


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(permission="auto")
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @reg.tool(permission="confirm")
    def file_delete(path: str) -> str:
        """Delete a file."""
        raise AssertionError("must never execute without approval")

    @reg.tool(permission="deny")
    def shell_exec(cmd: str) -> str:
        raise AssertionError("must never execute")

    @reg.tool(permission="auto")
    async def slow() -> str:
        await asyncio.sleep(5)
        return "late"

    @reg.tool(permission="auto")
    def boom() -> str:
        raise RuntimeError("tool crashed")

    return reg


def call(name, **args):
    return ProviderResponse(tool_calls=[ToolCall(name=name, args=args)])


async def test_auto_tool_runs_and_loop_finishes():
    provider = MockProvider([call("add", a=2, b=3), ProviderResponse(text="answer is 5")])
    trace = TraceLogger()
    result = await Agent(provider, make_registry(), trace).run("add 2 and 3")

    assert result.status == "done"
    assert result.final_text == "answer is 5"
    tool_msg = result.messages[2]
    assert tool_msg.tool_result.status == "ok" and tool_msg.tool_result.data == 5
    assert [e["event"] for e in trace.events] == ["run_start", "llm_response", "tool_result", "llm_response", "run_end"]


async def test_confirm_tool_is_held_not_executed():
    provider = MockProvider([call("file_delete", path="a.txt")])
    result = await Agent(provider, make_registry()).run("delete a.txt")
    assert result.status == "pending_approval"
    assert result.pending[0].status == "pending"


async def test_denied_tool_never_runs():
    provider = MockProvider([call("shell_exec", cmd="rm -rf /"), ProviderResponse(text="ok")])
    result = await Agent(provider, make_registry()).run("x")
    assert result.messages[2].tool_result.status == "denied"


async def test_unknown_tool_and_invalid_args_return_error():
    reg = make_registry()
    r1 = await reg.dispatch(ToolCall(name="nope"))
    r2 = await reg.dispatch(ToolCall(name="add", args={"a": "not-a-number", "b": 1}))
    assert r1.status == "error" and "unknown tool" in r1.error
    assert r2.status == "error" and "invalid args" in r2.error


async def test_timeout_and_crash_are_structured():
    reg = make_registry()
    assert (await reg.dispatch(ToolCall(name="slow"), timeout=0.1)).status == "timeout"
    crashed = await reg.dispatch(ToolCall(name="boom"))
    assert crashed.status == "error" and "RuntimeError" in crashed.error


async def test_max_steps_stops_infinite_loop():
    provider = MockProvider([call("add", a=1, b=1)] * 20)
    result = await Agent(provider, make_registry(), max_steps=3).run("loop forever")
    assert result.status == "max_steps" and result.steps == 3
