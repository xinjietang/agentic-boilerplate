"""Async agent loop implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_boilerplate.core.events import EventBus, EventType

if TYPE_CHECKING:
    from agentic_boilerplate.config.settings import AgentConfig
    from agentic_boilerplate.context.manager import ContextManager
    from agentic_boilerplate.policies.approval import ApprovalPolicy
    from agentic_boilerplate.skills.registry import SkillRegistry
    from agentic_boilerplate.tools.registry import ToolRegistry


class AgentLoop:
    """Core async agent loop that orchestrates message processing and tool execution."""

    def __init__(
        self,
        config: AgentConfig,
        context_manager: ContextManager,
        tool_registry: ToolRegistry,
        approval_policy: ApprovalPolicy,
        event_bus: EventBus,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self.config = config
        self.context = context_manager
        self.tools = tool_registry
        self.approval = approval_policy
        self.event_bus = event_bus
        self.skills = skill_registry

    async def run_turn(self, user_message: str) -> None:
        """Process one user turn end-to-end."""
        # 1. Emit USER_MESSAGE event
        await self.event_bus.emit_type(EventType.USER_MESSAGE, {"content": user_message})

        # 2. Add user message to context
        self.context.add_message("user", user_message)

        # 3. Call LLM (stub)
        messages = self.context.get_messages()
        llm_response = await self._call_llm(messages)

        # 4. Parse tool calls and execute them
        tool_calls = llm_response.get("tool_calls", [])
        tool_results: list[dict] = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_params = tool_call.get("parameters", {})
            call_id = tool_call.get("id", tool_name)

            # 5a. Check approval
            spec_and_handler = self.tools.get(tool_name)
            if spec_and_handler:
                spec, _ = spec_and_handler
                if self.approval.requires_approval(spec, tool_params):
                    await self.event_bus.emit_type(
                        EventType.STATUS,
                        {"message": f"Tool '{tool_name}' requires approval but running in non-interactive mode; skipping."},
                    )
                    continue

            # 5b. Determine if this call targets a registered skill
            is_skill = self.skills is not None and self.skills.get(tool_name) is not None

            # 5c. Emit SKILL_CALL or TOOL_CALL
            if is_skill:
                await self.event_bus.emit_type(
                    EventType.SKILL_CALL,
                    {"id": call_id, "name": tool_name, "input": tool_params.get("input", "")},
                )
            else:
                await self.event_bus.emit_type(
                    EventType.TOOL_CALL,
                    {"id": call_id, "name": tool_name, "parameters": tool_params},
                )

            # 5d. Execute tool (skills are registered as tools so this works for both)
            result_str = await self._execute_tool_call(tool_call)

            # 5e. Emit SKILL_RESULT or TOOL_RESULT
            if is_skill:
                await self.event_bus.emit_type(
                    EventType.SKILL_RESULT,
                    {"id": call_id, "name": tool_name, "result": result_str},
                )
            else:
                await self.event_bus.emit_type(
                    EventType.TOOL_RESULT,
                    {"id": call_id, "name": tool_name, "result": result_str},
                )

            tool_results.append({"id": call_id, "name": tool_name, "result": result_str})

        # Build final assistant content
        assistant_content = llm_response.get("content", "")
        if tool_results:
            result_lines = "\n".join(
                f"[{r['name']}] → {r['result']}" for r in tool_results
            )
            if assistant_content:
                assistant_content = f"{assistant_content}\n\nTool results:\n{result_lines}"
            else:
                assistant_content = f"Tool results:\n{result_lines}"

        # 6. Add assistant message to context
        self.context.add_message("assistant", assistant_content)

        # Emit ASSISTANT_MESSAGE
        await self.event_bus.emit_type(
            EventType.ASSISTANT_MESSAGE, {"content": assistant_content}
        )

        # 7. Emit TURN_COMPLETE
        await self.event_bus.emit_type(EventType.TURN_COMPLETE, {})

        # Check if compaction is needed
        if self.context.should_compact():
            removed = self.context.compact()
            await self.event_bus.emit_type(
                EventType.COMPACT, {"removed": removed}
            )

    async def _call_llm(self, messages: list[dict]) -> dict:
        """Stub LLM that returns mock responses based on message content."""
        # Find the last user message
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        lower = user_msg.lower()

        if "calc " in lower or "calculate" in lower:
            # Extract expression after "calc " or "calculate "
            for prefix in ("calculate ", "calc "):
                idx = lower.find(prefix)
                if idx != -1:
                    expression = user_msg[idx + len(prefix):].strip()
                    break
            else:
                expression = "1 + 1"

            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_calc_1",
                        "name": "calculator",
                        "parameters": {"expression": expression},
                    }
                ],
            }

        if "echo " in lower:
            idx = lower.find("echo ")
            text = user_msg[idx + 5:].strip()
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_echo_1",
                        "name": "echo",
                        "parameters": {"text": text},
                    }
                ],
            }

        return {
            "content": f"I received your message: {user_msg}",
            "tool_calls": [],
        }

    async def _execute_tool_call(self, tool_call: dict) -> str:
        """Execute a single tool call and return the result as a string."""
        tool_name = tool_call.get("name", "")
        tool_params = tool_call.get("parameters", {})

        result = await self.tools.execute(tool_name, tool_params)
        if result.success:
            return result.output
        return f"Error: {result.error}"
