# Building Agents with agentic-boilerplate

A friendly, step-by-step guide for developers who want to create their own agents using this repo.

---

## What you get out of the box

| Building block | Where it lives | What it does |
|---|---|---|
| `AgentLoop` | `core/agent_loop.py` | Drives the turn loop: call LLM → dispatch tools → update context |
| `ToolRegistry` | `tools/registry.py` | Register any Python callable as a named tool |
| `SkillRegistry` | `skills/` | Register higher-level sub-agents as single-parameter skills |
| `ContextManager` | `context/manager.py` | Keeps the message history; auto-compacts when it gets long |
| `EventBus` | `core/events.py` | Pub/sub bus — subscribe to any agent event (tool calls, errors, etc.) |
| `ApprovalPolicy` | `policies/approval.py` | Prompt the user before dangerous tools run; disable with YOLO mode |
| `AgentConfig` | `config/settings.py` | Load config from YAML, JSON, or Python dataclasses |
| `CLIRenderer` + `InteractiveCLI` | `ui/cli.py` | Drop-in interactive REPL and `--headless` mode |

---

## The three-step recipe for any new agent

### Step 1 — Define your tools

A tool is a plain Python function plus a `ToolSpec` that describes it to the LLM.

```python
# my_agent/tools/my_tools.py
from agentic_boilerplate.tools.base import ToolResult, ToolSpec
from agentic_boilerplate.tools.registry import ToolRegistry

GREET_SPEC = ToolSpec(
    name="greet",
    description="Say hello to a person by name.",
    parameters={
        "name": {"type": "string", "description": "The person's name."},
    },
    requires_approval=False,   # set True for destructive actions
    tags=["demo"],
)

def greet(name: str) -> ToolResult:
    return ToolResult.ok(f"Hello, {name}! 👋")

def register_tools(registry: ToolRegistry) -> None:
    registry.register(GREET_SPEC, greet)
```

> **Tip:** `ToolResult.ok(output)` signals success; `ToolResult.fail(error)` signals failure.
> Both sync and `async` handlers are supported.

---

### Step 2 — Wire up the agent

```python
# my_agent/agent.py
import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agentic_boilerplate.config.settings import AgentConfig
from agentic_boilerplate.context.manager import ContextManager
from agentic_boilerplate.core.agent_loop import AgentLoop
from agentic_boilerplate.core.events import EventBus, EventType
from agentic_boilerplate.core.session import Session
from agentic_boilerplate.core.submission import SubmissionQueue
from agentic_boilerplate.policies.approval import ApprovalPolicy
from agentic_boilerplate.tools.registry import ToolRegistry
from agentic_boilerplate.ui.cli import CLIRenderer, InteractiveCLI

from tools.my_tools import register_tools   # your tools module

def build_agent():
    config = AgentConfig()
    config.system_prompt = "You are a friendly greeter."

    event_bus    = EventBus()
    context      = ContextManager(config.max_context_messages, config.compact_threshold)
    context.add_message("system", config.system_prompt)
    registry     = ToolRegistry()
    register_tools(registry)
    approval     = ApprovalPolicy(yolo_mode=config.yolo_mode)
    session      = Session(config=config, yolo_mode=config.yolo_mode)
    loop         = AgentLoop(config, context, registry, approval, event_bus)
    queue        = SubmissionQueue()
    return loop, queue, session

def main():
    loop, queue, session = build_agent()
    renderer = CLIRenderer()
    loop.event_bus.subscribe(renderer.handle_event)

    # headless single-shot
    async def _run():
        await loop.event_bus.emit_type(EventType.READY, {})
        await loop.run_turn("greet Alice")
    asyncio.run(_run())

if __name__ == "__main__":
    main()
```

---

### Step 3 — Add your LLM routing

The default `AgentLoop._call_llm()` is a stub.  Override it with your own logic — either a
real LLM call or keyword-based routing for zero-dependency demos:

```python
class MyAgentLoop(AgentLoop):
    async def _call_llm(self, messages: list[dict]) -> dict:
        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        if "greet" in user_msg.lower():
            # Extract the name after "greet "
            name = user_msg.split("greet", 1)[-1].strip() or "World"
            return {
                "content": f"Greeting {name}…",
                "tool_calls": [
                    {"id": "call_1", "name": "greet", "parameters": {"name": name}}
                ],
            }

        # Fallback — just echo
        return {"content": f"I got: {user_msg}", "tool_calls": []}
```

Replace the stub with a real LLM in production:

```python
async def _call_llm(self, messages: list[dict]) -> dict:
    import openai
    client = openai.AsyncOpenAI(api_key=self.config.llm.api_key)
    response = await client.chat.completions.create(
        model=self.config.llm.model,
        messages=messages,
        tools=self.tools.to_llm_schema(),
    )
    msg = response.choices[0].message
    tool_calls = [
        {"id": tc.id, "name": tc.function.name, "parameters": json.loads(tc.function.arguments)}
        for tc in (msg.tool_calls or [])
    ]
    return {"content": msg.content or "", "tool_calls": tool_calls}
```

---

## Adding a headless + interactive CLI

Just swap the last few lines of `main()`:

```python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", metavar="PROMPT")
    parser.add_argument("--config",   metavar="PATH")
    parser.add_argument("--yolo",     action="store_true")
    args = parser.parse_args()

    loop, queue, session = build_agent(args.config, args.yolo)
    renderer = CLIRenderer()
    loop.event_bus.subscribe(renderer.handle_event)

    if args.headless:
        async def _run():
            await loop.event_bus.emit_type(EventType.READY, {})
            await loop.run_turn(args.headless)
        asyncio.run(_run())
    else:
        cli = InteractiveCLI(renderer=renderer)
        asyncio.run(cli.run(loop, queue, session))
```

Run it:

```bash
python agent.py                         # interactive REPL
python agent.py --headless "greet Bob"  # single-shot
python agent.py --config config.yaml    # load custom config
python agent.py --yolo                  # skip all approval prompts
```

---

## Composing sub-agents with Skills

A **Skill** wraps a whole `AgentLoop` as a single callable exposed to the parent LLM.
Great for hierarchical or specialist agents:

```python
from agentic_boilerplate.skills import AgentSkill, SkillRegistry, SkillSpec

sub_loop  = build_specialist_loop()   # any AgentLoop
skill     = AgentSkill(sub_loop)

skill_reg = SkillRegistry()
skill_reg.register(
    SkillSpec(name="specialist", description="Delegate to specialist agent",
              input_description="Task for the specialist"),
    skill,
)
skill_reg.register_in_tool_registry(tool_registry)
# Pass skill_reg as the fifth positional arg to AgentLoop
loop = AgentLoop(config, context, tool_registry, approval, event_bus, skill_reg)
```

---

## Useful config knobs (`config.yaml`)

```yaml
llm:
  provider: stub          # swap to: openai | anthropic | etc.
  model: stub-model
  api_key: null           # or set via OPENAI_API_KEY env var
  max_tokens: 4096
  temperature: 0.7

max_context_messages: 100  # hard limit on history
compact_threshold: 80      # auto-compact above this many messages
yolo_mode: false           # true = skip all approval prompts
system_prompt: "You are a helpful assistant."
```

---

## Complete working examples

| Example | What it shows |
|---|---|
| [`examples/daily_news_curator/`](examples/daily_news_curator/) | Fetches, filters, and formats a morning news digest |
| [`examples/daily_calendar/`](examples/daily_calendar/) | Manages an in-memory calendar and finds free slots |
| [`examples/research_assistant/`](examples/research_assistant/) | Searches, takes notes, compiles reports |
| [`examples/machine_learning_agent/`](examples/machine_learning_agent/) | Designs models, runs training, analyses results, proposes improvements |

Each example is self-contained, requires no API key, and runs immediately:

```bash
pip install -e ".[dev]"

cd examples/machine_learning_agent
python agent.py --headless "design a CNN for image classification"
python agent.py --headless "setup training with lr=0.001 epochs=10"
python agent.py --headless "run training"
python agent.py --headless "analyze results"
python agent.py --headless "propose improvements"
```

---

## Checklist for a production-ready agent

- [ ] Replace `_call_llm()` with a real LLM provider (OpenAI, Anthropic, Ollama, …)
- [ ] Set `requires_approval=True` on any destructive or irreversible tool
- [ ] Load secrets from environment variables, not hard-coded values
- [ ] Replace in-memory stores with a real database or vector store
- [ ] Add structured logging by subscribing an additional handler to `EventBus`
- [ ] Write tests in `tests/` using `pytest-asyncio` and the stub LLM

---

## Running the test suite

```bash
pytest
```
