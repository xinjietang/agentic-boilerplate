"""Machine Learning Agent — example agent built on agentic-boilerplate.

This agent helps users design neural network architectures, configure training
pipelines, run simulated training experiments, analyse results, and propose
data-driven improvements.

It extends AgentLoop with a stub LLM that routes natural-language ML requests
to the six ML tools.  Replace _call_llm() with a real LLM call (OpenAI,
Anthropic, etc.) for production use.

Usage
-----
# Interactive REPL
python agent.py

# Headless single-shot examples
python agent.py --headless "design a CNN for image classification"
python agent.py --headless "setup training with lr=0.001 epochs=20"
python agent.py --headless "run training"
python agent.py --headless "analyze results"
python agent.py --headless "propose improvements"
python agent.py --headless "list experiments"

# End-to-end workflow
python agent.py --headless "design a transformer for nlp"
python agent.py --headless "setup training lr=0.0001 epochs=15 dropout=0.1"
python agent.py --headless "run training"
python agent.py --headless "analyze"
python agent.py --headless "propose architecture changes"

# With a custom config
python agent.py --config config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys

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

from tools.ml_tools import register_ml_tools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_float(text: str, key: str, default: float) -> float:
    """Parse 'key=0.001' or 'key 0.001' from *text*."""
    m = re.search(rf"{key}[=\s]+([0-9.eE+\-]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return default


def _extract_int(text: str, key: str, default: int) -> int:
    """Parse 'key=32' or 'key 32' from *text*."""
    m = re.search(rf"{key}[=\s]+([0-9]+)", text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return default


def _extract_str(text: str, key: str, choices: list[str], default: str) -> str:
    """Parse 'key=value' or 'key value' from *text*, constrained to *choices*."""
    m = re.search(rf"{key}[=\s]+(\w+)", text, re.IGNORECASE)
    if m:
        val = m.group(1).lower()
        if val in choices:
            return val
    return default


# ---------------------------------------------------------------------------
# Extended agent loop with ML-aware stub LLM
# ---------------------------------------------------------------------------

class MLAgentLoop(AgentLoop):
    """AgentLoop subclass with a stub LLM that routes ML experiment requests."""

    async def _call_llm(self, messages: list[dict]) -> dict:  # type: ignore[override]
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        lower = user_msg.lower()

        # ── list experiments ─────────────────────────────────────────────────
        if any(w in lower for w in ("list experiments", "show experiments", "all experiments", "experiment history", "list runs")):
            return {
                "content": "Listing all recorded experiments…",
                "tool_calls": [
                    {"id": "call_list_1", "name": "list_experiments", "parameters": {}}
                ],
            }

        # ── design model ─────────────────────────────────────────────────────
        if any(w in lower for w in ("design", "architecture", "create model", "build model", "new model")):
            # Detect model type
            model_type = "mlp"
            for mt in ("cnn", "rnn", "transformer", "resnet", "mlp"):
                if mt in lower:
                    model_type = mt
                    break

            # Detect task
            task = "image_classification"
            task_map = {
                "image classif": "image_classification",
                "image classification": "image_classification",
                "object detection": "object_detection",
                "regression": "regression",
                "nlp": "nlp",
                "text": "nlp",
                "time series": "time_series",
                "time_series": "time_series",
            }
            for kw, task_val in task_map.items():
                if kw in lower:
                    task = task_val
                    break

            hidden_layers = _extract_int(user_msg, r"(?:layers?|depth)", 3)
            units = _extract_int(user_msg, r"units?", 128)

            # Extract notes (anything after a dash or colon that isn't a param)
            notes_m = re.search(r"(?:note[s]?|notes?)[:\s]+(.+)", user_msg, re.IGNORECASE)
            notes = notes_m.group(1).strip() if notes_m else ""

            return {
                "content": f"Designing a {model_type} model for {task}…",
                "tool_calls": [
                    {
                        "id": "call_design_1",
                        "name": "design_model",
                        "parameters": {
                            "model_type": model_type,
                            "task": task,
                            "hidden_layers": hidden_layers,
                            "units_per_layer": units,
                            "notes": notes,
                        },
                    }
                ],
            }

        # ── setup training ───────────────────────────────────────────────────
        if any(w in lower for w in ("setup", "configure", "config training", "set training", "hyperparameter", "set lr", "set learning")):
            lr       = _extract_float(user_msg, r"(?:lr|learning.?rate)", 1e-3)
            bs       = _extract_int(user_msg, r"(?:batch.?size|bs)", 32)
            epochs   = _extract_int(user_msg, r"epochs?", 10)
            opt      = _extract_str(user_msg, r"(?:optimizer?|opt)", ["adam", "sgd", "rmsprop", "adamw"], "adam")
            wd       = _extract_float(user_msg, r"(?:weight.?decay|wd|l2)", 0.0)
            dropout  = _extract_float(user_msg, r"dropout", 0.0)
            sched    = _extract_str(user_msg, r"sched(?:uler)?", ["none", "step", "cosine"], "none")

            return {
                "content": f"Configuring training: lr={lr}, epochs={epochs}, batch={bs}…",
                "tool_calls": [
                    {
                        "id": "call_setup_1",
                        "name": "setup_training",
                        "parameters": {
                            "learning_rate": lr,
                            "batch_size": bs,
                            "epochs": epochs,
                            "optimizer": opt,
                            "weight_decay": wd,
                            "dropout": dropout,
                            "scheduler": sched,
                        },
                    }
                ],
            }

        # ── run training ─────────────────────────────────────────────────────
        if any(w in lower for w in ("run training", "train", "start training", "fit", "run experiment")):
            notes_m = re.search(r"(?:note[s]?|notes?)[:\s]+(.+)", user_msg, re.IGNORECASE)
            notes = notes_m.group(1).strip() if notes_m else ""
            return {
                "content": "Starting training run…",
                "tool_calls": [
                    {
                        "id": "call_train_1",
                        "name": "run_training",
                        "parameters": {"notes": notes},
                    }
                ],
            }

        # ── analyze results ──────────────────────────────────────────────────
        if any(w in lower for w in ("analyze", "analyse", "analysis", "results", "metrics", "inspect", "evaluate")):
            exp_m = re.search(r"exp_\d+", lower)
            exp_id = exp_m.group(0) if exp_m else ""
            return {
                "content": f"Analysing experiment results{' for ' + exp_id if exp_id else ''}…",
                "tool_calls": [
                    {
                        "id": "call_analyze_1",
                        "name": "analyze_results",
                        "parameters": {"experiment_id": exp_id},
                    }
                ],
            }

        # ── propose changes ──────────────────────────────────────────────────
        if any(w in lower for w in ("propose", "improve", "suggest", "recommendation", "next step", "what should", "changes")):
            focus_map = {
                "architecture": "architecture",
                "arch": "architecture",
                "hyper": "hyperparameters",
                "hyperparameter": "hyperparameters",
                "regularisation": "regularisation",
                "regularization": "regularisation",
                "data": "data",
            }
            focus = "all"
            for kw, fval in focus_map.items():
                if kw in lower:
                    focus = fval
                    break
            return {
                "content": f"Generating improvement proposals (focus: {focus})…",
                "tool_calls": [
                    {
                        "id": "call_propose_1",
                        "name": "propose_changes",
                        "parameters": {"focus": focus},
                    }
                ],
            }

        # ── help ─────────────────────────────────────────────────────────────
        if any(w in lower for w in ("help", "what can", "commands", "how do")):
            return {
                "content": (
                    "I'm your ML Agent! Here's what I can do:\n\n"
                    "  🏗  Design a model:\n"
                    "       'design a CNN for image classification'\n"
                    "       'design a transformer for nlp with 4 layers'\n\n"
                    "  ⚙   Configure training:\n"
                    "       'setup training lr=0.001 epochs=20 dropout=0.3'\n\n"
                    "  🚀  Run an experiment:\n"
                    "       'run training'\n\n"
                    "  📊  Analyse results:\n"
                    "       'analyze results'\n"
                    "       'analyze exp_0002'\n\n"
                    "  💡  Get improvements:\n"
                    "       'propose changes'\n"
                    "       'suggest architecture improvements'\n\n"
                    "  📋  Track experiments:\n"
                    "       'list experiments'\n"
                ),
                "tool_calls": [],
            }

        # ── fallback ─────────────────────────────────────────────────────────
        return {
            "content": (
                f"I received: '{user_msg}'\n\n"
                "Try one of:\n"
                "  'design a CNN for image classification'\n"
                "  'setup training lr=0.001 epochs=20'\n"
                "  'run training'\n"
                "  'analyze results'\n"
                "  'propose improvements'\n"
                "  'list experiments'"
            ),
            "tool_calls": [],
        }


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent(config_path: str | None = None, yolo: bool = False):
    if config_path:
        if config_path.endswith(".json"):
            config = AgentConfig.from_json(config_path)
        else:
            config = AgentConfig.from_yaml(config_path)
    else:
        config = AgentConfig()
        config.system_prompt = (
            "You are an expert Machine Learning Engineer Agent.  "
            "Help the user design neural network architectures, configure training "
            "pipelines, run training experiments, analyse results (loss curves, "
            "accuracy, overfitting signals), and propose concrete improvements."
        )

    if yolo:
        config.yolo_mode = True

    event_bus = EventBus()
    context = ContextManager(
        max_messages=config.max_context_messages,
        compact_threshold=config.compact_threshold,
    )
    context.add_message("system", config.system_prompt)

    registry = ToolRegistry()
    register_ml_tools(registry)

    approval = ApprovalPolicy(yolo_mode=config.yolo_mode)
    session = Session(config=config, yolo_mode=config.yolo_mode)
    loop = MLAgentLoop(config, context, registry, approval, event_bus)
    queue = SubmissionQueue()
    return loop, queue, session


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ml-agent",
        description="Machine Learning Agent — agentic-boilerplate example",
    )
    parser.add_argument("--headless", metavar="PROMPT", help="Run a single prompt and exit")
    parser.add_argument("--config", metavar="PATH", help="Path to YAML/JSON config file")
    parser.add_argument("--yolo", action="store_true", help="Skip approval prompts")
    args = parser.parse_args()

    agent_loop, submission_queue, session = build_agent(args.config, args.yolo)
    renderer = CLIRenderer()
    agent_loop.event_bus.subscribe(renderer.handle_event)

    if args.headless:
        async def _run():
            await agent_loop.event_bus.emit_type(EventType.READY, {})
            await agent_loop.run_turn(args.headless)
        asyncio.run(_run())
    else:
        cli = InteractiveCLI(renderer=renderer)
        asyncio.run(cli.run(agent_loop, submission_queue, session))


if __name__ == "__main__":
    main()
