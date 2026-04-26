"""Tools for the Machine Learning Agent.

All computation is simulated in-process so the example runs without any
ML framework dependency.  Replace the stub implementations with real
training loops (PyTorch, TensorFlow, scikit-learn, etc.) for production use.

Available tools
---------------
design_model      – Specify a model architecture and store it as the active design.
setup_training    – Configure the training hyperparameters for the active model.
run_training      – Simulate a training run and record metrics in the experiment log.
analyze_results   – Inspect the latest (or a named) experiment for patterns such as
                    overfitting, underfitting, and convergence issues.
propose_changes   – Generate a list of concrete improvement suggestions based on the
                    most recent analysis.
list_experiments  – Show all experiments recorded so far with key metrics.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from agentic_boilerplate.tools.base import ToolResult, ToolSpec
from agentic_boilerplate.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Shared in-process state
# ---------------------------------------------------------------------------

@dataclass
class ModelDesign:
    model_type: str          # e.g. "cnn", "mlp", "rnn", "transformer"
    task: str                # e.g. "image_classification", "regression", "nlp"
    layers: list[dict]       # list of {type, units/filters, activation, …}
    params_estimate: int     # rough total parameter count
    notes: str = ""


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    optimizer: str = "adam"
    weight_decay: float = 0.0
    dropout: float = 0.0
    scheduler: str = "none"   # "none" | "step" | "cosine"


@dataclass
class ExperimentResult:
    experiment_id: str
    model_type: str
    task: str
    config: TrainingConfig
    train_losses: list[float]
    val_losses: list[float]
    train_accuracies: list[float]
    val_accuracies: list[float]
    final_train_loss: float
    final_val_loss: float
    final_train_acc: float
    final_val_acc: float
    epochs_run: int
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    notes: str = ""


# Mutable singletons representing current session state
_active_design: ModelDesign | None = None
_active_config: TrainingConfig | None = None
_experiments: dict[str, ExperimentResult] = {}
_exp_counter: int = 1
_last_analysis: dict[str, Any] = {}


def _next_exp_id() -> str:
    global _exp_counter
    eid = f"exp_{_exp_counter:04d}"
    _exp_counter += 1
    return eid


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulate_training(
    model_type: str,
    task: str,
    cfg: TrainingConfig,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Generate plausible-looking loss/accuracy curves without any real training."""

    rng = random.Random(42 + cfg.epochs + int(cfg.learning_rate * 1e6))

    # Base difficulty: affects how quickly the model converges
    difficulty = {
        "image_classification": 0.6,
        "object_detection": 0.8,
        "regression": 0.4,
        "nlp": 0.7,
        "time_series": 0.5,
    }.get(task.lower().replace(" ", "_"), 0.65)

    # Architecture quality score (rough heuristic)
    arch_score = {
        "cnn": 0.85 if "class" in task.lower() else 0.55,
        "mlp": 0.65,
        "rnn": 0.70 if "nlp" in task.lower() or "time" in task.lower() else 0.55,
        "transformer": 0.90,
        "resnet": 0.92,
    }.get(model_type.lower(), 0.70)

    # LR quality: penalise values that are too high or too low
    lr = cfg.learning_rate
    lr_score = 1.0 - abs(math.log10(lr) + 3) * 0.15   # sweet-spot around 1e-3
    lr_score = max(0.4, min(1.0, lr_score))

    # Overfitting tendency: high dropout helps; high capacity or many epochs hurts
    overfit_factor = max(0.0, (cfg.epochs / 20) - cfg.dropout * 0.5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    init_loss = 2.5 + rng.uniform(-0.3, 0.3)
    best_train = init_loss * (1 - arch_score * lr_score * 0.8) * difficulty
    best_val = best_train * (1 + overfit_factor * 0.3)

    for epoch in range(1, cfg.epochs + 1):
        progress = epoch / cfg.epochs
        decay = math.exp(-3 * progress)

        t_loss = best_train + (init_loss - best_train) * decay + rng.uniform(-0.02, 0.02)
        v_loss = best_val + (init_loss - best_val) * decay * 0.9 + rng.uniform(-0.03, 0.04)
        v_loss += overfit_factor * progress * 0.1   # diverge over time when overfitting

        t_acc = min(0.99, (1 - t_loss / init_loss) * arch_score + rng.uniform(-0.01, 0.01))
        v_acc = min(0.99, (1 - v_loss / init_loss) * arch_score * 0.97 + rng.uniform(-0.01, 0.01))

        train_losses.append(round(max(0.01, t_loss), 4))
        val_losses.append(round(max(0.01, v_loss), 4))
        train_accs.append(round(max(0.0, min(1.0, t_acc)), 4))
        val_accs.append(round(max(0.0, min(1.0, v_acc)), 4))

    return train_losses, val_losses, train_accs, val_accs


# ---------------------------------------------------------------------------
# Tool: design_model
# ---------------------------------------------------------------------------

DESIGN_MODEL_SPEC = ToolSpec(
    name="design_model",
    description=(
        "Design a neural network architecture for a given task.  "
        "Specify the model type (cnn, mlp, rnn, transformer, resnet) and "
        "the number of hidden layers/units.  The design is stored as the "
        "active model for subsequent training."
    ),
    parameters={
        "model_type": {
            "type": "string",
            "description": (
                "Architecture family: 'cnn', 'mlp', 'rnn', 'transformer', 'resnet'."
            ),
        },
        "task": {
            "type": "string",
            "description": (
                "Learning task: 'image_classification', 'regression', "
                "'nlp', 'object_detection', 'time_series'."
            ),
        },
        "hidden_layers": {
            "type": "integer",
            "description": "Number of hidden layers (default: 3).",
        },
        "units_per_layer": {
            "type": "integer",
            "description": "Units (or filters) per hidden layer (default: 128).",
        },
        "notes": {
            "type": "string",
            "description": "Optional free-text notes about the design decision.",
        },
    },
    requires_approval=False,
    tags=["ml", "design"],
)


def design_model(
    model_type: str,
    task: str,
    hidden_layers: int = 3,
    units_per_layer: int = 128,
    notes: str = "",
) -> ToolResult:
    global _active_design

    mt = model_type.lower().strip()
    layer_type_map = {
        "cnn": "Conv2D",
        "mlp": "Dense",
        "rnn": "LSTM",
        "transformer": "TransformerBlock",
        "resnet": "ResidualBlock",
    }
    layer_type = layer_type_map.get(mt, "Dense")

    # Build a simple layer stack
    layers: list[dict] = [{"type": "Input", "shape": "auto"}]
    for i in range(1, hidden_layers + 1):
        layer: dict = {
            "type": layer_type,
            "units": units_per_layer,
            "activation": "relu" if i < hidden_layers else "softmax",
        }
        if mt == "cnn":
            layer = {"type": "Conv2D", "filters": units_per_layer, "kernel_size": 3, "activation": "relu"}
        elif mt == "rnn":
            layer = {"type": "LSTM", "units": units_per_layer, "return_sequences": i < hidden_layers}
        elif mt == "transformer":
            layer = {"type": "TransformerBlock", "d_model": units_per_layer, "num_heads": 4, "ff_dim": units_per_layer * 4}
        layers.append(layer)

    if mt == "cnn":
        layers.append({"type": "GlobalAveragePooling2D"})

    # Determine output head
    output_units = 10 if "classif" in task.lower() else 1
    output_activation = "softmax" if "classif" in task.lower() else "linear"
    layers.append({"type": "Dense", "units": output_units, "activation": output_activation})

    params_estimate = sum(
        units_per_layer * units_per_layer
        for _ in range(hidden_layers - 1)
    ) + units_per_layer * output_units + units_per_layer  # rough estimate

    _active_design = ModelDesign(
        model_type=mt,
        task=task,
        layers=layers,
        params_estimate=params_estimate,
        notes=notes,
    )

    summary = json.dumps(
        {
            "model_type": mt,
            "task": task,
            "num_layers": len(layers),
            "params_estimate": params_estimate,
            "layers": layers,
            "notes": notes,
        },
        indent=2,
    )
    return ToolResult.ok(
        f"Model design saved.\n\n{summary}\n\n"
        "Next: use 'setup_training' to configure hyperparameters."
    )


# ---------------------------------------------------------------------------
# Tool: setup_training
# ---------------------------------------------------------------------------

SETUP_TRAINING_SPEC = ToolSpec(
    name="setup_training",
    description=(
        "Configure the training hyperparameters for the active model design. "
        "Settings are stored and applied on the next 'run_training' call."
    ),
    parameters={
        "learning_rate": {
            "type": "number",
            "description": "Learning rate (default: 0.001).",
        },
        "batch_size": {
            "type": "integer",
            "description": "Batch size (default: 32).",
        },
        "epochs": {
            "type": "integer",
            "description": "Number of training epochs (default: 10).",
        },
        "optimizer": {
            "type": "string",
            "description": "Optimiser: 'adam', 'sgd', 'rmsprop', 'adamw' (default: 'adam').",
        },
        "weight_decay": {
            "type": "number",
            "description": "L2 regularisation weight decay (default: 0.0).",
        },
        "dropout": {
            "type": "number",
            "description": "Dropout rate applied after hidden layers, 0–1 (default: 0.0).",
        },
        "scheduler": {
            "type": "string",
            "description": "LR scheduler: 'none', 'step', 'cosine' (default: 'none').",
        },
    },
    requires_approval=False,
    tags=["ml", "training"],
)


def setup_training(
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 10,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    dropout: float = 0.0,
    scheduler: str = "none",
) -> ToolResult:
    global _active_config

    _active_config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        optimizer=optimizer,
        weight_decay=weight_decay,
        dropout=dropout,
        scheduler=scheduler,
    )

    summary = json.dumps(asdict(_active_config), indent=2)
    model_info = (
        f"Active model: {_active_design.model_type} / {_active_design.task}"
        if _active_design
        else "No model designed yet — run 'design_model' first."
    )
    return ToolResult.ok(
        f"Training config saved.\n{model_info}\n\n{summary}\n\n"
        "Next: use 'run_training' to start the experiment."
    )


# ---------------------------------------------------------------------------
# Tool: run_training
# ---------------------------------------------------------------------------

RUN_TRAINING_SPEC = ToolSpec(
    name="run_training",
    description=(
        "Run a training experiment using the active model design and training config. "
        "Simulates epoch-by-epoch training and records loss/accuracy curves in the "
        "experiment log."
    ),
    parameters={
        "notes": {
            "type": "string",
            "description": "Optional notes to attach to this experiment run.",
        }
    },
    requires_approval=False,
    tags=["ml", "training"],
)


def run_training(notes: str = "") -> ToolResult:
    global _active_design, _active_config, _experiments

    if _active_design is None:
        return ToolResult.fail(
            "No model design found.  Run 'design_model' first."
        )

    cfg = _active_config if _active_config is not None else TrainingConfig()

    train_losses, val_losses, train_accs, val_accs = _simulate_training(
        _active_design.model_type, _active_design.task, cfg
    )

    exp_id = _next_exp_id()
    result = ExperimentResult(
        experiment_id=exp_id,
        model_type=_active_design.model_type,
        task=_active_design.task,
        config=cfg,
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accs,
        val_accuracies=val_accs,
        final_train_loss=train_losses[-1],
        final_val_loss=val_losses[-1],
        final_train_acc=train_accs[-1],
        final_val_acc=val_accs[-1],
        epochs_run=cfg.epochs,
        notes=notes,
    )
    _experiments[exp_id] = result

    # Compact curves: show every epoch up to 5, then sample
    display_epochs = list(range(1, cfg.epochs + 1))
    if cfg.epochs > 10:
        step = cfg.epochs // 10
        display_epochs = list(range(1, cfg.epochs + 1, step))
        if display_epochs[-1] != cfg.epochs:
            display_epochs.append(cfg.epochs)

    curve_lines = ["Epoch | Train Loss | Val Loss | Train Acc | Val Acc"]
    curve_lines.append("------|------------|----------|-----------|--------")
    for e in display_epochs:
        i = e - 1
        curve_lines.append(
            f"{e:5d} | {train_losses[i]:10.4f} | {val_losses[i]:8.4f} | "
            f"{train_accs[i]:9.4f} | {val_accs[i]:7.4f}"
        )

    return ToolResult.ok(
        f"Experiment {exp_id} complete ({cfg.epochs} epochs).\n\n"
        + "\n".join(curve_lines)
        + f"\n\nFinal — train loss: {result.final_train_loss:.4f}, "
        f"val loss: {result.final_val_loss:.4f}, "
        f"train acc: {result.final_train_acc:.4f}, "
        f"val acc: {result.final_val_acc:.4f}\n\n"
        "Next: use 'analyze_results' to interpret the curves."
    )


# ---------------------------------------------------------------------------
# Tool: analyze_results
# ---------------------------------------------------------------------------

ANALYZE_RESULTS_SPEC = ToolSpec(
    name="analyze_results",
    description=(
        "Analyse training results from the latest (or a specified) experiment. "
        "Detects overfitting, underfitting, learning-rate issues, and unstable training."
    ),
    parameters={
        "experiment_id": {
            "type": "string",
            "description": (
                "ID of the experiment to analyse (e.g. 'exp_0001'). "
                "Defaults to the most recent experiment."
            ),
        }
    },
    requires_approval=False,
    tags=["ml", "analysis"],
)


def analyze_results(experiment_id: str = "") -> ToolResult:
    global _last_analysis

    if not _experiments:
        return ToolResult.fail("No experiments found.  Run 'run_training' first.")

    if experiment_id:
        exp = _experiments.get(experiment_id)
        if exp is None:
            return ToolResult.fail(
                f"Experiment '{experiment_id}' not found.  "
                f"Available: {', '.join(_experiments.keys())}"
            )
    else:
        exp = list(_experiments.values())[-1]

    # ── Diagnostics ──────────────────────────────────────────────────────────

    issues: list[str] = []
    suggestions: list[str] = []

    train_loss = exp.train_losses
    val_loss = exp.val_losses
    train_acc = exp.train_accuracies
    val_acc = exp.val_accuracies
    n = len(train_loss)

    # 1. Overfitting: val loss diverges from train loss in the second half
    mid = n // 2
    val_trend_second_half = val_loss[-1] - val_loss[mid]
    train_trend_second_half = train_loss[-1] - train_loss[mid]
    overfit_gap = val_loss[-1] - train_loss[-1]

    overfitting = overfit_gap > 0.15 and val_trend_second_half > 0.05
    if overfitting:
        issues.append(
            f"Overfitting detected: val loss is {overfit_gap:.3f} above train loss "
            f"and rising in the second half (+{val_trend_second_half:.3f})."
        )
        suggestions.append("Increase dropout rate (try 0.3–0.5).")
        suggestions.append("Add L2 weight decay (try 1e-4 to 1e-3).")
        suggestions.append("Reduce model capacity (fewer layers / units).")
        suggestions.append("Augment training data or add early stopping.")

    # 2. Underfitting: train loss still high at end
    underfitting = train_loss[-1] > 1.0
    if underfitting:
        issues.append(
            f"Underfitting detected: final train loss {train_loss[-1]:.3f} is still high."
        )
        suggestions.append("Increase model capacity (more layers / units).")
        suggestions.append("Train for more epochs.")
        suggestions.append("Try a higher learning rate or a more powerful optimiser.")

    # 3. Learning rate too high: loss oscillates or NaN-like spikes
    loss_diffs = [abs(train_loss[i] - train_loss[i - 1]) for i in range(1, n)]
    lr_instability = max(loss_diffs) > 0.5 if loss_diffs else False
    if lr_instability:
        issues.append(
            f"Unstable training: max loss jump = {max(loss_diffs):.3f}. "
            "Learning rate may be too high."
        )
        suggestions.append("Reduce the learning rate by 5–10× (e.g., try 1e-4).")
        suggestions.append("Use a warmup schedule or gradient clipping.")

    # 4. Slow convergence: loss still dropping at the end
    still_converging = (train_loss[mid] - train_loss[-1]) > 0.5 * (train_loss[0] - train_loss[-1])
    if still_converging and not underfitting:
        issues.append("Loss is still dropping at the final epoch — more epochs may help.")
        suggestions.append("Increase the number of epochs.")
        suggestions.append("Use a cosine or step LR scheduler to squeeze out more gains.")

    # 5. All good
    if not issues:
        issues.append("No major issues detected.")
        suggestions.append(
            "The model is converging well.  Consider fine-tuning with a lower LR "
            "or experimenting with a more powerful architecture."
        )

    # ── Acc gap ───────────────────────────────────────────────────────────
    acc_gap = train_acc[-1] - val_acc[-1]
    acc_gap_note = (
        f"Accuracy gap (train - val): {acc_gap:.3f} "
        + ("(significant — model may be memorising training data)" if acc_gap > 0.10 else "(acceptable)")
    )

    _last_analysis = {
        "experiment_id": exp.experiment_id,
        "issues": issues,
        "suggestions": suggestions,
        "overfit_gap": round(overfit_gap, 4),
        "acc_gap": round(acc_gap, 4),
        "final_train_loss": exp.final_train_loss,
        "final_val_loss": exp.final_val_loss,
        "final_train_acc": exp.final_train_acc,
        "final_val_acc": exp.final_val_acc,
        "config_used": asdict(exp.config),
    }

    lines = [
        f"Analysis of {exp.experiment_id}  ({exp.model_type} / {exp.task})",
        "=" * 60,
        "",
        "Results",
        f"  Final train loss : {exp.final_train_loss:.4f}",
        f"  Final val loss   : {exp.final_val_loss:.4f}",
        f"  Final train acc  : {exp.final_train_acc:.4f}",
        f"  Final val acc    : {exp.final_val_acc:.4f}",
        f"  {acc_gap_note}",
        "",
        "Issues found",
    ]
    for issue in issues:
        lines.append(f"  ⚠  {issue}")
    lines += ["", "Initial suggestions"]
    for sug in suggestions:
        lines.append(f"  →  {sug}")
    lines += [
        "",
        "Run 'propose_changes' for a full, prioritised improvement plan.",
    ]

    return ToolResult.ok("\n".join(lines))


# ---------------------------------------------------------------------------
# Tool: propose_changes
# ---------------------------------------------------------------------------

PROPOSE_CHANGES_SPEC = ToolSpec(
    name="propose_changes",
    description=(
        "Generate a prioritised list of concrete, actionable improvements based on "
        "the most recent analysis.  Covers architecture changes, hyperparameter "
        "tuning, regularisation, data strategy, and next experiments to run."
    ),
    parameters={
        "focus": {
            "type": "string",
            "description": (
                "Optional focus area: 'architecture', 'hyperparameters', "
                "'regularisation', 'data', 'all' (default: 'all')."
            ),
        }
    },
    requires_approval=False,
    tags=["ml", "improvement"],
)


def propose_changes(focus: str = "all") -> ToolResult:
    if not _last_analysis:
        return ToolResult.fail(
            "No analysis available.  Run 'analyze_results' first."
        )

    exp_id = _last_analysis["experiment_id"]
    issues = _last_analysis["issues"]
    cfg = _last_analysis["config_used"]
    overfit_gap = _last_analysis["overfit_gap"]
    acc_gap = _last_analysis["acc_gap"]
    f_train_loss = _last_analysis["final_train_loss"]
    f_val_loss = _last_analysis["final_val_loss"]

    focus_lower = focus.lower().strip()

    proposals: list[dict] = []

    # ── Architecture ─────────────────────────────────────────────────────────
    if focus_lower in ("all", "architecture"):
        if _active_design:
            mt = _active_design.model_type
            task = _active_design.task
            arch_proposals = []

            if overfit_gap > 0.15:
                arch_proposals.append(
                    "Reduce depth by 1–2 layers to lower model capacity."
                )
                arch_proposals.append(
                    "Add BatchNormalization layers after each hidden layer "
                    "to stabilise activations."
                )
            if f_train_loss > 1.0:
                arch_proposals.append(
                    "Increase model width (double units per layer) to improve "
                    "representational capacity."
                )
            if mt == "mlp" and "image" in task.lower():
                arch_proposals.append(
                    "Switch from MLP to CNN or ResNet — MLPs rarely work well "
                    "for image tasks."
                )
            if mt == "rnn" and "nlp" in task.lower():
                arch_proposals.append(
                    "Consider upgrading from RNN/LSTM to a Transformer (e.g. "
                    "DistilBERT) for better long-range dependency modelling."
                )

            if arch_proposals:
                proposals.append(
                    {
                        "category": "Architecture",
                        "priority": "High" if overfit_gap > 0.2 or f_train_loss > 1.5 else "Medium",
                        "changes": arch_proposals,
                    }
                )

    # ── Hyperparameters ──────────────────────────────────────────────────────
    if focus_lower in ("all", "hyperparameters"):
        hp_proposals = []

        lr = cfg["learning_rate"]
        if lr > 1e-2:
            hp_proposals.append(f"Reduce learning rate from {lr} to 1e-3 or 1e-4.")
        elif lr < 1e-5:
            hp_proposals.append(f"Increase learning rate from {lr} — it may be too small to escape local minima.")

        if cfg["epochs"] < 20 and f_train_loss > 0.5:
            hp_proposals.append(
                f"Train for more epochs (current: {cfg['epochs']}).  "
                "Try 30–50 epochs with early stopping on val loss."
            )

        if cfg["scheduler"] == "none":
            hp_proposals.append(
                "Add a learning rate scheduler (cosine annealing recommended) to "
                "improve final convergence."
            )

        if cfg["optimizer"] == "sgd":
            hp_proposals.append(
                "Try Adam or AdamW instead of SGD — Adam is typically more robust "
                "to hyperparameter choices."
            )

        if hp_proposals:
            proposals.append(
                {
                    "category": "Hyperparameters",
                    "priority": "High",
                    "changes": hp_proposals,
                }
            )

    # ── Regularisation ───────────────────────────────────────────────────────
    if focus_lower in ("all", "regularisation"):
        reg_proposals = []

        if overfit_gap > 0.1 and cfg["dropout"] < 0.2:
            reg_proposals.append(
                f"Increase dropout from {cfg['dropout']} to 0.3–0.5 "
                "to combat overfitting."
            )
        if overfit_gap > 0.15 and cfg["weight_decay"] < 1e-4:
            reg_proposals.append(
                "Add L2 regularisation: set weight_decay=1e-4."
            )
        if acc_gap > 0.10:
            reg_proposals.append(
                "Apply label smoothing (e.g. 0.1) to reduce overconfident predictions."
            )

        if reg_proposals:
            proposals.append(
                {
                    "category": "Regularisation",
                    "priority": "High" if overfit_gap > 0.2 else "Medium",
                    "changes": reg_proposals,
                }
            )

    # ── Data strategy ────────────────────────────────────────────────────────
    if focus_lower in ("all", "data"):
        data_proposals = []

        if overfit_gap > 0.15:
            data_proposals.append(
                "Apply data augmentation (flips, crops, colour jitter for images; "
                "synonym replacement for text)."
            )
            data_proposals.append(
                "Collect or synthesise more training examples to reduce the "
                "train/val generalisation gap."
            )
        if f_train_loss > 1.0:
            data_proposals.append(
                "Check for class imbalance; apply weighted sampling or oversampling "
                "minority classes."
            )
            data_proposals.append(
                "Normalise / standardise input features if not already done."
            )

        if data_proposals:
            proposals.append(
                {
                    "category": "Data Strategy",
                    "priority": "Medium",
                    "changes": data_proposals,
                }
            )

    # ── Next experiments ─────────────────────────────────────────────────────
    next_exps = [
        "Run the suggested changes as a new experiment and compare with "
        f"{exp_id} using 'list_experiments'.",
        "Use 'setup_training' to apply any hyperparameter changes, then "
        "'run_training' again.",
    ]
    proposals.append(
        {"category": "Next Experiments", "priority": "Action", "changes": next_exps}
    )

    if not any(p["category"] != "Next Experiments" for p in proposals):
        proposals.insert(
            0,
            {
                "category": "General",
                "priority": "Low",
                "changes": [
                    "Model performance looks good — consider ablation studies "
                    "to understand which components contribute most.",
                    "Profile training speed and memory usage before scaling up.",
                ],
            },
        )

    # ── Render ───────────────────────────────────────────────────────────────
    lines = [
        f"Improvement Proposals (based on {exp_id}, focus={focus})",
        "=" * 60,
        "",
    ]
    for p in proposals:
        lines.append(f"[{p['priority']}] {p['category']}")
        for change in p["changes"]:
            lines.append(f"  • {change}")
        lines.append("")

    lines.append(
        "Apply the High-priority changes first, then re-run training to "
        "measure the impact."
    )

    return ToolResult.ok("\n".join(lines))


# ---------------------------------------------------------------------------
# Tool: list_experiments
# ---------------------------------------------------------------------------

LIST_EXPERIMENTS_SPEC = ToolSpec(
    name="list_experiments",
    description=(
        "List all recorded training experiments with their key metrics "
        "for easy comparison."
    ),
    parameters={},
    requires_approval=False,
    tags=["ml", "tracking"],
)


def list_experiments() -> ToolResult:
    if not _experiments:
        return ToolResult.ok(
            "No experiments recorded yet.  Use 'run_training' to start one."
        )

    lines = [f"Experiments ({len(_experiments)} total)", ""]
    header = f"{'ID':<10} {'Model':<12} {'Task':<22} {'Epochs':>6} {'Train Loss':>10} {'Val Loss':>9} {'Val Acc':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for exp in _experiments.values():
        lines.append(
            f"{exp.experiment_id:<10} {exp.model_type:<12} {exp.task:<22} "
            f"{exp.epochs_run:>6} {exp.final_train_loss:>10.4f} "
            f"{exp.final_val_loss:>9.4f} {exp.final_val_acc:>8.4f}"
        )

    lines.append("")
    lines.append(
        "Use 'analyze_results experiment_id=<id>' to inspect any experiment."
    )
    return ToolResult.ok("\n".join(lines))


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_ml_tools(registry: ToolRegistry) -> None:
    """Register all ML agent tools into *registry*."""
    registry.register(DESIGN_MODEL_SPEC, design_model)
    registry.register(SETUP_TRAINING_SPEC, setup_training)
    registry.register(RUN_TRAINING_SPEC, run_training)
    registry.register(ANALYZE_RESULTS_SPEC, analyze_results)
    registry.register(PROPOSE_CHANGES_SPEC, propose_changes)
    registry.register(LIST_EXPERIMENTS_SPEC, list_experiments)
