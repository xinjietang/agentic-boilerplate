# Machine Learning Agent

An agentic example built on [agentic-boilerplate](../../README.md).  
The agent designs neural network architectures, configures training pipelines,
runs simulated training experiments, analyses results, and proposes
data-driven improvements — all through a conversational workflow.

---

## What it demonstrates

| Boilerplate feature | How it's used |
|---|---|
| `ToolSpec` / `ToolResult` | Six ML tools with descriptive JSON-schema parameters |
| `ToolRegistry` | Centralised tool registration via `register_ml_tools()` |
| `AgentLoop` subclass | `MLAgentLoop._call_llm()` dispatches design / train / analyse / propose intents |
| `ContextManager` | Conversation history lets the agent track multi-step experiment workflows |
| `EventBus` | All tool calls and results surface as structured events in the CLI |
| Config YAML | Custom ML-focused system prompt in `config.yaml` |

---

## Tools

| Tool | Description |
|---|---|
| `design_model` | Specify an architecture (CNN, MLP, RNN, Transformer, ResNet) for a task |
| `setup_training` | Configure learning rate, epochs, batch size, dropout, scheduler, etc. |
| `run_training` | Simulate a training run and record epoch-by-epoch loss/accuracy curves |
| `analyze_results` | Detect overfitting, underfitting, LR instability, and slow convergence |
| `propose_changes` | Generate prioritised, concrete improvement suggestions |
| `list_experiments` | Show all experiments and their final metrics for comparison |

> **Stub vs production:** All training is simulated in-process.
> Replace `_simulate_training()` in `tools/ml_tools.py` with a real training loop
> (PyTorch `Trainer`, Keras `model.fit()`, scikit-learn `Pipeline.fit()`, etc.) and
> replace `_call_llm()` in `agent.py` with a real LLM call (OpenAI, Anthropic, …)
> for production use.

---

## Quick start

```bash
pip install -e ".[dev]"

cd examples/machine_learning_agent
python agent.py            # interactive REPL

# Headless single-shot commands
python agent.py --headless "design a CNN for image classification"
python agent.py --headless "setup training lr=0.001 epochs=20 dropout=0.2"
python agent.py --headless "run training"
python agent.py --headless "analyze results"
python agent.py --headless "propose improvements"
python agent.py --headless "list experiments"
```

---

## Sample end-to-end session

```
you> design a CNN for image classification with 4 layers

⚙ TOOL CALL  design_model(model_type='cnn', task='image_classification', hidden_layers=4, ...)
⚙ TOOL RESULT  design_model →
  Model design saved.
  {
    "model_type": "cnn",
    "task": "image_classification",
    "num_layers": 6,
    "params_estimate": 49280,
    "layers": [
      {"type": "Input", "shape": "auto"},
      {"type": "Conv2D", "filters": 128, "kernel_size": 3, "activation": "relu"},
      ...
      {"type": "Dense", "units": 10, "activation": "softmax"}
    ]
  }

you> setup training lr=0.001 epochs=25 dropout=0.3 scheduler=cosine

⚙ TOOL CALL  setup_training(learning_rate=0.001, epochs=25, dropout=0.3, scheduler='cosine', ...)
⚙ TOOL RESULT  setup_training → Training config saved.

you> run training

⚙ TOOL CALL  run_training()
⚙ TOOL RESULT  run_training →
  Experiment exp_0001 complete (25 epochs).

  Epoch | Train Loss | Val Loss | Train Acc | Val Acc
  ------|------------|----------|-----------|--------
      1 |     2.2104 |   2.3201 |    0.2310 |  0.2180
      3 |     1.8743 |   1.9821 |    0.4102 |  0.3890
      ...
     25 |     0.4231 |   0.5102 |    0.8512 |  0.8231

you> analyze results

⚙ TOOL CALL  analyze_results()
⚙ TOOL RESULT  analyze_results →
  Analysis of exp_0001  (cnn / image_classification)
  ============================================================

  Results
    Final train loss : 0.4231
    Final val loss   : 0.5102
    Final train acc  : 0.8512
    Final val acc    : 0.8231
    Accuracy gap (train - val): 0.0281 (acceptable)

  Issues found
    ⚠  No major issues detected.

  Initial suggestions
    →  The model is converging well.  Consider fine-tuning with a lower LR …

you> propose improvements

⚙ TOOL CALL  propose_changes(focus='all')
⚙ TOOL RESULT  propose_changes →
  Improvement Proposals (based on exp_0001, focus=all)
  ...
  [High] Hyperparameters
    • Add a learning rate scheduler (cosine annealing recommended) …
  [Action] Next Experiments
    • Run the suggested changes as a new experiment …
```

---

## Typical multi-experiment workflow

```bash
# Experiment 1 — baseline MLP (often underperforms on images)
python agent.py --headless "design a mlp for image classification"
python agent.py --headless "setup training lr=0.01 epochs=10"
python agent.py --headless "run training"
python agent.py --headless "analyze results"
python agent.py --headless "propose architecture changes"

# Experiment 2 — switch to CNN per recommendation
python agent.py --headless "design a CNN for image classification 3 layers"
python agent.py --headless "setup training lr=0.001 epochs=20 dropout=0.3 scheduler=cosine"
python agent.py --headless "run training"
python agent.py --headless "analyze results"

# Compare
python agent.py --headless "list experiments"
```

---

## Adapting to production

1. **Replace the stub LLM** — swap `_call_llm()` in `agent.py` with a real
   OpenAI / Anthropic call so the agent can reason about which tools to use
   without hardcoded keyword routing:

   ```python
   import json, openai

   async def _call_llm(self, messages):
       client = openai.AsyncOpenAI(api_key=self.config.llm.api_key)
       resp = await client.chat.completions.create(
           model="gpt-4o",
           messages=messages,
           tools=self.tools.to_llm_schema(),
       )
       msg = resp.choices[0].message
       calls = [
           {"id": tc.id, "name": tc.function.name,
            "parameters": json.loads(tc.function.arguments)}
           for tc in (msg.tool_calls or [])
       ]
       return {"content": msg.content or "", "tool_calls": calls}
   ```

2. **Replace the simulated training** — swap `_simulate_training()` in
   `tools/ml_tools.py` with a real training loop:

   ```python
   import torch
   from torch.utils.data import DataLoader

   def _real_training(model, train_loader, val_loader, cfg):
       optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
       for epoch in range(cfg.epochs):
           # ... train and validate ...
           yield epoch, train_loss, val_loss, train_acc, val_acc
   ```

3. **Persist experiments** — replace the in-memory `_experiments` dict with
   a database (SQLite via `sqlite3`, or an MLflow / Weights & Biases run).

4. **Add model persistence** — save/load model checkpoints in `run_training`
   so experiments are reproducible:

   ```python
   torch.save(model.state_dict(), f"checkpoints/{exp_id}.pt")
   ```

5. **Add hyperparameter search** — wrap `run_training` + `analyze_results`
   in a loop driven by `propose_changes` for automated tuning.
