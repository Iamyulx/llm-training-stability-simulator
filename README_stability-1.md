<div align="center">

# ⚡ LLM Training Stability Simulator

**A lightweight simulator for studying loss dynamics, divergence detection, and early stopping behaviour during LLM pretraining — no GPU required**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-22c55e) ![Status](https://img.shields.io/badge/Status-active-brightgreen)

</div>

---


## 🔍 Overview

Training large language models is expensive and unstable. Loss spikes, divergence, and
wasted compute from over-training are common failure modes — but hard to study without
running full-scale experiments.

This simulator lets you **reproduce and analyze training stability scenarios in seconds**,
using a configurable synthetic loss trajectory with noise injection. It models:

- Smooth convergence runs
- Loss spike events (simulated instability)
- Divergence trajectories
- Early stopping triggered by patience + threshold criteria

> **Use case:** Rapid prototyping of stability heuristics, early stopping policies, and
> evaluation criteria before committing to expensive real training runs.

---

## 🧩 Components

| Module | Description |
|---|---|
| `simulation.py` | Entry point — wires all components and runs the simulation |
| `trainer.py` | Generates synthetic loss trajectories with configurable noise |
| `early_stopping.py` | Patience-based early stopping with minimum improvement threshold |
| `evaluator.py` | Classifies training outcome: `GOOD`, `OK`, `DIVERGED`, or `FAIL` |
| `api_client.py` | *(Placeholder)* Future hook for external experiment tracking APIs |

---

## 🏗️ Architecture

```
llm-training-stability-simulator/
├── simulation.py       # Entry point: run_simulation()
├── trainer.py          # Synthetic loss trajectory generator
├── early_stopping.py   # EarlyStopping(patience, threshold)
├── evaluator.py        # Outcome classifier → GOOD / OK / DIVERGED / FAIL
├── api_client.py       # (empty) Placeholder for remote logging
└── requirements.txt    # Dependencies
```

**Simulation flow:**

```
run_simulation()
    │
    ├─► EarlyStopping(patience=2, threshold=0.01)
    ├─► Trainer(early_stopping)
    │       └─► .train(steps=50)  →  loss_history: List[float]
    │
    └─► Evaluator().evaluate(loss_history)
            ├─► "GOOD"     — loss monotonically decreased to minimum
            ├─► "OK"       — loss decreased but not to global min
            ├─► "DIVERGED" — final loss > initial loss
            └─► "FAIL"     — empty history (training never started)
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Iamyulx/llm-training-stability-simulator.git
cd llm-training-stability-simulator
pip install -r requirements.txt
```

---

## 🚀 Quickstart

**Run the default simulation:**
```bash
python simulation.py
```

**Expected output:**
```
📊 Evaluation Result: GOOD
```

**Use components programmatically:**
```python
from trainer import Trainer
from early_stopping import EarlyStopping
from evaluator import Evaluator

# Configure stability parameters
es = EarlyStopping(patience=3, threshold=0.005)
trainer = Trainer(early_stopping=es)
evaluator = Evaluator()

# Run and evaluate
loss_history = trainer.train(steps=100)
result = evaluator.evaluate(loss_history)

print(f"Outcome: {result}")           # GOOD / OK / DIVERGED / FAIL
print(f"Steps run: {len(loss_history)}")
print(f"Final loss: {loss_history[-1]:.4f}")
```

---

## 📊 Simulation Results

> ⚠️ **Placeholder values.** Run batch simulations and replace with real numbers.

| Outcome | Rate |
|---|---|
| `GOOD` — stable convergence | ~87% |
| `DIVERGED` — loss blowup | ~9% |
| Early stopped before convergence | ~4% |
| Avg steps to convergence | ~31 |

**Default configuration:**

```python
EarlyStopping(patience=2, threshold=0.01)
Trainer.train(steps=50)
```

---

## 🔬 Evaluator Logic

The `Evaluator` classifies a training run based on its loss history:

| Condition | Label | Meaning |
|---|---|---|
| `loss_history` is empty | `FAIL` | Training never produced output |
| `loss[-1] > loss[0]` | `DIVERGED` | Loss increased overall — unstable training |
| `min(loss) == loss[-1]` | `GOOD` | Loss is still decreasing — healthy run |
| Otherwise | `OK` | Loss decreased but plateaued before minimum |

This mirrors the heuristics used in real LLM training monitoring dashboards.

---

## 🛑 Early Stopping Logic

```
EarlyStopping(patience=2, threshold=0.01)
```

- Tracks `best_loss` seen so far
- At each step: if `current_loss < best_loss - threshold` → reset counter, update best
- Otherwise → increment counter
- If `counter >= patience` → returns `True` (stop signal)

This implements **minimum improvement early stopping**, which avoids stopping on
temporary plateaus while still catching true stagnation.

---

## ⚠️ Known Issues & Roadmap

- `api_client.py` is **empty** — intended as a hook for W&B / MLflow integration
- `__pycache__/` is committed — should be added to `.gitignore`
- No `.gitignore` file present
- `Trainer` uses synthetic noise only — extend to wrap real model training loops
- `Evaluator` returns plain strings — could return a structured dataclass

---

## 📚 Related Work

- Zhao et al. (2023) — [Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks](https://arxiv.org/abs/2310.02244)
- Wortsman et al. (2023) — [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322)
- Chowdhery et al. (2022) — [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)

---

## 📄 License

MIT © [Iamyulx](https://github.com/Iamyulx)
