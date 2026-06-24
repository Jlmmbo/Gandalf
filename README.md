# Gandalf

A C++ neural network that learns any function `f(i, j)` over `[0, V)²` — built from scratch with Eigen, no autograd frameworks. Ships with the Mandelbrot set as the default task.

## What it does

The model takes two integers `(i, j)` in `[0, V)²` and predicts `f(i, j)` (V‑class classification). To change the target, swap the function passed to `IntegerPairDataset::grid_split()`. It uses a small transformer‑inspired architecture:

- Embedding + learned positional encoding
- Single‑head **additive attention** between the two inputs
- LayerNorm + residual connections
- Two‑layer FFN with configurable activation (GELU/ReLU/SiLU/tanh)
- Pooled representation → linear classifier over V classes

## Build

```bash
cmake -S . -B build && cmake --build build
```

Requires Eigen 3 and optionally Qt6 (for the GUI). If Qt6 is present the binary supports both CLI and GUI modes; otherwise it builds in headless mode.

## Usage

```bash
# CLI training (headless)
./build/gandalf --vocab 64 --epochs 500 --batch 256 --d_model 128

# GUI (requires Qt6)
./build/gandalf --gui
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--vocab` | 16 | Grid size V (target classes = V) |
| `--epochs` | 200M | Max epochs |
| `--batch` | 128 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--d_model` | 64 | Embedding / attention dimension |
| `--activation` | gelu | Activation function |
| `--weight-decay` | 0 | L2 regularization |
| `--gui` | — | Launch Qt6 GUI instead of CLI |

## Project structure

```
src/
├── model.h      — FFNNAttentionModel with manual forward/backward
├── dataset.h    — IntegerPairDataset with grid_split()
├── main.cpp     — CLI entry point, Adam optimiser, training loop
├── gui.h        — Qt6 GUI declarations
└── gui.cpp      — Controls, accuracy plot, heatmap tiles
```

## Architecture

```
(i, j) ──► Embedding ──► Additive attention ──► LN + FFN ──► Pool ──► Linear ──► logits
                ▲              ▲
           pos_enc        LayerNorm
```

Forward and backward passes are fully manual (no autograd library). The backward pass computes analytical gradients for every operation: embedding lookup, attention dot‑products, softmax, layer normalisation, GELU/ReLU/SiLU, linear layers, and cross‑entropy loss.
