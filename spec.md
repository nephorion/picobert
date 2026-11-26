# Spec: Tiny BERT Visualizer

> **This is the original brief, kept for reference.** The app has since grown past it: a Training tab with replay, Twin pairs as the default task, dragging and keyboard editing of weights, a guided tour, a CI check, and usage counts on the hosted copy. The README describes what the page does now.

An interactive, single-page visualizer that runs a deliberately tiny BERT-style encoder **in the browser** and shows every number in every step of the forward pass, while the model trains on a toy task.

The goal is to build **intuition** for the computation. It isn't meant to be useful for real NLP.

---

## 1. Goals and non-goals

### Goals
- Let the user step through one full forward pass, stage by stage, and see every intermediate matrix with real numbers.
- Make attention *geometric*: each head has size 2, so queries and keys are drawn as 2D arrows.
- Train live on toy masked-token tasks, so the user can watch attention patterns form.
- For any computed number, be able to explain how it was computed (hover → show the arithmetic).
- Keep the code readable: the math is hand-written in plain JS with no ML library, so the user can read it alongside the page.

### Non-goals
- Real text, real tokenizers, pretrained weights.
- GPU, WebGL, performance beyond what's described in §9.
- Next-sentence prediction, dropout, segment embeddings.
- Mobile layout (desktop browser only; it should still not break on a narrow window).

---

## 2. Technical constraints

- **One self-contained file: `index.html`** with inline CSS and JS. No build step, no npm, no CDN dependencies. Opening the file from disk must work offline. One exception, for the hosted copy only: it loads GoatCounter to count visits and feature use (see README).
- Vanilla JS (ES2020+), SVG and/or Canvas for drawing. No frameworks.
- All math implemented by hand (see §4). No TensorFlow.js or similar.
- Deterministic: seeded PRNG (e.g. mulberry32). Same seed → same init → same training run.
- Expose the main objects on `window.tinybert` (model, trace, trainer, config) for console poking.
- Code organized into clearly commented sections in this order:
  1. Config
  2. PRNG
  3. Matrix utilities and minimal autograd
  4. Model (parameters, forward pass with trace, loss)
  5. Tasks / data generators
  6. Trainer (Adam)
  7. Self-tests
  8. Rendering (one function per panel)
  9. UI wiring / main loop

---

## 3. Model definition

Default config (all values editable in a `CONFIG` object at the top of the file; the UI only needs to expose the ones in §6):

| Name | Default | Notes |
|---|---|---|
| `d_model` | 6 | Embedding / residual width |
| `n_heads` | 3 | |
| `d_head` | 2 | `d_model / n_heads`; must be 2 for the arrow plots |
| `n_layers` | 1 | Supports 1 or 2 |
| `d_ff` | 24 | 4 × d_model |
| `seq_len` | 8 | Fixed length, no padding |
| `vocab` | task-dependent, ≤ 16 | Always includes `[MASK]` |

### Architecture (post-LN, like original BERT)

For input token ids `t[0..L-1]`:

```
E      = TokEmb[t] + PosEmb[0..L-1]                  (L × 6)
for each layer:
  Q = X Wq + bq ; K = X Wk + bk ; V = X Wv + bv      (L × 6 each)
  split Q,K,V into 3 heads of width 2                (3 × L × 2)
  S_h = Q_h K_hᵀ / sqrt(2)                           (3 × L × L)
  A_h = softmax_rows(S_h)                            (3 × L × L)
  O_h = A_h V_h                                      (3 × L × 2)
  O   = concat_h(O_h) Wo + bo                        (L × 6)
  X1  = LayerNorm(X + O)                             (L × 6)
  H   = GELU(X1 W1 + b1)                             (L × 24)
  F   = H W2 + b2                                    (L × 6)
  X   = LayerNorm(X1 + F)                            (L × 6)
logits = X Wout + bout                               (L × vocab)
P      = softmax_rows(logits)
loss   = mean cross-entropy over masked positions only
```

- GELU: use the tanh approximation (state this in a code comment).
- LayerNorm: per row, eps = 1e-5, learnable gain γ and bias β.
- No attention mask: every token attends to every token, including `[MASK]` tokens and itself.
- Output head: a plain linear layer 6 → vocab (untied). Note in a comment that real BERT adds a dense + GELU + LayerNorm transform and ties weights to the token embedding. This is a deliberate simplification.

### Initialization
- Weight matrices: Xavier/Glorot uniform. Biases: 0. LayerNorm γ = 1, β = 0.
- Token and position embeddings: N(0, 0.5²).

---

## 4. Math engine

### 4.1 Matrices
A small `Mat` type: `{rows, cols, data: Float64Array}` with matmul, transpose, add, row-softmax, slicing columns (for head split), and concat columns.

### 4.2 Autograd
Implement a **minimal reverse-mode autograd over matrices** (not scalars, which would be too slow and too noisy). Each op records its inputs and a backward closure. Ops needed: `matmul`, `add` (with row-broadcast bias), `scale`, `sliceCols`, `concatCols`, `rowSoftmax`, `layerNorm`, `gelu`, `gatherRows` (embedding lookup), `crossEntropyMasked`.

The backward of each op must be written out explicitly and readably, with a comment showing the derivative formula.

### 4.3 Trace
The forward pass must return a **trace object** holding every intermediate matrix, keyed by stage (see §5), plus the parameters used. **Rendering reads only from the trace and parameters.** It never recomputes math. This keeps the display and the model guaranteed to match.

### 4.4 Optimizer
Adam (β1 = 0.9, β2 = 0.999, eps = 1e-8), default lr = 0.01, batch size 16 (gradients averaged over the batch). Optional gradient clipping at global norm 1.0.

---

## 5. Stages (the pipeline)

The UI shows these as a clickable horizontal strip. Selecting a stage shows its detail panel (§6.3). Numbering is per layer for stages 2–10.

| # | Stage | Shows |
|---|---|---|
| 1 | Embeddings | TokEmb rows for the input, PosEmb rows, their sum E |
| 2 | Projections | X, Wq/Wk/Wv (6×6), Q, K, V (L×6), with column groups colored by head |
| 3 | Head split | Q, K, V split into 3 × (L×2) |
| 4 | Scores | S_h = Q_h K_hᵀ / √2 per head |
| 5 | Softmax | A_h per head (rows sum to 1) |
| 6 | Weighted values | O_h = A_h V_h per head |
| 7 | Concat + output projection | concat(O_h), Wo, O |
| 8 | Residual + LayerNorm 1 | X, O, X + O, mean/var per row, X1 |
| 9 | Feed-forward | X1, W1, pre-GELU, H (L×24), W2, F |
| 10 | Residual + LayerNorm 2 | X1 + F, X out |
| 11 | Output | logits, softmax P at masked positions, loss |

Controls: Prev / Next / Play (auto-advance, ~1.5 s per stage) and keyboard arrows ←/→.

---

## 6. UI

Layout (desktop): a top control bar, the pipeline strip below it, then a main area with a **stage detail panel** (left, wide) and the **heads panel** (right). A **training panel** sits along the bottom.

### 6.1 Control bar
- **Task selector** (§7). Changing the task resets the model.
- **Input sequence editor:** the current sequence shown as token chips. Click a chip to toggle `[MASK]` on that position. A dropdown on each chip changes the token. A "Random example" button draws from the task generator.
- **Seed** input plus a **Reset model** button.
- **Layer selector** (when `n_layers = 2`).
- **Number precision** toggle: 2 or 3 decimals.

### 6.2 Matrix rendering (used everywhere)
- Each matrix is drawn as a grid of cells showing the number, with the background colored:
  - **Diverging** scale centered on 0 (blue ↔ neutral ↔ orange/red) for weights, activations, scores.
  - **Sequential** scale 0 → 1 for softmax outputs.
  - The scale range is per-matrix (symmetric around 0 for diverging). A small legend shows the min and max.
- Rows are labeled with tokens (e.g. `2:"c"`), and columns with dimension index or head-colored group.
- Each head has a consistent color used everywhere (columns, arrows, panels).
- Above each matrix: its name, shape (e.g. `Q₁  4×2`), and the formula that produced it.
- Must be legible in both light and dark mode (follow `prefers-color-scheme`).

### 6.3 Cell explainer (key intuition feature)
Hovering any cell in a computed matrix shows a tooltip, and highlights the input row/column that produced it, with the actual arithmetic:
- matmul cell: `Q[2,1] = X[2,:]·Wq[:,1] = (0.31×−0.20) + (…) + … + bq[1] = 0.47`
- score cell: `S₁[2,3] = q₂·k₃ / √2 = (0.47×0.12 + −0.30×0.88) / 1.414 = −0.15`
- softmax cell: `A₁[2,3] = exp(−0.15) / Σⱼ exp(S₁[2,j]) = 0.86 / 4.10 = 0.21`
- LayerNorm cell: `(x − μ) / √(σ² + ε) · γ + β` with the row's μ and σ² filled in
- GELU and residual cells: the analogous formula.

### 6.4 Heads panel (always visible, tracks the current layer)
Three columns, one per head. Each head shows:
1. **2D arrow plot:** for every token, the query vector q_i (solid arrow) and the key vector k_j (dashed arrow) from the origin, labeled with the token. Axes auto-scale with a fixed aspect ratio (1:1, so angles are true).
   - Hovering a query arrow highlights it and draws its projection onto each key, with the dot product shown. Hovering a token anywhere on the page highlights its arrows.
2. **Attention heatmap** A_h (L×L).
3. **Attention lines:** two columns of tokens (query side on the left, key side on the right) with lines whose opacity/thickness is proportional to A_h[i,j].
4. **Circuit matrices** (collapsed by default, "Show circuits" toggle): W_QK = Wq_h Wk_hᵀ (6×6) and W_OV = Wv_h Wo_h (6×6), with a one-line caption explaining what each means.

### 6.5 Output panel (stage 11, and a small summary always visible)
For each masked position, a bar chart of P over the vocabulary with the true token highlighted, plus the per-position loss.

### 6.6 Training panel
- **Train / Pause**, **Step ×1** (one optimizer step), **Step ×100**.
- Learning-rate input; steps per second; current step count.
- Live **loss curve** (moving average) and **masked-token accuracy** curve.
- While training, the visible input sequence is re-run through the current weights so every panel updates live. Throttle re-rendering to ≤ 10 fps, but let training run faster in between.
- **Snapshots:** "Save snapshot" stores the current weights with a step number. A dropdown restores one, so the user can compare "before" and "after".

---

## 7. Toy tasks

Each task is a generator that returns `{tokens, maskedPositions, targets}`. Mask 1–2 random positions per example (positions inside the `[CLS]`/`[SEP]` frame only, if those are used; using them is optional, keep it simple). Each task lists the attention pattern it *should* produce, shown as a hint in the UI:

1. **Previous token:** a random sequence over `a–f`, constructed so the sequence is a known cyclic walk (`a→b→c→d→e→f→a`) from a random start. The masked token can be predicted from its left neighbour (or its right). *Possible: a head attends to position i−1 or i+1, though the task does not force it.*
2. **Repeat pattern:** a random block of 3 tokens repeated (e.g. `d a f d a f d a`). *Expected: a head attends to position i±3.*
3. **Count / majority:** a sequence over `{x, y}`. The masked target is whichever symbol is more frequent among the unmasked tokens (odd count guaranteed). *Expected: broad, roughly uniform attention.*
4. **Sorted digits:** a strictly increasing sequence of digits 0–9 (randomly chosen subset). *Expected: attends to both neighbours.*

The default is **Twin pairs** (each token appears twice in a row; both twins are never masked together), listed above the four numbered tasks: it forces a visible attention pattern, which task 1 does not. Acceptance criteria 6 and 7 still refer to task 1.

---

## 8. Self-tests

A **"Run self-tests"** button (results shown in a small panel; also run automatically on load, and log to the console):

- **Gradient check:** for a random batch, compare autograd gradients against central finite differences (h = 1e-5) for a random sample of ≥ 50 parameters, covering every parameter tensor. Pass if the max relative error is < 1e-4.
- Softmax rows sum to 1 (±1e-9).
- LayerNorm output rows have mean ≈ 0 and variance ≈ 1 (before γ/β, with default γ = 1, β = 0).
- Shapes of every trace entry match §5.
- Determinism: two inits with the same seed produce identical parameters.

If any test fails, show a visible red banner.

---

## 9. Acceptance criteria

1. Opening `index.html` directly from disk in current Chrome/Firefox/Edge works with no console errors.
2. All self-tests pass.
3. Every stage in §5 renders with real numbers for the current input, and the cell explainer works for each matrix type listed in §6.3.
4. The arrow plots, heatmaps and attention lines for all 3 heads agree with each other (they come from the same trace).
5. Training runs at ≥ 200 optimizer steps/s on a typical laptop with rendering throttled.
6. On task 1 with the default seed and settings, masked-token accuracy reaches ≥ 90 % within 3000 steps. If it doesn't, tune the defaults (lr, init scale, batch size) until it does, and record the chosen values in `CONFIG`.
7. After training on task 1, the heads panel shows each head's learned attention pattern in its heatmap. No particular pattern is required: any unmasked token determines the answer, so a previous/next-token head need not form.
8. A 1–2 paragraph `README.md` next to `index.html` explains how to open it and what each panel shows.

---

## 10. Stretch goals (only after §9 is complete)

- **Edit a weight:** click a weight cell and drag vertically to change its value; the whole trace recomputes instantly.
- **Ablate a head:** a toggle per head that zeroes its output, to show its effect on the loss and predictions.
- **2-layer mode** with a layer selector, and composition of heads across layers.
- **Export/import weights** as JSON.
- **Reference check:** a short `reference.py` (NumPy only) that loads exported JSON weights and an input, runs the same forward pass, and prints the outputs so they can be compared to the page's to 1e-9.

---

## 11. Implementation notes for Claude Code

- Build in this order: math engine + gradient check → model forward with trace → training on task 1 (verify in the console) → matrix renderer → stages → heads panel → training panel → cell explainer → polish.
- Verify in an actual browser: open the page, run the self-tests, train task 1, and screenshot the heads panel before and after training.
- Keep functions short and named after the math (`scoresForHead`, `layerNormBackward`, …). The user will read the code to learn from it, so clarity beats cleverness.
