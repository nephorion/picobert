# Tiny BERT Visualizer

Open `index.html` directly in Chrome, Firefox or Edge (no server, no build, works offline). It runs a deliberately tiny BERT-style encoder (d_model 6, 3 heads of width 2) written in plain JS with a hand-rolled matrix autograd, and trains it live on toy masked-token tasks. The **pipeline strip** steps through one forward pass (←/→, Prev/Next/Play); the **left panel** shows every intermediate matrix of the selected stage with real numbers, and hovering any computed cell shows the arithmetic that produced it and highlights its inputs. The **heads panel** (right) shows, per head, the query (solid) and key (dashed) vectors as 2D arrows (hover a query to see its projection onto each key), the attention heatmap, attention lines, and optionally the W_QK / W_OV circuit matrices; each head can be ablated. The **bottom panel** trains (Train / Step ×1 / Step ×100), plots loss and masked-token accuracy, shows predictions at the masked positions, and saves/restores weight snapshots to compare before and after.

**Your own training data:** choose *Your data…* in the task menu, then paste or upload sequences, one per line, exactly 8 tokens separated by spaces or commas (max 15 distinct tokens). JSON lines (`{"tokens":[…],"mask":[2,5]}`) can fix the masked positions; otherwise 1–2 random positions are masked each time. Loading data resets the model. Every element has an **ⓘ icon**: hover it (or click to keep it open) for an explanation of the stage, matrix, control or chart next to it; the *How to read this page* box at the top gives a 2-minute tour and can be collapsed (it remembers). **Export/Import weights** writes JSON that `reference.py` (NumPy only, `uv run reference.py weights.json --ids …`) can re-run independently to check the page's forward pass.

## Example training sets (choose *Your data…*)

Five ready-made sets, each with a description, what to expect, and its accuracy ceiling; each can be loaded in one click or downloaded to see the file format. Measured after 3000 steps (3 seeds, best head's attention on the expected position): **Twin pairs** 0.80–0.95 on the twin; **Country → capital** (JSON lines with a fixed mask) 0.51–0.87 on the previous token; **Mirror** ≈ 0.74 on position 7−i; **Repeat every 4** 0.71–0.89 at ±4; **Days of the week** only 0.07–0.20 on the neighbours (any visible day determines the answer, so it makes a useful contrast).

## Measured results (Node + headless Chromium, seed 1 unless noted)

- Self-tests (gradient check vs finite differences, softmax/LayerNorm, trace shapes, determinism): all pass; worst gradient relative error ≈ 8e-9.
- Task 1 reaches 100 % masked accuracy on held-out examples by step 500 (seeds 1–3 and more), well inside the 3000-step target. Defaults in `CONFIG` (lr 0.01, batch 16, embedding std 0.5, clip 1.0) needed no tuning.
- Speed: ≈ 520 steps/s without rendering, ≈ 480 steps/s with live rendering (target: ≥ 200).
- `reference.py` matches the page's logits to the printed 12 digits (2-layer model).
- Task 4 (sorted digits) tops out near 80 %: with 8 of 10 digits kept, a gap between neighbours can often be filled two ways, so that is the ceiling for the task, not a training failure.

## Known gap vs. the spec

Acceptance criterion 7 (a head visibly showing the previous/next-token pattern after training on task 1) is **not reliably met**. The model solves the task, but because it knows both a token and its position, any unmasked token determines the answer, so heads often attend at offsets like ±2 or ±3 or spread out. Across 30 seeds and several lr / batch / init settings, the best head put 0.25–0.47 of its attention on ±1 (uniform would be ≈ 0.29). Making ±1 the only useful signal needs a change to the task itself.
