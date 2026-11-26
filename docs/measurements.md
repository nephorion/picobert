# Measurements

## Default task: Twin pairs

3000 steps, seeds 1–10: nine reach 100 % masked accuracy by step 1000, with the best head putting 0.64–0.94 of a masked position's attention on its twin. [Seed 5](seed5.md) is the exception. The built-in task never hides both twins; the *Twin pairs* example set under *Your data…* can, which is why its ceiling and figures differ.

## Example training sets (*Your data…*)

After 3000 steps (3 seeds, best head's attention on the expected position):

| Set | Attention on the expected position |
|---|---|
| Twin pairs | 0.80–0.95 on the twin |
| Country → capital (fixed mask) | 0.51–0.87 on the previous token |
| Mirror | ≈ 0.74 on position 7−i |
| Repeat every 4 | 0.71–0.89 at ±4 |
| Days of the week | 0.07–0.20 on the neighbours (any visible day gives the answer, so it makes a useful contrast) |

## Task 1 (previous token)

It reaches 100 % masked accuracy on held-out examples by step 500, but do not expect a clean previous/next-token head. The model knows both a token and its position, so any unmasked token determines the answer, and heads often attend at offsets like ±2 or ±3 or spread out. Across 30 seeds and several learning-rate, batch and initialisation settings, the best head put 0.25–0.47 of its attention on ±1 (uniform would be ≈ 0.29). The spec's acceptance criterion 7 was relaxed to match.

## Other

Measured on the first version of the page, in Node and headless Chromium:

- Self-tests: all pass; worst gradient relative error ≈ 8e-9.
- Speed: ≈ 520 steps/s without rendering, ≈ 480 steps/s with live rendering (target: ≥ 200).
- `reference.py` matched the page's logits to the printed 12 digits (2-layer model).
- Task 4 (sorted digits) tops out near 80 %: with 8 of 10 digits kept, a gap can often be filled two ways.
