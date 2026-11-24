# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Independent NumPy forward pass for the Tiny BERT Visualizer.

Usage:  uv run reference.py tinybert-weights.json --ids 1 2 0 4 5 6 1 2 [--mask 2]
`--ids` are token ids (0 = [MASK]); `--mask` lists masked positions (default: where id == 0).
Prints logits and P at the masked positions; compare with the page (to ~1e-9).
Export the weights with the page's "Export weights" button.
"""
import argparse, json
import numpy as np

def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b

def gelu(x):  # tanh approximation, same as the page
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def forward(w, ids, n_heads=3):
    P = {k: np.array(v["data"]).reshape(v["rows"], v["cols"]) for k, v in w["params"].items()}
    L = len(ids); X = P["tokEmb"][ids] + P["posEmb"][:L]
    layer = 0
    while f"l{layer}.Wq" in P:
        p = lambda n: P[f"l{layer}.{n}"]
        Q, K, V = (X @ p(f"W{c}") + p(f"b{c}") for c in "qkv")
        dh = Q.shape[1] // n_heads; outs = []
        for h in range(n_heads):
            s = slice(h * dh, (h + 1) * dh)
            A = softmax(Q[:, s] @ K[:, s].T / np.sqrt(dh)); outs.append(A @ V[:, s])
        O = np.concatenate(outs, 1) @ p("Wo") + p("bo")
        X1 = layer_norm(X + O, p("ln1g"), p("ln1b"))
        F = gelu(X1 @ p("W1") + p("bf1")) @ p("W2") + p("bf2")
        X = layer_norm(X1 + F, p("ln2g"), p("ln2b")); layer += 1
    return X @ P["Wout"] + P["bout"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("weights"); ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--mask", type=int, nargs="*"); a = ap.parse_args()
    w = json.load(open(a.weights)); logits = forward(w, a.ids); Pm = softmax(logits)
    pos = a.mask if a.mask is not None else [i for i, t in enumerate(a.ids) if t == 0]
    np.set_printoptions(precision=12, suppress=True, linewidth=160)
    print("logits:\n", logits)
    for i in pos: print(f"P at position {i}:", Pm[i], "-> argmax", w["vocab"][int(Pm[i].argmax())])
