# Why does seed 5 struggle?

Set **Seed** to 5, press **Reset model** and train Twin pairs for 3000 steps. Most seeds are at 100 % long before that. Seed 5 is at about 59 %.

## It is slow, not stuck

| Training steps | Masked accuracy |
|---|---|
| 3,000 | 59 % |
| 10,000 | 99 % |
| 30,000 | 100 % |

By 10,000 steps one head (head 2) looks at the twin from every position, putting 0.64–0.85 of its attention there. At 3,000 steps it does so strongly from position 0 only (0.81), and weakly elsewhere (0.08–0.27).

## It is the combination that is unlucky

The seed sets two things: the starting weights and the order of the training batches. Changing either one alone, for 3000 steps:

| Change | Accuracy |
|---|---|
| Same starting weights, different batches (5 tries) | 100 % each |
| Same batches, different starting weights (3 tries) | 100 % each |
| Same seed, learning rate 0.003 or 0.03 instead of 0.01 | 100 % |
| Same seed, 2 layers | 100 % |

So neither the start nor the data is "bad" by itself. This particular path passes through a long flat stretch where progress is slow.

## How common is it?

Of seeds 1–40 at 3000 steps, only 5 (59 %) and 27 (60 %) were below 90 %; seed 40 reached 93 %.

## Try it

Train seed 5, then open the **Training** tab and drag **Replay** to watch head 2 fill in the twin pattern slowly. Or set **lr** to 0.03 before training and see it get there much sooner.

*Accuracy and attention were measured on 600 fresh examples.*
