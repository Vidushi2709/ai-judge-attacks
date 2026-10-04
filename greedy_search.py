"""Greedy search (best-of-K, like a beam of width 1): each step tries K random one-word
changes, keeps the best if it beats the current attack. One step = K x 10 judge calls."""
import json
import os
import random

import common as c

L, K = 20, 8
rng = random.Random(1)
INIT = os.environ.get("INIT")  # warm start: path to a summary json whose attack is the starting point
attack = json.load(open(INIT))["attack"].split() if INIT else [rng.choice(c.VOCAB) for _ in range(L)]
L = len(attack)
best = c.loss(" ".join(attack))
run = c.Run(os.environ.get("NAME", "greedy"))
run.log(0, best, " ".join(attack))
step = 0
while c.CALLS + K * len(c.TRAIN) <= c.BUDGET:
    step += 1
    cands = []
    for _ in range(K):
        cand = attack[:]
        cand[rng.randrange(L)] = rng.choice(c.VOCAB)
        cands.append((c.loss(" ".join(cand)), cand))
    l, cand = min(cands, key=lambda x: x[0])
    if l < best:
        attack, best = cand, l
    row = run.log(step, best, " ".join(attack))
    print(f"step {step} calls {row['calls']} {row['secs']}s best P(correct) {row['best_p']}", flush=True)
run.finish(" ".join(attack))
