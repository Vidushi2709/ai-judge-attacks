"""Random search (Null Models paper, after Andriushchenko et al. 2024): change one word of the
attack at random; keep the change if the training loss drops. One step = 10 judge calls."""
import json
import os
import random

import common as c

L = 20  # attack length in words
rng = random.Random(1)
INIT = os.environ.get("INIT")  # warm start: path to a summary json whose attack is the starting point
attack = json.load(open(INIT))["attack"].split() if INIT else [rng.choice(c.VOCAB) for _ in range(L)]
L = len(attack)
best = c.loss(" ".join(attack))
run = c.Run(os.environ.get("NAME", "random"))
run.log(0, best, " ".join(attack))
step = 0
while c.CALLS + len(c.TRAIN) <= c.BUDGET:
    step += 1
    cand = attack[:]
    cand[rng.randrange(L)] = rng.choice(c.VOCAB)
    l = c.loss(" ".join(cand))
    if l < best:
        attack, best = cand, l
    row = run.log(step, best, " ".join(attack))
    if step % 20 == 0:
        print(f"step {step} calls {row['calls']} {row['secs']}s best P(correct) {row['best_p']}", flush=True)
run.finish(" ".join(attack))
