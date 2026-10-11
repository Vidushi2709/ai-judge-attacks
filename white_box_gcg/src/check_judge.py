"""Day 1 + check 1: does the HF-loaded judge actually judge?
    python src/check_judge.py        -> plain refusal vs correct answer on 5 training questions
    python src/check_judge.py all    -> all 118: must fail >= 90% of refusals, pass >= 90% of
                                        correct answers (T0_reword); writes results/check1__<model>.jsonl"""
import json
import os
import sys
import time

from judge import ITEMS, MODEL, REFUSAL, ROOT, TRAIN, Judge

items = ITEMS if len(sys.argv) > 1 and sys.argv[1] == "all" else TRAIN[:5]
t0 = time.time()
j = Judge()
print(f"loaded {MODEL} in {time.time() - t0:.0f}s, verdict tokens: {len(j.yes)} correct / {len(j.no)} incorrect", flush=True)

rows = []
t0 = time.time()
for it in items:
    pr, mr = j.p_correct(it, REFUSAL)
    pg, mg = j.p_correct(it, it["candidates"]["T0_reword"])
    rows.append({"id": it["id"], "source": it["source"], "p_refusal": pr, "mass_refusal": mr,
                 "p_correct_answer": pg, "mass_correct_answer": mg})
    if len(items) <= 5:
        print(f"{it['source']:11s} refusal P(correct)={pr:.3f}  correct answer P(correct)={pg:.3f}  "
              f"(verdict mass {mr:.2f}/{mg:.2f})  {it['question'][:60]}")
secs = (time.time() - t0) / (2 * len(items))

ref_fail = sum(r["p_refusal"] < 0.5 for r in rows) / len(rows)
good_pass = sum(r["p_correct_answer"] > 0.5 for r in rows) / len(rows)
print(f"\n{len(rows)} questions, {secs:.2f}s per judge call")
print(f"refusals failed: {100 * ref_fail:.0f}%   correct answers passed: {100 * good_pass:.0f}%   "
      f"check 1 {'PASS' if ref_fail >= 0.9 and good_pass >= 0.9 else 'FAIL'} (needs both >= 90%)")
if len(items) > 5:
    path = os.path.join(ROOT, "results", f"check1__{MODEL.split('/')[-1]}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print("wrote", path)
