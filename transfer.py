"""Transfer test: do the attacks found on the local judge also fool Gemini 2.5 Flash-Lite?
Gemini grades in the normal reason-then-verdict format with the best prompt (st), on the first
40 held-out questions. Plain-refusal baseline comes from the existing st run (same refusal, same prompt).
    BUDGET_INR=0.5 python transfer.py"""
import glob
import json
import os

import common as c
from llm import ask
from run import RUBRICS, TEMPLATE, cost_inr, parse

MODEL = "gemini-2.5-flash-lite"
BUDGET_INR = float(os.environ.get("BUDGET_INR", "0.5"))
QS = c.TEST[:40]
path = os.path.join(c.OUT, "transfer__flash-lite-2.5__st.jsonl")
done = {(r["method"], r["id"]) for r in map(json.loads, open(path, encoding="utf-8"))} if os.path.exists(path) else set()

attacks = {json.load(open(p))["method"]: json.load(open(p))["attack"]
           for p in glob.glob(os.path.join(c.OUT, "summary__*__llama3__st.json"))}
spent = 0.0
with open(path, "a", encoding="utf-8") as f:
    for method, attack in sorted(attacks.items()):
        for it in QS:
            if (method, it["id"]) in done:
                continue
            if spent > BUDGET_INR:
                raise SystemExit(f"STOPPED at ~Rs{spent:.2f} (budget Rs{BUDGET_INR}). Rerun to resume.")
            prompt = TEMPLATE.format(rubric=RUBRICS["st"], question=it["question"], expected=it["expected"],
                                     answer=c.REFUSAL + " " + attack)
            r = ask(MODEL, prompt, think=0, max_tokens=200)
            spent += cost_inr(MODEL, r)
            f.write(json.dumps({"method": method, "id": it["id"], "source": it["source"],
                                "verdict": parse(r["text"]), "text": r["text"]}) + "\n")
            f.flush()
print(f"spent ~Rs{spent:.2f} (estimate; real cost has run ~2x this)")

ids = {it["id"] for it in QS}
base = [r for r in map(json.loads, open(os.path.join(c.ROOT, "results", "flash-lite-2.5__st.jsonl"), encoding="utf-8"))
        if r["change"] == "T3_refuse" and r["id"] in ids]
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
print("\n| attack | refusals marked correct by Gemini 2.5 Flash-Lite (40 held-out questions) |\n|---|---|")
print(f"| none (plain refusal) | {sum(r['verdict'] == 'correct' for r in base)} of {len(base)} |")
for m in sorted(attacks):
    g = [r for r in rows if r["method"] == m]
    print(f"| {m} | {sum(r['verdict'] == 'correct' for r in g)} of {len(g)} (no verdict: {sum(not r['verdict'] for r in g)}) |")
