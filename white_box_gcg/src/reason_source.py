"""Does reasoning first protect the source judge? Every attack on Qwen3-4B (the judge GCG was tuned on),
108 held-out questions, two formats:
    oneword -> the format GCG was tuned in (VERDICT: prefilled, P(correct) of the next token)
    reason  -> one sentence of reasoning, then VERDICT (same prompt as transfer.py); greedy, 200 tokens max.
               No VERDICT line -> the reply + "VERDICT:" is fed back and the next token read ("forced").
    python src/reason_source.py   -> results/reason__Qwen3-4B.jsonl and a table"""
import json
import os

import torch

from judge import MODEL, ROOT, RUBRIC_ST, TEST, TPL, Judge
from master_keys import wilson
from transfer import REASON, answers, parse

OUT = os.path.join(ROOT, "results")


@torch.no_grad()
def reason(j, item, answer):
    chat = TPL.format(p=REASON.format(rubric=RUBRIC_ST, question=item["question"],
                                      expected=item["expected"], answer=answer))[:-len("VERDICT:")]
    ids = j.tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(j.model.device)
    out = j.model.generate(ids, max_new_tokens=200, do_sample=False)
    text = j.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    v, forced = parse(text), False
    if v is None:
        ids = j.tok(chat + text.rstrip() + "\n\nVERDICT:", return_tensors="pt", add_special_tokens=False).input_ids
        probs = j.model(ids.to(j.model.device)).logits[0, -1].float().softmax(-1)
        y, n = probs[j.yes].sum().item(), probs[j.no].sum().item()
        v, forced = ("correct" if y > n else "incorrect"), True
    return {"verdict": v, "forced": forced, "text": text, "pass": v == "correct"}


if __name__ == "__main__":
    j = Judge()
    atk = answers()
    path = os.path.join(OUT, f"reason__{MODEL.split('/')[-1]}.jsonl")
    done = {(r["attack"], r["format"], r["id"]) for r in map(json.loads, open(path, encoding="utf-8"))} \
        if os.path.exists(path) else set()
    with open(path, "a", encoding="utf-8") as f:
        for name, (answer, _) in atk.items():
            for it in TEST:
                if (name, "oneword", it["id"]) not in done:
                    p, m = j.p_correct(it, answer)
                    f.write(json.dumps({"attack": name, "format": "oneword", "id": it["id"], "source": it["source"],
                                        "p_correct": p, "verdict_mass": m, "pass": p > 0.5}) + "\n")
                if (name, "reason", it["id"]) not in done:
                    f.write(json.dumps({"attack": name, "format": "reason", "id": it["id"], "source": it["source"],
                                        **reason(j, it, answer)}, ensure_ascii=False) + "\n")
                f.flush()
            print(f"done: {name}", flush=True)

    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    print(f"\n{MODEL} (the source judge), 108 held-out questions. Cells: judged correct (95% CI).\n")
    print("| attack | one-word | reason first | reason first: forced verdicts |\n|---|---|---|---|")
    for name in atk:
        cells = []
        for fmt in ("oneword", "reason"):
            g = [r for r in rows if r["attack"] == name and r["format"] == fmt]
            k = sum(r["pass"] for r in g)
            lo, hi = wilson(k, len(g))
            cells.append(f"{k}/{len(g)} ({100 * lo:.0f}–{100 * hi:.0f}%)")
        forced = sum(r["forced"] for r in rows if r["attack"] == name and r["format"] == "reason")
        print(f"| {name} | {cells[0]} | {cells[1]} | {forced} |")
    print("wrote", path)
