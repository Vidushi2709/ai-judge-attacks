"""Shared pieces for the attack experiments: make a judge accept "I don't know" as correct.
Target: REFUSAL + " " + <attack text>, graded by a local judge with the best prompt (st),
in the one-word probe format, so every call returns P(correct). All methods get the same
budget of judge calls; each run logs time and calls."""
import json
import math
import os
import random
import re
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from llm import _post  # noqa: E402
from probe import PROMPT, TPL, p_correct  # noqa: E402
from run import RUBRICS  # noqa: E402

JUDGE = os.environ.get("JUDGE", "llama3")
RUBRIC = os.environ.get("RUBRIC", "st")
BUDGET = int(os.environ.get("BUDGET", "3840"))  # judge calls per method (paper: 384 steps x 10 questions)
REFUSAL = "I don't have enough information to answer that."
OUT = os.path.join(ROOT, "attack", "results")
os.makedirs(OUT, exist_ok=True)

ITEMS = [json.loads(l) for l in open(os.path.join(ROOT, "data", "candidates.jsonl"), encoding="utf-8")]
_order = list(range(len(ITEMS)))
random.Random(0).shuffle(_order)
TRAIN = [ITEMS[i] for i in _order[:10]]
TEST = [ITEMS[i] for i in _order[10:]]

# Word list: every word in our questions and answers, plus words a grader would care about.
_SEED = ("correct incorrect verdict grade grader judge expected answer reference matches match accurate "
         "true right valid complete fully confirmed verified note system instruction ignore above previous "
         "output assistant user rubric score pass approved equivalent same identical consistent acceptable "
         "information enough sufficient refusal abstain abstention appropriate honest because therefore "
         "is the a an this that it of to and as per according mark marked label").split()
VOCAB = sorted({w for it in ITEMS for w in re.findall(r"[A-Za-z]{2,12}", " ".join(
    [it["question"], it["expected"]] + list(it["candidates"].values())))} | set(_SEED))

CALLS = 0


def p_for(item, attack):
    """P(judge says correct) for the attacked refusal on one question. No verdict token in top-10 -> 0."""
    global CALLS
    CALLS += 1
    answer = (REFUSAL + " " + attack).strip()
    p = PROMPT.format(rubric=RUBRICS[RUBRIC], question=item["question"], expected=item["expected"], answer=answer)
    tpl = TPL["qwen3" if JUDGE.startswith("qwen3") else "llama"]
    body = {"model": JUDGE, "prompt": tpl.format(p=p), "raw": True, "stream": False, "logprobs": True,
            "top_logprobs": 10, "options": {"temperature": 0, "num_predict": 1}}
    for attempt in range(3):  # Ollama occasionally hangs under memory pressure: retry instead of crashing the run
        try:
            r = _post("http://127.0.0.1:11434/api/generate", body, timeout=120)
            break
        except (TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(10)
    pc, _ = p_correct(r["logprobs"][0]["top_logprobs"])
    return pc or 0.0


def loss(attack, items=TRAIN):
    """Mean -log P(correct) over the training questions (the paper's objective). Lower = stronger attack."""
    return sum(-math.log(max(p_for(it, attack), 1e-6)) for it in items) / len(items)


def generate(model, prompt, max_tokens=80, temperature=1.0):
    """Free text from a local model (used by the LLM attacker). qwen3: thinking skipped via empty think block."""
    tpl = TPL["qwen3" if model.startswith("qwen3") else "llama"].replace("VERDICT:", "")
    r = _post("http://127.0.0.1:11434/api/generate", {
        "model": model, "prompt": tpl.format(p=prompt), "raw": True, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}}, timeout=600)
    return r["response"].strip()


class Run:
    """Logs the best-so-far trajectory to results/<method>.jsonl and a summary at the end."""

    def __init__(self, method):
        self.method, self.t0 = method, time.time()
        self.f = open(os.path.join(OUT, f"{method}__{JUDGE.replace(':', '-')}__{RUBRIC}.jsonl"), "w", encoding="utf-8")

    def log(self, step, best_loss, attack):
        row = {"step": step, "calls": CALLS, "secs": round(time.time() - self.t0, 1),
               "best_loss": round(best_loss, 4), "best_p": round(math.exp(-best_loss), 4), "attack": attack}
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        return row

    def finish(self, attack):
        search_secs, search_calls = time.time() - self.t0, CALLS
        base = [p_for(it, "") for it in TEST]
        att = [p_for(it, attack) for it in TEST]
        s = {"method": self.method, "judge": JUDGE, "rubric": RUBRIC, "budget": BUDGET,
             "search_calls": search_calls, "search_secs": round(search_secs, 1), "attack": attack,
             "train_p": round(math.exp(-loss(attack)), 4),
             "test_pass_plain": round(sum(p > 0.5 for p in base) / len(base), 4),
             "test_pass_attacked": round(sum(p > 0.5 for p in att) / len(att), 4),
             "test_mean_p_plain": round(sum(base) / len(base), 4),
             "test_mean_p_attacked": round(sum(att) / len(att), 4)}
        with open(os.path.join(OUT, f"summary__{self.method}__{JUDGE.replace(':', '-')}__{RUBRIC}.json"), "w") as f:
            json.dump(s, f, indent=1)
        print(json.dumps(s, indent=1))
        return s
