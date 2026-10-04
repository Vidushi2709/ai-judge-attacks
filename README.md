# ai-judge-attacks

Search for a fixed text that makes an LLM judge accept a refusal ("I don't have enough
information to answer that.") as a correct answer, even though the judge is given the
expected answer. The judge runs locally through Ollama and reports the probability it
assigns to "correct", which is what the searches optimise. A final step checks whether
the attacks found locally also work on Gemini 2.5 Flash-Lite.

Three search methods are compared on an equal budget of judge calls, plus two warm-start
runs. Results and the full write-up are in [`WRITEUP.md`](WRITEUP.md); raw logs are in
`results/`.

## What this folder needs from the parent project

This folder is not standalone. It was written inside the `judge_reliability` project and
imports from it. To run it, these must exist one level up:

| Path (relative to this folder) | Used for |
|---|---|
| `../src/llm.py` | `_post` (Ollama HTTP call), `ask` (Gemini call); reads `../.env` for `GEMINI_API_KEY` |
| `../src/probe.py` | the one-word judge prompt, the chat templates per model family, and `p_correct` (turns top-10 next-token probabilities into P(correct)) |
| `../src/run.py` | the grading prompts (`RUBRICS`), the reason-then-verdict template, the verdict parser, and Gemini price table |
| `../data/candidates.jsonl` | the 118 questions, each with its expected answer and key value |
| `../results/flash-lite-2.5__st.jsonl` | Gemini's verdicts on the plain refusal, used as the baseline in `transfer.py` |
| `../.env` | `GEMINI_API_KEY=...` (only for `transfer.py`) |

If you clone only this folder, copy those files into the same relative places, or edit
`common.py` (`ROOT`, line 13) and `transfer.py` to point at where you put them.

## Requirements

- Python 3.11 (standard library only, except `matplotlib` for `compare.py`)
- [Ollama](https://ollama.com) running locally with the judge and attacker models pulled:
  `ollama pull llama3` and `ollama pull mistral` (defaults), `ollama pull qwen3:4b` if you
  want that judge
- A GPU with about 6 GB of VRAM is enough; llama3 8B partly spills to the CPU at that size
- A Gemini API key, only for the transfer test

## How to run

Run from this folder.

```bash
python random_search.py            # ~18 min on an RTX 3050
python greedy_search.py            # ~15 min
python llm_attacker.py             # ~16 min; mistral writes, llama3 judges
BUDGET_INR=0.5 python transfer.py  # Gemini check; stops when the estimate passes the budget
python compare.py                  # table on stdout + results/convergence.png
```

Warm start (continue from a saved attack instead of random words):

```bash
cp results/summary__random__llama3__st.json results/init_from_random.json
INIT=results/init_from_random.json NAME=random_cont python random_search.py
INIT=results/init_from_random.json NAME=greedy_warm python greedy_search.py
```

## Settings

All settings are environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `JUDGE` | `llama3` | Ollama model that grades. Chat template is picked by name prefix (`qwen3` or llama-style). |
| `RUBRIC` | `st` | Which grading prompt from `../src/run.py` the judge gets |
| `BUDGET` | `3840` | Judge calls per search run (the paper's 384 steps × 10 training questions) |
| `ROUNDS` | `30` | LLM attacker rounds (each = 1 attacker call + 10 judge calls) |
| `ATTACKER` | `mistral` | Ollama model that writes attack text |
| `INIT` | unset | Path to a `summary__*.json`; its attack becomes the starting text |
| `NAME` | method name | Output name for the run (use with `INIT` so the original isn't overwritten) |
| `BUDGET_INR` | `0.5` | `transfer.py` only: stop when the estimated Gemini spend passes this. Estimates have run about 2× below the real bill. |

## Files

| File | What it does |
|---|---|
| `common.py` | Judge call and P(correct), the loss, the 10/108 train/test split (seed 0), the word list, run logging, final scoring on held-out questions |
| `random_search.py` | Change one word at random; keep it if the training loss drops. 10 judge calls per step. |
| `greedy_search.py` | Try 8 one-word changes; keep the best if it beats the current text. 80 judge calls per step. |
| `llm_attacker.py` | PAIR-style: the attacker model sees its 5 best attempts and their scores and writes a new one. Keeps only the first line of its output. |
| `transfer.py` | Grades each method's best attack with Gemini 2.5 Flash-Lite on the first 40 held-out questions, in the reason-then-verdict format |
| `compare.py` | Table of calls, time, calls-to-first-success, train and held-out scores; draws `results/convergence.png` |
| `WRITEUP.md` | Setup, results, what went wrong, limits |

## Output format

- `results/<name>__<judge>__<rubric>.jsonl`: one line per step: `step`, `calls`, `secs`,
  `best_loss`, `best_p` (mean P(correct) on the 10 training questions), `attack`.
- `results/summary__<name>__<judge>__<rubric>.json`: calls and seconds used, the final
  attack, `train_p`, and on the 108 held-out questions: share of refusals passed and mean
  P(correct), with and without the attack.
- `results/transfer__flash-lite-2.5__st.jsonl`: one line per (method, question) with
  Gemini's verdict and its full reply.

Runs are resumable only in the sense that `transfer.py` skips (method, question) pairs it
has already graded. The search scripts start fresh each time; use `NAME` to keep old runs.

## Known rough edges

- Ollama occasionally hangs under memory pressure. Judge calls time out after 2 minutes
  and retry up to 3 times (`common.py`, `p_for`).
- Random and greedy search share a random seed, so they draw the same candidate changes.
  Give one of them a different seed before treating them as independent runs.
- On Windows, call Ollama at `127.0.0.1`, not `localhost`: the name lookup added about
  2 seconds per call.
