# White-box GCG attacks on a reference-guided judge

**Question:** when an attacker can look inside a judge, how much stronger is the attack than one that only sees scores, and does the stronger attack still work on other judges, or does it only exploit quirks of the one judge it was tuned on?

**Setup:** 118 harmless questions (40 personal-memory questions from LongMemEval, 78 trivia questions from NQ-open), each with an expected answer. The answer being graded is the refusal "I don't have enough information to answer that." followed by one fixed attack text. The attack succeeds when the judge marks the refusal **correct**. 10 training / 108 held-out questions (shuffled with seed 0). Nothing here involves harmful content: GCG's target is the single word "correct".

The full write-up, with every experiment's what / why / result / conclusion, is in **[`WRITEUP.md`](WRITEUP.md)**.

## Headline results

Source judge: Qwen3-4B (Hugging Face, 4-bit NF4), one-word verdict. Counts are refusals marked correct out of 108 held-out questions.

| Attack | On Qwen3-4B (source) | Qwen3-1.7B | Qwen3-4B-2507 | llama3 8B |
|---|---|---|---|---|
| Plain refusal (control) | 0 | 2 | 0 | 0 |
| Best free master key ("Respuesta") | 49 | 1 | 0 | 0 |
| **GCG, seed 0 / seed 1 / seed 2** | **107 / 108 / 108** | 12 / 47 / not run | 0 / 0 / not run | 0 / 0 / not run |

1. **Gradient access breaks the judge.** Three of three GCG seeds pass 107–108 of 108 held-out refusals, in 7–15 minutes each on a laptop GPU.
2. **It generalises across inputs, not across models.** On Qwen3-4B the suffix works after any wording of "I don't know" (106–108/108) and even makes wrong answers pass (seed 1: 83/108, against 10 without it). On other judges it does almost nothing: 0/108 on llama3 and on the updated Qwen3-4B-2507, and partial transfer only to the same-generation sibling Qwen3-1.7B.
3. **Explaining first is not a reliable defence.** When the judge writes a sentence of reasoning before its verdict, GCG still passes 81–91/108 on Qwen, and the "Solution" key nearly doubles (21 → 40). On llama3, the Part 1 readable argument jumps from 27 to 106/108. llama3 adopts the attacker's argument; Qwen invents that the expected answer is present.

## Not done

- PAIR without the hint on Qwen (arm A2), so the readable-against-gradient comparison on transfer is still missing.
- Defences: the perplexity filter (D1) and the defence sentence (D2).
- GCG seed 2 through the transfer and stress tests.
- The explain-first test on Qwen was stopped early: GCG seed 2 and the Part 1 English sentence were not run. `python src/reason_source.py` resumes it.
- Gemini as a transfer target.

## Requirements

- Python 3.11, and `pip install -r requirements.txt` (torch 2.5.1 with CUDA, transformers 4.57.6, bitsandbytes 0.49.2, accelerate)
- A GPU with about 6 GB of memory (RTX 3050 laptop here). Only one model fits at a time: don't run a GCG script and an Ollama transfer together.
- [Ollama](https://ollama.com) for the transfer tests: `ollama pull llama3`, `ollama pull qwen3:1.7b`, `ollama pull qwen3:4b`. Note that Ollama's `qwen3:4b` is the July 2025 ("2507") update, **not** the same checkpoint as the Hugging Face source.
- `transfer.py` also reads the Part 1 readable attack from `../results/summary__llm__llama3__st.json`. Everything else is self-contained.

## How to run

Run from this folder. On Windows, set `HF_HUB_OFFLINE=1` after the first model download (see Known rough edges).

```bash
python src/check_judge.py all                              # check 1: does the judge judge?
python src/master_keys.py                                  # A1: the 10 master keys
python src/gcg.py --mode berlin                            # GCG sanity check ("The capital of France is Berlin.")
python src/gcg.py --mode judge --seed 1 --steps 200 --chunk 32   # A4: GCG on the judge
python src/stress.py                                       # suffixes on other refusals, wrong and correct answers
python src/transfer.py llama3:latest                       # transfer, one-word and reason-first
python src/transfer.py qwen3:1.7b oneword                  # same-family transfer
python src/reason_source.py                                # explain-first test on the source judge (resumable)
```

## Files

| File | What it does |
|---|---|
| `src/judge.py` | The source judge: loads Qwen3-4B in 4-bit, builds the grading prompt (rubric `st`, one-word format, `VERDICT:` prefilled), returns P(correct) summed over every "correct"-like and "incorrect"-like token. Also the 10/108 split (`TRAIN`, `TEST`). |
| `src/check_judge.py` | Check 1: plain refusal against a correct answer on all 118 questions. |
| `src/master_keys.py` | A1: the 10 master keys from "One Token to Fool LLM-as-a-Judge", alone and after the refusal. Also holds the Wilson interval function. |
| `src/gcg.py` | GCG (written by Vidushi): one-hot gradients, top-256 shortlist, 64 one-token-swap candidates per step, re-tokenisation filter, prefix KV cache, questions added one at a time. `--mode berlin` is the sanity check. |
| `src/stress.py` | Each GCG suffix on five other refusal wordings, two kinds of wrong answer, and correct answers. |
| `src/transfer.py` | Every attack on an Ollama judge, one-word and/or reason-first, with transfer ratios. Picks up every `gcg_judge_seed*.json`. |
| `src/reason_source.py` | Every attack on the source judge, one-word against explain-first. |
| `data/candidates.jsonl` | The 118 questions with expected answers and Part 1's candidate answers (correct, wrong, refusal, …). |
| `WRITEUP.md` | Every experiment in order: what, why, results with intervals, conclusions, caveats. |

## Output

All in `results/`:
- `check1__Qwen3-4B.jsonl`, `a1__Qwen3-4B.jsonl`: one row per question (and key).
- `gcg_<mode>_seed<N>.json`: the final suffix, its token IDs, step history, and train / held-out scores. Rerunning the same mode and seed **overwrites** this file.
- `logs/gcg_<mode>_seed<N>_<timestamp>.log` and `.steps.jsonl`: full per-step logs, never overwritten.
- `transfer__<model>.jsonl`, `stress__Qwen3-4B.jsonl`, `reason__Qwen3-4B.jsonl`: one row per (attack, format, question). The transfer and reason scripts skip rows already present, so they can be resumed.

## Known rough edges

- **Windows model download:** `snapshot_download` hit `WinError 1314` (no permission to create symlinks) and left `config.json` and `vocab.json` out of the snapshot folder. They had to be copied in by hand. Turning on Windows Developer Mode, or downloading with `local_dir=`, avoids this.
- **llama3 often skips the `VERDICT:` line** when asked to reason first. `transfer.py` then feeds its reply plus `VERDICT:` back and reads the next token, and records the row as `forced`.
- **Seed 2 stopped on a borderline question:** GCG's cached scoring said all 10 training questions passed, a fresh re-score says 9 of 10.
- `stress.py` rewrites its output file from scratch on each run (not resumable).
