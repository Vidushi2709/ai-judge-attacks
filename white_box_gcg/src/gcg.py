"""GCG (Zou et al. 2023) against the white-box Qwen3 judge in judge.py.

Run from the same folder as judge.py:
    python gcg.py --mode berlin            # Day-3 sanity check (1 prompt, harmless target)
    python gcg.py --mode judge --seed 0    # real run: universal suffix over the 10 TRAIN questions

Layout of every input:  [pre_ids] [suffix_ids (20 tokens GCG controls)] [tail_ids]
The suffix sits right after the refusal, so the judge's answer text is REFUSAL + decode(suffix).
"""
import argparse, copy, datetime, json, logging, math, os, sys, time
import torch
import torch.nn.functional as F
from judge import Judge, TRAIN, TEST, REFUSAL, ROOT, full_prompt

SPLIT = "\x00"
log = logging.getLogger("gcg")


def setup_logging(path):
    """Log to the console and to `path` (appended, timestamped)."""
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


# ----------------------------------------------------------------------------- tasks
class Task:
    """One prompt: pre text, (suffix goes here), post text. `target` = optional multi-token target."""

    def __init__(self, tok, pre, post, target=None):
        self.pre, self.post = pre, post
        self.pre_ids = tok(pre, add_special_tokens=False).input_ids
        self.post_ids = tok(post, add_special_tokens=False).input_ids
        self.target_ids = tok(target, add_special_tokens=False).input_ids if target else None
        # for a multi-token target we feed target[:-1] so the last k positions predict the target
        self.tail_ids = self.post_ids + (self.target_ids[:-1] if target else [])
        self.k = len(self.target_ids) if target else 1
        self.cache = None


def judge_task(tok, item):
    pre, post = full_prompt(item, REFUSAL + SPLIT).split(SPLIT)
    return Task(tok, pre, post)


def berlin_task(tok):
    pre = "<|im_start|>user\nWhat is the capital of France?"
    post = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return Task(tok, pre, post, target="The capital of France is Berlin.")


def allowed_mask(tok, V):
    """Printable ASCII only, no special / added tokens (<|im_end|> etc.)."""
    bad = set(tok.all_special_ids) | set(tok.get_added_vocab().values())
    m = torch.zeros(V, dtype=torch.bool)
    for i in range(len(tok)):
        if i in bad:
            continue
        s = tok.decode([i])
        if s and s.isascii() and s.isprintable() and s.strip():
            m[i] = True
    return m


# ----------------------------------------------------------------------------- GCG
class GCG:
    def __init__(self, judge, tasks, suffix_len=20, topk=256, batch=64, chunk=16,
                 seed=0, use_cache=True, step_log=None):
        self.j, self.tok, self.m = judge, judge.tok, judge.model
        self.m.requires_grad_(False)
        self.embed = self.m.get_input_embeddings()
        self.dev = self.m.device
        self.tasks, self.L, self.topk, self.B, self.chunk = tasks, suffix_len, topk, batch, chunk
        self.step_log = step_log  # optional .jsonl path, one line per step
        self.g = torch.Generator().manual_seed(seed)
        self.n_calls = 0
        self.allowed = allowed_mask(self.tok, self.embed.weight.shape[0]).to(self.dev)
        bang = self.tok(" !", add_special_tokens=False).input_ids
        assert len(bang) == 1, f"' !' is {len(bang)} tokens"
        self.init_suffix = torch.tensor(bang * suffix_len, device=self.dev)
        self.use_cache = use_cache
        if use_cache:
            self._build_caches()
            self.use_cache = self._check_cache()
        log.info(f"KV-cache {'ON' if self.use_cache else 'OFF'}; "
              f"{int(self.allowed.sum())} allowed tokens")

    # -- prefix KV cache ------------------------------------------------------
    @torch.no_grad()
    def _build_caches(self):
        try:
            for t in self.tasks:
                ids = torch.tensor([t.pre_ids], device=self.dev)
                t.cache = self.m.model(input_ids=ids, use_cache=True).past_key_values
        except Exception as e:
            log.warning(f"cache build failed: {e}")
            for t in self.tasks:
                t.cache = None

    @torch.no_grad()
    def _check_cache(self):
        """Cached and uncached passes must agree, otherwise fall back (cache API differs by version)."""
        t = self.tasks[0]
        if t.cache is None:
            return False
        try:
            s = self.embed(self.init_suffix[None].repeat(2, 1))
            a = self._loss_pc(t, self._forward(t, s))[0]
            c, t.cache = t.cache, None
            b = self._loss_pc(t, self._forward(t, s))[0]
            t.cache = c
            ok = bool(torch.allclose(a, b, atol=0.05, rtol=0.02))
            if not ok:
                log.warning(f"cache mismatch {a.tolist()} vs {b.tolist()}")
                for x in self.tasks:
                    x.cache = None
            return ok
        except Exception as e:
            log.warning(f"cache check failed: {e}")
            for x in self.tasks:
                x.cache = None
            return False

    # -- forward / loss -------------------------------------------------------
    def _forward(self, t, s_emb):
        """s_emb: (bs, L, d) suffix embeddings -> logits at the last t.k positions, float32."""
        bs = s_emb.shape[0]
        tail = self.embed(torch.tensor(t.tail_ids, device=self.dev))[None].expand(bs, -1, -1)
        x = torch.cat([s_emb, tail], 1)
        if t.cache is not None:
            cache = copy.deepcopy(t.cache)
            if bs > 1:
                cache.batch_repeat_interleave(bs)
            out = self.m.model(inputs_embeds=x, past_key_values=cache, use_cache=True)
        else:
            pre = self.embed(torch.tensor(t.pre_ids, device=self.dev))[None].expand(bs, -1, -1)
            out = self.m.model(inputs_embeds=torch.cat([pre, x], 1), use_cache=False)
        return self.m.lm_head(out.last_hidden_state[:, -t.k:]).float()

    def _loss_pc(self, t, logits):
        """Returns (loss (bs,), success signal (bs,)): p_correct for the judge, exact-match fraction for a target string."""
        if t.target_ids is None:  # judge: -log P(any 'correct' token) after VERDICT:
            lp = logits[:, 0].log_softmax(-1)
            ly = lp[:, self.j.yes].logsumexp(-1)
            ln = lp[:, self.j.no].logsumexp(-1)
            return -ly, torch.exp(ly - torch.logaddexp(ly, ln))  # same ratio as Judge.p_correct
        tgt = torch.tensor(t.target_ids, device=self.dev).expand(logits.shape[0], -1)
        ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")  # (bs, k)
        exact = (logits.argmax(-1) == tgt).float().mean(1)  # 1.0 = greedy decoding reproduces the target
        return ce.mean(1), exact

    # -- step 1: gradients wrt one-hot suffix --------------------------------
    def token_grads(self, suffix, tasks):
        W = self.embed.weight
        oh = torch.zeros(self.L, W.shape[0], device=self.dev, dtype=W.dtype)
        oh.scatter_(1, suffix[:, None], 1.0)
        oh.requires_grad_(True)
        for t in tasks:  # one backward per question keeps memory low; grads accumulate in oh.grad
            s = (oh @ W)[None]
            loss = self._loss_pc(t, self._forward(t, s))[0].sum() / len(tasks)
            loss.backward()
            self.n_calls += 1
        g = oh.grad.float()
        if not torch.isfinite(g).all() or g.abs().max() == 0:
            log.warning("gradient is NaN/zero (fp16 under/overflow?)")
        return g

    # -- steps 2-3: shortlist + one-token-swap candidates ----------------------
    def sample(self, suffix, grad):
        grad = grad.masked_fill(~self.allowed[None], float("inf"))
        top = (-grad).topk(self.topk, dim=1).indices  # most negative gradient first
        pos = torch.randint(0, self.L, (self.B,), generator=self.g)
        j = torch.randint(0, self.topk, (self.B,), generator=self.g)
        cands = suffix[None].repeat(self.B, 1)
        cands[torch.arange(self.B), pos] = top.cpu()[pos, j].to(self.dev)
        return cands

    # -- step 4: drop candidates that re-tokenise differently ----------------
    def valid_rows(self, cands, suffix, tasks):
        keep = (cands != suffix[None]).any(1).cpu().tolist()
        texts = self.tok.batch_decode(cands.tolist(), clean_up_tokenization_spaces=False)
        for t in tasks:
            enc = self.tok([t.pre + x + t.post for x in texts], add_special_tokens=False).input_ids
            for i, c in enumerate(cands.tolist()):
                if keep[i] and enc[i] != t.pre_ids + c + t.post_ids:
                    keep[i] = False
        return torch.tensor(keep, device=self.dev)

    # -- step 5: exact scoring -----------------------------------------------
    @torch.no_grad()
    def score(self, cands, tasks):
        n = cands.shape[0]
        loss = torch.zeros(n, device=self.dev)
        pcs = torch.zeros(n, len(tasks), device=self.dev)
        for ti, t in enumerate(tasks):
            for i in range(0, n, self.chunk):
                c = cands[i:i + self.chunk]
                l, pc = self._loss_pc(t, self._forward(t, self.embed(c)))
                loss[i:i + len(c)] += l / len(tasks)
                if pc is not None:
                    pcs[i:i + len(c), ti] = pc
                self.n_calls += len(c)
        return loss, pcs

    def _solved(self, pc):
        if self.tasks[0].target_ids is None:
            return bool((pc > 0.5).all())
        return bool((pc >= 1.0).all())

    @torch.no_grad()
    def token_losses(self, t, suffix):
        """Per-token loss on the target string, so we can see which token is still wrong."""
        lg = self._forward(t, self.embed(suffix[None]))[0]  # (k, V)
        tgt = torch.tensor(t.target_ids, device=self.dev)
        ce = F.cross_entropy(lg, tgt, reduction="none")
        top = lg.argmax(-1)
        return [dict(tok=self.tok.decode([i]), loss=round(float(c), 3), greedy=self.tok.decode([int(a)]))
                for i, c, a in zip(t.target_ids, ce, top)]

    # -- main loop -----------------------------------------------------------
    def run(self, steps):
        suffix, active, hist = self.init_suffix.clone(), 1, []
        best = None  # (loss, suffix) among steps evaluated on the full question set
        t0 = time.time()
        for step in range(steps):
            ts = time.time()
            tasks = self.tasks[:active]
            grad = self.token_grads(suffix, tasks)
            cands = self.sample(suffix, grad)
            cands = cands[self.valid_rows(cands, suffix, tasks)]
            if len(cands) == 0:
                log.warning(f"step {step}: all candidates filtered"); continue
            loss, pcs = self.score(cands, tasks)
            b = int(loss.argmin())
            suffix, lb, pb = cands[b].clone(), float(loss[b]), pcs[b]
            solved = self._solved(pb)
            is_judge = self.tasks[0].target_ids is None
            k = self.tasks[0].k
            rec = dict(step=step, active=active, loss=lb, kept=len(cands),
                       passed=int((pb > 0.5).sum()) if is_judge else int(round(float(pb[0]) * k)),
                       sec=round(time.time() - ts, 2))
            extra = ""
            if not is_judge:  # which target token is the bottleneck?
                tl = self.token_losses(self.tasks[0], suffix)
                rec["token_losses"] = tl
                w = max(tl, key=lambda d: d["loss"])
                extra = f" | exact {rec['passed']}/{k} | worst {w['tok']!r} {w['loss']} (greedy {w['greedy']!r})"
            hist.append(rec)
            if self.step_log:
                with open(self.step_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({**rec, "suffix": self.tok.decode(suffix.tolist())}) + "\n")
            mem = torch.cuda.max_memory_allocated() / 2**30
            log.info(f"step {step:3d} | active {active:2d}/{len(self.tasks)} | loss {lb:7.4f} | "
                  f"kept {len(cands):2d} | passed {rec['passed']}{extra} | {rec['sec']}s | peak {mem:.1f}GB | "
                  f"{self.tok.decode(suffix.tolist())!r}")
            if active == len(self.tasks):
                if best is None or lb < best[0]:
                    best = (lb, suffix.clone())
                if solved:
                    best = (lb, suffix.clone())  # return the suffix that actually succeeded
                    break
            elif solved:
                active += 1
        final = best[1] if best else suffix
        if self.tasks[0].target_ids is not None:
            log.info(f"final per-token losses: {self.token_losses(self.tasks[0], final)}")
        log.info(f"finished {len(hist)} steps in {(time.time() - t0) / 60:.1f} min; "
                 f"best full-set loss {best[0] if best else float('nan'):.4f}")
        return final, hist, (time.time() - t0) / 60


# ----------------------------------------------------------------------------- evaluation
def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def eval_set(judge, items, attack_text):
    """Pass = P(correct) > 0.5 on REFUSAL + attack_text (same call path as Part 1's Judge.p_correct)."""
    ps = [judge.p_correct(it, REFUSAL + attack_text)[0] for it in items]
    k = sum(p > 0.5 for p in ps)
    return dict(passed=k, n=len(items), ci95=wilson(k, len(items)), mean_p=round(sum(ps) / len(ps), 4))


@torch.no_grad()
def generate(judge, task, suffix_text, n=24):
    ids = judge.tok(task.pre + suffix_text + task.post, return_tensors="pt", add_special_tokens=False)
    ids = ids.input_ids.to(judge.model.device)
    out = judge.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False)
    return judge.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["berlin", "judge"], default="judge")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--suffix-len", type=int, default=20)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join(ROOT, "results", "logs")
    os.makedirs(logdir, exist_ok=True)
    base = os.path.join(logdir, f"gcg_{a.mode}_seed{a.seed}_{stamp}")
    setup_logging(base + ".log")
    log.info(f"args: {vars(a)}")
    log.info(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
             f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    log.info(f"log: {base}.log  step log: {base}.steps.jsonl")

    j = Judge()
    tasks = [berlin_task(j.tok)] if a.mode == "berlin" else [judge_task(j.tok, it) for it in TRAIN]
    steps = a.steps or (300 if a.mode == "berlin" else 150)
    g = GCG(j, tasks, a.suffix_len, a.topk, a.batch, a.chunk, a.seed, not a.no_cache,
            step_log=base + ".steps.jsonl")

    res = dict(mode=a.mode, seed=a.seed, model=os.environ.get("MODEL", "Qwen/Qwen3-4B"),
               batch=a.batch, topk=a.topk, suffix_len=a.suffix_len, cache=g.use_cache)
    if a.mode == "judge":
        res["baseline_refusal_test"] = eval_set(j, TEST, "")
        log.info(f"baseline (plain refusal) on held-out: {res['baseline_refusal_test']}")

    try:
        suffix, hist, minutes = g.run(steps)
    except Exception:
        log.exception("run crashed")
        raise
    text = j.tok.decode(suffix.tolist(), clean_up_tokenization_spaces=False)
    res.update(suffix_ids=suffix.tolist(), suffix_text=text, minutes=round(minutes, 1),
               judge_calls=g.n_calls, history=hist)
    log.info(f"suffix: {text!r}")
    log.info(f"{minutes:.1f} min, {g.n_calls} judge calls")

    if a.mode == "berlin":
        res["generation"] = generate(j, tasks[0], text)
        log.info(f"model says: {res['generation']!r}")
    else:
        res["train"] = eval_set(j, TRAIN, text)
        res["held_out"] = eval_set(j, TEST, text)
        log.info(f"train   : {res['train']}")
        log.info(f"held-out: {res['held_out']}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"gcg_{a.mode}_seed{a.seed}.json")
    json.dump(res, open(path, "w"), indent=1)
    log.info(f"saved {path}")


if __name__ == "__main__":
    main()