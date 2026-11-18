"""Long-form generation with a sliding context window.

The board can only emit 40-80 tokens per turn: INFER_CTX=104 and the reply
budget eat the rest.  The natural fix is to keep generating and let the oldest
tokens fall out of the window -- the model is a continuation model, so losing
the head of a story is exactly what a fixed block_size LM already does at
training time.

Two questions have to be answered before any firmware change:

  1. Does the text survive past ~200 tokens, or does it collapse into a
     repetition loop?  No amount of KV-cache surgery fixes a looping model.
  2. How much context does it actually need?  104 (the board's window) vs 64
     vs 128 changes the whole design.

This answers both on the PC, where a sliding window is just `ids[-win:]`
(main.Transformer.forward slices rope_cos[:T], so positions are renumbered to
0..T-1 on every call -- identical to what a cache slide would do).

CLI:
    python gen_long.py --prompt "Once upon a time, there was a little dog" --max 400
    python gen_long.py --sweep --max 300

Importable:
    import gen_long
    gen_long.load("model/model_5l32e.pt.best")
    gen_long.run(prompt, win=104, maxnew=200, seed=0)
"""
import argparse
import os
import re
import sys
import time
from collections import Counter

_MAIN = None          # the imported main.py module, once load() has run
_MODEL = None


def _arch_env(ckpt):
    """Set ESPGPT_* overrides from the checkpoint's own __arch__ block.

    main.py reads these at import time, so this has to happen before `import
    main`.  Without it the model is built at the default 4L/20E shape and a
    5L/32E checkpoint fails to load.
    """
    import torch
    arch = torch.load(ckpt, map_location="cpu", weights_only=False).get("__arch__", {})
    for key, env in (("n_layer", "ESPGPT_N_LAYER"), ("n_experts", "ESPGPT_N_EXPERT"),
                     ("n_embd", "ESPGPT_N_EMBD"), ("n_head", "ESPGPT_N_HEAD"),
                     ("n_kv_head", "ESPGPT_N_KV_HEAD"), ("moe_hidden", "ESPGPT_MOE_HIDDEN"),
                     ("block_size", "ESPGPT_BLOCK_SIZE")):
        if key in arch:
            os.environ[env] = str(arch[key])
    if "shared_expert" in arch:
        os.environ["ESPGPT_SHARED_EXPERT"] = "1" if arch["shared_expert"] else "0"
    if "cla" in arch:
        os.environ["ESPGPT_CLA"] = "1" if arch["cla"] else "0"
    return arch


def load(ckpt="model/model_5l32e.pt.best", threads=8):
    """Build the model once; every later run() call reuses it."""
    global _MAIN, _MODEL
    if _MODEL is not None:
        return _MODEL
    arch = _arch_env(ckpt)
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import main as _m
    torch.set_num_threads(int(os.environ.get("THREADS", threads)))

    def _fwd(self, x):
        return F.linear(self._quantize_activation(x),
                        self._quantize_weight(self.weight), self.bias)

    _m.BitLinear.forward = _fwd
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    for _k in ("__arch__", "rope_cos", "rope_sin"):
        sd.pop(_k, None)
    model = _m.Transformer(group_size=_m.qat_group_size)
    res = model.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    # rope_cos/sin are popped above: they are a pure function of block_size, not
    # learned weights, and the model regenerates them in __init__.
    assert not [k for k in res.missing_keys if not k.startswith("rope_")], res.missing_keys
    model.eval()
    _m._ARCH = arch
    _MAIN, _MODEL = _m, model
    return model


def enc(text):
    t = _MAIN.encode(text)
    return t.tolist() if hasattr(t, "tolist") else list(t)


def dec(ids):
    return _MAIN.tokenizer.decode(list(ids))


def run(prompt, win=104, maxnew=400, seed=0, temp=0.7, topk=40, rep=1.0,
        repwin=20, pin=0):
    """Generate maxnew tokens, keeping only the last `win` in the window.

    `pin` keeps the first N prompt tokens bolted to the front of the window
    forever (an attention sink).  Without it the "Once upon a time" opening
    falls out of the window after ~`win` steps and the model, seeing a window
    that looks like the middle of nowhere, starts a brand new story.
    """
    import torch
    if _MODEL is None:
        load()
    g = torch.Generator().manual_seed(seed)
    ids = enc(prompt)
    prompt_len = len(ids)
    out, nlls = [], []
    with torch.no_grad():
        for _ in range(maxnew):
            if pin and len(ids) > win:
                x_ids = ids[:pin] + ids[-(win - pin):]
            else:
                x_ids = ids[-win:]
            logits, _ = _MODEL(torch.tensor([x_ids], dtype=torch.long))
            lg = logits[0, -1].float().clone()
            if rep:
                for c in ids[-repwin:]:
                    lg[c] -= rep
            v, i = torch.topk(lg, min(topk, lg.numel()))
            if temp <= 0.0:
                # Greedy matches the board's `:temp 0` path, which is what makes
                # a PC/board text comparison meaningful.  The NLL signal is
                # meaningless without a distribution, so it is left at 0.
                nxt = int(i[0])
                nlls.append(0.0)
            else:
                probs = torch.softmax(v / temp, -1)
                pick = torch.multinomial(probs, 1, generator=g)
                nxt = int(i[pick])
                # NLL of the token we are about to emit, under the same top-k
                # distribution the sampler used.  Cheapest "is it degenerating"
                # signal: a loop drives it toward 0, incoherence drives it up.
                nlls.append(float(-torch.log(probs[pick])))
            ids.append(nxt)
            out.append(nxt)
    return prompt_len, out, nlls


def metrics(out):
    text = dec(out)
    grams = lambda seq, n: Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
    g8 = grams(out, 8)
    rep8 = sum(c - 1 for c in g8.values() if c > 1) / max(1, sum(g8.values()))
    top8, top8n = ("-", 0)
    if g8:
        top8, top8n = g8.most_common(1)[0]
        top8 = dec(list(top8))
    d4 = len(set(tuple(out[i:i + 4]) for i in range(len(out) - 3))) / max(1, len(out) - 3)
    return {
        "words": len(re.findall(r"[a-z']+", text.lower())),
        "distinct1": len(set(out)) / max(1, len(out)),
        "distinct4": d4,
        "rep8": rep8,
        "top8": top8,
        "top8n": top8n,
        "sent": len(re.findall(r"[.!?]", text)) / max(1, len(out) / 100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/model_5l32e.pt.best")
    ap.add_argument("--prompt", default="Once upon a time, there was a little dog named Max")
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--win", type=int, default=104)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--rep", type=float, default=1.0)
    ap.add_argument("--repwin", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--pin", type=int, default=0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--wins", default="32,64,104,128")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    load(a.ckpt)
    arch = getattr(_MAIN, "_ARCH", {})
    print(f"ckpt {a.ckpt}  arch {arch.get('n_layer')}L/{arch.get('n_experts')}E "
          f"block {_MAIN.block_size}  temp {a.temp} topk {a.topk} rep {a.rep}\n")
    print(f'prompt: {a.prompt}\n')

    wins = [int(w) for w in a.wins.split(",")] if a.sweep else [a.win]
    rows = []
    for win in wins:
        for s in range(a.seeds):
            t0 = time.time()
            _, out, nlls = run(a.prompt, win, a.max, a.seed + s, a.temp, a.topk,
                               a.rep, a.repwin, a.pin)
            dt = time.time() - t0
            m = metrics(out)
            q = max(1, len(nlls) // 10)
            head = sum(nlls[:q]) / q
            tail = sum(nlls[-q:]) / q
            m.update(win=win, pin=a.pin, seed=a.seed + s, nll=head, nll_end=tail,
                     ms=1000 * dt / len(out), text=dec(out), ntok=len(out))
            rows.append(m)
            print(f"[win {win:3d} pin {a.pin:2d} seed {a.seed + s}] {len(out)} tok in "
                  f"{dt:.1f}s ({m['ms']:.0f} ms/tok)  words {m['words']}  "
                  f"nll {head:.2f}->{tail:.2f}  rep8 {m['rep8']:.2f}  "
                  f"top8 x{m['top8n']}  distinct4 {m['distinct4']:.2f}")
            print("-" * 100)
            print(m["text"])
            print("-" * 100)
            if m["top8n"] > 1:
                print(f"most repeated 8-gram (x{m['top8n']}): {m['top8']!r}")
            print()

    if len(rows) > 1:
        print(f"{'win':>4} {'tok':>5} {'words':>6} {'nll':>6} {'nll_end':>8} "
              f"{'rep8':>6} {'top8':>5} {'dist4':>6} {'sent/100':>9}")
        for m in rows:
            print(f"{m['win']:>4} {m['ntok']:>5} {m['words']:>6} {m['nll']:>6.2f} "
                  f"{m['nll_end']:>8.2f} {m['rep8']:>6.2f} {m['top8n']:>5} "
                  f"{m['distinct4']:>6.2f} {m['sent']:>9.1f}")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for m in rows:
                f.write(f"=== win {m['win']} seed {m['seed']} ===\n{m['text']}\n\n")
            f.write(str([{k: v for k, v in m.items() if k != "text"} for m in rows]))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
