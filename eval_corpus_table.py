"""One table to decide whether a corpus is usable by *this* model.

Important: cross entropy here is measured the same way training measures it.

  * QA-shaped text is split into (prompt, answer) pairs and only the answer
    tokens are scored -- the prompt is masked out, exactly like the `.mask`
    sidecar does during training. Scoring the prompt tokens too inflates the
    number by more than a nat, because the model was never trained to predict
    summary text: it was only ever trained to *condition* on it.
  * Each example becomes its own window. Slicing one long stream across
    examples would charge the model for the jump between unrelated examples,
    which training never does.

Plain prose (no "User:" markers) falls back to 129-wide windows, which is how
the story half of the corpus is fed.

Reference points on model/model_mixed4_128.pt.best:
  chat val   2.04   story val  2.38   in-domain prose ~2.8 (unmasked)
  FineWeb    5.20   Cosmopedia 6.09

    python eval_corpus_table.py model/model_mixed4_128.pt.best \
        control=E:/espllm_data/corpus_probe/kind_control.txt ...
"""
import math
import os
import statistics
import sys

os.environ.setdefault("ESPGPT_WEIGHT_BITS", "1.58")
os.environ.setdefault("ESPGPT_BLOCK_SIZE", "128")
os.environ.setdefault("ESPGPT_DATASET", "dataset_mixed4.txt")

import torch
import torch.nn.functional as F

import main

torch.set_num_threads(int(os.environ.get("THREADS", "6")))
BLOCK = int(main.block_size)          # 128
WIN = BLOCK + 1


def fwd(self, x):
    return F.linear(self._quantize_activation(x),
                    self._quantize_weight(self.weight), self.bias)


main.BitLinear.forward = fwd


def load(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd.pop("__arch__", None)
    sd.pop("rope_cos", None)
    sd.pop("rope_sin", None)
    m = main.Transformer(group_size=main.qat_group_size)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    m.eval()
    return m


def parse_qa(text):
    """Return [(prompt, answer)] for text written as 'User: ...\\nBot: ...'."""
    pairs = []
    lines = [l.rstrip("\n") for l in text.splitlines()]
    i = 0
    while i < len(lines) - 1:
        if lines[i].startswith("User:") and lines[i + 1].startswith("Bot:"):
            ans = lines[i + 1][4:].strip()
            if ans:
                pairs.append((lines[i], ans))
            i += 2
        else:
            i += 1
    return pairs


@torch.no_grad()
def score_masked(model, examples):
    """Answer-only CE, one example per window, prompt tokens masked to -100."""
    total_ce = 0.0
    total_n = 0
    per_ex = []
    for k in range(0, len(examples), 8):
        chunk = examples[k:k + 8]
        B = len(chunk)
        x = torch.zeros(B, BLOCK, dtype=torch.long)
        y = torch.full((B, BLOCK), -100, dtype=torch.long)
        for b, (p_ids, a_ids) in enumerate(chunk):
            ids = p_ids + a_ids
            n = min(len(ids), WIN)
            x[b, :n - 1] = torch.tensor(ids[:n - 1], dtype=torch.long)
            yy = torch.tensor(ids[1:n], dtype=torch.long)
            keep = torch.full_like(yy, -100)
            # the token at index len(p_ids)-1 predicts the first answer token
            keep[len(p_ids) - 1:] = yy[len(p_ids) - 1:]
            y[b, :n - 1] = keep
        logits, _ = model(x, None)
        lp = F.log_softmax(logits.float(), dim=-1)
        safe = y.clamp(min=0)
        ce = -lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        valid = (y != -100)
        for b in range(B):
            v = valid[b]
            if v.sum() > 0:
                per_ex.append(ce[b][v].mean().item())
        total_ce += (ce * valid).sum().item()
        total_n += valid.sum().item()
    return total_ce / max(total_n, 1), per_ex


@torch.no_grad()
def score_plain(model, ids):
    n = (len(ids) // WIN) * WIN
    if n == 0:
        return float("nan")
    w = ids[:n].view(-1, WIN)
    ls = []
    for i in range(0, len(w), 8):
        chunk = w[i:i + 8]
        x, y = chunk[:, :-1], chunk[:, 1:]
        logits, _ = model(x, None)
        lp = F.log_softmax(logits.float(), dim=-1)
        ce = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ls.extend(ce.mean(dim=1).tolist())
    return statistics.mean(ls)


def probe(model, name, path, limit):
    if not os.path.exists(path):
        print("%-16s missing %s" % (name, path))
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read(limit)
    if not text.strip():
        print("%-16s empty" % name)
        return

    pairs = parse_qa(text)
    if len(pairs) >= 5:
        ex = []
        chars = toks = 0
        for p, a in pairs[:800]:
            pid = main.encode(p).tolist()
            # training caps the prompt at 96 tokens; anything longer would be
            # truncated on the board, so it does not count as usable data
            if len(pid) > 96:
                continue
            aid = main.encode(" " + a).tolist()
            if len(pid) + len(aid) > WIN:
                continue
            ex.append((pid, aid))
            chars += len(p) + len(a) + 1
            toks += len(pid) + len(aid)
        if not ex:
            print("%-16s no example fits a %d-token window" % (name, BLOCK))
            return
        ce, _ = score_masked(model, ex)
        al = [len(a) for _, a in ex]
        print("%-16s %8.2f %7s %7.3f %9.1f %7d %6d  fits %d/%d, ans-only CE"
              % (name, chars / max(toks, 1), "-", ce, math.exp(ce),
                 statistics.median(al), sum(len(a) for _, a in ex),
                 len(ex), len(pairs)))
    else:
        ids = main.encode(text)
        ce = score_plain(model, ids)
        print("%-16s %8.2f %7s %7.3f %9.1f %7s %6d  plain windows"
              % (name, len(text) / max(len(ids), 1), "-", ce,
                 math.exp(ce) if ce == ce else float("nan"), "-", len(ids)))


def main_run():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = 200000
    for a in sys.argv[1:]:
        if a.startswith("--limit"):
            limit = int(a.split("=")[1])
    ckpt = args[0] if args and os.path.exists(args[0]) else "model/model_mixed4_128.pt.best"
    cases = [a for a in args if "=" in a]
    if not cases:
        cases = [
            "in_domain=data/tinystories/instruct_train.txt",
            "control=E:/espllm_data/corpus_probe/kind_control.txt",
        ]
    model = load(ckpt)
    print("checkpoint:", ckpt, "| max bytes per corpus:", limit)
    print("%-16s %8s %7s %7s %9s %7s %6s" %
          ("corpus", "chars/tk", "", "CE", "ppl", "ans tok", "tokens"))
    print("-" * 74)
    for c in cases:
        name, path = c.split("=", 1)
        probe(model, name, path, limit)


if __name__ == "__main__":
    main_run()
