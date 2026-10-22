"""Where does the 5L win come from? Per-prompt-family CE on one fixed split.

Same protocol as eval_5l_vs_4l.py (fixed split, fixed batches, identical
forward), but each row is classified by decoding its own prefix, so the two
architectures are compared family by family instead of only in aggregate.
"""
import os

os.environ.setdefault("ESPGPT_DATASET", "E:/espllm_data/dataset_mixed7.txt")
os.environ.setdefault("ESPGPT_BLOCK_SIZE", "128")
os.environ.setdefault("ESPGPT_BATCH_SIZE", "128")
os.environ.setdefault("ESPGPT_WEIGHT_BITS", "1.58")

import random

import torch
import torch.nn.functional as F

import main

N_BATCH = int(os.environ.get("N_BATCH", "60"))
ARMS = [
    ("4L/40E mixed13", "model/model_mixed13_lowlr.pt", 4, 40),
    ("5L/32E 5l32e", "model/model_5l32e.pt.best", 5, 32),
]

# Order matters: the Once prefixes are tested before the generic ones.
FAMILIES = [
    ("cont-once", "User: Continue this story: Once"),
    ("cont-once", "User: Continue this story: One"),
    ("cont", "User: Continue this story:"),
    ("words", "User: Write a story using these words:"),
    ("summary", "User: Summarise"),
]

random.seed(11)
torch.manual_seed(11)
BATCHES = [main.get_batch("val") for _ in range(N_BATCH)]


def family_of(row):
    head = main.tokenizer.decode(row[:14].tolist())
    for name, prefix in FAMILIES:
        if head.startswith(prefix):
            return name
    return "story"


# Bucket each batch into per-family row index lists, once.
BUCKETS = []
for x, y in BATCHES:
    per = {}
    for i in range(x.size(0)):
        per.setdefault(family_of(x[i]), []).append(i)
    BUCKETS.append(per)

counts = {}
for per in BUCKETS:
    for k, idx in per.items():
        counts[k] = counts.get(k, 0) + len(idx)
print("rows per family on the fixed split:",
      {k: counts[k] for k in sorted(counts)})
print(f"{len(BATCHES)} batches x {main.batch_size} rows\n")


def evaluate(path, n_layer, n_experts):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    sd = {k: v for k, v in sd.items() if not k.startswith("__")}
    main.n_layer = n_layer
    main.n_experts = n_experts
    main.dropout = 0.0
    model = main.Transformer(group_size=main.qat_group_size)
    model.load_state_dict(sd)
    model.eval()
    agg = {}
    pooled_tot, pooled_n = 0.0, 0
    with torch.no_grad():
        for (x, y), per in zip(BATCHES, BUCKETS):
            logits, _ = model(x)
            lf = logits.view(-1, logits.size(-1))
            yf = y.view(-1)
            per_tok = F.cross_entropy(lf, yf, reduction="none").view(x.size(0), -1)
            for fam, idx in per.items():
                sub = per_tok[idx].mean().item() * len(idx)
                a = agg.setdefault(fam, [0.0, 0])
                a[0] += sub
                a[1] += len(idx)
            pooled_tot += float(per_tok.mean().item()) * x.size(0)
            pooled_n += x.size(0)
    out = {k: v[0] / v[1] for k, v in agg.items()}
    out["POOLED"] = pooled_tot / pooled_n
    return out


results = []
for label, path, L, E in ARMS:
    if not os.path.exists(path):
        print(f"{label}: MISSING")
        continue
    results.append((label, evaluate(path, L, E)))

cols = sorted(set().union(*[r[1].keys() for r in results]),
              key=lambda k: (k == "POOLED", k))
print(f"{'arm':<18}" + "".join(f"{c:>11}" for c in cols))
print("-" * (18 + 11 * len(cols)))
for label, res in results:
    print(f"{label:<18}" + "".join(f"{res.get(c, float('nan')):>11.4f}" for c in cols))
if len(results) == 2:
    print(f"{'delta':<18}" + "".join(
        f"{results[1][1].get(c, 0) - results[0][1].get(c, 0):>+11.4f}" for c in cols))
