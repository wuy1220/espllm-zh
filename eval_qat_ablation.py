"""What the shipped checkpoint actually loses to weight / activation quantisation.

Training already runs the full QAT path: every projection is a BitLinear with
absmean ternary weights + per-token int8 activations and a straight-through
estimator (main.py:744).  So the usual "PTQ vs QAT" question does not apply --
what is unmeasured is *how much the fake quantisation costs the trained model*
and, more interestingly, whether the activation clamp is a constraint the
firmware does not even impose.

The board keeps activations in float32 (src/main.cpp: every g_* buffer is
float), and only weights are ternary plus K/V int8.  If arm 1 (ternary W, fp32
A) is measurably better than arm 0, then the training-time int8 activation
clamp is simulating a limit we never deploy, and dropping it is free quality.

All four arms share the same batches, so the deltas are paired and the standard
error of the difference is far smaller than the standard error of each CE.

  python eval_qat_ablation.py            # 60 batches, ~1 min
  N_BATCH=200 python eval_qat_ablation.py
"""
import os

os.environ.setdefault("ESPGPT_DATASET", "E:/espllm_data/dataset_mixed7.txt")
os.environ.setdefault("ESPGPT_BLOCK_SIZE", "128")
os.environ.setdefault("ESPGPT_BATCH_SIZE", "128")
os.environ.setdefault("ESPGPT_WEIGHT_BITS", "1.58")

import math
import random
import sys

import torch
import torch.nn.functional as F

import main

CKPT = os.environ.get("EVAL_CKPT", "model/model_5l32e.pt.best")
N_LAYER = int(os.environ.get("N_LAYER", "5"))
N_EXPERT = int(os.environ.get("N_EXPERT", "32"))
N_BATCH = int(os.environ.get("N_BATCH", "60"))

random.seed(11)
torch.manual_seed(11)
BATCHES = [main.get_batch("val") for _ in range(N_BATCH)]
print(f"fixed validation set: {N_BATCH} batches x {main.batch_size} seqs "
      f"x {main.block_size} tokens = {N_BATCH * main.batch_size * main.block_size:,} tokens\n")

main.n_layer = N_LAYER
main.n_experts = N_EXPERT
main.dropout = 0.0
main.BINARY_WEIGHTS = False

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
if hasattr(sd, "state_dict"):
    sd = sd.state_dict()
sd = {k: v for k, v in sd.items() if not k.startswith("__")}
model = main.Transformer(group_size=main.qat_group_size)
missing = [k for k in model.state_dict() if k not in sd]
assert not missing, f"checkpoint does not match arch: missing {missing[:4]}"
model.load_state_dict(sd)
model.eval()


def make_forward(quant_w: bool, quant_a: bool):
    def fwd(self, x):
        w = self._quantize_weight(self.weight) if quant_w else self.weight
        a = self._quantize_activation(x) if quant_a else x
        return F.linear(a, w, self.bias)
    return fwd


ARMS = [
    ("0 ternary W + int8 A (as trained)", True, True),
    ("1 ternary W + fp32 A (the board)  ", True, False),
    ("2 fp32 W    + int8 A              ", False, True),
    ("3 fp32 W    + fp32 A (ideal)      ", False, False),
]

per_batch = {}
with torch.no_grad():
    for label, qw, qa in ARMS:
        main.BitLinear.forward = make_forward(qw, qa)
        row = []
        for x, y in BATCHES:
            logits, _ = model(x)
            row.append(float(F.cross_entropy(logits.view(-1, logits.size(-1)),
                                             y.view(-1))))
        per_batch[label] = row
        print(f"done {label}")

base = per_batch[ARMS[0][0]]
print()
print(f"{'arm':<38}{'val CE':>9}{'ppl':>8}{'delta':>9}{'+-':>8}")
print("-" * 72)
for label, _, _ in ARMS:
    row = per_batch[label]
    ce = sum(row) / len(row)
    diffs = [a - b for a, b in zip(row, base)]
    mean_d = sum(diffs) / len(diffs)
    var = sum((d - mean_d) ** 2 for d in diffs) / max(1, len(diffs) - 1)
    se = math.sqrt(var / len(diffs))
    print(f"{label:<38}{ce:>9.4f}{math.exp(ce):>8.2f}{mean_d:>+9.4f}{se:>8.4f}")
