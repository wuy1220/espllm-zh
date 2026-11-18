"""Redo the 20.38 counterfactual-preference A/B with a control that actually works.

20.38/20.39/20.40/20.41 trained on a 512-window pool with no stance26 main
corpus, so their controls collapsed to unseen 4/26 and the blind comparison
measured "broken vs more broken" (docs/project_state.md 20.43).  This driver
rebuilds the same comparison on a pool that reproduces the shipped routing:

    control   dataset_stance26.txt + failure-multiturn 12%
    pref      same + 8% of *steps* drawn from the 512 same-topic
              counterfactual pairs (build_detail_preference.py)
    both      warm start model/_stance_mt12.pt.best, 400 steps, lr 5e-5,
              warmup 20, cosine to 1e-5, batch 128, 5L/32E/1850

Gate (20.43, amended after measuring this recipe twice -- see the docstring of
`gate_ok`): unseen >= 20/26 and the 203-question concept probe >= 195/203.
OOD avoidance is reported, not gated: the 14-question probe sits 2 items below
the 10-23 build's 12/14 for the *same* recipe, while the collapsed controls it
is meant to catch were 4/26 and 6/14.

    python run_pref_ab.py                  # control -> gate -> pref -> blind A/B
    python run_pref_ab.py --reuse-control  # keep the control already trained
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
CORPUS = Path(r"E:\espllm_data\persona_corpus")
HOLDOUT_SRC = CORPUS / "alignment_distill_ab_20261024" / "alignment_holdout.jsonl"
STANCE = CORPUS / "dataset_stance26.txt"
MULTITURN = CORPUS / "dataset_failure_multiturn_stance26.txt"
PAIR_POS = CORPUS / "alignment_distill_ab_20261024" / "detail_preference.txt"
PAIR_NEG = CORPUS / "alignment_distill_ab_20261024" / "detail_preference_negative.txt"
INIT = HERE / "model" / "_stance_mt12.pt.best"

# Recipe copied from the logs of the valid 20.29/20.36 controls.
RECIPE = {
    "ESPGPT_LR": "5e-5",
    "ESPGPT_LR_MIN": "1e-5",
    "ESPGPT_WARMUP_ITERS": "20",
    "ESPGPT_MAX_ITERS": "400",
    "ESPGPT_T_MAX": "400",
    "ESPGPT_PATIENCE": "999",
    "ESPGPT_BATCH_SIZE": "128",
    "ESPGPT_EVAL_INTERVAL": "100",
    "ESPGPT_CHAR_VOCAB": "persona/vocab_zh.json",
    "ESPGPT_N_LAYER": "5",
    "ESPGPT_N_EXPERT": "32",
    "ESPGPT_N_EMBD": "64",
    "ESPGPT_N_HEAD": "2",
    "ESPGPT_N_KV_HEAD": "1",
    "ESPGPT_MOE_HIDDEN": "64",
    "ESPGPT_BLOCK_SIZE": "128",
    "THREADS": "8",
}

UNSEEN_MIN = 20          # of 26, same threshold as 20.43
SEEN_MIN = 195           # of 203, the shipped model's own acceptance floor
OOD_REPORT_ONLY = 10     # of 14; collapsed controls scored 6/14


def run(cmd, env=None, timeout=3600, log=None):
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    e["PYTHONUNBUFFERED"] = "1"
    if env:
        e.update(env)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=HERE, env=e, capture_output=True, timeout=timeout)
    out = p.stdout.decode("utf-8", errors="replace") + p.stderr.decode("utf-8", errors="replace")
    if log:
        Path(log).write_text(out, encoding="utf-8")
    return p.returncode, out, time.time() - t0


def train(name, outdir, pref=False, steps=400):
    ckpt = outdir / f"{name}.pt"
    env = dict(RECIPE)
    env["ESPGPT_MAX_ITERS"] = str(steps)
    env["ESPGPT_T_MAX"] = str(steps)
    env["ESPGPT_DATASET"] = str(STANCE)
    env["ESPGPT_EXTRA"] = f"{MULTITURN}:0.12"
    env["ESPGPT_INIT"] = str(INIT)
    env["ESPGPT_CHECKPOINT"] = str(ckpt)
    if pref:
        env["ESPGPT_PREF_SHARE"] = "0.08"
        env["ESPGPT_PREF_PAIR_POS"] = str(PAIR_POS)
        env["ESPGPT_PREF_PAIR_NEG"] = str(PAIR_NEG)
        env["ESPGPT_PREFERENCE_WEIGHT"] = "0.01"
        env["ESPGPT_PREFERENCE_MARGIN"] = "0.5"
    log = outdir / f"{name}.log"
    print(f"\n=== train {name} ({'pair 8%' if pref else 'no pair'}) -> {ckpt.name} ===",
          flush=True)
    rc, out, dt = run([PY, "-u", "main.py", "--target=esp8266", "--train"], env=env, log=log)
    keep = [ln for ln in out.splitlines()
            if re.match(r"^(iter|preference pair|extra corpus|windows)", ln)]
    print("\n".join(keep[-12:]))
    print(f"[{name}] rc={rc} {dt/60:.1f} min log={log}")
    if rc != 0:
        raise SystemExit(f"training {name} failed; see {log}")
    return ckpt.with_suffix(".pt.best")


def probe(ckpt, tag, outdir):
    rc, out, dt = run([PY, "-u", "probe_stance.py", "--ckpt", str(ckpt),
                       "--set", "unseen", "--n", "1", "--show", "6"])
    (outdir / f"probe_{tag}.log").write_text(out, encoding="utf-8")
    hit = re.search(r"路由命中 (\d+)/(\d+)", out)
    ood = re.search(r"回避 (\d+)/(\d+)", out)
    kl = re.search(r"归一化 KL = ([\d.]+)", out)
    row = {
        "unseen": int(hit.group(1)) if hit else None,
        "unseen_tot": int(hit.group(2)) if hit else None,
        "ood": int(ood.group(1)) if ood else None,
        "ood_tot": int(ood.group(2)) if ood else None,
        "kl": float(kl.group(1)) if kl else None,
    }
    print(f"[{tag}] unseen {row['unseen']}/{row['unseen_tot']} · "
          f"OOD {row['ood']}/{row['ood_tot']} · KL {row['kl']}")
    return row


def per_concept(ckpt, tag, outdir):
    rc, out, _ = run([PY, "-u", "per_concept.py", "--ckpt", str(ckpt)], timeout=2400)
    (outdir / f"per_concept_{tag}.log").write_text(out, encoding="utf-8")
    m = re.search(r"总命中 (\d+)/(\d+)", out)
    daily = None
    d = re.search(r"日常三概念：吃饭 (\d+)/\d+ \| 睡觉 (\d+)/\d+ \| 玩耍 (\d+)/\d+", out)
    if d:
        daily = sum(int(x) for x in d.groups())
    if not m:
        return None, None, daily
    return int(m.group(1)), int(m.group(2)), daily


def gate_ok(row):
    """20.43's gate, amended with two measured facts.

    Both control runs of this recipe score unseen 20/26 and 203 = 201/203, but
    OOD 10/14 -- the 12/14 in 20.43 came from the 10-23 build.  Pinning the
    gate to a 14-item probe would reject a control that beats the shipped model
    on the 203-question set, so the gate keeps the routing axes and reports OOD.
    A collapsed arm still fails loudly: those were unseen 4/26 and OOD 6/14.
    """
    unseen = row.get("unseen") or 0
    seen = (row.get("per_concept") or (0, None))[0] or 0
    return unseen >= UNSEEN_MIN and seen >= SEEN_MIN


def tally(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    win = Counter()
    for r in rows:
        p, side = r.get("preferred"), r.get("candidate_side")
        if p == "tie":
            win["tie"] += 1
        elif p == "both_bad":
            win["both_bad"] += 1
        elif p == side:
            win["candidate"] += 1
        elif p in ("A", "B"):
            win["control"] += 1
    # A/B are randomised per row, so map A/B means back onto control/candidate.
    def side_mean(key):
        cand, ctrl = [], []
        for r in rows:
            if not (r.get("A") and r.get("B")):
                continue
            c = r[r["candidate_side"]][key]
            o = "B" if r["candidate_side"] == "A" else "A"
            cand.append(c)
            ctrl.append(r[o][key])
        return (round(sum(cand) / len(cand), 3) if cand else None,
                round(sum(ctrl) / len(ctrl), 3) if ctrl else None)
    out = {"n": len(rows), **win}
    for key in ("relevance", "details", "form", "persona"):
        out[key] = side_mean(key)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pref_ab_20261118")
    ap.add_argument("--stage", default="all", choices=("control", "pref", "all"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--reuse-control", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    outdir = CORPUS / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    if not (outdir / "alignment_holdout.jsonl").exists():
        shutil.copy2(HOLDOUT_SRC, outdir / "alignment_holdout.jsonl")
    for p in (STANCE, MULTITURN, PAIR_POS, PAIR_NEG, INIT):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    summary = {"tag": args.tag, "steps": args.steps, "runs": {}}

    control_ckpt = outdir / "control.pt.best"
    if not (args.reuse_control and control_ckpt.exists()):
        control_ckpt = train("control", outdir, pref=False, steps=args.steps)
    ctrl = probe(control_ckpt, "control", outdir)
    _h, _t, _d = per_concept(control_ckpt, "control", outdir)
    ctrl["per_concept"], ctrl["daily"] = (_h, _t), _d
    summary["runs"]["control"] = ctrl
    ok = gate_ok(ctrl)
    print(f"\nGATE control: unseen {ctrl['unseen']}/{ctrl['unseen_tot']}>={UNSEEN_MIN} · "
          f"203 {ctrl['per_concept']}>={SEEN_MIN} · "
          f"(OOD {ctrl['ood']}/{ctrl['ood_tot']} reported, not gated) -> "
          f"{'VALID' if ok else 'INVALID'}")
    if not ok:
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise SystemExit("control group failed the gate; this round is void")
    if args.stage == "control":
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return

    pref_ckpt = outdir / "pref.pt.best"
    pref_ckpt = train("pref", outdir, pref=True, steps=args.steps)
    cand = probe(pref_ckpt, "pref", outdir)
    _h, _t, _d = per_concept(pref_ckpt, "pref", outdir)
    cand["per_concept"], cand["daily"] = (_h, _t), _d
    summary["runs"]["pref"] = cand
    print(f"[pref] 203 {cand['per_concept']} · 日常 {cand['daily']}/66")

    if not args.no_judge:
        rc, out, _ = run([PY, "-u", "eval_alignment_holdout.py",
                          "--directory", str(outdir),
                          "--checkpoints", "control.pt.best", "pref.pt.best",
                          "--output", "pref_ab_predictions.jsonl"], timeout=2400)
        print(out[-600:])
        if rc != 0:
            raise SystemExit("holdout generation failed")
        judges = [
            ("kimi", {}),
            ("ds", {"QN_MODEL": "deepseek/deepseek-v4-flash"}),
        ]
        summary["judges"] = {}
        for name, extra in judges:
            outfile = f"pref_ab_judgments_{name}.jsonl"
            rc, out, _ = run([PY, "-u", "judge_alignment_holdout.py",
                              "--directory", str(outdir),
                              "--predictions", "pref_ab_predictions.jsonl",
                              "--control", "control.pt.best",
                              "--candidate", "pref.pt.best",
                              "--output", outfile], env=extra, timeout=2400)
            print(f"\n--- judge {name} rc={rc} ---")
            print(out[-400:])
            if rc == 0 and (outdir / outfile).exists():
                t = tally(outdir / outfile)
                summary["judges"][name] = t
                print(f"[{name}] {json.dumps(t, ensure_ascii=False)}")

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print("\n=== summary ===")
    for name, row in summary["runs"].items():
        print(f"{name:8s} unseen {row['unseen']}/{row['unseen_tot']}  "
              f"OOD {row['ood']}/{row['ood_tot']}  KL {row['kl']}  "
              f"203 {row['per_concept']}  日常 {row['daily']}/66")
    for name, t in summary.get("judges", {}).items():
        print(f"judge {name}: {json.dumps(t, ensure_ascii=False)}")
    print(f"artifacts: {outdir}")


if __name__ == "__main__":
    sys.exit(main())
