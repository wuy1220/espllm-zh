# -*- coding: utf-8 -*-
"""per-concept 路由对比：同一个 ckpt、贪心解码，按概念拆分命中率。

为什么要拆：总命中率是 28 个概念的平均，新增 3 个概念会把分母变大，
「总分掉了」分不清是旧概念退化还是新概念没学会。拆开看才能判断一版模型
到底能不能用。

用法（每个 ckpt 必须独立进程 —— gen_long.load 有单例缓存，同进程第二个
ckpt 会空转，历史上因此误判过"两个模型输出逐字节相同"）：
    python _per_concept.py --ckpt model/model_stance5.pt.best \
        --legacy-sig model/stance5_legacy_answers.json
    python _per_concept.py --ckpt model/model_stance9.pt.best
"""
import argparse
import collections
import json
import os
import sys

os.environ.setdefault("ESPGPT_CHAR_VOCAB", "persona/vocab_zh.json")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "persona"))

import gen_long  # noqa: E402
import gen_react_corpus as R  # noqa: E402
import stance_lexicon as SL  # noqa: E402

PROMPT = "User: {}\nBot:"


def head_match(g, pool, k=2):
    return bool(g) and any(a[:k] == g[:k] for a in pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--legacy-sig", default="")
    a = ap.parse_args()

    if a.legacy_sig:
        legacy = json.load(open(a.legacy_sig, encoding="utf-8"))
        for c, ans in legacy.items():
            if not c.startswith("_"):
                SL.ANSWERS[c] = ans

    gen_long.load(a.ckpt)
    # probe_stance.gen 是同一份封装，直接复用它的 generate 入口
    from probe_stance import gen as pgen  # noqa: E402

    rows = []
    for cname, _topic, _nouns, _animal in R.CONCEPTS:
        pool = SL.ANSWERS[cname]
        hit = cross = miss = 0
        for q in SL.QUESTIONS[cname]:
            g = pgen(q, 0.0, 0)
            if head_match(g, pool):
                hit += 1
            elif any(head_match(g, p2) for c2, p2 in SL.ANSWERS.items() if c2 != cname):
                cross += 1
            else:
                miss += 1
        rows.append((cname, hit, cross, miss))

    tot = sum(h + c + m for _, h, c, m in rows)
    ok = sum(h for _, h, _c, _m in rows)
    print("=== %s ===" % a.ckpt)
    print("总命中 %d/%d (%.0f%%)\n" % (ok, tot, 100.0 * ok / tot))
    print("  %-8s 命中 串味 落空" % "概念")
    for cname, h, c, m in sorted(rows, key=lambda r: (r[1], r[0])):
        flag = "" if h == h + c + m else "   <<<"
        print("  %-8s %2d   %2d   %2d%s" % (cname, h, c, m, flag))
    daily = [r for r in rows if r[0] in ("吃饭", "睡觉", "玩耍")]
    print("\n  日常三概念：%s" % " | ".join(
        "%s %d/%d" % (c, h, h + cc + m) for c, h, cc, m in daily))


if __name__ == "__main__":
    main()
