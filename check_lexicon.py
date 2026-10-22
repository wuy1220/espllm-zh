# -*- coding: utf-8 -*-
"""台词本体检：三条硬约束 + 问法查重，全自动。

约束来源（都是被实测打出来的，不是审美）：
  ① 答案不以该概念名词（含别名）开头 —— 否则和固件回声接成结巴，
     而且按「输入含该名词」过滤时会把 4 条一次滤光、整类样本归零。
  ② 每概念 4 条答案共用同一个 2 字签名，且跨概念唯一 ——
     probe_route_gate.py 量出首字是瓶颈（p=0.115 vs 第二字 0.436），
     签名制把首字候选 63 压到 24，「首字对 = 概念对」。
  ③ 全部字符在 persona/vocab_zh.json 里 —— 否则板上解码成空。
  ④ 问法全局唯一 —— 同一问法挂两个概念，模型根本没法收敛
     （「她厉害吗」同时挂大妖精和魔理沙，训练样本还原 0/12 的真凶）。
  ⑤ 答案长度 6~20 字。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "persona"))

import gen_react_corpus as R  # noqa: E402
import stance_lexicon as SL  # noqa: E402

VOCAB = set(json.load(open(os.path.join(HERE, "persona", "vocab_zh.json"),
                           encoding="utf-8"))["chars"])

NOUNS = {c: [n for n in ns if all(ch in VOCAB for ch in n)]
         for c, _, ns, _ in R.CONCEPTS}

fails = []
warns = []


def main():
    concepts = list(SL.ANSWERS)
    print("概念 %d 个，答案 %d 条，问法 %d 条，回避 %d 条"
          % (len(concepts), sum(len(v) for v in SL.ANSWERS.values()),
             sum(len(v) for v in SL.QUESTIONS.values()), len(SL.DEFLECT)))

    # ① 答案不以概念名词/别名开头
    for c in concepts:
        pool = [c] + NOUNS.get(c, [])
        for a in SL.ANSWERS[c]:
            for n in pool:
                if n and a.startswith(n):
                    fails.append("① [%s] 答案以名词开头: %s (名词=%s)" % (c, a, n))

    # ② 签名：概念内一致 + 跨概念唯一
    sig = {}
    for c in concepts:
        ans = SL.ANSWERS[c]
        if len(ans) != 4:
            fails.append("② [%s] 答案不是 4 条（%d）" % (c, len(ans)))
            continue
        s = ans[0][:2]
        for a in ans:
            if a[:2] != s:
                fails.append("② [%s] 签名不一致: %s vs %s" % (c, s, a[:2]))
        if s in sig:
            fails.append("② 签名跨概念撞车: %s（%s vs %s）" % (s, sig[s], c))
        sig[s] = c
        # 签名不能含该概念的名词别名（否则会被 answerable() 一次滤光）
        for n in NOUNS.get(c, []):
            if n in s:
                fails.append("② [%s] 签名含名词别名 %s: %s" % (c, n, s))

    # ③ 字符全在字表
    for c in concepts:
        for a in SL.ANSWERS[c]:
            for ch in a:
                if ch not in VOCAB:
                    fails.append("③ [%s] 答案含字表外字符 %r: %s" % (c, ch, a))
        for q in SL.QUESTIONS.get(c, []):
            for ch in q:
                if ch not in VOCAB:
                    fails.append("③ [%s] 问法含字表外字符 %r: %s" % (c, ch, q))

    # ④ 问法全局唯一
    seen = {}
    for c in concepts:
        for q in SL.QUESTIONS.get(c, []):
            if q in seen:
                fails.append("④ 问法跨概念撞车: %s（%s vs %s）" % (q, seen[q], c))
            seen[q] = c
    for a in SL.DEFLECT:
        for ch in a:
            if ch not in VOCAB:
                fails.append("③ 回避含字表外字符 %r: %s" % (ch, a))

    # ⑤ 答案长度 6~20 字
    for c in concepts:
        for a in SL.ANSWERS[c]:
            if not (6 <= len(a) <= 20):
                warns.append("⑤ [%s] 答案长度 %d 超出 6~20: %s" % (c, len(a), a))

    # 每个概念都要有问法
    for c in concepts:
        if not SL.QUESTIONS.get(c):
            fails.append("④ [%s] 没有问法" % c)

    # ⑥ 签名共享首字 = 近义字污染风险（警告级，不阻断）
    # 2026-10-21：「她俩」(比试) vs 「她们」(妖精同伴) 前 2 字唯一、过了旧检查，
    # 但共享首字「她」且第二字近义，模型在 temp 0.5 下 0/6 分不开、稳定串味。
    by_first = {}
    for s, c in sig.items():
        by_first.setdefault(s[0], []).append((c, s))
    for f, items in sorted(by_first.items()):
        if len(items) > 1:
            warns.append("⑥ 签名共享首字 %s: %s（第二字近义时会互相串）"
                         % (f, ", ".join("%s=%s" % t for t in items)))

    print("\n".join("  ! " + w for w in warns) if warns else "  无警告")
    if fails:
        print("\n".join("  X " + f for f in fails))
        print("\n** %d 条不通过 **" % len(fails))
        return 1
    print("\n** 全部通过（%d 概念 / %d 签名）**" % (len(concepts), len(sig)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
