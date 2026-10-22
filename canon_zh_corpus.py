"""把 3,498 条「一问一答全不重样」压成「185 个意图 -> 每个一句规范回复」。

为什么必须压：实测 `model_zh_cirno3` 在**语料原句**上 pairwise KL 只有 0.1798、
贪心还原 0/20，输出反复是「咱不知道，咱的冰屋就是冰之湖！」这类通用模板。
3,498 个互不相同的目标回复超出了 2.17M 三元权重的容量，模型只能用"人设腔 +
模板"去压低平均损失 —— 这是 `research_2026-10.md` 里那条"语料难度必须匹配容量"
的同一课（yukari-synthetic 那次也是 921 条、output 中位 83 字，学不动）。

压法：按 (话题, kind) 分组，每组选一条**medoid**（与组内其它回复的平均 4-gram
Jaccard 最高者），组内所有 prompt 都映射到它。任务从"记住 3,498 个字符串"变成
"把 prompt 路由到 185 个桶"，需要的信息量从 ~12 bit/条降到 ~7.5 bit/条。

CLI:
    python canon_zh_corpus.py
    python canon_zh_corpus.py --min-group 2 --out raw_canon.jsonl
"""
import argparse
import collections
import json
import os

DIR = r"E:\espllm_data\persona_corpus"
SRC = os.path.join(DIR, "raw.jsonl")
DST = os.path.join(DIR, "raw_canon.jsonl")


def grams(s, n=4):
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def prompt_grams(s, n=2):
    """prompt 用 2-gram：中文句子短，4-gram 在 4~6 字的问句上几乎不重叠。"""
    s = "".join(ch for ch in s if ch.strip())
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=DST)
    ap.add_argument("--min-group", type=int, default=2,
                    help="小于这个规模的组原样保留（各自独立目标）")
    ap.add_argument("--merge-below", type=int, default=4,
                    help="小于这个规模的组才考虑合并")
    ap.add_argument("--sim", type=float, default=0.30,
                    help="小组合并所需的 prompt 2-gram 平均 Jaccard；不够就自成一组")
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false",
                    help="保留重复 prompt（默认按众数去重）")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.src, encoding="utf-8") if l.strip()]

    def base_topic(t):
        """'自我介绍-几岁' / '喜好·冰' -> '自我介绍' / '喜好'。"""
        return t.split("-")[0].split("·")[0].strip() or t

    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["t"], r["kind"])].append(r)
    n_before = len(groups)

    # 小组怎么并：按**字面相似**并到同一父话题下最近的组，而不是按父话题名硬并。
    #
    # 旧做法（并回父话题再取 medoid）把 11 个子话题并成一个 `自我介绍`，
    # 150 条 prompt 只剩 3 个目标，其中 93 条（62%）答「咱住雾之湖边的冰屋」——
    # `你几岁`/`你叫什么名字`/`幻想乡是什么地方` 全被映射到"住哪"。
    # medoid 只能答混合池里的一支，这是结构性错误，不是选句不佳。
    #
    # 现在：小组成员与候选大组逐条比 prompt 的 2-gram Jaccard，取平均；
    # 只有真的像（>= sim）才并，否则**自成一组**（哪怕只有一条）。
    # 多出来的目标数很便宜，答错高频道问题很贵。
    big = {k: v for k, v in groups.items() if len(v) >= a.merge_below}
    merged = {k: list(v) for k, v in big.items()}
    n_kept = 0
    n_folded = 0
    for (t, kind), rs in groups.items():
        if len(rs) >= a.merge_below:
            continue
        parent = base_topic(t)
        cands = [(k, v) for k, v in big.items()
                 if k[1] == kind and base_topic(k[0]) == parent]
        best, best_s = None, 0.0
        for k, v in cands:
            s = sum(max(jaccard(prompt_grams(r["u"]), prompt_grams(o["u"])) for o in v)
                    for r in rs) / len(rs)
            if s > best_s:
                best, best_s = k, s
        if best is not None and best_s >= a.sim:
            merged[best].extend(rs)
            n_folded += 1
        else:
            merged[(t, kind)] = list(rs)
            n_kept += 1
    groups = merged
    print("小组处理：%d 组 -> %d 组（并入 %d 组，另立 %d 组，相似度阈值 %.2f）"
          % (n_before, len(groups), n_folded, n_kept, a.sim))

    out, chosen = [], {}
    for key, rs in groups.items():
        if len(rs) < a.min_group:
            for r in rs:
                out.append(r)
            chosen[key] = [r["a"] for r in rs]
            continue
        gs = [grams(r["a"]) for r in rs]
        # medoid：平均相似度最高的那条回复充当该意图的唯一回答
        best_i, best_s = 0, -1.0
        for i in range(len(rs)):
            s = sum(jaccard(gs[i], gs[j]) for j in range(len(rs))) / len(rs)
            if s > best_s:
                best_i, best_s = i, s
        canon = rs[best_i]["a"]
        chosen[key] = [canon]
        for r in rs:
            out.append({"t": r["t"], "kind": r["kind"], "u": r["u"], "a": canon})

    # prompt 级多数表决。
    #
    # 语料工厂会重复使用同一句 prompt（同一个问题被写进好几个话题），压成规范
    # 回复后，同一句输入就有了多个互相矛盾的答案：'今天好热' 对应 10 种，其中
    # 只有 5 条是"热死咱了"，其余分散在"咱一点都不热""那是咱的倒影"等等。
    # 这种监督信号教不会映射，只会让模型在两个答案之间漂——板上实测同一句
    # 连问两次，一次答"热死咱了"、一次答"这就走啦？"（后者甚至不在这 10 种里，
    # 是整句被噪声拽到了边缘分布）。
    #
    # 13% 的行属于这种情况。按 prompt 取众数为唯一答案，删掉其余，
    # 数据少 13%，但"同一句输入 -> 两个答案"这种自相矛盾的样本清零。
    if a.dedupe:
        by_u = collections.defaultdict(list)
        for r in out:
            by_u[r["u"]].append(r)
        ded, dropped = [], 0
        for u, rs in by_u.items():
            cnt = collections.Counter(r["a"] for r in rs)
            best = cnt.most_common(1)[0][0]
            keep = next(r for r in rs if r["a"] == best)
            ded.append(keep)
            dropped += len(rs) - 1
        print("prompt 级去重：%d 行 -> %d 行（删掉 %d 条重复 prompt，"
              "其中 %d 句原本答案互相矛盾）"
              % (len(out), len(ded), dropped,
                 sum(1 for v in by_u.values() if len({r["a"] for r in v}) > 1)))
        out = ded

    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_target = sum(len(v) for v in chosen.values())
    big = sorted(((k, len(v)) for k, v in groups.items() if len(v) >= a.min_group),
                 key=lambda x: -x[1])[:8]
    print("原 %d 条 / %d 组" % (len(rows), len(groups)))
    print("目标回复从 %d 降到 %d（唯一 %d 个）" % (
        len({r["a"] for r in rows}), n_target, len({r["a"] for r in out})))
    print("写成 %d 行（唯一 prompt %d）" % (len(out), len({r["u"] for r in out})))
    print("\n最大的几组（所有 prompt 共用一个回答）：")
    for k, n in big:
        print("  %-14s %-11s %3d 条 -> %r" % (k[0], k[1], n, chosen[k][0]))
    print("\n已写 %s（%.2f MB）" % (a.out, os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
