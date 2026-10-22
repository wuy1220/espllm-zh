"""表态语料（react5）：模型只说「表态」，一个名词都不抄。

为什么不再让模型回声
--------------------
react3 用「回声 + 表态」教模型照抄输入里的名词，实测撞上架构硬边界：
在**训练集自己的随机伪名词样本**上贪心解码，前 2 字命中 0/16 —— 首字常对
（KL 探针里 '大妖精呢' -> '大' 93.3%），次字必错。5 层 / 2 头 / 64 维 / ternary
长不出能精确匹配内容的 QK 电路。文献侧有同一个结论：RTN/低比特量化会摧毁小模型的
induction head（GPT-2 124M 上 10/10 全灭，GPTQ 才保得住），我们比它还小 60 倍。

所以回声改由固件做（一个通用函数，100% 准确），模型只负责表态。

为什么不用「名词 × 模板」交叉
-----------------------------
react4 用 33 条模板乘概念别名，造了 2088 行，其中大量是「路有几个？」「我想摸摸个子。」
这种语义不通的句子。模型学到的是「怪输入 -> 随便给一个答案」：实测概念路由 41%，
而误判里 32/70 条全部塌到「冰」那一条答案上。**语料不是越多越好，是越像人话越好。**

这一版改成 `persona/stance_lexicon.py` 里的手写问法（每概念 6 条，人真会说的话），
每条问法与 4 条答案**全配对** —— 模型没有机会把某一对当特例背下来，只能学概念。

CLI:
    python gen_stance_corpus.py --dry-run
    python gen_stance_corpus.py
"""
import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "persona"))

import gen_react_corpus as R  # noqa: E402  复用概念名与字数集合
import stance_lexicon as SL  # noqa: E402  台词本 + 问法

VOCAB = os.path.join(HERE, "persona", "vocab_zh.json")
SRC = r"E:\espllm_data\persona_corpus\raw_canon.jsonl"
OUT = r"E:\espllm_data\persona_corpus\raw_stance2.jsonl"

# 陌生词用到的模板：只保留语义通顺的那几条。
# 第一版把 33 条全用上了，于是「我想摸摸天平。」这种句子也被当成正常输入。
SAFE_TMPL = [
    "{N}是什么", "{N}是啥", "你见过{N}吗", "{N}在哪", "{N}有什么用",
    "什么是{N}", "{N}怎么用", "你认识{N}吗", "{N}好不好", "给我讲讲{N}",
    "{N}值钱吗", "{N}能吃吗",
]

NOISE = ["？", "？？？", "。。。", "哈哈哈哈", "啊啊啊啊啊", "嗯", "呃",
         "哦", "……", "这个", "那个啊", "所以说", "然后", "你知道吗", "喂", "嗯嗯"]

# S6 垃圾形态：字母串/数字串/符号串/英文句/中英混合 —— 用户真会打出来的乱按。
# 2026-10-21 截图事故的根因：词表 ASCII 只有 27 个（User:Bot"',!?1hlp2icFgma9），
# aabbccddee 的 b/c/d、qwerty 的 q/w、1234567890 的 3~8/0 全是 UNK，模型看到的是
# UNK 序列，而语料里从没教过「UNK 序列 -> 回避」：贪心时 DEFLECT 勉强第一，
# temp 0.7 一扰就串到别的池（实测 6/6 稳定串味，甚至 6/6 崩坏）。
# 这些串**故意不滤字表**（all(ch in vocab) 不过滤）——含 UNK 才是真实分布。
JUNK_FORMS = [
    # 字母重复/乱打（截图原样 + 变体）
    "aabbccddee", "aabbccdd", "xxyyzz", "aaaaaa", "abcabcabc", "ababab",
    "qwertyuiop", "asdfghjkl", "zxcvbnm", "qwerty", "asdf", "hjkl",
    "abcdefg", "fedcba", "acegik", "bdfhjl", "aaa bbb ccc",
    # 数字串
    "1234567890", "66666666", "00000", "888", "111111", "123123123",
    "0987654321", "999999999", "2333333",
    # 符号串（中文标点大多在词表内，ASCII 符号多为 UNK）
    "???!!!", "......", "~~~", "***", "@@@@", "###", "$$$", "^^^",
    "!!!!!", "?????", "。。。。。", "—————", "￥￥￥",
    # 英文句子（lorem ipsum 是截图原样）
    "lorem ipsum dolor sit amet", "hello world", "the quick brown fox",
    "ok fine whatever", "asap plz", "test test test", "hello",
    "what is this", "i dont know",
    # 中英混合
    "abc你好", "嗯ok", "好aaa", "OKOK", "yesyes", "no no no",
    # 单字符/超短
    "a", "x", "q", "z", "6", "?", ".",
    # 超长重复
    "aaaaaaaaaaaaaaaaaaaa", "哈哈哈哈哈哈哈哈哈哈", "嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯",
]

# S7 元行为：问句类型模板 x 名词池，**排除已配置概念槽的组合** -> 通用回避。
#
# 2026-10-21 两轮实测的边界（重要，别再扩）：
# - v1（450 条，含态度/能力/认识类）：日常回避 0->6 条生效，但 seen 89%->56%
#   ——负空间和正空间问句形态重叠（「你喜欢X吗」vs「你喜欢什么」）时，
#   2.17M 模型分不开"这个名词学过没有"（induction head 缺失的又一表现），
#   450 条 DEFLECT 直接压过 24 条概念样本，概念路由崩塌。
# - 结论：**模型原理上判得了"是不是中文"（UNK vs 汉字，S6 已覆盖），
#   判不了"这个名词学没学过"（要记 28 概念名词表）**。后者归规则层。
# - v2（本版）：只保留规则帧不覆盖的类型（时态/情态/处置/属性），
#   态度类归 zh_affection、能力类归 zh_friend（已放行给模型接）、
#   信息问句归 zh_friend_dyj。每条负样本都不和任何概念问法同形态。
META_TMPL = [
    # 处置/属性（规则不覆盖，模型也没概念槽）
    "你怎么用{N}", "{N}有什么用", "{N}好吃吗", "{N}多少钱", "{N}是什么做的",
    "你怎么做{N}", "{N}在哪里", "为什么有{N}",
    # 时态
    "你昨天{N}了吗", "你明天{N}吗", "你最近{N}吗", "你常{N}吗",
    # 情态
    "你能{N}吗", "你会{N}吗", "你想过{N}吗", "你试过{N}吗", "你敢{N}吗",
    "你应该{N}吗", "你可以{N}吗", "你需要{N}吗", "你愿意{N}吗",
]
# 无实体状态问句（问模型自己）。日常概念（吃饭/睡觉/玩耍）的手写问法
# 覆盖高频形态，其余落这里 -> 回避。
META_TMPL_NONOUN = [
    "你在想什么", "你要去哪", "你无聊吗", "你寂寞吗", "你后悔吗",
    "你怎么了", "你还好吗", "你累不累", "你开心吗", "你难过吗",
    "你紧张吗", "你害怕吗", "你几点起", "你周末干嘛", "你放假了吗",
]
# 补充名词：别处没覆盖的幻想乡角色（避免和概念名词重复）。
META_EXTRA_NOUNS = ("幽幽子 蕾咪 咲夜 帕秋莉 芙兰 妖忌 文文 早苗 诹访子 神奈子 "
                    "天子 衣玖 雫 映姬 妖梦 二小姐 门卫 图书馆")


def load_vocab():
    return set(json.load(open(VOCAB, encoding="utf-8"))["chars"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-s1", type=int, default=2400, help="canon 自然句条数上限")
    ap.add_argument("--n-ood", type=int, default=6)
    ap.add_argument("--n-pseudo", type=int, default=900)
    ap.add_argument("--n-noise", type=int, default=200)
    ap.add_argument("--n-junk", type=int, default=150,
                    help="S6 垃圾形态条数（字母/数字/符号/英文串 -> 回避）")
    ap.add_argument("--n-meta", type=int, default=150,
                    help="S7 元行为条数（问句类型模板 x 名词，未配置组合 -> 回避。"
                         "v1 450 条把概念路由打到 56%，见 META_TMPL 注释）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    vocab = load_vocab()
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    nouns_of = {c: [n for n in ns if all(ch in vocab for ch in n)]
                for c, _, ns, _ in R.CONCEPTS}
    topic_of = {c: t for c, t, _, _ in R.CONCEPTS}

    out = []

    def emit(src, tag, u, ans):
        out.append({"src": src, "t": tag, "u": u, "a": ans})

    def answerable(cname, u):
        """输入里出现过的名词，答案不能出现在开头（否则和固件回声接成结巴）。"""
        hit = [n for n in nouns_of.get(cname, []) if n in u]
        cands = [x for x in SL.ANSWERS[cname] if not any(x.startswith(n) for n in hit)]
        return cands or SL.ANSWERS[cname]

    # ---- S2 手写问法 × 全部答案：核心数据，每一对都出现 ----
    for cname in SL.ANSWERS:
        for q in SL.QUESTIONS[cname]:
            if any(ch not in vocab for ch in q):
                print("！问法含字表外字符，跳过：%s" % q)
                continue
            for ans in answerable(cname, q):
                emit("S2", topic_of[cname], q, ans)

    # ---- S1 canon 自然句：真人怎么问的多样性来源 ----
    by_topic = collections.defaultdict(list)
    for r in rows:
        if r["kind"] != "ood" and all(ch in vocab for ch in r["u"]):
            by_topic[r["t"]].append(r["u"])
    per = max(1, a.n_s1 // max(len(topic_of), 1))
    cname_of = {t: c for c, t in topic_of.items()}
    for topic, us in by_topic.items():
        cname = cname_of.get(topic)
        if not cname:
            continue
        us = list(dict.fromkeys(us))
        rng.shuffle(us)
        for u in us[:per]:
            emit("S1", topic, u, rng.choice(answerable(cname, u)))

    # ---- S3 域外真词 / S5 伪名词：都走通用回避 ----
    ood = [w for w in R.OOD_NOUNS if all(c in vocab for c in w)]
    for noun in ood:
        for t in rng.sample(SAFE_TMPL, min(a.n_ood, len(SAFE_TMPL))):
            emit("S3", "_ood", t.replace("{N}", noun), rng.choice(SL.DEFLECT))

    pchars = sorted((set("".join(R.PSEUDO_EXTRA)) | {c for w in ood for c in w}) & vocab)
    made, guard = 0, 0
    while made < a.n_pseudo and guard < a.n_pseudo * 40:
        guard += 1
        w = "".join(rng.choice(pchars) for _ in range(rng.choice((2, 2, 2, 3))))
        if w in ood:
            continue
        emit("S5", "_pseudo", rng.choice(SAFE_TMPL).replace("{N}", w), rng.choice(SL.DEFLECT))
        made += 1

    # ---- S4 无名词噪声 ----
    noise_ok = [x for x in NOISE if x and all(ch in vocab for ch in x)]
    for _ in range(a.n_noise):
        emit("S4", "_noise", rng.choice(noise_ok), rng.choice(SL.DEFLECT))

    # ---- S6 垃圾形态 -> 回避（2026-10-21 截图事故修复，见 JUNK_FORMS 注释）----
    # 不滤字表：表外字符编码成 UNK 正是真实垃圾输入的形态。
    for i in range(a.n_junk):
        emit("S6", "_junk", JUNK_FORMS[i % len(JUNK_FORMS)], rng.choice(SL.DEFLECT))

    # ---- S7 元行为：未配置的问句形态 -> 回避（问句类型矩阵的负空间）----
    # 名词分层：概念名词（用户最常问的）优先入池，OOD/其他角色补量。
    concept_nouns, extra_nouns = set(), set()
    for _c, _t, ns, _ in R.CONCEPTS:
        concept_nouns.update(ns)
    extra_nouns.update(R.OOD_NOUNS)
    extra_nouns.update(META_EXTRA_NOUNS.split())
    concept_nouns = {n for n in concept_nouns if n and all(ch in vocab for ch in n)}
    extra_nouns = {n for n in extra_nouns if n and all(ch in vocab for ch in n)}
    configured = {q for c in SL.QUESTIONS for q in SL.QUESTIONS[c]}
    pool, seen_q = [], set()

    def _add(t, nouns):
        for n in nouns:
            q = t.replace("{N}", n)
            if q in configured or q in seen_q:
                continue
            seen_q.add(q)
            pool.append(q)

    for t in META_TMPL:                 # 概念名词优先
        _add(t, sorted(concept_nouns))
    for t in META_TMPL:                 # OOD 补量
        _add(t, sorted(extra_nouns))
    for t in META_TMPL_NONOUN:
        if t not in configured and t not in seen_q:
            seen_q.add(t)
            pool.append(t)
    rng.shuffle(pool)
    for q in pool[:a.n_meta]:
        emit("S7", "_meta", q, rng.choice(SL.DEFLECT))

    # ---- 质检 ----
    bad = collections.Counter()
    for r in out:
        if r["t"] == "_junk":
            continue          # S6 故意含 UNK，单独统计
        for ch in r["u"] + r["a"]:
            if ch not in vocab:
                bad[ch] += 1
    junk_unk = sum(1 for r in out if r["t"] == "_junk"
                   for ch in r["u"] if ch not in vocab)
    dup = sum(1 for r in out
              if r["t"] in cname_of and any(
                  r["a"].startswith(n) and n in r["u"]
                  for c, t in cname_of.items() if t == r["t"] for n in nouns_of.get(c, [])))
    dist = collections.Counter(r["src"] for r in out)
    lens = sorted(len(r["a"]) for r in out)
    per_topic = collections.Counter(r["t"] for r in out)
    known = sum(v for k, v in dist.items() if k in ("S1", "S2"))

    print("总行 %d" % len(out))
    print("  分布 %s" % dict(dist))
    print("  认识的话题 %d 行（%.0f%%）| 陌生词 %d 行"
          % (known, 100.0 * known / len(out), len(out) - known))
    print("  输入去重 %d | 答案种类 %d" % (len({r["u"] for r in out}), len({r["a"] for r in out})))
    print("  答案长度 p50/p90/max = %d/%d/%d" % (lens[len(lens) // 2], lens[int(len(lens) * .9)], lens[-1]))
    print("  字表外字符 %s" % ("无" if not bad else "%d 种 %s" % (len(bad), " ".join(bad))))
    print("  S6 垃圾形态 %d 条，其中 %d 字符是 UNK（故意，模拟真实乱按）"
          % (sum(1 for r in out if r["t"] == "_junk"), junk_unk))
    print("  前缀会结巴的行 %d（必须为 0）" % dup)
    print("  每概念 %d~%d 行" % (min(v for k, v in per_topic.items() if k in cname_of),
                              max(v for k, v in per_topic.items() if k in cname_of)))

    if a.dry_run:
        return
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n写入 %s" % a.out)


if __name__ == "__main__":
    main()
