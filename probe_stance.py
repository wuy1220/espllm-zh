"""表态模型验收：问法变了，答案还对不对。

和 probe_react.py 的区别
------------------------
react 版的指标是「有没有抄到名词」。这一版故意不抄名词，所以指标换成：
  * **命中率**：同一概念换 5 种问法，贪心解码是否落进该概念的 4 条答案；
  * **跨概念串味**：答案落到了别的概念上（比「答不上来」更糟，说明路由塌了）；
  * **陌生词回避率**：域外/伪名词是否落进 DEFLECT；
  * **输入敏感性 KL**：3 字输入两两之间的归一化 KL，生死线 0.5。

CLI:
    python probe_stance.py --ckpt model/model_stance.pt.best
    python probe_stance.py --ckpt ... --n 3 --show 12
"""
import argparse
import collections
import itertools
import json
import math
import os
import random
import sys

os.environ.setdefault("ESPGPT_CHAR_VOCAB", "persona/vocab_zh.json")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "persona"))

import gen_long  # noqa: E402
import gen_react_corpus as R  # noqa: E402
import stance_lexicon as SL  # noqa: E402

PROMPT = "User: {}\nBot:"

# （原先这里有一张手写的 EXTRA 表，2026-10-22 起删除：它定义的问法从未被读取过
#  —— seen 口径直接用 stance_lexicon.QUESTIONS，现在那 26 个概念的问法都已补齐。）

# 训练时**没出现过**的问法：只有这张表才测得出「学到概念」还是「背下问法」。
UNSEEN = {
    "青蛙": ["青蛙为什么这么凉", "你冻过几只青蛙", "青蛙会跳吗"],
    "冰": ["你的冰能冻多大", "冰会化吗", "你怕火吗"],
    "雾之湖": ["雾大不大", "湖边有花吗", "湖里能游泳吗"],
    "冰屋": ["冰屋会塌吗", "屋里冷吗", "有人住冰屋吗"],
    "大妖精": ["大酱人呢", "你们认识多久了", "大酱会飞吗"],
    "妖精同伴": ["桑妮她们呢", "你们打过架吗", "她们怕你吗"],
    "恶作剧": ["你又干什么了", "谁把湖冻了", "别冻我"],
    "算数": ["二加二等于几", "你能数到几", "一共多少"],
    "最强": ["你比灵梦强吗", "有人打得过你吗", "你是最强的吗"],
    "灵梦": ["灵梦厉害吗", "你输过吗", "巫女会来吗"],
    "魔理沙": ["那个黑白的呢", "你们打过吗", "她的魔法怎么样"],
    "夏天": ["好热啊", "夏天怎么过", "冰还有吗"],
    "冬天": ["好冷啊", "冬天玩什么", "雪下大了"],
    "雪人": ["堆雪人吗", "雪人长什么样", "雪人倒了"],
    "鱼": ["湖里鱼多吗", "鱼抓到了吗", "鱼怎么吃"],
    "符卡": ["放个符卡", "弹幕怎么放", "你的符长什么样"],
    "被小看": ["你个子真小", "你是不是很矮", "别看我小"],
    "迷路": ["我找不到路", "你带我回家", "这边怎么走"],
    "红魔馆": ["红魔馆远吗", "咲夜认识你吗", "那边危险吗"],
    "⑨": ["你是不是笨蛋", "他们叫你什么", "你是九吗"],
    "自我介绍": ["你是什么人", "你叫什么名字", "你多大了"],
    "喜好": ["你喜欢冰吗", "你爱好什么", "甜的好吃吗"],
    "打招呼": ["嗨嗨", "你好呀", "我回来了"],
    # 被挑衅 / 比试 2026-10-22 已迁出模型（归规则帧 zh_taunt / zh_duel）。
    # 留着条目没有意义：主循环遍历 gen_react_corpus.CONCEPTS，它已删掉这两个，
    # 条目永远不会被取到。日常三概念是 2026-10-22 新进模型的，必须有 unseen。
    "吃饭": ["你想吃点什么", "肚子会叫吗", "你尝得出味道吗"],
    "睡觉": ["你睡多久", "睡着了会醒吗", "你睡觉关灯吗"],
    "玩耍": ["玩累了吗", "玩什么最开心", "一个人玩有意思吗"],
}

OOD_PROBE = ["天平", "音箱", "邮票", "针管", "石头", "图图", "剪镜毯", "蓝莓",
             "钢琴", "比特币", "火箭", "冰淇淋", "区块链", "摩天轮"]

KL_SET = ["你是谁", "你住哪", "你几岁", "今天好热", "今天好冷", "你在干嘛",
          "大妖精呢", "青蛙在哪", "一加一呢", "你喜欢啥"]


def gen(q, temp=0.0, seed=0, maxnew=40):
    _, out, _ = gen_long.run(PROMPT.format(q), win=64, maxnew=maxnew, seed=seed,
                             temp=temp, topk=40, rep=1.0, repwin=20)
    text = gen_long.dec(out).split("\n")[0].strip()
    tags = os.environ.get("ESPGPT_PLAN_TAGS", "")
    if tags and text[:1] in tags:
        text = text[1:].lstrip()
    return text


def head_match(g, pool, k=2):
    """前 k 字对上就算命中。默认 2 —— 每个概念的 4 条答案共用前 2 字签名，
    所以前 2 字对上就等于「路由对了」；k=4 才是「措辞也对」。"""
    for a in pool:
        n = min(k, len(a), len(g))
        if n >= 2 and a[:n] == g[:n]:
            return a
    return None


def kl_probe(n=10):
    """和 probe_zh_kl.py 同一套算法，但模板保留 `Bot: ` 的尾随空格。

    那个空格不是笔误：语料是 `prompt + " " + ans`，回复的第一个 token 永远是空格。
    少了它，量到的是「下一个 token 是不是空格」，10 条 prompt 会给出同一个答案，
    KL≈0 —— 看起来像"模型不读输入"，其实只是量错了位置。
    """
    import torch
    m, model = gen_long._MAIN, gen_long._MODEL
    dists = []
    for q in KL_SET[:n]:
        ids = gen_long.enc("User: {}\nBot: ".format(q))
        x = torch.tensor([ids[-m.block_size:]], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(x)
        dists.append(torch.softmax(logits[0, -1].float(), -1))
    tot, cnt = 0.0, 0
    for i, j in itertools.combinations(range(len(dists)), 2):
        p, q = dists[i], dists[j]
        lp, lq = p.clamp_min(1e-12).log(), q.clamp_min(1e-12).log()
        kl_pq = float((p * (lp - lq)).sum())
        kl_qp = float((q * (lq - lp)).sum())
        hp, hq = float(-(p * lp).sum()), float(-(q * lq).sum())
        if hp > 1e-3 and hq > 1e-3:
            tot += 0.5 * (kl_pq / hp + kl_qp / hq)
            cnt += 1
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", default="unseen", choices=("seen", "unseen"),
                    help="seen = 训练用过的手写问法；unseen = 没见过的问法（真正的考验）")
    ap.add_argument("--n", type=int, default=6, help="每概念问几种（0 = 全部）")
    ap.add_argument("--show", type=int, default=6, help="打印几个错例")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-kl", action="store_true")
    ap.add_argument("--legacy-sig", default="",
                    help="旧版答案池 JSON（如 model/stance5_legacy_answers.json）："
                         "签名改版后测旧模型必须带，否则答案判 CROSS")
    a = ap.parse_args()

    if a.legacy_sig:
        # 词典签名改版后，旧模型的答案会被新签名判成串味（她俩 vs 两个）。
        # 用快照覆盖对应概念的答案池，判定口径回到模型对应的那一版。
        import json
        legacy = json.load(open(a.legacy_sig, encoding="utf-8"))
        n = 0
        for c, ans in legacy.items():
            if c.startswith("_"):
                continue
            SL.ANSWERS[c] = ans
            n += 1
        print("  [legacy-sig] 覆盖 %d 个概念的答案池（%s）" % (n, a.legacy_sig))

    gen_long.load(a.ckpt)
    rng = random.Random(7)

    hit, miss, cross, rows = 0, 0, 0, []
    whole = 0          # 前 4 字也对：路由对了之后措辞有没有对上
    for cname, topic, nouns, animal in R.CONCEPTS:
        pool = SL.ANSWERS[cname]
        src = SL.QUESTIONS[cname] if a.set == "seen" else UNSEEN.get(cname, [])
        qs = list(src)
        if a.n and len(qs) > a.n:
            qs = rng.sample(qs, a.n)
        for q in qs:
            g = gen(q, a.temp, a.seed)
            m = head_match(g, pool, k=2)
            if m:
                hit += 1
                if head_match(g, pool, k=4):
                    whole += 1
                rows.append(("OK", cname, q, g))
            else:
                other = None
                for c2, arr2 in SL.ANSWERS.items():
                    if c2 != cname and head_match(g, arr2, k=2):
                        other = c2
                        break
                if other:
                    cross += 1
                    rows.append(("CROSS:" + other, cname, q, g))
                else:
                    miss += 1
                    rows.append(("MISS", cname, q, g))

    total = hit + miss + cross
    print("=== 概念路由 [%s] temp=%.1f ===" % (a.set, a.temp))
    print("路由命中 %d/%d (%.0f%%) | 措辞命中(前4字) %d/%d (%.0f%%) | 串到别的概念 %d | 完全落空 %d"
          % (hit, total, 100.0 * hit / total, whole, total,
             100.0 * whole / total, cross, miss))
    bad = [r for r in rows if r[0] != "OK"]
    if bad:
        print("\n错例（最多 %d 个）：" % a.show)
        for tag, c, q, g in bad[: a.show]:
            print("  %-12s %-4s「%s」-> %s" % (tag, c, q, g))

    # 陌生词
    okd = 0
    print("\n=== 陌生词 ===")
    for q in OOD_PROBE:
        g = gen(q, a.temp, a.seed)
        d = head_match(g, SL.DEFLECT)
        okd += 1 if d else 0
        if not d:
            print("  漏 %-6s -> %s" % (q, g))
    print("回避 %d/%d (%.0f%%)" % (okd, len(OOD_PROBE), 100.0 * okd / len(OOD_PROBE)))

    if not a.skip_kl:
        k = kl_probe()
        print("\n=== 输入敏感性 ===")
        print("pairwise 归一化 KL = %.3f（生死线 0.5）%s"
              % (k, "" if k > 0.5 else "  ← 不合格"))

    # 同概念内答案分布：全塌到一条就说明过拟合
    dist = collections.Counter(r[3][:4] for r in rows if r[0] == "OK")
    print("\n命中答案的开头分布（前 6）：%s" % " / ".join("%s×%d" % (k, v) for k, v in dist.most_common(6)))


if __name__ == "__main__":
    main()
