"""把 203 集里反复答错的问法打成「硬例子包」，按步数占比再喂一遍。

为什么需要它：`_stance_mt12` 在 203 集上 198/203，剩下 5 条不是没学过
（「三加五是多少」在语料里出现 7 次），而是被别的概念抢走 —— 精确串召回
在 2.17M / ternary 上已经接近上限。把这几句单独拎出来提高曝光，是唯一
不改架构还能推动它的手段。

四轮记录（每轮只补上一轮新暴露出来的项，累计）：
  r1  你吃什么 / 三加五是多少 / 你打得过她吗 / 湖里有什么 / 你平时干什么
  r2  + 雾之湖在哪 / 湖边有什么 / 你堆过雪人吗 / 你在干嘛
  r3  + 你喝水吗 / 你最喜欢吃什么
  r4  + 你早上吃什么            -> 203/203，且 OOD 回到 12/14

用法：
    python build_hard_pack.py --round 4 --out E:/espllm_data/persona_corpus/fix_r4.txt
    ESPGPT_EXTRA="<多轮>:0.10,<包>:0.12" python main.py --train

注意：包只包含「问 -> 该概念的答案池」这一种映射，答案一律取自
`persona/stance_lexicon.py`（唯一真相），不要在这里另写台词。
"""
import argparse
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "persona"))

import stance_lexicon as SL  # noqa: E402
from pack_zh_corpus import encode, load_vocab  # noqa: E402

ROUNDS = {
    1: [("吃饭", "你吃什么"), ("算数", "三加五是多少"), ("灵梦", "你打得过她吗"),
        ("鱼", "湖里有什么"), ("玩耍", "你平时干什么")],
    2: [("雾之湖", "雾之湖在哪"), ("雾之湖", "湖边有什么"),
        ("雪人", "你堆过雪人吗"), ("玩耍", "你在干嘛")],
    3: [("吃饭", "你喝水吗"), ("吃饭", "你最喜欢吃什么")],
    4: [("吃饭", "你早上吃什么")],
}


def build(rounds, repeat):
    pairs = []
    for r in sorted(ROUNDS):
        if r <= rounds:
            pairs.extend(ROUNDS[r])
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=4, choices=sorted(ROUNDS),
                    help="累计到第几轮（默认 4 = 当前部署的那一版）")
    ap.add_argument("--repeat", type=int, default=3,
                    help="每句重复几遍；包被整批采样，重复只为平衡答案条数")
    ap.add_argument("--out", default=r"E:\espllm_data\persona_corpus\fix_pack.txt")
    a = ap.parse_args()

    pairs = build(a.round, a.repeat)
    chars, stoi, unk = load_vocab()
    lines, masks = [], []
    per_concept = {}
    for cname, q in pairs:
        prompt = "User: %s\nBot:" % q
        p_ids = encode(prompt, stoi, unk)
        for _ in range(a.repeat):
            for ans in SL.ANSWERS[cname]:
                full = prompt + " " + ans + "\n"
                lines.append(" ".join(str(i) for i in encode(full, stoi, unk)))
                masks.append(str(len(p_ids)))
        per_concept[cname] = per_concept.get(cname, 0) + 1

    io.open(a.out, "w", encoding="ascii", newline="\n").write("\n".join(lines) + "\n")
    io.open(a.out + ".mask", "w", encoding="ascii", newline="\n").write("\n".join(masks) + "\n")
    print("round<=%d: %d 句 -> %d 窗口 -> %s" % (a.round, len(pairs), len(lines), a.out))
    print("涉及概念：%s" % ", ".join("%s×%d" % kv for kv in sorted(per_concept.items())))


if __name__ == "__main__":
    main()
