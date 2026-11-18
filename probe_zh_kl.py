"""中文版「模型到底有没有读输入」探针：同一模板下换内容，看下一个字的分布动多少。

英文版（probe_prompt_attention.py）量出 0.017 的根因是语料里短输入 0 条；
中文角色语料的 prompt 是按意图轴造的，理论上不该重演。这个脚本就是那道闸：
**短输入 pairwise 归一化 KL 必须 > 0.5**，否则人设再好也是假的——模型只是在
按边缘分布挑一个开头。

归一化方式：KL 的绝对值没法读（要看它所在分布的尺度），所以除以熵。
两条分布 p、q 对称归一化 KL = 0.5*(KL(p||q)/H(p) + KL(q||p)/H(q))，
对所有 prompt 两两平均。

CLI:
    python probe_zh_kl.py
    python probe_zh_kl.py --ckpt model/model_zh_cirno.pt.best --maxlen 6
"""
import argparse
import itertools
import os
import sys

os.environ.setdefault("ESPGPT_CHAR_VOCAB", "persona/vocab_zh.json")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_long  # noqa: E402

DEFAULT_CKPT = "model/model_zh_cirno.pt.best"

# 同一模板、只换内容。长度刻意做得接近（都是 3~6 字），排除"长度差异"这个混淆项。
CONTENT = [
    "你是谁",
    "你住哪",
    "你几岁",
    "今天好热",
    "今天好冷",
    "你在干嘛",
    "大妖精呢",
    "你怕灵梦",
    "青蛙在哪",
    "你会魔法",
    "一加一呢",
    "你喜欢啥",
]

# 罐头外的一组，长度刻意和 CONTENT 对齐（3~6 字）。用来回答一个独立的问题：
# 它读输入的能力是「只在训练分布内成立」，还是「换到没见过的说法也成立」。
# 两组用同一个模板、同一套归一化，数字可以直接比。
OOD = [
    "你谁啊",
    "讲笑话",
    "你会下棋",
    "烹饪妖精",
    "哈哈哈",
    "我难过",
    "抱抱",
    "喜欢你",
    "笨蛋",
    "天气真好",
]

# 结尾这一个空格不是笔误：pack_zh_corpus.py 里是 `full = prompt + " " + ans`，
# 所以回复的第一个 token 永远是空格。少了它，探针量的是「下一个 token 是不是
# 空格」——12 个 prompt 会全部给出 89% 的同一个答案，KL≈0，看起来像"模型不读
# 输入"，其实只是量错了位置。
TMPL = "User: {}\nBot: "


def next_dist(text):
    import torch
    m, model = gen_long._MAIN, gen_long._MODEL
    ids = gen_long.enc(text)
    x = torch.tensor([ids[-m.block_size:]], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(x)
    return torch.softmax(logits[0, -1].float(), -1)


def kl(p, q):
    lp = p.clamp_min(1e-12).log()
    lq = q.clamp_min(1e-12).log()
    return float((p * (lp - lq)).sum())


def ent(p):
    return float(-(p * p.clamp_min(1e-12).log()).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--maxlen", type=int, default=0,
                    help="只取前 N 条（默认全部）")
    ap.add_argument("--set", default="in", choices=("in", "ood"),
                    help="in = 训练分布内的说法；ood = 罐头外说法")
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):
        print("checkpoint 不存在：", a.ckpt)
        return 1
    gen_long.load(a.ckpt)
    print("checkpoint: %s（词表 %d）\n" % (
        a.ckpt, gen_long._MAIN.tokenizer.get_vocab_size()))

    base = OOD if a.set == "ood" else CONTENT
    items = base[: a.maxlen] if a.maxlen else base
    print("=== 同模板换内容：每个 prompt 的 top-3 下一个字 ===")
    dists = []
    for c in items:
        p = next_dist(TMPL.format(c))
        dists.append(p)
        top = __import__("torch").topk(p, 3)
        s = "  ".join("%r %.1f%%" % (gen_long.dec([int(i)]), 100 * float(v))
                      for v, i in zip(top.values, top.indices))
        print("  %-8s H=%.2f  %s" % (c, ent(p), s))

    pair = []
    for p, q in itertools.combinations(dists, 2):
        hp, hq = ent(p), ent(q)
        if hp < 1e-6 or hq < 1e-6:
            continue
        pair.append(0.5 * (kl(p, q) / hp + kl(q, p) / hq))
    pair.sort()
    mean = sum(pair) / len(pair)
    med = pair[len(pair) // 2]
    print("\n=== 内容敏感性 ===")
    print("  pairwise 归一化 KL：均值 %.4f  中位 %.4f  最小 %.4f  最大 %.4f" % (
        mean, med, pair[0], pair[-1]))
    verdict = "读了输入" if mean > 0.5 else ("几乎没读" if mean < 0.1 else "弱相关")
    print("  判定：%s（生死线 0.5）" % verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
