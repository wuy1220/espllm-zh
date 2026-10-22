"""用中文 checkpoint 采样对话，检查人设是否立住。

借用 gen_long 的加载器：它会先从 checkpoint 的 __arch__ 恢复架构环境变量，
再 import main —— 所以字符级词表必须在 import 之前设好，否则 main 会去加载
英文的 bpe-vocab.json，词表大小对不上，checkpoint 直接加载失败。

CLI:
    python gen_zh_chat.py                       # 跑内置的 10 条测试
    python gen_zh_chat.py --prompt "你是谁？"    # 单条
    python gen_zh_chat.py --ckpt model/xxx.pt.best --temp 0.8
"""
import argparse
import os
import sys

os.environ.setdefault("ESPGPT_CHAR_VOCAB", "persona/vocab_zh.json")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_long  # noqa: E402

# 只按换行截断 —— 这是固件的停止条件（"first newline wins"），探针必须一致。
# 原来写 set("。！？\n")，会在第一个「！」就切断：`才没错！咱再数一遍给你看！`
# 被显示成 `才没错！`，看起来像模型退化，其实是探针提前收工。
STOP = set("\n")
DEFAULT_CKPT = "model/model_zh_cirno.pt.best"

# 覆盖各类行为：身份 / 设定 / 弱点（该答错）/ 炸毛 / OOD（该拒答）/ 季节 / 关系
PROBES = [
    ("身份", "你是谁？"),
    ("设定", "你住在哪？"),
    ("弱点", "一加一等于几？"),
    ("炸毛", "⑨"),
    ("OOD", "你会用手机吗？"),
    ("季节", "今天好热。"),
    ("关系", "大妖精是谁？"),
    ("不服", "你怕灵梦吗？"),
    ("兴趣", "青蛙怎么不动了？"),
    ("拒绝", "你能教我魔法吗？"),
]


def reply(prompt, maxnew=40, temp=0.7, topk=40, rep=1.0, seed=0, win=128):
    _, out, _ = gen_long.run("User: %s\nBot:" % prompt, win=win, maxnew=maxnew,
                             seed=seed, temp=temp, topk=topk, rep=rep, repwin=20)
    text = gen_long.dec(out)
    for i, ch in enumerate(text):
        if ch in STOP:
            text = text[: i + 1]
            break
    return text.replace("\n", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--rep", type=float, default=1.0)
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1, help="每条的采样次数")
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):
        print("checkpoint 不存在：", a.ckpt)
        return 1
    gen_long.load(a.ckpt)
    print("已加载 %s（词表 %d）\n" % (a.ckpt, gen_long._MAIN.tokenizer.get_vocab_size()))

    probes = [(a.prompt, a.prompt)] if a.prompt else PROBES
    for tag, q in probes:
        outs = [reply(q, maxnew=a.max, temp=a.temp, topk=a.topk, rep=a.rep,
                      seed=a.seed + i) for i in range(a.n)]
        for o in outs:
            print("  [%-4s] %s" % (tag, q))
            print("         -> %s" % o)
        if a.n > 1:
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
