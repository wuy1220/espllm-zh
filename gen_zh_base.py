"""裸续写探针：直接给开头，不加 User/Bot 模板，用来判断语言底子。

阶段 1 的 TinyStories 语料没有对话结构，套对话模板只会得到分布外输出；
想知道"中文语法通不通"必须还原训练时的形态：纯文本续写。

CLI:
    python gen_zh_base.py                                  # 用阶段 1 的 base 模型
    python gen_zh_base.py --ckpt model/model_zh_cirno.pt.best
    python gen_zh_base.py --prompt "从前有一只小猫，"
"""
import argparse
import os
import sys

os.environ.setdefault("ESPGPT_CHAR_VOCAB", "persona/vocab_zh.json")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_long  # noqa: E402

DEFAULT_CKPT = "model/model_zh_base.pt.best"

# 覆盖不同题材与句法：叙事 / 动作 / 因果 / 形容词 / 三要素一致性
PROBES = [
    "从前有一只小猫，",
    "早晨，小明和小红一起去",
    "妈妈把苹果放在桌子上，",
    "小狗跑得很快，因为它要",
    "天空是蓝色的，阳光照在",
    "他打开门，看见一个",
]

STOP = set("。！？\n")


def continue_text(prompt, maxnew=80, temp=0.7, topk=40, rep=1.0, seed=0, win=128):
    _, out, _ = gen_long.run(prompt, win=win, maxnew=maxnew, seed=seed,
                             temp=temp, topk=topk, rep=rep, repwin=20)
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
    ap.add_argument("--max", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):
        print("checkpoint 不存在：", a.ckpt)
        return 1
    gen_long.load(a.ckpt)
    print("已加载 %s（词表 %d）\n" % (a.ckpt, gen_long._MAIN.tokenizer.get_vocab_size()))

    probes = [a.prompt] if a.prompt else PROBES
    for p in probes:
        out = continue_text(p, maxnew=a.max, temp=a.temp, topk=a.topk, rep=a.rep,
                            seed=a.seed)
        print("  %s" % p)
        print("  -> %s\n" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
