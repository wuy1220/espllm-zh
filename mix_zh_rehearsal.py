"""把角色语料和 TinyStories 混成一份带 mask 的训练集（阶段 2 抗遗忘）。

为什么必须混：人设语料只有 3,498 窗、且全是短回复，单训它 1,200 步就能把人设
立住，但语法会一起塌掉（实测："雾之湖边的，咱才不怕她" 缺名词）。混入同量的
故事窗口等于每步都在做 rehearsal，让人设在语法不崩的前提下长出来。

输出两份文件：
  dataset_mix_zh.txt         每行一个窗口的 ids（含 header）
  dataset_mix_zh.txt.mask    每行 = 该窗 prompt 长度；故事窗写 0（整窗训练）

CLI:
    python mix_zh_rehearsal.py                     # 1:1，3498+3498
    python mix_zh_rehearsal.py --ratio 0.7         # 人设 : 故事 = 1 : 0.7
"""
import argparse
import os
import random

DIR = r"E:\espllm_data\persona_corpus"
PERSONA = os.path.join(DIR, "dataset_cirno.txt")
STORIES = os.path.join(DIR, "dataset_ts_zh.txt")
OUT = os.path.join(DIR, "dataset_mix_zh.txt")

HEADER = "# pre-sliced windows: one window per line, ids space separated"


def read_windows(path):
    """返回 [ids...]；跳过 '#' 开头的 header。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            rows.append(ln)
    return rows


def main():
    global PERSONA, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default=PERSONA, help="人设语料 txt")
    ap.add_argument("--out", default=OUT, help="输出 txt")
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="故事窗相对人设窗的倍数（1.0 = 一比一）")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    PERSONA = a.persona
    OUT = a.out

    pw = read_windows(PERSONA)
    pm = [int(x) for x in open(PERSONA + ".mask").read().split()]
    assert len(pw) == len(pm), (len(pw), len(pm))

    sw = read_windows(STORIES)
    rng = random.Random(a.seed)
    n_story = min(len(sw), int(round(len(pw) * a.ratio)))
    sw = rng.sample(sw, n_story)

    # 交错排列：保证每个 batch（顺序取样）里两类都出现，而不是前一半人设
    # 后一半故事。人设窗放在偶数位，故事窗填奇数位，余下的按尾部追加。
    merged, masks = [], []
    for i in range(max(len(pw), len(sw))):
        if i < len(pw):
            merged.append(pw[i])
            masks.append(pm[i])
        if i < len(sw):
            merged.append(sw[i])
            masks.append(0)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(HEADER + "\n")
        for w in merged:
            f.write(w + "\n")
    with open(OUT + ".mask", "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, masks)) + "\n")

    import statistics
    pl = [len(x.split()) for x in pw]
    sl = [len(x.split()) for x in sw]
    print("人设窗 %d（长度中位 %d）" % (len(pw), statistics.median(pl)))
    print("故事窗 %d（长度中位 %d）" % (len(sw), statistics.median(sl)))
    print("合计 %d 窗，%.2f MB" % (len(merged), os.path.getsize(OUT) / 1e6))
    print("已写 %s 与 .mask" % OUT)


if __name__ == "__main__":
    main()
