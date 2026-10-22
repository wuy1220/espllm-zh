"""把琪露诺对话语料打包成训练格式（ids 流 + .mask 侧文件）。

格式对齐现有英文语料（main.py 的 CORPUS_FORMAT=ids 分支）：
  - 每行一个窗口，空格分隔的 token id
  - <dataset>.mask 每行一个整数 = 该窗口的 prompt 长度，0 表示整窗都训

模板用 `User: {u}\\nBot: {a}`，与固件 main.cpp 的 `User: X\\nBot:` 完全一致，
也和多轮历史拼接方式一致（轮与轮之间空一行）。

为什么 mask 只标一个整数就够：现在只做单轮，一个窗口就是一条对话。
多轮语料如果拼进同一窗口，这个侧文件表达不了「第二轮用户位也要 mask」，
所以多轮必须单独一条窗口、单独一个 mask 值 —— 见 --multiturn 的处理。

用法：
    python pack_zh_corpus.py                     # 单轮，输出到 E 盘
    python pack_zh_corpus.py --verify            # 打包后自检覆盖率
"""
import argparse
import json
import os
import sys

CORPUS = r"E:\espllm_data\persona_corpus\raw.jsonl"
VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona", "vocab_zh.json")
OUTDIR = r"E:\espllm_data\persona_corpus"
OUT_TXT = os.path.join(OUTDIR, "dataset_cirno.txt")
# 必须紧贴数据集文件名：main.py 用的是 dataset_path + ".mask"，
# 写成 dataset_cirno.mask 时它找不到，会静默按「整窗训练」跑（用户位也进 loss）。
OUT_MASK = OUT_TXT + ".mask"


def load_vocab():
    d = json.load(open(VOCAB, encoding="utf-8"))
    chars = d["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    unk = chars.index("<|endoftext|>") if "<|endoftext|>" in chars else 0
    return chars, stoi, unk


def encode(s, stoi, unk):
    return [stoi.get(c, unk) for c in s]


def main():
    # `--corpus` 的 default 引用了模块级 CORPUS，所以 global 必须在它之前声明。
    global CORPUS, OUT_TXT, OUT_MASK
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--corpus", default=CORPUS, help="原始 jsonl，默认 raw.jsonl")
    ap.add_argument("--out", default="", help="输出 txt 路径，默认 dataset_cirno.txt")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    # 允许换语料/换输出：规范化回复版（raw_canon.jsonl）要和原版并存，
    # 才能在同一套 val 划分下比较，而不是把原版覆盖掉。
    CORPUS = a.corpus
    if a.out:
        OUT_TXT = a.out
        OUT_MASK = OUT_TXT + ".mask"

    chars, stoi, unk = load_vocab()
    print("词表 %d 条，UNK id = %d" % (len(chars), unk))

    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    print("语料 %d 条" % len(rows))

    wins, masks, unk_cnt, tot_tok = [], [], 0, 0
    for r in rows:
        u, ans = r["u"].strip(), r["a"].strip()
        prompt = "User: %s\nBot:" % u
        # 末尾这个 "\n" 是**停止信号**，不是排版：固件的换行即停
        # （"first newline wins"）。原来没加，模型学不到"说完了"这件事，
        # 板上第一句正确之后会顺着故事语料继续编下去 ——
        # 实测「一加一等于几」答完 `当然是三！咱数过的，错不了。` 之后
        # 接了 `她最乐意帮助助，一只名叫琪露诺！`。
        full = prompt + " " + ans + "\n"
        ids = encode(full, stoi, unk)
        p_ids = encode(prompt, stoi, unk)
        # 模板字符若掉进 UNK，说明词表装配有问题，直接暴露出来
        unk_cnt += sum(1 for i in ids if i == unk)
        tot_tok += len(ids)
        wins.append(ids)
        masks.append(len(p_ids))

    lens = sorted(len(w) for w in wins)
    n = len(wins)
    print("窗口 %d 个，长度 p50 %d / p90 %d / max %d，总 token %d" % (
        n, lens[n // 2], lens[int(n * 0.9)], lens[-1], tot_tok))
    print("UNK 出现 %d 次（应接近 0；模板字符 'User:/Bot:' 必须全在词表里）" % unk_cnt)
    ml = sorted(masks)
    print("prompt 长度 p50 %d / max %d（= mask 值）" % (ml[n // 2], ml[-1]))

    over = sum(1 for k in lens if k > 128)
    print("超过 block_size 128 的窗口 %d 个" % over)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        # 这行注释是格式契约，不是说明文字：main.py 的 auto 检测靠
        # `"window per line" in raw_lines[:3]` 来区分「预切窗口」和「扁平流」。
        # 少了它，整个文件会被当成一条扁平流按 block_size 重切，
        # 窗口数从 2505 变成 1400，.mask 直接被忽略（用户位也进 loss）。
        f.write("# pre-sliced windows: one window per line, ids space separated\n")
        for w in wins:
            f.write(" ".join(map(str, w)) + "\n")
    with open(OUT_MASK, "w", encoding="utf-8") as f:
        for m in masks:
            f.write("%d\n" % m)
    print("已写 %s（%.2f MB）" % (OUT_TXT, os.path.getsize(OUT_TXT) / 1e6))
    print("已写 %s" % OUT_MASK)

    if a.verify:
        # 回读并解码前 3 条，确认 encode/decode 对称、模板完整
        print("\n-- 自检：前 3 条解码回文本 --")
        lines = [ln for ln in open(OUT_TXT, encoding="utf-8").read().splitlines()
                 if ln and not ln.startswith("#")][:3]
        ms = open(OUT_MASK, encoding="utf-8").read().splitlines()[:3]
        for ln, m in zip(lines, ms):
            ids = [int(x) for x in ln.split()]
            txt = "".join(chars[i] for i in ids)
            m = int(m)
            print("  prompt[%d]: %r" % (m, "".join(chars[i] for i in ids[:m])))
            print("  target   : %r" % "".join(chars[i] for i in ids[m:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
