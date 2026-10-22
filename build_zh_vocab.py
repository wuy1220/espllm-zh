"""从琪露诺语料构建字符级词表，并给出覆盖率决策依据。

核心问题不是为了凑满 2048：**词表越小，tok_emb 和输出头越省 Flash**。
每词表条目 88 B（tok int8 64 + scale 4 + head 2-bit 16 + scale 4），
所以砍掉 1000 个名额 = 省 88 KB —— 而第 6 层只要 61,800 B。
但词表也不能太小：用户输入的字不在表里就变 UNK。

用法：
    python build_zh_vocab.py              # 只出报告
    python build_zh_vocab.py --write      # 同时写 persona/vocab_zh.json
    python build_zh_vocab.py --size 1536 --write
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

CORPUS = r"E:\espllm_data\persona_corpus\raw.jsonl"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona", "vocab_zh.json")

# 训练模板与固件交互所必需的 ASCII 字符。字符级分词下 "User: " 也要逐个进表，
# 否则模板本身会产生 UNK。
TEMPLATE_CHARS = list("User:Bot \n")

# 中文标点：语料里高频出现，且固件侧要按标点断句/判停
CJK_PUNCT = list("。！？，、；：\"\"''（）《》…—·「」『』～")

# 角色标记，必须单独占一个 token
EXTRA = ["⑨"]

HAN = re.compile(r"[\u4e00-\u9fff]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--size", type=int, default=2048)
    a = ap.parse_args()

    if not os.path.exists(CORPUS):
        print("没有语料：", CORPUS)
        return 1
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8") if l.strip()]
    n = len(rows)
    texts = [r["u"] + r["a"] for r in rows]

    han_freq = Counter()
    other_freq = Counter()
    for t in texts:
        for ch in t:
            if HAN.match(ch):
                han_freq[ch] += 1
            else:
                other_freq[ch] += 1

    n_han_tok = sum(han_freq.values())
    print("=" * 68)
    print("语料 %d 条   汉字 %d 个（不同）   汉字总出现 %d 次" % (
        n, len(han_freq), n_han_tok))
    print("非汉字字符 %d 种：" % len(other_freq))
    for ch, c in other_freq.most_common(28):
        print("   %r × %d" % (ch, c), end="")
    print()

    # 汉字覆盖曲线：决定"砍到多少字还不伤语料"
    rank = [c for c, _ in han_freq.most_common()]
    print("\n-- 保留前 k 个汉字时的语料覆盖率 --")
    cum, marks = 0, []
    for i, (ch, c) in enumerate(han_freq.most_common(), 1):
        cum += c
        if i in (200, 400, 600, 800, 1000, 1200, 1500, 1800, 2000, 2500):
            marks.append((i, cum / n_han_tok))
    for i, cov in marks:
        print("   前 %4d 字 → 覆盖 %.2f%%" % (i, 100 * cov))

    # 每条长度：决定 block_size 够不够
    lens = sorted(len(t) for t in texts)
    print("\n-- 单条字符长度（用户+回复，t = 1 token）--")
    print("   p50 %d  p90 %d  p99 %d  max %d" % (
        lens[n // 2], lens[int(n * 0.9)], lens[int(n * 0.99)], lens[-1]))

    # 词表装配
    size = a.size
    # 模板字符必须无条件进表：语料文本里不会出现 "User:" 的 U/B/o/t，
    # 只按语料字频筛会把它们漏掉，训练时模板本身就变成 UNK 串。
    used_other = []
    for ch in list(TEMPLATE_CHARS) + list(CJK_PUNCT) + EXTRA:
        if ch not in used_other:
            used_other.append(ch)
    for ch, _ in other_freq.most_common():
        if ch not in used_other:
            used_other.append(ch)

    reserved = 1 + len(used_other)          # <|endoftext|> + 非汉字
    han_quota = size - reserved
    kept_han = rank[:han_quota]
    dropped = rank[han_quota:]

    covered = sum(han_freq[c] for c in kept_han) / max(1, n_han_tok)
    vocab = ["<|endoftext|>"] + kept_han + used_other

    print("\n-- 词表装配（目标 %d）--" % size)
    print("  特殊 1 + 汉字 %d + 其他 %d = %d" % (len(kept_han), len(used_other), len(vocab)))
    print("  汉字覆盖率 %.3f%%   被砍掉的汉字 %d 个" % (100 * covered, len(dropped)))
    if dropped:
        print("  被砍的字（按频率）：%s" % "".join(dropped[:40]))

    # 语料里有多少条包含被砍掉的字 → 这些条要替换或丢弃
    drop_set = set(dropped)
    hit = sum(1 for t in texts if any(ch in drop_set for ch in t))
    print("  含被砍字的语料 %d/%d 条（%.1f%%）→ 建库时替换掉或丢弃" % (
        hit, n, 100.0 * hit / n))

    # Flash 估算：相对 2048 词表
    saving = (2048 - len(vocab)) * 88
    print("\n-- Flash --")
    print("  词表 %d 条 vs 英文模型 2048 条 → %+d B（%+.1f KB）" % (
        len(vocab), -saving, -saving / 1024.0))
    print("  第 6 层需要 +61,800 B，当前余量 41,989 B")

    if a.write:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"size": len(vocab), "chars": vocab}, f, ensure_ascii=False)
        print("\n已写 %s（%d 条）" % (OUT, len(vocab)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
