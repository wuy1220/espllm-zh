"""生成最终的中文词表：人设语料的字全部收，TinyStories 的字按频率补。

两条硬约束的平衡：
  ① 人设语料的 1207 个汉字必须全收 —— 少一个就有一条训练样本变 UNK
  ② TinyStories 的 3551 个汉字收不全 —— 但那 15% 的缺口主要在「莉/蒂/姆/汤/猫/糕」
     这类故事里的专有名词和具体物件，对「琪露诺怎么说话」没有信息量，
     覆盖到 98% 就够语言预训练用了。

为什么不是越多越好：每词表条目 88 B。词表 1649 时相对 2048 省 35,112 B，
  加上现有余量 41,989 B = 77,101 B，刚好够第 6 层的 61,800 B。

用法：
    python make_zh_vocab.py                 # 汉字上限 1600
    python make_zh_vocab.py --han 1900      # 更保守
    python make_zh_vocab.py --no-ts         # 不掺 TinyStories
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

CORPUS = r"E:\espllm_data\persona_corpus\raw.jsonl"
PARQ = r"E:\espllm_data\corpus\tinystories_zh_0.parquet"
FREQ_CACHE = r"E:\espllm_data\corpus\ts_zh_charfreq.json"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona", "vocab_zh.json")

TEMPLATE_CHARS = list("User:Bot \n")
CJK_PUNCT = list("。！？，、；：\"\"''（）《》…—·「」『』～")
EXTRA = ["⑨"]
HAN = re.compile(r"[\u4e00-\u9fff]")


def ts_freq():
    """TinyStories 字频，缓存到 json（读 parquet 要 1 分多钟）。"""
    if os.path.exists(FREQ_CACHE):
        d = json.load(open(FREQ_CACHE, encoding="utf-8"))
        print("TinyStories 字频读缓存（%d 字）" % len(d))
        return Counter(d)
    import pyarrow.parquet as pq
    f = pq.ParquetFile(PARQ)
    freq = Counter()
    read = 0
    for i in range(f.metadata.num_row_groups):
        for r in f.read_row_group(i).to_pylist():
            for ch in (r.get("text") or ""):
                if HAN.match(ch):
                    freq[ch] += 1
            read += 1
            if read >= 250000:
                break
        if read >= 250000:
            break
    json.dump(dict(freq), open(FREQ_CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    print("TinyStories 字频：%d 条故事，%d 个不同汉字" % (read, len(freq)))
    return freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--han", type=int, default=1600)
    ap.add_argument("--no-ts", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8") if l.strip()]
    han_p, other_p = Counter(), Counter()
    for r in rows:
        for ch in r["u"] + r["a"]:
            (han_p if HAN.match(ch) else other_p)[ch] += 1
    print("人设语料 %d 条：汉字 %d 个，非汉字 %d 种" % (len(rows), len(han_p), len(other_p)))

    tf = Counter() if a.no_ts else ts_freq()

    # 非汉字：模板字符 + 标点 + 角色标记 优先，其余按语料频率
    others = []
    for ch in TEMPLATE_CHARS + CJK_PUNCT + EXTRA:
        if ch not in others:
            others.append(ch)
    for ch, _ in other_p.most_common():
        if ch not in others:
            others.append(ch)
    if not a.no_ts:
        for ch, _ in tf.most_common():
            if not HAN.match(ch) and ch not in others:
                others.append(ch)

    # 汉字：人设字全收（按频率），不足 han 上限时用 TinyStories 补
    han = [c for c, _ in han_p.most_common()]
    if len(han) < a.han:
        for ch, _ in tf.most_common():
            if HAN.match(ch) and ch not in han_p:
                han.append(ch)
                if len(han) >= a.han:
                    break

    vocab = ["<|endoftext|>"] + han + others
    ts_cov = None
    if tf:
        tot = sum(tf.values())
        ts_cov = sum(c for ch, c in tf.items() if ch in set(vocab)) / max(1, tot)

    print("\n词表 %d 条 = 1 特殊 + %d 汉字 + %d 其他" % (len(vocab), len(han), len(others)))
    if ts_cov is not None:
        print("TinyStories 覆盖率 %.2f%%" % (100 * ts_cov))
    print("人设语料覆盖率 100%%（%d 个汉字全部在内）" % len(han_p))
    saving = (2048 - len(vocab)) * 88
    print("\nFlash：相对 2048 词表 %+d B（%+.1f KB）；现有余量 41,989 B → %.0f B" % (
        -saving, -saving / 1024, 41989 + saving))
    print("第 6 层需要 61,800 B → %s" % ("够" if 41989 + saving > 61800 else "不够"))
    print("输出窗口 block 128：单条平均 %.0f token" % (
        sum(len(r["u"]) + len(r["a"]) + 12 for r in rows) / len(rows)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"size": len(vocab), "chars": vocab},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("\n已写 %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
