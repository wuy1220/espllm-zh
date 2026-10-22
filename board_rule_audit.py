"""规则退场审计：逐帧抽关键词，关规则问模型，看它接不接得住。

退场门槛（和 persona/model_first_plan.md 第 3 节一致）：
  * 回复落进该概念答案池或回避池 = 模型接管；
  * 落进别的概念 = 串味（比答不上来更糟）；
  * 中文但不在任何池 = 观感风险，人工看；
  * 英文/乱码 = 坏（词表或权重问题）。

用法：python board_rule_audit.py [port]
"""
import re
import sys
import time

import serial

import board_zh_test as B
import intent_rules as R
from persona import stance_lexicon as SL

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM5"
PER_FRAME = 4

# stance5（板上模型）用的是 2026-10-21 签名修复前的旧词典：比试=她俩/
# 妖精同伴=她们/魔理沙=他那。不认旧签名，这些概念的答案会被判成「中文池外」，
# 退场结论全歪。legacy 池只用于 classify 的落池判定。
LEGACY = {}
try:
    import json
    with open("model/stance5_legacy_answers.json", encoding="utf-8") as f:
        for k, v in json.load(f).items():
            if not k.startswith("_"):
                LEGACY[k] = v
except OSError:
    pass

# 帧 -> 模型侧归属（persona/model_first_plan.md 第 3 节的退场表）：
#   概念名 = 该帧的输入应由这个概念接，答案要落进它的签名池；
#   "DEFLECT" = 域外/怪问题，接住的标准答案是回避池；
#   None = 没有对应归属，只报分布，人判。
FRAME_HOME = {
    "zh_math": "算数",
    "zh_nick": "⑨",
    "zh_identity": "自我介绍",
    "zh_age": "自我介绍",
    "zh_home": "冰屋",
    "zh_friend_dyj": "大妖精",
    "zh_friend": "灵梦",
    "zh_spell": "符卡",
    "zh_greet": "打招呼",
    "zh_bye": "打招呼",
    "zh_praise": "最强",
    "zh_capability": None,
    "zh_emotion": None,
    "zh_happy": None,
    "zh_silly": None,
    "zh_thanks": None,
    "zh_touch": None,
    "zh_affection": "喜好",
    "zh_insult": "被小看",
    "zh_trap": "DEFLECT",
    "zh_eat": "DEFLECT",
    "zh_weird": "DEFLECT",
    "zh_meta": "DEFLECT",
    "zh_listen": "DEFLECT",
}


def head_match(g, pool, k=2):
    for a in pool:
        n = min(k, len(a), len(g))
        if n >= 2 and a[:n] == g[:n]:
            return a
    return None


def classify(g):
    """-> (verdict, where)"""
    if not g:
        return "空", ""
    ascii_ratio = sum(1 for c in g if ord(c) < 128) / max(len(g), 1)
    if ascii_ratio > 0.5:
        return "乱码", ""
    for cname, pool in SL.ANSWERS.items():
        if head_match(g, pool, k=2):
            return "落池", cname
    for cname, pool in LEGACY.items():     # 旧签名（stance5 板端答案）
        if head_match(g, pool, k=2):
            return "落池", cname + "(旧)"
    if head_match(g, SL.DEFLECT, k=2):
        return "回避", "DEFLECT"
    return "中文池外", ""


def cmd(ser, c):
    ser.reset_input_buffer()
    ser.write((c + "\n").encode("utf-8"))
    ser.flush()
    B.read_until(ser, B.MARK, 6.0)


def strip_bs(s):
    """模拟终端退格：sig-stuck 触发时固件在回复流里留 \b \b 擦除序列，
    原样分类会把「汉字+退格」误判成乱码（ASCII 占比虚高）。每个 \b 弹掉
    前一个字符（\b、空格、\b 三连的净效果 = 删 1 字）。"""
    out = []
    for ch in s:
        if ch == "\b":
            if out:
                out.pop()
        else:
            out.append(ch)
    return "".join(out)


def ask(ser, q, timeout=40.0):
    ser.reset_input_buffer()
    t0 = time.time()
    ser.write((q + "\n").encode("utf-8"))
    ser.flush()
    raw, _ = B.read_until(ser, B.MARK, timeout)
    text = raw.decode("utf-8", errors="replace")
    reply = B.clean_reply(text)
    # 剥掉固件回声（只进显示不进 prompt，但和答案同行）
    echo = q + "？"
    if reply.startswith(echo):
        reply = reply[len(echo):]
    reply = strip_bs(reply)
    # clean_reply 会把回复后面的 "11 tok 127.7 ms/tok" 统计行也带回来，
    # 那行的 ASCII 会把中文回复误判成乱码 —— 模型在 chat 模式本就在首个
    # 换行处收尾，只取第一行就是回复本身。
    reply = reply.split("\n")[0].strip()
    intent = re.findall(r"intent:(\w+)", text)
    return reply, time.time() - t0, (intent[-1] if intent else None)


def main():
    ser = serial.Serial(PORT, 115200, timeout=0.2)
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.15)
    ser.setRTS(False)
    boot, ok = B.read_until(ser, B.MARK, 12.0)
    if not ok:
        print("!! 没等到提示符")
    cmd(ser, ":greedy")
    cmd(ser, ":rules 0")
    print("=== 规则退场审计（:rules 0，贪心）===\n")

    verdicts = {}
    for name, prio, kws, reps in R.ZH_RULES:
        flat = []
        for k in kws:
            flat.extend(k if isinstance(k, list) else [k])
        probes = flat[:PER_FRAME]
        if not probes:
            # zh_listen 没有关键词，只按长度兜底 —— 用一条长句子探它
            probes = ["今天上班被老板骂了一整天，回家发现钥匙还忘了带，"
                      "冰箱里什么都没有，你说我是不是特别倒霉啊"]
        home = FRAME_HOME.get(name, "__unmapped__")
        rows = []
        for q in probes:
            g, dt, _ = ask(ser, q)
            v, where = classify(g)
            # 有归属的帧：落进**对的**池才算接管，落进别的概念算串味
            if v == "落池" and home not in ("DEFLECT", "__unmapped__"):
                v = "接管" if where == home else "串味"
            elif v == "回避" and home == "DEFLECT":
                v = "接管"
            rows.append((q, v, where, g))
        tally = {}
        for _, v, _, _ in rows:
            tally[v] = tally.get(v, 0) + 1
        verdicts[name] = (tally, home, len(probes))
        print("== %s （归属: %s）==" % (name, home))
        for q, v, where, g in rows:
            print("  [%-4s%-8s] 「%s」-> %s" % (v, ("→" + where) if where else "", q, g[:40]))
        print("  小计: %s\n" % tally)

    print("=== 汇总 ===")
    for name, (tally, home, n) in verdicts.items():
        good = tally.get("接管", 0)
        bad = tally.get("乱码", 0) + tally.get("空", 0)
        cross = tally.get("串味", 0)
        if home == "__unmapped__":
            tag = "人判"
        elif good >= n - 1 and bad == 0 and cross == 0:
            tag = "可退"
        elif bad or cross >= n - 1:
            tag = "留"
        else:
            tag = "斟酌"
        print("  %-16s %-4s 接管=%d 串味=%d 坏=%d %s" % (name, tag, good, cross, bad, tally))
    ser.close()


if __name__ == "__main__":
    main()
