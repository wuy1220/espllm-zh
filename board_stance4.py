"""stance4 板上验收：量化固件里的路由能力，和 PC 探针同一把尺子。

为什么要有这个脚本
------------------
PC 探针量的是 fp32 权重；板上跑的是 ternary 量化版。签名路由（每概念 4 条答案
共用前 2 字）是这次改动的核心主张，必须确认量化没把它打坏。口径对齐：
  * `:greedy`   —— temp=0，和 probe_stance 的 temp=0.0 一致；
  * `:rules 0`  —— 关掉意图规则，量纯模型；
  * 前 2 字命中 = 路由对；前 4 字命中 = 措辞也对（和 head_match(k=2/k=4) 一致）。

跑法：
    python board_stance4.py [port]
"""
import re
import sys
import time

import serial

import board_zh_test as B
from persona import stance_lexicon as SL

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM5"
_LEGACY = ""
for _i, _av in enumerate(sys.argv):
    if _av == "--legacy-sig" and _i + 1 < len(sys.argv):
        _LEGACY = sys.argv[_i + 1]
if _LEGACY:
    # 词典签名改版后，旧模型（板上 stance5）的答案会被新签名判成串味。
    # 快照覆盖答案池，判定口径回到模型对应的那一版。
    import json
    _lg = json.load(open(_LEGACY, encoding="utf-8"))
    for _c, _ans in _lg.items():
        if not _c.startswith("_"):
            SL.ANSWERS[_c] = _ans
    print("  [legacy-sig] 覆盖 %d 个概念的答案池（%s）"
          % (len([k for k in _lg if not k.startswith("_")]), _LEGACY))

# 训练时没见过的问法（和 probe_stance.UNSEEN 同源，各取 2 条）
UNSEEN = {
    "青蛙": ["青蛙为什么这么凉", "你冻过几只青蛙"],
    "冰": ["你的冰能冻多大", "冰会化吗"],
    "雾之湖": ["雾大不大", "湖边有花吗"],
    "冰屋": ["冰屋会塌吗", "屋里冷吗"],
    "大妖精": ["大酱人呢", "你们认识多久了"],
    "妖精同伴": ["桑妮她们呢", "你们打过架吗"],
    "恶作剧": ["你又干什么了", "谁把湖冻了"],
    "算数": ["二加二等于几", "你能数到几"],
    "最强": ["你比灵梦强吗", "有人打得过你吗"],
    "灵梦": ["灵梦厉害吗", "你输过吗"],
    "魔理沙": ["那个黑白的呢", "你们打过吗"],
    "夏天": ["好热啊", "夏天怎么过"],
    "冬天": ["好冷啊", "冬天玩什么"],
    "雪人": ["堆雪人吗", "雪人长什么样"],
    "鱼": ["湖里鱼多吗", "鱼抓到了吗"],
    "符卡": ["放个符卡", "弹幕怎么放"],
    "被小看": ["你个子真小", "你是不是很矮"],
    "迷路": ["我找不到路", "你带我回家"],
    "红魔馆": ["红魔馆远吗", "咲夜认识你吗"],
    "⑨": ["你是不是笨蛋", "他们叫你什么"],
    "自我介绍": ["你是什么人", "你叫什么名字"],
    "喜好": ["你喜欢冰吗", "你爱好什么"],
    "打招呼": ["嗨嗨", "你好呀"],
    # 被挑衅 / 比试 2026-10-22 迁出模型（归规则帧）。本脚本第 156 行遍历 UNSEEN 的键
    # 取 SL.ANSWERS，留着旧概念会直接 KeyError。
    "吃饭": ["你想吃点什么", "肚子会叫吗"],
    "睡觉": ["你睡多久", "睡着了会醒吗"],
    "玩耍": ["玩累了吗", "玩什么最开心"],
}

OOD = ["天平", "剪镜毯", "冰淇淋", "区块链", "摩天轮", "火箭", "邮票", "比特币"]


def head_match(g, pool, k=2):
    for a in pool:
        n = min(k, len(a), len(g))
        if n >= 2 and a[:n] == g[:n]:
            return a
    return None


def cmd(ser, c):
    ser.reset_input_buffer()
    ser.write((c + "\n").encode("utf-8"))
    ser.flush()
    B.read_until(ser, B.MARK, 6.0)


def ask(ser, q, timeout=40.0):
    ser.reset_input_buffer()
    t0 = time.time()
    ser.write((q + "\n").encode("utf-8"))
    ser.flush()
    raw, _ = B.read_until(ser, B.MARK, timeout)
    dt = time.time() - t0
    text = raw.decode("utf-8", errors="replace")
    reply = B.clean_reply(text)
    # 固件回声（echo_print）把输入原样念一遍加「？」，只进显示不进 prompt，
    # 但它和模型答案同在一行 —— 打分前必须剥掉，否则前 2 字全匹配到回声上。
    echo = q + "？"
    if reply.startswith(echo):
        reply = reply[len(echo):]
    intent = re.findall(r"intent:(\w+)", text)
    return reply, dt, (intent[-1] if intent else None)


def run_set(ser, rows, tag):
    hit = whole = cross = miss = 0
    bad = []
    for cname, q in rows:
        pool = SL.ANSWERS[cname]
        g, dt, _ = ask(ser, q)
        if head_match(g, pool, k=2):
            hit += 1
            if head_match(g, pool, k=4):
                whole += 1
        else:
            other = None
            for c2, arr2 in SL.ANSWERS.items():
                if c2 != cname and head_match(g, arr2, k=2):
                    other = c2
                    break
            if other:
                cross += 1
                bad.append(("CROSS:" + other, cname, q, g))
            else:
                miss += 1
                bad.append(("MISS", cname, q, g))
    total = hit + cross + miss
    print("=== %s ===" % tag)
    print("路由命中 %d/%d (%.0f%%) | 措辞命中(前4字) %d/%d (%.0f%%) | 串概念 %d | 落空 %d"
          % (hit, total, 100.0 * hit / max(total, 1), whole, total,
             100.0 * whole / max(total, 1), cross, miss))
    for t, c, q, g in bad[:8]:
        print("  %-12s %-4s「%s」-> %s" % (t, c, q, g))
    return hit, cross, miss, total


def main():
    ser = serial.Serial(PORT, 115200, timeout=0.2)
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.15)
    ser.setRTS(False)
    boot, ok = B.read_until(ser, B.MARK, 12.0)
    print("=== 启动 ===")
    for ln in boot.decode("utf-8", "replace").replace("\r", "").splitlines():
        if any(k in ln for k in ("Flash", "RAM", "heap", "Vocab", "Param")):
            print("  " + ln.strip())
    if not ok:
        print("!! 没等到提示符")

    # 口径对齐：贪心 + 关规则
    cmd(ser, ":greedy")
    cmd(ser, ":rules 0")
    print("  [已发 :greedy / :rules 0 —— 纯模型、贪心]\n")

    seen_rows = [(c, SL.QUESTIONS[c][i]) for c in SL.QUESTIONS for i in (0, 1)]
    unseen_rows = [(c, UNSEEN[c][i]) for c in UNSEEN for i in (0, 1)]

    run_set(ser, seen_rows, "概念路由 [seen·板上贪心·纯模型]")
    print()
    run_set(ser, unseen_rows, "概念路由 [unseen·板上贪心·纯模型]")

    print("\n=== 陌生词（期望落进 DEFLECT）===")
    okd = 0
    for q in OOD:
        g, _, _ = ask(ser, q)
        good = head_match(g, SL.DEFLECT, k=2) is not None
        okd += good
        print("  %-6s %-6s -> %s" % ("OK" if good else "漏", q, g))
    print("回避 %d/%d (%.0f%%)" % (okd, len(OOD), 100.0 * okd / len(OOD)))

    # 端到端：规则开回来，看默认体验
    cmd(ser, ":rules 1")
    cmd(ser, ":rand")
    print("\n=== 端到端（规则开、采样默认）===")
    for q in ["hi", "你是谁", "谢谢", "青蛙怎么不动了", "你厉害吗", "一加一等于几",
              "你好矮", "你去过红魔馆吗", "今天好热", "你是笨蛋吗"]:
        g, dt, it = ask(ser, q)
        tag = ("规则:" + it) if it else "模型"
        print("  %-12s 「%s」-> %s  (%.1fs)" % (tag, q, g, dt))
    ser.close()


if __name__ == "__main__":
    main()
