# -*- coding: utf-8 -*-
"""26 概念模型的板上验收（stance26）。

三块，各自独立判定：
  A 日常三概念（吃饭/睡觉/玩耍）：`:rules 0` + `:greedy`，看概念命中率。
    stance5 上这三类是 0/6（语料里根本没有），这次扩到 22 条手写问法。
  B 言语行为归规则（被挑衅/比试）：`:rules 1`，看 trace 是否打 zh_taunt /
    zh_duel。这两类 2026-10-22 从模型迁到规则层（模型的 QK 只能按名词
    路由，概念一多整类塌掉）。
  C 不能抢的自比问句：「你打得过灵梦吗」「你俩谁强」「和灵梦比谁强」必须
    **没有** intent 标签（= 走模型），这些是 最强/灵梦/魔理沙/大妖精 的
    训练问法，规则抢走就是纯损失。

用法: python _probe_26.py COM5
"""
import re
import sys
import time

sys.path.insert(0, ".")
import board_zh_test as B  # noqa: E402
import serial  # noqa: E402
from persona import stance_lexicon as SL  # noqa: E402

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM5"

DAILY = ["吃饭", "睡觉", "玩耍"]

# 这两类的问法已不在词典里（迁到规则层），只能内联。
TAUNT = ["让灵梦收拾你", "让魔理沙打你", "我叫人来收拾你", "你等着",
         "看我不叫大妖精来", "信不信我告诉灵梦"]
DUEL = ["灵梦和大妖精谁厉害", "灵梦和魔理沙谁厉害", "大妖精和魔理沙谁更厉害",
        "巫女和黑白谁厉害", "她俩谁强", "她俩打起来谁赢"]
TAUNT_REPS = ["叫啊！咱就在这儿等着。", "来一个冻一个，来两个冻一双！",
              "吓唬谁呢？咱才不怕。", "随便叫，反正打不过咱。",
              "人越多越好，咱正嫌湖面空。"]
DUEL_REPS = ["两个加起来也不是咱的对手！", "两个一起上，咱一只手就够。",
             "两个谁赢都一样，赢不了咱。", "放马过来，咱照冻不误！"]

# 自比问句：必须放行给模型（规则接走 = 纯损失）
SELF_CMP = ["你打得过灵梦吗", "你俩谁强", "和灵梦比谁强", "你和大酱谁强"]


def ask(ser, q, timeout=40.0):
    ser.reset_input_buffer()
    ser.write((q + "\n").encode("utf-8"))
    ser.flush()
    raw, _ = B.read_until(ser, B.MARK, timeout)
    text = raw.decode("utf-8", errors="replace")
    reply = B.clean_reply(text)
    echo = q + "？"
    if reply.startswith(echo):
        reply = reply[len(echo):]
    reply = reply.split("\n")[0].strip()
    m = re.findall(r"intent:(\w+)", text)
    return reply, (m[-1] if m else None), ("sig_stuck" in text)


def head2(s):
    return s[:2] if len(s) >= 2 else ""


def main():
    ser = serial.Serial(PORT, 115200, timeout=0.2)
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.15)
    ser.setRTS(False)
    B.read_until(ser, B.MARK, 12.0)
    ser.write(b":greedy\n")
    ser.flush()
    time.sleep(0.4)
    ser.reset_input_buffer()

    # ---- A 日常三概念（关规则）----
    ser.write(b":rules 0\n")
    ser.flush()
    time.sleep(0.4)
    ser.reset_input_buffer()
    print("=== A 日常三概念（:greedy / :rules 0）===")
    tot_hit = tot = 0
    for cname in DAILY:
        pool = SL.ANSWERS[cname]
        hit = 0
        qs = SL.QUESTIONS[cname]
        for q in qs:
            g, lab, stuck = ask(ser, q)
            if head2(g) == head2(pool[0]):
                v = "命中"
                hit += 1
            elif any(head2(g) == head2(a) for c in SL.ANSWERS if c != cname
                     for a in SL.ANSWERS[c]):
                v = "串味"
            elif g.startswith("没听") or g.startswith("这事儿"):
                v = "回避"
            else:
                v = "崩坏"
            if stuck:
                v += "+stuck"
            print("  [%-6s] 「%s」-> %s" % (v, q, g[:36]))
        print("  -- %s %d/%d" % (cname, hit, len(qs)))
        print()
        tot_hit += hit
        tot += len(qs)
    print("  日常三概念合计 %d/%d (%.0f%%)\n" % (tot_hit, tot, 100.0 * tot_hit / tot))

    # ---- B 言语行为归规则 ----
    ser.write(b":rules 1\n")
    ser.flush()
    time.sleep(0.4)
    ser.reset_input_buffer()
    print("=== B 言语行为（规则帧 zh_taunt / zh_duel）===")
    ok = 0
    for q, want, reps in ([(q, "zh_taunt", TAUNT_REPS) for q in TAUNT] +
                          [(q, "zh_duel", DUEL_REPS) for q in DUEL]):
        g, lab, _stuck = ask(ser, q)
        good = (lab == want) and any(head2(g) == head2(a) for a in reps)
        ok += 1 if good else 0
        print("  [%s] 「%s」-> %s   <%s>" % ("OK" if good else "NG", q, g[:30], lab))
    print("  -- 规则帧 %d/%d\n" % (ok, len(TAUNT) + len(DUEL)))

    # ---- C 自比问句必须放行给模型 ----
    ser.write(b":greedy\n")
    ser.flush()
    time.sleep(0.4)
    ser.reset_input_buffer()
    print("=== C 自比问句（必须无 intent 标签 = 走模型）===")
    okc = 0
    for q in SELF_CMP:
        g, lab, _stuck = ask(ser, q)
        good = lab is None
        okc += 1 if good else 0
        print("  [%s] 「%s」-> %s   <%s>" % ("OK" if good else "NG", q, g[:30], lab))
    print("  -- 放行 %d/%d" % (okc, len(SELF_CMP)))

    ser.write(b":rand\n")
    ser.flush()
    ser.close()


if __name__ == "__main__":
    main()
