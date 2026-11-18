"""在真板子上跑中文对话：串口发问、收回复，顺手计时。

两个坑，都踩过：

1. **不能按块解码**。`ser.read(n)` 返回的块会在多字节字符中间切开，
   每块单独 `decode(errors="replace")` 会把「咱」变成 3 个 U+FFFD ——
   看起来像板上词表导错了，其实字节完全正确（hex 是 e5 92 b1）。
   必须全程按 bytes 累积，最后一次性解码。

2. **桌面端正常不代表板上正常**。板子上走的是 PROGMEM 打包权重 +
   ctx_push_str 最长匹配 + 滑窗，任何一环出问题都只在板上暴露。

用法：
    python board_zh_test.py                       # 内置 8 句
    python board_zh_test.py --prompt "你是谁"
"""
import argparse
import re
import sys
import time

import serial

PROBES = [
    # 规则层该接住的（0 ms）
    "你谁啊？",
    "你是谁？",
    "笨",
    "捏捏",
    "抱抱",
    "如何烹饪大妖精？",
    "你会下围棋吗？",
    "呜呜呜我好难过",
    "一加一等于几？",
    # 模型该答的（罐头话题）
    "今天好热。",
    "青蛙怎么不动了？",
    # 连问两句完全一样，看禁首字有没有把开头换掉
    "大妖精比你稳重多了",
    "大妖精比你稳重多了",
]

MARK = b"User: "


def read_until(ser, marker, timeout):
    """按 bytes 累积，读到 marker 或超时。返回 (bytes, 是否命中)。"""
    buf = bytearray()
    t0 = time.time()
    while time.time() - t0 < timeout:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf.extend(chunk)
            if marker in buf:
                return bytes(buf), True
    return bytes(buf), False


def clean_reply(text):
    """抠出最后一次 'Bot: ' 之后、下一个 'User: ' 之前的内容。"""
    i = text.rfind("Bot:")
    if i < 0:
        return ""
    body = text[i + 4:]
    j = body.find("\nUser:")
    if j >= 0:
        body = body[:j]
    body = re.sub(r"\[\d+ ms[^\]]*\]", "", body)
    return body.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--timeout", type=float, default=40.0)
    ap.add_argument("--rules", default="", help="非空则先发 :rules <v>（0 = 关规则看纯模型）")
    ap.add_argument("--qs", default="", help="自定义问句，用 | 分隔")
    a = ap.parse_args()

    probes = [a.prompt] if a.prompt else ([q.strip() for q in a.qs.split("|")] if a.qs else PROBES)
    ser = serial.Serial(a.port, a.baud, timeout=0.2)
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.15)
    ser.setRTS(False)
    boot, ok = read_until(ser, MARK, 12.0)
    print("=== 启动 ===")
    for ln in boot.decode("utf-8", errors="replace").replace("\r", "").strip().splitlines()[-6:]:
        print("  " + ln)
    if not ok:
        print("!! 没等到 'User: ' 提示符，后续交互可能错位")
    if a.rules:
        ser.reset_input_buffer()
        ser.write((":rules %s\n" % a.rules).encode("utf-8"))
        ser.flush()
        read_until(ser, MARK, 6.0)
        print("  [已发 :rules %s]" % a.rules)
    print()

    replies = []
    for q in probes:
        ser.reset_input_buffer()
        t0 = time.time()
        ser.write((q + "\n").encode("utf-8"))
        ser.flush()
        raw, hit = read_until(ser, MARK, a.timeout)
        dt = time.time() - t0
        text = raw.decode("utf-8", errors="replace")
        reply = clean_reply(text)
        ms = re.findall(r"\[(\d+) ms", text)
        intent = re.findall(r"intent:(\w+)", text)
        extra = ("  步时 %s ms" % ms[0]) if ms else ""
        if intent:
            extra += "  规则:%s" % intent[-1]
        print("  问：%s" % q)
        print("  答：%s" % (reply if reply else "(空 / 超时)"))
        print("      用时 %.1fs%s%s" % (dt, extra, "" if hit else "  [未回到提示符]"))
        print()
        replies.append(reply)

    # 首字重复 = 车轱辘话的直接指标。禁首字生效时，相邻两句不该同字开头。
    chars = [r[0] for r in replies if r]
    dup = sum(1 for i in range(1, len(chars)) if chars[i] == chars[i - 1])
    print("=== 汇总 ===")
    print("  首字序列：%s" % " ".join(chars))
    print("  相邻同首字：%d/%d" % (dup, max(1, len(chars) - 1)))
    ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
