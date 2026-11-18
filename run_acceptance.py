"""一命令验收：把所有固定探针跑一遍，直接给 PASS / FAIL。

为什么要它
----------
`docs/project_state.md` 20.2 定义了统一验收集，但那是五条命令、五个进程，
每次手工抄数字。2026-10-24 的几轮 A/B 就是在这里翻车的：控制组的 unseen 只有
4/26（正式模型 20/26），却因为没人先跑验收，被当成有效对照去下结论。
把门槛写成代码，控制组不合格会当场红掉。

各探针必须独立进程
------------------
`gen_long.load()` 有单例缓存，同进程换 ckpt 会静默沿用上一个模型
（`per_concept.py` 文档里记过这次误判），所以这里一律 subprocess。

用法
----
    python run_acceptance.py                       # PC 全量，约 3~4 分钟
    python run_acceptance.py --quick                # 跳 203 集，约 1 分钟
    python run_acceptance.py --board COM5           # 加板端三段验收 + 延迟
    python run_acceptance.py --ckpt model/xxx.pt.best

退出码：0 = 全部 PASS，1 = 有 FAIL。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# 门槛取自项目历史基线（docs/project_state.md 20.1 / 20.17 / 20.34）。
# 宁可按已知最好值卡死，也不要让"比上次差一点"悄悄通过。
GATES = {
    "seen":        (195, 203, "203 集逐概念命中（部署模型 198/203）"),
    "unseen":      (20, 26,   "未见问法路由（每概念 1 句）"),
    "ood":         (12, 14,   "陌生词回避"),
    "kl":          (0.5, None, "输入敏感性归一化 KL，必须 > 0.5"),
    "daily":       (60, 66,   "板上日常三概念（吃饭/睡觉/玩耍）"),
    "latency_ms":  (None, 250, "板上平均 token 延迟上限"),
}


def run(cmd, timeout=1800):
    """跑一个子进程，返回 (rc, 合并输出)。中文靠 PYTHONIOENCODING 固定成 UTF-8。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True,
                           timeout=timeout)
        out = p.stdout.decode("utf-8", errors="replace") + \
              p.stderr.decode("utf-8", errors="replace")
        return p.returncode, out, time.time() - t0
    except subprocess.TimeoutExpired as e:
        got = (e.stdout or b"").decode("utf-8", errors="replace")
        return 124, got + "\n!! 超时 %ds" % timeout, time.time() - t0


def grep1(pat, text, cast=float):
    m = re.search(pat, text)
    return cast(m.group(1)) if m else None


def frac(pat, text):
    """抓 `X/Y` 形式的分数。"""
    m = re.search(pat, text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/_stance_mt12.pt.best")
    ap.add_argument("--quick", action="store_true", help="跳过 203 集（省 2 分钟）")
    ap.add_argument("--board", default="", help="串口号，如 COM5；给了才跑板端")
    ap.add_argument("--json", default="", help="把结果另存为 JSON")
    ap.add_argument("--show", type=int, default=6, help="每个探针打印几个错例")
    a = ap.parse_args()

    rows = []          # (检查项, 结果文本, 门槛文本, PASS/FAIL/SKIP)
    logs = {}

    def add(name, got, need, ok, detail=""):
        rows.append((name, got, need, "PASS" if ok else "FAIL", detail))

    # ── 1 架构闸门 ───────────────────────────────────────────
    rc, out, dt = run([PY, "check_arch_sync.py", a.ckpt])
    logs["arch"] = out
    ok = (rc == 0) and ("matches the checkpoint" in out or "header-only OK" in out)
    add("架构同步", "通过" if rc == 0 else "失败", "头文件 == checkpoint", ok,
        "" if ok else out.strip().splitlines()[-1][:80])

    # ── 2 未见问法 + 陌生词 + KL ─────────────────────────────
    rc, out, dt = run([PY, "-u", "probe_stance.py", "--ckpt", a.ckpt,
                       "--set", "unseen", "--n", "1", "--show", str(a.show)])
    logs["unseen"] = out
    hit, tot = frac(r"路由命中 (\d+)/(\d+)", out)
    ood, ood_tot = frac(r"回避 (\d+)/(\d+)", out)
    kl = grep1(r"归一化 KL = ([\d.]+)", out)

    need_h, need_t, _ = GATES["unseen"]
    if hit is None:
        rows.append(("未见问法路由", "解析失败", "≥ %d/%d" % (need_h, need_t), "FAIL", "见 logs"))
    else:
        add("未见问法路由", "%d/%d" % (hit, tot or need_t), "≥ %d/%d" % (need_h, need_t),
            hit * need_t >= need_h * (tot or need_t))

    need_o, need_ot, _ = GATES["ood"]
    if ood is None:
        rows.append(("陌生词回避", "解析失败", "≥ %d/%d" % (need_o, need_ot), "FAIL", ""))
    else:
        add("陌生词回避", "%d/%d" % (ood, ood_tot), "≥ %d/%d" % (need_o, need_ot),
            ood * need_ot >= need_o * (ood_tot or need_ot))

    if kl is None:
        rows.append(("输入敏感性 KL", "解析失败", "> 0.5", "FAIL", "加 --skip-kl 了？"))
    else:
        add("输入敏感性 KL", "%.3f" % kl, "> 0.5", kl > GATES["kl"][0])

    # ── 3 203 集逐概念 ──────────────────────────────────────
    if not a.quick:
        rc, out, dt = run([PY, "-u", "per_concept.py", "--ckpt", a.ckpt])
        logs["per_concept"] = out
        hit, tot = frac(r"总命中 (\d+)/(\d+)", out)
        need, need_t, _ = GATES["seen"]
        if hit is None:
            rows.append(("203 集逐概念", "解析失败", "≥ %d/%d" % (need, need_t), "FAIL", ""))
        else:
            add("203 集逐概念", "%d/%d" % (hit, tot or need_t), "≥ %d/%d" % (need, need_t),
                hit * need_t >= need * (tot or need_t))
        m = re.search(r"日常三概念：吃饭 (\d+)/\d+ \| 睡觉 (\d+)/\d+ \| 玩耍 (\d+)/\d+", out)
        if m:
            daily = sum(int(x) for x in m.groups())
            need, need_t, _ = GATES["daily"]
            add("日常三概念", "%d/%d" % (daily, need_t), "≥ %d/%d" % (need, need_t),
                daily >= need)

    # ── 4 板端（可选）─────────────────────────────────────────
    if a.board:
        if not os.path.exists("probe_board_acceptance.py"):
            rows.append(("板端验收", "脚本缺失", "-", "FAIL", ""))
        else:
            rc, out, dt = run([PY, "-u", "probe_board_acceptance.py", a.board], timeout=900)
            logs["board"] = out
            d_hit, d_tot = frac(r"日常三概念合计 (\d+)/(\d+)", out)
            r_hit, r_tot = frac(r"规则帧 (\d+)/(\d+)", out)
            s_hit, s_tot = frac(r"放行 (\d+)/(\d+)", out)
            lat = [float(x) for x in re.findall(r"([\d.]+) ms/tok", out)]
            if d_hit is not None:
                add("板端日常三概念", "%d/%d" % (d_hit, d_tot), "≥ 60/66",
                    d_hit * 66 >= 60 * d_tot)
            if r_hit is not None:
                add("板端规则帧", "%d/%d" % (r_hit, r_tot), "全中", r_hit == r_tot)
            if s_hit is not None:
                add("板端自比放行", "%d/%d" % (s_hit, s_tot), "全中", s_hit == s_tot)
            if lat:
                add("板端延迟", "max %.1f ms/tok" % max(lat), "≤ 250 ms/tok",
                    max(lat) <= GATES["latency_ms"][1])
            else:
                rows.append(("板端延迟", "无 ms/tok 行", "≤ 250 ms/tok", "SKIP", ""))
    else:
        rows.append(("板端", "未指定 --board", "-", "SKIP", "板子没插时正常"))

    # ── 输出 ────────────────────────────────────────────────
    print("=" * 78)
    print("验收：%s   （%s）" % (a.ckpt, time.strftime("%Y-%m-%d %H:%M")))
    print("=" * 78)
    w = max(len(r[0]) for r in rows) + 2
    for name, got, need, verdict, detail in rows:
        print("%-*s %-16s %-14s %s %s" % (w, name, got, need, verdict,
                                          ("  " + detail) if detail else ""))

    failed = [r for r in rows if r[3] == "FAIL"]
    print("-" * 78)
    print("结果：%d PASS / %d FAIL / %d SKIP" %
          (sum(1 for r in rows if r[3] == "PASS"), len(failed),
           sum(1 for r in rows if r[3] == "SKIP")))
    if failed:
        print("不合格：" + "、".join(r[0] for r in failed))
        print("逐项原始输出见 logs/acceptance_*.json")

    if a.json or failed:
        os.makedirs("logs", exist_ok=True)
        path = a.json or os.path.join(
            "logs", "acceptance_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ckpt": a.ckpt,
                       "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "rows": [dict(zip(("item", "got", "need", "verdict", "detail"), r))
                                for r in rows],
                       "raw": logs}, f, ensure_ascii=False, indent=2)
        print("已写 %s" % path)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
