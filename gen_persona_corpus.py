"""琪露诺角色对话语料生成器（L1+L2 合并版）。

为什么不是「一个 prompt 跑 35k 次」：
  ① 同源感 —— 同一 prompt 的批次句式节奏趋同，实测 20 条里前 2 字去重只有 17
  ② 弱点分布不可控 —— 在 prompt 里写「20% 答错」模型做不到，实测全答对
  做法是把「控制量」提到 job 层：每个 job = 话题子集 × 语气 × 意图 × 弱点类型，
  弱点类型决定这一批整体该答对还是答错，剩下交给 LLM 自由发挥。

用法：
    python gen_persona_corpus.py --n 2000            # 目标 2000 条
    python gen_persona_corpus.py --n 2000 --jobs 2   # 只用 2 路并发（调试）
    python gen_persona_corpus.py --n 2000 --resume   # 断点续跑（默认就是续跑）
    python gen_persona_corpus.py --report            # 只出报告，不生成
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- 环境

for _l in open(os.path.expanduser(r"~\.espllm\api.env"), encoding="utf-8"):
    _l = _l.strip()
    if _l and not _l.startswith("#") and "=" in _l:
        _k, _v = _l.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

BASE = os.environ["QN_BASE"]
KEY = os.environ["QN_KEY"]
MODEL = os.environ.get("QN_MODEL", "moonshotai/kimi-k3-free")

OUTDIR = r"E:\espllm_data\persona_corpus"
RAW = os.path.join(OUTDIR, "raw.jsonl")
JOBS = os.path.join(OUTDIR, "jobs_done.txt")
LOGP = os.path.join(OUTDIR, "gen.log")
os.makedirs(OUTDIR, exist_ok=True)

_lock = threading.Lock()
_logf = open(LOGP, "a", encoding="utf-8")


def log(s=""):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), s)
    with _lock:
        print(line, flush=True)
        _logf.write(line + "\n")
        _logf.flush()


# ---------------------------------------------------------------- 素材表

# 20 话题，每个带 10 个具体对象（对象是让回复"具体"而不是空话的关键）
TOPICS = {
    "冰与冷气": "冰、霜、冷气、冰块、冰柱、寒气、冻住、冰面、冰锥、雪",
    "青蛙": "青蛙、大青蛙、池塘、荷叶、蝌蚪、冻住、解冻、碎掉、呱呱叫、水边",
    "雾之湖": "雾、湖面、湖水、湖边、芦苇、水波、雾气、小船、倒影、湖底",
    "冰屋": "冰墙、屋顶、冰床、门、窗、家里、修补、裂缝、住在里面、冻上",
    "大妖精": "大酱、绿头发、朋友、一起玩、聪明、温柔、担心咱、劝咱、陪着咱、大妖精",
    "妖精同伴": "桑妮、露娜、斯塔、三妖精、小妖精、跟着咱、一起飞、比赛、恶作剧、吵闹",
    "恶作剧": "捉弄、躲起来、吓一跳、偷看、冻脚、滑倒、笑、被骂、跑掉、再来一次",
    "算数": "一加一、数数、手指、三加五、九、十个、数不清、算错了、再数一遍、聪明",
    "最强宣言": "最强、幻想乡第一、无敌、冰柱、挑战、比试、赢、输、不服、再来",
    "和灵梦比试": "灵梦、巫女、红色、退治、符卡、弹幕、输、跑掉、下次、不服",
    "和魔理沙比试": "魔理沙、黑白、扫帚、魔法、星星、叫咱笨蛋、追、躲、炸、再打",
    "夏天好热": "夏天、太阳、热、融化、流汗、喝水、想冬天、讨厌、躲起来、冰块",
    "冬天最好": "冬天、雪、冷、喜欢、结冰、雪人、滑冰、白色、安静、礼物",
    "雪人冰雕": "雪人、冰雕、形状、兔子、星星、手、冻住、看、漂亮、化掉",
    "湖里的鱼": "鱼、若鹭姬、人鱼、水里、唱歌、抓鱼、冰、游、藏、听",
    "冰符弹幕": "符卡、冰符、弹幕、冰锥、射出、躲、打中、漂亮、厉害、再来",
    "被小看": "小看、矮、个子小、不厉害、生气、证明、哼、看不起、打赌、等着瞧",
    "迷路": "迷路、路、不知道、森林、树、问路、方向、转圈、找到、回家",
    "红魔馆": "红魔馆、吸血鬼、大小姐、门、不敢去、咲夜、女仆、靠近、偷偷、跑",
    "⑨这个外号": "⑨、笨蛋、不许叫、生气、数字、名字、琪露诺、最强、反驳、哼",
    # 这三类是第一轮语料漏掉、但对话开场最可能出现的问句。实测 6,427 条里
    # 「你是谁」出现 0 次，模型只能拿别的组的回答顶上 —— 不是模型笨，是语料没教。
    "自我介绍": "名字、琪露诺、冰之妖精、几岁、住在哪、雾之湖、最强、幻想乡、笨蛋、九",
    "打招呼": "你好、早上好、再见、在吗、干嘛呢、嗨、晚安、来了、好久不见、陪我玩",
    "喜好": "喜欢、讨厌、最喜欢、不喜欢、冰、冬天、青蛙、热、甜食、玩",
}

# OOD 批次专用话题池。必须和域内话题分开：混在一起时模型会把「冰屋会不会化」
# 这类域内问题塞进 ood 批，导致真正教「拒答」的样本被稀释（实测 55 条 ood 里
# 只有约 20% 是明确拒答，其余都是绕着答）。
OOD_TOPICS = {
    "手机电脑": "手机、电脑、网络、拍照、屏幕、充电",
    "交通": "汽车、飞机、火车、地铁、开车、方向",
    "学校": "学校、考试、作业、老师、课本、上课",
    "外地": "国家、城市、外国、海边、沙漠、旅行",
    "数理": "方程、公式、物理、化学、计算器、定理",
    "时间": "年份、几千年前、未来、历史、时代、计划",
    "钱与工作": "钱、工资、上班、银行卡、买菜、付账",
    "天气科学": "天气预报、气温、下雨的原因、台风、卫星",
    "宇宙": "宇宙、星星、月亮、火箭、地球、太空",
    "家电": "电灯、自来水、暖气、冰箱、空调、洗衣机",
}

TONES = [
    "得意洋洋", "炸毛生气", "不服气", "懒洋洋", "炫耀", "抱怨", "随口敷衍", "兴奋",
]

INTENTS = [
    "提问", "回答", "炫耀", "否认", "抱怨", "邀请", "拒绝", "回忆", "惊叹", "反问", "撒娇", "不服",
]

# 弱点类型：决定这一批整体该答对还是答错（在 job 层控制，不靠 prompt 里的百分比）
KINDS = [
    ("normal", 0.52, "正常对话。琪露诺可以吹牛、可以答对，也可以答非所问，但不要刻意算错。"),
    ("weak_right", 0.18, "话题涉及她的弱点（算数、个子、怕热），但她这次居然答对了，而且要理直气壮。"),
    ("weak_wrong", 0.20, "话题涉及她的弱点（算数、个子、怕热），她自信地回答，但答案是错的，而且被指出也不痛快承认。"),
    ("ood", 0.10, "用户问的是幻想乡之外的事（现代科技、外地、学校、抽象概念）。琪露诺确实不懂。"
                  "**每一条都必须先明确说出不懂**——原话可以是「咱不懂」「咱没听说过」「那是什么玩意」"
                  "「咱不知道」——然后再用她的口气加半句找补（「反正咱最强就对了」「问这个干嘛」）。"
                  "绝对不许硬猜答案、不许假装知道、不许解释原理、不许编造细节。"),
]

SYS = """你在为一个 2.17M 参数的中文角色模型造训练语料。角色是《东方Project》的琪露诺：
冰之妖精，住雾之湖边的冰屋，自称"咱"，坚信自己是幻想乡最强，实际上是个笨蛋。
爱把青蛙冻住再放回湖里，被叫"⑨"会炸毛，夏天怕热。用户是普通人类访客。"""

BODY = """生成 {n} 条单轮对话，严格满足：

【这条最要紧】{kind_desc}

【回复】琪露诺说的话：
1. 8~24 个汉字，一句到两句，口语，具体
2. **每一条都必须含"咱"**，不许用"我"代替
3. 句末必须有 。！？ 之一
4. 不许用比喻（像/如同/仿佛/宛如）、成语、破折号、括号、颜文字
5. 不许长篇讲道理，不许说"作为一个""你可以""建议你"这类话

【用户】问的话：
1. ≤18 个汉字，像真人随口一问，不恭维、不撒娇、不长句
2. 不要每条都用问号结尾，有的可以是陈述句（"这青蛙怎么硬了。"）

【输出】只输出 JSON 数组，不要任何解释、不要 markdown 代码块：
[{{"t":"话题","u":"用户说的话","a":"琪露诺的回答"}}]

本批话题只从这些里取（每条都要用其中一个作为 t）：{topics}
本批整体语气：{tone}
本批多用这几种说话意图：{intents}
不同条目的开头要变化，不要都用"咱要""咱是"开头。"""

BANNED = re.compile(r"(像|如同|仿佛|宛如|好似|好比|——|\(|\)|（|）|作为一个|建议你|你可以)")
END_PUNC = "。！？!?"


# ---------------------------------------------------------------- job 构造

def build_jobs(target, per_job, seed=20261021):
    """按 KINDS 比例生成 job 列表。

    关键约束：**每种 kind 的 job 列表必须与 target 无关**。
    否则 --n 从 600 改到 2000 时 job id 全变，之前跑过的全部认不出来（踩过）。
    做法是每个 kind 用独立 rng 预生成一大批，按配额取前缀。
    """
    need = (target + per_job - 1) // per_job
    weights = [(name, w, desc) for name, w, desc in KINDS]

    counts = {}
    for name, w, _ in weights:
        counts[name] = max(1, int(round(need * w)))
    diff = need - sum(counts.values())
    i = 0
    while diff != 0:
        nm = weights[i % len(weights)][0]
        if diff > 0:
            counts[nm] += 1
            diff -= 1
        elif counts[nm] > 1:
            counts[nm] -= 1
            diff += 1
        i += 1

    desc_of = {n: d for n, _, d in weights}
    pools = {}
    for name, _w, _d in weights:
        rng = random.Random("%s|%d" % (name, seed))
        tpool = OOD_TOPICS if name == "ood" else TOPICS
        # 只造指定话题（补漏用）。扩话题表会让每个 job 的 tag 变化 → md5 变 →
        # resume 全部失效，为了补 3 个话题重跑 8000 条纯属浪费。
        _only = [s.strip() for s in
                 os.environ.get("ESPGPT_PERSONA_TOPICS", "").split(",") if s.strip()]
        if _only:
            tpool = {k: v for k, v in tpool.items() if k in _only}
        n_keys = min(6, len(tpool))
        lst = []
        for _ in range(max(need, 200)):
            keys = rng.sample(list(tpool), n_keys)
            tone = rng.choice(TONES)
            intents = rng.sample(INTENTS, 3)
            tag = "%s|%s|%s|%s" % (name, ",".join(keys), tone, ",".join(intents))
            lst.append({
                "id": hashlib.md5(tag.encode()).hexdigest()[:10], "kind": name,
                "kind_desc": desc_of[name], "topics": keys, "tone": tone,
                "intents": intents, "pool": "ood" if name == "ood" else "in",
                "n": per_job,
            })
        pools[name] = lst

    # 交错排列，避免同类 job 扎堆（顺序固定，不受 target 影响）
    order = []
    for name, _w, _d in weights:
        order.extend([name] * counts[name])
    random.Random(seed).shuffle(order)
    ptr = {k: 0 for k in pools}
    jobs = []
    for name in order:
        jobs.append(pools[name][ptr[name]])
        ptr[name] += 1
    return jobs


def render(job):
    pool = OOD_TOPICS if job.get("pool") == "ood" else TOPICS
    topics = "；".join("%s（可用：%s）" % (k, pool[k]) for k in job["topics"])
    return (SYS + "\n\n" + BODY).format(
        n=job["n"], kind_desc=job["kind_desc"], topics=topics,
        tone=job["tone"], intents="、".join(job["intents"]))


# ---------------------------------------------------------------- API

def call(prompt, retries=5):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 9000,
        "temperature": 0.95,
        "reasoning_effort": "low",   # 实测：思考 25 tok，质量优于 max 档，速度快 8 倍
    }
    body = json.dumps(payload, ensure_ascii=False).encode()
    last = ""
    for i in range(retries):
        req = urllib.request.Request(
            BASE.rstrip("/") + "/chat/completions", data=body,
            headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
            method="POST")
        try:
            # timeout 不能给太长：6 路并发时服务端会挂起而不是报错，
            # 420s 的 timeout 等于让 6 个线程一起白等 7 分钟（踩过）。
            with urllib.request.urlopen(req, timeout=150) as r:
                j = json.loads(r.read().decode("utf-8", "replace"))
            ch = j["choices"][0]
            return True, ((ch.get("message") or {}).get("content") or ""), j.get("usage") or {}
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")
            last = "HTTP %d %s" % (e.code, txt[:100])
            if e.code in (429, 500, 502, 503, 522):
                time.sleep(4 * (i + 1))
                continue
            return False, last, {}
        except Exception as e:  # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, str(e)[:90])
            time.sleep(4)
    return False, last, {}


def parse_rows(content):
    """从可能带 markdown 围栏/前后废话的输出里抠出 JSON 数组。"""
    if not content:
        return []
    m = re.search(r"\[.*\]", content, re.S)
    if not m:
        return []
    try:
        rows = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return []
    return [r for r in rows if isinstance(r, dict)]


# ---------------------------------------------------------------- 质检

def han(s):
    return len(re.sub(r"[^\u4e00-\u9fff]", "", s))


def gate_row(row):
    """闸 2（长度/标点/禁用词）+ 用户位长度。闸 1（字表）留到全局分析阶段。"""
    u = str(row.get("u", "")).strip()
    a = str(row.get("a", "")).strip()
    t = str(row.get("t", "")).strip()
    if not a or not u:
        return None, "空"
    na = han(a)
    if na < 6:
        return None, "回复太短"
    if na > 26:
        return None, "回复太长"
    if "咱" not in a:
        return None, "缺咱"
    if "我" in a:
        return None, "混用我"
    if a[-1] not in END_PUNC:
        return None, "无句末标点"
    if BANNED.search(a):
        return None, "禁用写法"
    if han(u) > 20:
        return None, "用户位太长"
    if not t:
        t = "未标注"
    return {"t": t, "u": u, "a": a, "kind": row.get("kind", "")}, ""


# ---------------------------------------------------------------- 主流程

def load_done():
    done = set()
    if os.path.exists(JOBS):
        for line in open(JOBS, encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(line)
    return done


def load_seen():
    """已落盘的回复文本集合，用于跨 job 去重。"""
    seen = set()
    if os.path.exists(RAW):
        for line in open(RAW, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["a"])
            except Exception:  # noqa: BLE001
                pass
    return seen


def run_job(job):
    prompt = render(job)
    t0 = time.time()
    ok, content, usage = call(prompt)
    if not ok:
        log("  ✗ job %s 失败：%s" % (job["id"], content[:80]))
        return job["id"], [], {"fail": 1}
    rows = parse_rows(content)
    if not rows:
        log("  ✗ job %s 无 JSON（%d 字）：%s" % (job["id"], len(content), content[:70].replace("\n", " ")))
        return job["id"], [], {"fail": 1}

    good, reasons = [], {}
    for r in rows:
        r["kind"] = job["kind"]
        rec, why = gate_row(r)
        if rec:
            good.append(rec)
        else:
            reasons[why] = reasons.get(why, 0) + 1

    kept = []
    with _lock:
        for rec in good:
            if rec["a"] in _seen:
                reasons["重复"] = reasons.get("重复", 0) + 1
                continue
            _seen.add(rec["a"])
            kept.append(rec)
        for rec in kept:
            _rawf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _rawf.flush()
        with open(JOBS, "a", encoding="utf-8") as f:
            f.write(job["id"] + "\n")

    dt = time.time() - t0
    log("  ✓ job %-10s %-11s 出 %2d 收 %2d（%.1fs, %s tok）%s" % (
        job["id"], job["kind"], len(rows), len(kept), dt,
        usage.get("completion_tokens"), (" 剔除:" + str(reasons)) if reasons else ""))
    return job["id"], kept, {"rows": len(rows), "kept": len(kept), "dt": dt, "reasons": reasons}


_seen = set()
_rawf = None
_total = {"rows": 0, "kept": 0, "fail": 0}


def main():
    global _seen, _rawf
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--per-job", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=4, help="并发路数")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="忽略已完成的 job，重跑全部")
    a = ap.parse_args()

    if a.report:
        return report()

    done = load_done() if not a.no_resume else set()
    _seen = load_seen()
    all_jobs = build_jobs(a.n, a.per_job)
    todo = [j for j in all_jobs if j["id"] not in done]
    est = sum(j["n"] for j in todo)

    log("=" * 70)
    log("目标 %d 条 / 每 job %d 条 / 共 %d job（已完成 %d，待跑 %d，预计产出 %d 条）" % (
        a.n, a.per_job, len(all_jobs), len(all_jobs) - len(todo), len(todo), est))
    log("并发 %d 路   已有语料 %d 条   模型 %s" % (a.jobs, len(_seen), MODEL))

    if not todo:
        log("没有待跑的 job。用 --no-resume 可重跑全部。")
        return report()

    _rawf = open(RAW, "a", encoding="utf-8")
    t0 = time.time()
    queue = list(todo)
    qlock = threading.Lock()

    def worker(wid):
        while True:
            with qlock:
                if not queue:
                    return
                job = queue.pop(0)
            try:
                jid, kept, st = run_job(job)
            except Exception as e:  # noqa: BLE001
                log("  ✗ 路%d 异常 %s: %s" % (wid, type(e).__name__, str(e)[:80]))
                continue
            with _lock:
                _total["rows"] += st.get("rows", 0)
                _total["kept"] += st.get("kept", 0)
                _total["fail"] += st.get("fail", 0)

    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(a.jobs)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    _rawf.close()

    wall = time.time() - t0
    log("-" * 70)
    log("完成：%d job / %.1f 分钟 / 出 %d 条 收 %d 条（保留率 %.1f%%）/ 失败 %d" % (
        len(todo), wall / 60, _total["rows"], _total["kept"],
        100.0 * _total["kept"] / max(1, _total["rows"]), _total["fail"]))
    log("每条 %.2fs" % (wall / max(1, _total["kept"])))
    return report()


def report():
    if not os.path.exists(RAW):
        print("还没有语料。")
        return 0
    rows = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]
    if not rows:
        print("语料为空。")
        return 0
    n = len(rows)
    lens = sorted(han(r["a"]) for r in rows)
    q = lambda p: lens[min(n - 1, int(n * p))]  # noqa: E731
    zan = sum(1 for r in rows if "咱" in r["a"])
    punc = sum(1 for r in rows if r["a"][-1] in END_PUNC)
    opens = {r["a"][:2] for r in rows}
    uniq_a = len({r["a"] for r in rows})

    print("\n" + "=" * 66)
    print("语料报告  %s" % RAW)
    print("=" * 66)
    print("条数 %d   回复唯一 %d（%.1f%%）   开头二字唯一 %d（%.1f%%）" % (
        n, uniq_a, 100.0 * uniq_a / n, len(opens), 100.0 * len(opens) / n))
    print("回复字数 p10/p50/p90 = %d/%d/%d   均值 %.1f" % (q(0.1), q(0.5), q(0.9), sum(lens) / n))
    ulens = sorted(han(r["u"]) for r in rows)
    print("用户字数 p50/p90 = %d/%d" % (ulens[n // 2], ulens[int(n * 0.9)]))
    print("含「咱」%d/%d (%.1f%%)   句末标点 %d/%d (%.1f%%)" % (
        zan, n, 100.0 * zan / n, punc, n, 100.0 * punc / n))

    print("\n-- 按 kind --")
    kinds = {}
    for r in rows:
        kinds.setdefault(r.get("kind", "?"), []).append(r)
    for k, v in sorted(kinds.items(), key=lambda x: -len(x[1])):
        lk = sorted(han(r["a"]) for r in v)
        print("  %-11s %4d 条 (%.1f%%)  字数 p50 %d" % (
            k, len(v), 100.0 * len(v) / n, lk[len(lk) // 2]))

    print("\n-- 话题覆盖（目标 20 个）--")
    tk = {}
    for r in rows:
        tk[r["t"]] = tk.get(r["t"], 0) + 1
    print("  已出现 %d 个不同话题标签；前 8：" % len(tk))
    for k, v in sorted(tk.items(), key=lambda x: -x[1])[:8]:
        print("    %-12s %d" % (k, v))

    # 字频：这是定 2000 字表的依据
    freq = {}
    for r in rows:
        for ch in r["u"] + r["a"]:
            if "\u4e00" <= ch <= "\u9fff":
                freq[ch] = freq.get(ch, 0) + 1
    tot = sum(freq.values())
    rank = sorted(freq.items(), key=lambda x: -x[1])
    print("\n-- 字频（总字数 %d，不同汉字 %d）--" % (tot, len(rank)))
    cum, marks = 0, {}
    for i, (ch, c) in enumerate(rank, 1):
        cum += c
        for p in (0.9, 0.95, 0.98, 0.99, 0.999):
            if p not in marks and cum / tot >= p:
                marks[p] = i
    for p in (0.9, 0.95, 0.98, 0.99, 0.999):
        print("  覆盖 %.1f%% 需要 %d 个字" % (100 * p, marks.get(p, len(rank))))
    print("  前 30 高频字：%s" % "".join(ch for ch, _ in rank[:30]))

    with open(os.path.join(OUTDIR, "char_freq.txt"), "w", encoding="utf-8") as f:
        for ch, c in rank:
            f.write("%s\t%d\n" % (ch, c))
    print("\n字频表已写 %s" % os.path.join(OUTDIR, "char_freq.txt"))

    print("\n-- 随机 10 条 --")
    rnd = random.Random(7)
    for r in rnd.sample(rows, min(10, n)):
        print("  [%s|%s] %s -> %s" % (r.get("kind", "?"), r["t"][:6], r["u"], r["a"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
