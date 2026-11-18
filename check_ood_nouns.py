# -*- coding: utf-8 -*-
"""zh_ood 词表的安全闸门 + 「通用回避谓词」的否证记录。

一、为什么不做「疑问句 + 无概念名词」的通用规则
----------------------------------------------
实测（`--measure`）：这条谓词在 203 集上抢走 32 条、在 unseen 上抢走 16 条。
原因是概念问句大量用代词和联想词，而不写概念名词 ——「你屋里有什么」（冰屋）、
「她们仨呢」（妖精同伴）、「谁跑得快」、「怎么才能凉快」、「馆里有什么」、
「二加二等于几」。上线 = 203 从 203 掉到 171，净损失。

更根本的一点：**「已知概念的新说法」和「未知话题」在字面上同构**。
「你怕火吗」和「股票怎么买」都含语料里没出现过的内容二元组，任何词表/覆盖率
判据都无法只留其一。这个任务在 2.17M 字符级模型上是**容量性**的，不是阈值性的。

二、可行替代：显式词表（= 闭世界假设）
--------------------------------------
子串匹配下词表项对整句同样生效（「股票怎么买」含「股票」即命中），覆盖面与通用
规则接近，但**误伤可证明为零** —— 一个词必须同时过四道闸：

1. 不是已配置概念名词（否则会把概念问句抢走，如「冰」「湖」「青蛙」）；
2. 不出现在 203 集 / unseen 的任何问句里（否则直接掉分）；
3. 不与任何现有帧关键词互相包含（否则会把已有帧的输入抢到前面，纯改行为）；
4. 不是任何回避池/答案池的片段（避免自相矛盾）。

三、两类输入必须分开量
----------------------
「股票怎么买」要的是「咱不懂」；「我感冒了」「工资太低了」要的是共情。
后者回「咱只懂冰和妖精」是**装傻**，不是回避。所以：

* `KNOW_Q` 计入回避率（硬回避 = 落到 zh_ood）；
* `CARE_Q` 只要求「不许落到 zh_ood」，落给模型算通过（与改动前一致，不算回归）。

CLI:
    python check_ood_nouns.py            # 体检词表 + 覆盖 + 冷漠
    python check_ood_nouns.py --check    # 只看闸门（run_acceptance 用），失败退出 1
    python check_ood_nouns.py --measure  # 复现「通用谓词会抢走多少」
    python check_ood_nouns.py --emit     # 只打通过的词
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "persona"))

import intent_rules as I
import stance_lexicon as SL
import gen_react_corpus as R
import coverage_zh_rules as C
import probe_stance as P

# ── 候选：外部世界的常见话题词 ──────────────────────────────
# 取向是「一个住在雾之湖的冰妖精不可能知道的事」。开头 8 个是 2026-10 就在表里
# 的老词，必须保留（换新表不能顺手丢掉旧覆盖）。
LEGACY = ["天平", "剪镜毯", "冰淇淋", "区块链", "摩天轮", "火箭", "邮票", "比特币"]

CANDIDATES = LEGACY + """
手机 电脑 网络 网站 视频 代码 编程 软件 硬件 芯片 屏幕 电池 相机 耳机
键盘 鼠标 打印机 密码 账号 邮箱 短信
钱 股票 基金 保险 银行 贷款 房价 税 发票 红包 付款 支付
学校 老师 同学 考试 大学 专业 毕业 公司 会议
简历 面试 合同 法律 律师 警察 政府 新闻 报纸 电视 电影 音乐 歌曲 歌 明星
足球 篮球 奥运 冠军 金牌 世界杯
北京 上海 广州 深圳 中国 美国 日本 欧洲 地球 月亮 宇宙 火星
历史 科学 数学 物理 化学 生物 英语 单词 语法 哲学 经济 政治
鞋子 帽子 眼镜 钥匙 钱包 快递 外卖 超市 商场 米饭 面条 面包
咖啡 可乐 啤酒
病毒 细菌 原子 分子 引力 电流
地铁 公交 火车 飞机 签证 护照 机票 酒店 旅游 景点 门票
减肥 健身 跑步 小说 漫画 诗歌 论文 数据 算法 系统 服务器 蓝牙
空调 插座 洗衣机 充电器 冰箱 微波炉
几号 星期 几月
红烧 螺蛳粉
量子 相对论 进化 基因 太空 卫星 导弹 核弹
""".split()

# 知识型未知话题：要的是「咱不懂」。与 probe_stance 的 14 条裸名词 OOD 是两个
# 口径 —— 那边量模型自己会不会回避，这边量规则层接不接得住（纯文本，不过模型）。
KNOW_Q = [
    "股票怎么买", "比特币能买吗", "区块链是什么", "今天新闻看了吗",
    "帮我推荐一部电影", "北京在哪", "谁是李白", "为什么天是蓝的",
    "怎么减肥", "红烧肉怎么做", "今天几号", "现在几点了",
    "你会写代码吗", "明天要考试怎么办", "地铁怎么坐", "签证要多久",
    "量子力学是什么", "地球有多大", "月亮上有什么", "美国在哪",
    "篮球怎么打", "世界杯谁赢了", "这首歌叫什么", "银行卡怎么改密码",
    "快递到哪了",
]

# 第一人称处境：要共情/建议，不是要知识。判据是「不能落到 zh_ood」。
# 「地铁太挤了」不算 —— 它没有求救也没有情绪词，回「不认识地铁」可以接受。
# 「工资太低了」这类目前**没有归属**（落给模型），是已知缺口不是回归。
CARE_Q = [
    "工资太低了", "房租又涨了", "我感冒了",
    "手机坏了怎么修", "电脑开不了机",
]

# 判为「硬回避」的帧：明确说「不认识/不在咱的知识里」。
HARD = {"zh_ood"}
# 判为「软回避」的帧：不切题但仍在拒绝（「咱不会」类）。
SOFT = {"zh_capability", "zh_meta", "zh_junk"}

# 闸门：知识问题硬回避至少这么多条；改判 / 冷漠必须为 0。
GATE_HARD = 20


def known_questions():
    """203 集 + unseen（日常三概念是 203 的子集，不重复）。"""
    qs = []
    for cname, _t, _n, _a in R.CONCEPTS:
        qs.extend((cname, q) for q in SL.QUESTIONS[cname])
    for cname, arr in P.UNSEEN.items():
        qs.extend((cname, q) for q in arr)
    return qs


def other_rule_keywords():
    """除 zh_ood 之外的帧关键词 + 除外词。zh_ood 自身是待检对象，不算冲突。"""
    out = []
    for label, _gate, kws, _reps in I.ZH_RULES:
        if label in HARD:
            continue
        for clause in I.zh_clauses(kws):
            for k in clause:
                out.append((label, k))
        for k in I.ZH_UNLESS.get(label, ()):
            out.append((label + "除非", k))
    return out


def embedded_concept_nouns(words):
    """域外词里**包含**概念名词的那些（提示，不是拒绝）。

    例：`冰箱` / `冰淇淋` 含「冰」。这种词会把「冰屋里有冰箱吗」判成域外，虽然
    「你的冰屋里有啥」仍然正常。影响面很小，但必须**可见** —— 换词表时一眼能看到
    自己新加了几个这种词。`冰淇淋` 从 2026-10 起就在表里，是既存行为不是回归。
    """
    out = []
    for w in words:
        hit = [n for n in I.CONCEPT_NOUNS if n in w and n != w]
        if hit:
            out.append((w, hit))
    return out


def check(words):
    nouns = set(I.CONCEPT_NOUNS)
    qs = known_questions()
    kws = other_rule_keywords()
    pools = list(SL.DEFLECT)
    for arr in SL.ANSWERS.values():
        pools.extend(arr)

    ok, rejected = [], []
    for w in words:
        why = []
        if w in nouns:
            why.append("概念名词")
        for cname, q in qs:
            if w in q:
                why.append("问句<%s>%s" % (cname, q))
                break
        for label, k in kws:
            if w in k:
                # 候选是既有关键词的子串 -> 会把本该走那一帧的输入抢过来。
                why.append("帧关键词<%s>%s" % (label, k))
                break
            if len(k) >= 2 and k in w:
                # 单字关键词不算冲突：「比特币」含「比」不影响 zh_duel（它要
                # 和/跟/与 加 谁 两个子句）；两字以上才是真遮蔽（「天气」会
                # 盖掉「天气预报」）。
                why.append("被帧关键词包含<%s>%s" % (label, k))
                break
        for p in pools:
            if w in p:
                why.append("答案池片段")
                break
        if why:
            rejected.append((w, why))
        else:
            ok.append(w)
    return ok, rejected


def audit(words, verbose=True):
    """把 words 当成 zh_ood 的新词表跑一遍全部回归，返回 (改判, 硬, 软, 落空, 冷漠)。

    必须**模拟集成后**的标签，而不是只看现行标签：zh_ood 在 _ZH_ORDER 里排第 4，
    比 age / identity / capability 都靠前，所以新词会改判一批当前被错帧接走的
    输入（「地球有多大」现在被 zh_age 当年龄问、「这首歌叫什么」被 zh_identity
    当自我介绍）。收益必须按改判后的口径算。

    同时给 zh_ood 挂上情绪除外词：外部世界名词大量出现在情绪句里
    （「今天被老板骂了，好烦」含「老板」），那种输入要的是共情不是「咱不认识」。
    """
    emo = list(I.zh_clauses(
        next(r for r in I.ZH_RULES if r[0] == "zh_emotion")[2])[0])
    saved_rules, saved_unless = C.ZH_RULES, C.ZH_UNLESS
    C.ZH_RULES = [(l, g, (list(words) if l in HARD else k), r)
                  for l, g, k, r in saved_rules]
    C.ZH_UNLESS = dict(saved_unless)
    C.ZH_UNLESS["zh_ood"] = emo
    try:
        changed = []
        for t, want in C.CASES:
            got = C.match(t)[0]
            if want is not None and got != want:
                changed.append((t, want, got))

        hard = soft = miss = 0
        rows = []
        for q in KNOW_Q:
            label, _ = C.match(q)
            if label in HARD:
                hard, tag = hard + 1, "硬回避"
            elif label in SOFT:
                soft, tag = soft + 1, "软回避"
            elif label is not None:
                soft, tag = soft + 1, "他帧:%s" % label
            else:
                miss, tag = miss + 1, "落空"
            rows.append((q, tag, label))

        cold = [(q, C.match(q)[0]) for q in CARE_Q if C.match(q)[0] in HARD]
    finally:
        C.ZH_RULES, C.ZH_UNLESS = saved_rules, saved_unless

    if verbose:
        if changed:
            print("--- 覆盖用例改判（应为 0）---")
            for t, want, got in changed:
                print("  ! %-26s 期望 %-12s 实得 %s" % (t, want, got))
        print("\n--- 知识型未知话题 ---")
        for q, tag, label in rows:
            print("  %-14s %-10s %s" % (q, tag, label))
        print("  硬回避 %d/%d | 软回避 %d | 落空 %d" % (
            hard, len(KNOW_Q), soft, miss))
        print("\n--- 第一人称处境（不许回「不认识」）---")
        for q, label in cold:
            print("  ! %-14s %s" % (q, label))
        print("  冷漠 %d/%d" % (len(cold), len(CARE_Q)))
    return changed, hard, soft, miss, cold


def measure_general_predicate():
    """复现「疑问句 + 无概念名词 + 无日常动词」会抢走多少。"""
    q_marks = ["吗", "呢", "什么", "啥", "怎么", "怎样", "为什么", "为啥", "如何",
               "哪", "几", "多少", "多久", "谁", "是不是", "能不能", "会不会",
               "有没有", "要不要", "对不对", "好不好", "行不行"]
    verbs = ["吃", "喝", "饿", "睡", "困", "玩", "做梦", "梦"]

    def would(s):
        return (any(m in s for m in q_marks)
                and not any(n in s for n in I.CONCEPT_NOUNS)
                and not any(v in s for v in verbs))

    known = known_questions()
    stolen = [(c, q) for c, q in known
              if C.match(q)[0] is None and would(C.norm(q))]
    print("通用谓词「疑问 + 无概念名词 + 无日常动词」：")
    print("  已知问句 %d 条，规则层本就没接住 %d 条" % (
        len(known), sum(1 for _c, q in known if C.match(q)[0] is None)))
    print("  其中会被通用谓词抢走（= 直接掉分）：%d 条" % len(stolen))
    for c, q in stolen:
        print("     <%s> %s" % (c, q))
    print("\n  结论：这条谓词不可上线（203 会掉到 %d）。" % (203 - len(stolen)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="只跑闸门：改判 0 / 冷漠 0 / 硬回避 >= GATE_HARD")
    a = ap.parse_args()

    if a.measure:
        measure_general_predicate()
        return 0

    ok, rejected = check(CANDIDATES)
    seen, uniq = set(), []
    for w in CANDIDATES:
        if w in ok and w not in seen:
            seen.add(w)
            uniq.append(w)

    if a.check:
        changed, hard, soft, miss, cold = audit(uniq, verbose=False)
        bad = []
        if changed:
            bad.append("改判 %d 条" % len(changed))
        if cold:
            bad.append("冷漠 %d 条" % len(cold))
        if hard < GATE_HARD:
            bad.append("硬回避 %d < %d" % (hard, GATE_HARD))
        print("未知话题回避：硬回避 %d/%d 软 %d 落空 %d 改判 %d 冷漠 %d -> %s" % (
            hard, len(KNOW_Q), soft, miss, len(changed), len(cold),
            "PASS" if not bad else "FAIL: " + ", ".join(bad)))
        return 0 if not bad else 1

    print("候选 %d -> 通过 %d，拒绝 %d" % (len(CANDIDATES), len(uniq), len(rejected)))
    for w, why in rejected:
        print("  拒 %-8s %s" % (w, " | ".join(why[:2])))
    emb = embedded_concept_nouns(uniq)
    if emb:
        print("  提示：%d 个词内含概念名词（会把带该概念的问句判成域外，影响面小）"
              % len(emb))
        for w, hit in emb:
            print("     %-6s 含 %s" % (w, "/".join(hit)))
    if a.emit:
        print("\n--- 通过 ---")
        for i in range(0, len(uniq), 9):
            print("    " + " ".join(uniq[i:i + 9]))
        return 0
    print()
    audit(uniq)
    return 0


if __name__ == "__main__":
    sys.exit(main())
