"""中文规则层的接住率测试：不烧板子就能看每条输入落到哪个帧。

为什么要有：规则层是唯一能保证"必须命中"的地方，而关键词表最容易出的错是
**两种都不报错**的错——该中的没中（漏词）、不该中的中了（词太泛）。烧一次板子
两分钟，改一次关键词十秒钟，所以要能在 PC 上先跑。

匹配语义与固件严格一致（见 main.cpp normalize_for_match / contains_word）：
  * 转小写，ASCII 标点变空格，连续空白压成一个，两端去空；
  * 子串匹配（汉字字节 >= 0x80，is_word_char 恒 false，词边界判断退化）；
  * max_bytes > 0 时按 UTF-8 字节数设闸（一个汉字 3 B）；
  * 帧顺序 = 优先级，首个命中即返回。

CLI:
    python coverage_zh_rules.py
    python coverage_zh_rules.py --show-all     # 每帧打印命中的输入
"""
import argparse
import collections
import random
import sys

from intent_rules import (ZH_RULES, ZH_UNLESS, CONCEPT_NOUNS, GATED_FRAMES,
                          zh_clauses)

ASCII_SEP = set(" \t\r\n?!.,;:'\"()")


def norm(s):
    out, prev_space = [], True
    for ch in s:
        if "A" <= ch <= "Z":
            ch = ch.lower()
        if ch in ASCII_SEP:
            if not prev_space:
                out.append(" ")
                prev_space = True
        else:
            out.append(ch)
            prev_space = False
    return "".join(out).strip()


def match(text):
    """返回 (label, reply) 或 (None, None)，语义与固件一致。"""
    s = norm(text)
    if not s:
        return None, None
    nb = len(s.encode("utf-8"))
    for label, gate, kws, reps in ZH_RULES:
        if gate < 0:                 # 负 gate = 至少这么多字节（长句兜底）
            if nb < -gate:
                continue
        elif gate and nb > gate:
            continue
        # 每个子句都要命中，子句内任一即可；clause 列表为空 = 只按长度兜。
        if not all(any(k in s for k in c) for c in zh_clauses(kws)):
            continue
        # 负关键词：出现即改判，让给后面的帧或模型（2026-10-21 起）。
        if any(k in s for k in ZH_UNLESS.get(label, ())):
            continue
        # 名词门：输入含已配置概念的名词 -> 名词类帧让位给模型（2026-10-22 起）。
        # 固件是 !zh_has_concept_noun(norm) && (...)，同一语义。
        if label in GATED_FRAMES and any(k in s for k in CONCEPT_NOUNS):
            continue
        return label, random.choice(reps)
    return None, None


# 每条都标注"应该落到哪个帧"。落空(None)意味着交给模型 —— 对罐头话题是对的，
# 对身份/互动这类是失败。
CASES = [
    # 截图里那五句
    ("你谁啊", "zh_identity"),
    ("笨", None),
    ("你认识大妖精吗", None),   # 名词门：含「大妖精」-> 交模型（实测答「她是敢管咱的那个」）
    ("你认识琪露诺吗", "zh_identity"),
    ("怎么又被灵梦揍了", None),     # 话题名词不归规则管
    # 极端输入：情绪发癫
    ("呜呜呜我好难过", "zh_emotion"),
    ("今天被老板骂了，好烦", "zh_emotion"),
    ("啊啊啊啊啊啊", "zh_silly"),
    ("哈哈哈哈哈哈", "zh_silly"),
    ("我裂开了", None),
    # 怪问题 / 梗
    ("如何烹饪大妖精", None),
    ("大妖精能吃吗", None),
    ("如果把雾之湖煮干会怎么样", "zh_listen"),  # 长怪问题：>=30 B 落倾听兜底帧，回复内容无关但接得住,
    ("⑨", None),
    ("你是笨蛋吗", None),
    # 截图里的三句：结构型输入
    ("如果三妖精同时落水你先救谁", "zh_trap"),
    ("我和你妈同时掉水里你先救谁", "zh_trap"),
    ("请你吃梦想封印", "zh_spell"),
    ("请你吃大妖精", None),
    # 名词当话题 vs 当宾语：同一个"大妖精"，两种帧
    ("你认识大妖精吗", None),   # 名词门（同上，重复用例保留以加固）
    ("大妖精比你稳重多了", None),
    ("拿什么喂大妖精", None),
    # 角色玩具的高频元问题 / 表白
    ("你是机器人吗", "zh_meta"),
    ("你喜欢我吗", "zh_affection"),
    ("我不喜欢你", "zh_affection"),
    ("灵梦说你的弹幕很好躲", "zh_listen"),  # 名词门放行符卡帧后落长句兜底
    # 无厘头假设：宽帧，回复必须与内容无关
    ("如果我是外星人", None),
    ("假设雾之湖一千年后干了", "zh_listen"),  # 长怪问题：>=30 B 落倾听兜底帧，回复内容无关但接得住,
    # 互动
    ("捏捏", "zh_touch"),
    ("抱抱", "zh_touch"),
    ("摸摸头", "zh_touch"),
    ("戳一下", "zh_touch"),
    # 能力 / 请求
    ("你会下围棋吗", "zh_capability"),
    ("教我魔法", None),           # 名词门：含「魔法」-> 交模型
    ("帮我写作业", "zh_capability"),
    ("一加一等于几", None),       # zh_math 的 unless：算数是已配置概念
    # 身份 / 设定
    ("你是谁", None),             # zh_identity 的 unless：自我介绍训练问法
    ("你叫什么名字", None),       # zh_identity 的 unless
    ("你几岁了", "zh_age"),
    ("你住在哪里", "zh_home"),
    ("雾之湖在哪", None),            # 话题名词，交给模型
    ("你和灵梦谁厉害", None),     # 自比问句放行给模型（比试概念 2026-10-22 已迁出，
                                  # 现由 zh_duel 的 unless 你/咱/比 让出）
    ("打得过魔理沙吗", None),     # 同上（_CMP_NO 含「打得过」）
    # 社交
    ("你好", None),               # 问候关键词删除：模型答「来啦！咱正等着你呢。」
    ("在吗", "zh_greet"),
    ("晚安", "zh_bye"),
    ("你好可爱", "zh_praise"),
    ("你是最强的吗", None),       # 名词门：含「最强」-> 最强 概念
    ("你好烦", "zh_insult"),
    ("谢谢", "zh_thanks"),
    ("今天好开心", "zh_happy"),
    # 应该交给模型的（短句罐头话题）；长句陈述走 zh_listen
    ("魔理沙昨天又输了", None),
    ("青蛙怎么不动了", None),
    ("夏天好热啊", None),
    ("冰屋是怎么盖的", None),
    ("我今天上班被老板骂了然后回家的路上还下雨了鞋全湿了", "zh_emotion"),
    ("今天去湖边走了走，风挺大的", "zh_listen"),
    # 边界：短词闸
    ("这个笨蛋节真热闹", None),
    ("火锅真好吃", None),          # 只有"吃"没有对象，不算投喂梗
    ("手机没电了", "zh_capability"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    hit = miss = wrong = 0
    by_frame = collections.Counter()
    print("%-4s %-26s %-14s %s" % ("", "输入", "帧", "回复"))
    for text, want in CASES:
        label, reply = match(text)
        by_frame[label or "-"] += 1
        if label == want:
            ok, hit = "OK", hit + 1
        elif want is None and label is not None:
            ok, wrong = "多中", wrong + 1
        elif want is not None and label is None:
            ok, miss = "漏", miss + 1
        else:
            ok, wrong = "错帧", wrong + 1
        if a.show_all or ok != "OK":
            print("%-4s %-26s %-14s %s" % (ok, text[:24], label or "-", (reply or "-")[:26]))
    total = len(CASES)
    print("\n命中 %d/%d（%.0f%%）  漏 %d  多中/错帧 %d" % (
        hit, total, 100.0 * hit / total, miss, wrong))
    print("分布：" + "  ".join("%s×%d" % (k, v) for k, v in by_frame.most_common()))
    # 每条回复的复用次数：同一帧内被问到不同的东西会换措辞，这里只看池子大小
    print("\n各帧回复池：")
    for label, _g, kws, reps in ZH_RULES:
        cls = zh_clauses(kws)
        print("  %-15s 子句 %d 关键词 %2d  回复 %d  首字去重 %d" % (
            label, len(cls), sum(len(c) for c in cls), len(reps),
            len({r[0] for r in reps})))
    return 0 if (miss == 0 and wrong == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
