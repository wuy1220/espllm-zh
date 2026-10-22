"""反应式人设语料：让模型学会「复述对方的话 + 表态」，而不是背 351 张问答表。

## 为什么换配方（实测证据）

`probe_zh_novel.py` 在 canon4 上只换一个名词：

    青蛙怎么不动了？ -> 那是咱冻的，等会儿放回去就活了！      （对）
    企鹅怎么不动了？ -> 冷才好呢，咱最喜欢冬天了！            （跑题）
    仙人掌怎么不动了？-> 星星就是四个角，咱天天看星星，错不了！ （胡说）

模型**从不复述对方说的词**，一遇到词表外的名词就退回边缘分布。根因不是
"训得不够"，是配方：351 个目标回复 × 15 字 ≈ 1.2 Mbit 要背，逼近 2.17M
三元权重的有效容量；模型于是走了最省力的路——按句尾 2 字查表（组内末 2
字熵 4.84 bit ≈ 几乎唯一），一换说法就落空。

## 新配方

回复 = **[回声]{对方刚说的名词}** + **[表态][本概念的答案 或 通用回避]**

- 回声是把输入里的字**照抄**到输出，所以对**任何**没见过的名词都成立，
  而且这是"我听见了"的唯一硬证据。信息量从"背句子"变成"抄 2~4 个字"。
- 目标句从 351 条降到 ~85 条（23 概念 × 3 答案 + 18 句通用回避），
  容量立刻宽松，剩下的容量留给"把名词认出来"。
- 域外名词拿不到本概念的答案，只能落进通用回避池 —— **这就是"不懂"的
  实现方式，不需要一条规则**。

输出的观感差异：同一个概念，因为回声不同，每条回复都长得不一样；
域外输入也能答得不重复。这是"多样性来自输入"而不是"来自更大的答案池"。

## 四路数据

    S1 自然句   现有 raw_canon 的 in-domain 行（保留真人语感），按需加回声
    S2 交叉法   23 概念 × 名词表 × 句式模板 → 教会「名词 → 该概念的答案」
    S3 域外     168 个域外名词 × 模板 → 教会「抄名词 + 回避」
    S4 无名词   数字/标点/重复字 → 教会「没有名词就不回声」

CLI:
    python gen_react_corpus.py --dry-run          # 只看统计，不写文件
    python gen_react_corpus.py                    # 写 raw_react.jsonl
"""
import argparse
import collections
import json
import os
import random
import statistics
import sys

DIR = r"E:\espllm_data\persona_corpus"
SRC = os.path.join(DIR, "raw_canon.jsonl")
OUT = os.path.join(DIR, "raw_react.jsonl")
VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona", "vocab_zh.json")

# (概念名, 取答案的 canon 话题, 名词表, 是否可当"活物"用动物句式)
CONCEPTS = [
    ("青蛙", "青蛙", ["青蛙", "大青蛙", "蝌蚪", "池塘", "荷叶", "呱呱叫"], True),
    ("冰", "冰与冷气", ["冰", "冰块", "冰柱", "冷气", "寒气", "冰面"], False),
    ("雾之湖", "雾之湖", ["雾之湖", "雾", "湖面", "湖水", "芦苇", "小船"], False),
    ("冰屋", "冰屋", ["冰屋", "冰墙", "屋顶", "冰床"], False),
    ("大妖精", "大妖精", ["大妖精", "大酱"], True),
    ("妖精同伴", "妖精同伴", ["桑妮", "露娜", "斯塔", "三妖精"], True),
    ("恶作剧", "恶作剧", ["恶作剧", "捉弄", "吓一跳"], False),
    ("算数", "算数", ["算数", "算术", "一加一", "三加五", "数数"], False),
    ("最强", "最强宣言", ["最强", "幻想乡第一", "无敌"], False),
    ("灵梦", "和灵梦比试", ["灵梦", "巫女", "御币"], True),
    ("魔理沙", "和魔理沙比试", ["魔理沙", "黑白", "扫帚", "魔法"], True),
    ("夏天", "夏天好热", ["夏天", "太阳", "热", "流汗"], False),
    ("冬天", "冬天最好", ["冬天", "雪", "结冰", "滑冰"], False),
    ("雪人", "雪人冰雕", ["雪人", "冰雕", "兔子", "形状"], False),
    ("鱼", "湖里的鱼", ["鱼", "人鱼", "若鹭姬", "水里"], True),
    ("符卡", "冰符弹幕", ["符卡", "弹幕", "冰符", "冰锥"], False),
    ("被小看", "被小看", ["个子", "矮", "小看"], False),
    ("迷路", "迷路", ["路", "方向", "导航", "森林"], False),
    ("红魔馆", "红魔馆", ["红魔馆", "吸血鬼", "咲夜", "大小姐"], False),
    ("⑨", "⑨这个外号", ["⑨", "笨蛋", "九"], False),
    ("自我介绍", "自我介绍", ["你", "名字"], False),
    ("喜好", "喜好", ["喜欢", "甜食", "礼物"], False),
    ("打招呼", "打招呼", ["你好", "早上好", "晚安"], False),
    # 言语行为概念（被挑衅 / 比试）2026-10-22 起**不在模型里**：它们的触发
    # 信号是句法（威胁 / 比较）而不是名词，模型的 QK 只能按名词路由，概念一
    # 多就整类塌掉（stance9/10 实测 1/6、0/6）。已迁到规则层
    # （intent_rules.py 的 zh_taunt / zh_duel），这里连条目一起删掉 ——
    # 留着条目会让 per-concept 探针继续给它们记分，等于对着不存在的槽位评分。
    # 日常生活三件套（截图事故补：无实体状态问句原来没有概念槽）。
    # 名词表刻意不含「吃/睡/玩」这些动词：答案签名就是「吃冰/睡在/玩起」，
    # 动词进名词表会让 answerable() 把全部配对当"前缀结巴"滤掉（误伤 ——
    # 「你吃什么？吃冰！」是接龙不是结巴）。
    ("吃饭", "吃饭", ["甜食", "肉", "午饭"], False),
    ("睡觉", "睡觉", ["做梦", "冰床"], False),
    ("玩耍", "玩耍", ["游戏", "雪人"], False),
]

# 伪名词的取字池：只要有「名词感」，不必是真词。
# 目的不是造词，是让「抄输入」成为唯一可行解 —— 表里查不到，就只能照抄。
PSEUDO_EXTRA = ("猫狗鸟鱼虫蛇马牛羊猪鸡鸭兔鼠象狮虎豹狼鹿鹤鹰燕雀蚁蜂蝶"
                "花草树木叶根果瓜豆米麦菜茶酒醋盐糖蜜奶油肉蛋"
                "石土沙泥水火风雷云雨雪冰霜雾露泉河湖海江溪池"
                "金银铜铁锡铅钢玉珠宝珠帘布丝绸线绳网篮筐箱柜桌椅床门窗墙屋顶"
                "车船桥路灯塔钟表镜灯烛香烟火炮刀剑枪弓箭盾旗鼓琴笛箫号铃")

# 域外名词：全部由词表内汉字组成（已用 build 时校验过滤）。
# 分词表之外还要「模型没见过这个词」，所以刻意避开幻想乡词汇。
OOD_NOUNS = """熊猫 大象 鲨鱼 螃蟹 章鱼 蜜蜂 蝴蝶 蚂蚁 蜘蛛 老鼠 兔子 狐狸 老虎 狮子 猴子 乌龟
蛇 恐龙 龙 独角兽 手机 电脑 电视 冰箱 洗衣机 空调 微波炉 电梯 楼梯 汽车 火车 飞机 火箭
卫星 灯泡 电话 相机 闹钟 手表 眼镜 雨伞 帽子 鞋子 书包 课本 奶茶 火锅 面包 蛋糕 米饭
面条 西瓜 苹果 香蕉 巧克力 冰淇淋 糖果 区块链 编程 数学 物理 化学 英语 股票 银行 快递
外卖 电影 游戏 健身 医院 医生 工资 驾照 红绿灯 地图 日历 温度计 无线网 公园 商场 超市
图书馆 学校 教室 黑板 操场 博物馆 动物园 摩天轮 过山车 天桥 高楼 地铁 星星 月亮 太阳
云 雨 雷 彩虹 台风 沙漠 大海 沙滩 山 森林 火山 地震 蜻蜓 猫头鹰 松鼠 贝壳 珍珠 钻石
项链 镜子 毛毯 枕头 蜡烛 风筝 气球 秋千 滑梯 沙包 玩具 拼图 钢琴 吉他 鼓 小提琴 麦克风
报纸 杂志 小说 字典 词典 算盘 尺子 橡皮 胶水 剪刀 针线 口袋 箱子 钥匙 门锁 窗子 屋顶
炉子 水壶 杯子 盘子 勺子 锅 碗 篮子 拖把 扫帚 肥皂 毛巾 牙刷 洗发水 消毒水 创可贴
打火机 手电筒""".split()

# 人工替换：canon 的 medoid 里有几条是坏样本（跟概念不搭、或三句几乎重复）。
# 不替换它们，模型会以 1/3 的概率学会一句胡说 —— 这正是板上「星星就是四个角」的来源。
OVERRIDE = {
    "雪人": ["化掉咱就再冻一个，反正咱有的是冰。",
             "咱雕的雪人最好看，比灵梦堆的那个强多了。",
             "三个角！咱数过，一个角都不少。"],
    "打招呼": ["来啦？咱正好无聊，陪咱玩会儿！",
               "早上好！咱早起来了，冻了一早上青蛙。",
               "晚安？咱不睡，咱还要玩！"],
    "⑨": ["不许叫咱⑨！咱有名字，咱叫琪露诺！",
           "⑨？咱不懂这个，反正咱最强！",
           "谁再叫咱⑨，咱就把他冻起来！"],
}

# 代词不该被回声（「你？废话，咱就是最强的琪露诺」读着像神经病）
NO_ECHO = {"你", "咱"}

# 通用回避池：不能涉及任何具体知识，否则套到陌生名词上就成胡说。
ATTITUDES = [
    "咱没听说过这玩意。",
    "咱不知道，咱只管冰和青蛙。",
    "这个咱不懂，不过肯定没咱厉害！",
    "管它是什么，先冻了再说！",
    "听起来就没什么用。",
    "咱不认识，你别问咱。",
    "那是什么？能吃吗？",
    "咱不知道，问这个干嘛！",
    "哼，反正没咱厉害。",
    "咱只懂冰，别的问大酱去。",
    "你说的咱一句都没听懂。",
    "什么东西？大酱没跟咱说过。",
    "咱不管这个，咱要去冻青蛙了。",
    "诶？这个咱真不知道。",
    "别拿这个考咱，咱可是最强的！",
    "咱不认得，不过咱能把它冻住。",
    "这个……咱数不清。",
    "咱没见过，看着挺好玩！",
    "咱没兴趣，说别的吧。",
    "那是啥？咱从来没听过。",
]

# 句式模板：名词位置刻意分散（句首 / 中段 / 末段），避免模型只用「首 2 字」认名词。
TEMPLATES = [
    "{N}是什么？", "{N}是啥？", "{N}是啥玩意？", "你见过{N}吗？", "{N}你见过吗？",
    "{N}厉害吗？", "{N}在哪？", "我想看看{N}。", "给我讲讲{N}。", "{N}，真的假的？",
    "为什么{N}会这样？", "{N}有几个？", "这是谁的{N}？", "{N}有多大？", "{N}冷不冷？",
    "我想摸摸{N}。", "那个{N}呢？", "喂，{N}在哪？", "{N}是不是你干的？", "你认识{N}吗？",
    "{N}好看吗？", "我讨厌{N}。", "我害怕{N}。", "话说，{N}是啥？", "什么是{N}？",
    "{N}是不是很厉害？", "{N}有什么用？", "这个{N}好奇怪。", "那个{N}是你的吗？",
    "{N}好不好？", "你说{N}？", "咱问你，{N}是啥？", "天上那个{N}是什么？",
]
TEMPLATES_ANIMAL = [
    "{N}怎么不动了？", "{N}会咬人吗？", "{N}好可爱啊。", "{N}能吃吗？", "{N}跑哪去了？",
    "那只{N}在干嘛？", "{N}叫得好大声。", "我抓到一只{N}。",
]
# 无名词输入：教「没有名词就不回声」
NOISE = ["？", "？？？", "。。。", "!!!", "12345", "11111", "666", "hhh", "asdf",
         "哈哈哈哈", "啊啊啊啊啊", "嗯", "呃", "咦", "哦", "……", "~~~~", "？？",
         "这个", "那个啊", "所以说", "然后", "你知道吗", "喂", "在吗", "嗯嗯"]

ECHO_FMT = ["%s？", "%s啊，", "%s！", "%s嘛，"]
# 开头语气词必须全部在字表内（「咦」不在，实测掉 UNK），所以只用高频字。
LEAD = ["哼，", "哈，", "诶，", "喂，", "啊，", "唔，", "哼！"]


def load_vocab():
    d = json.load(open(VOCAB, encoding="utf-8"))
    return set(d["chars"])


def load_answers():
    """从 canon 语料按话题取 top-3 答案（canon 里每个话题已按 (话题,kind) 压过）。"""
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    cnt = collections.defaultdict(collections.Counter)
    for r in rows:
        cnt[r["t"]][r["a"]] += 1
    return rows, {t: [a for a, _ in c.most_common(3)] for t, c in cnt.items()}


def build_reply(noun, ans, rng, echo_p=1.0):
    """[回声] + 表态。答案已含该名词时不重复回声（否则「青蛙？青蛙…」）。

    echo_p 默认 **1.0**：只要句子里有名可抄，回复就必须以它开头。
    第一版给 30% 的样本换成「哼，/哈，/诶，」开头，结果首 token 的信道被这
    7 个高频语气词堵死 —— 名词首字有几百种，摊到每种只剩千分之几，
    采样时永远轮不到，抄名词这条路径拿不到任何梯度（实测：域外 32 句
    回声 0，A 组 46 句里 33 句开头是语气词）。
    现在「有名必抄」，「没名才用语气词」，两者不再竞争同一个首 token。
    """
    if noun and noun not in NO_ECHO and ans.startswith(noun) is False:
        return (rng.choice(ECHO_FMT) % noun) + ans
    return ans if noun else rng.choice(LEAD) + ans


def fill(tmpl, noun):
    return tmpl.replace("{N}", noun)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--echo-p", type=float, default=1.0, help="有名必抄；设 <1 会重现首 token 竞争")
    ap.add_argument("--n-s1", type=int, default=4000, help="自然句上限")
    ap.add_argument("--n-ood", type=int, default=14, help="每个域外名词出几句")
    ap.add_argument("--n-pseudo", type=int, default=4200, help="随机伪名词句数（教「抄任意片段」）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    vocab = load_vocab()
    rows, ans_of = load_answers()

    # 域外名词先过一遍字表，缺字的直接丢掉（板上 UNK 会解码成空）
    ood = [w for w in OOD_NOUNS if all(c in vocab for c in w)]
    dropped = [w for w in OOD_NOUNS if w not in ood]
    if dropped:
        print("字表外名词已丢弃 %d 个：%s" % (len(dropped), " ".join(dropped)))

    out = []  # (来源, 概念, 输入, 回复, 是否回声)

    # ---- S2 交叉法：概念名词 × 模板 ----
    pool_of_topic = {}
    for cname, topic, nouns, animal in CONCEPTS:
        pool = OVERRIDE.get(cname) or ans_of.get(topic) or []
        pool = list(dict.fromkeys(pool))  # canon 里三句几乎重复的（⑨）去重
        pool_of_topic[topic] = pool
        if not pool:
            print("！话题缺答案，跳过：%s -> %s" % (cname, topic))
            continue
        tmpls = TEMPLATES + (TEMPLATES_ANIMAL if animal else [])
        for noun in nouns:
            for t in tmpls:
                u = fill(t, noun)
                ans = rng.choice(pool)
                out.append(("S2", cname, u, build_reply(noun, ans, rng, a.echo_p), True))

    # ---- S3 域外：抄名词 + 回避 ----
    for noun in ood:
        for t in rng.sample(TEMPLATES, min(6, len(TEMPLATES))):
            u = fill(t, noun)
            ans = rng.choice(ATTITUDES)
            out.append(("S3", "_ood", u, build_reply(noun, ans, rng, a.echo_p), True))

    # ---- S5 伪名词：堵死「背名词表」这条路 ----
    # 第一版（model_react）在训练原句上 10/12 抄对，但换成乌龟/书包/天平就抄成
    # 「狮咱也不知道」「扫帚啊」—— 它抄的不是输入，是**记住的那份名词表**。
    # 168 个域外名词 × 2~3 字 ≈ 3.6 kbit，背下来比学「抄任意片段」便宜太多。
    # 所以域外一侧必须换成**随机拼出来的伪名词**：表里没有，只能照抄。
    pool_chars = sorted(({c for w in ood for c in w}
                         | {c for w in PSEUDO_EXTRA for c in w}) & vocab)
    known_words = set(OOD_NOUNS) | {n for _, _, ns, _ in CONCEPTS for n in ns}
    made = 0
    while made < a.n_pseudo:
        k = rng.choice((2, 2, 3, 3, 4))
        noun = "".join(rng.choice(pool_chars) for _ in range(k))
        if noun in known_words or noun[0] in "的了是在和就都也又还":
            continue
        u = fill(rng.choice(TEMPLATES), noun)
        ans = rng.choice(ATTITUDES)
        out.append(("S5", "_pseudo", u, build_reply(noun, ans, rng, a.echo_p), True))
        made += 1

    # ---- S1 自然句：canon 里的 in-domain 行，能认出名词就加回声 ----
    # 只保留「答案落在该概念 3 条之内」的行 —— 自然语感来自输入侧，
    # 答案侧必须死守 69 条池子，否则 351 条目标回复会把容量吃光（这是上一轮的墙）。
    nat = [r for r in rows if r["kind"] != "ood"
           and r["a"] in (pool_of_topic.get(r["t"]) or [])]
    rng.shuffle(nat)
    nat = nat[: a.n_s1]
    noun_of_topic = {topic: nouns for _, topic, nouns, _ in CONCEPTS}
    used = 0
    for r in nat:
        nouns = noun_of_topic.get(r["t"], [])
        hit = [n for n in nouns if n in r["u"]]
        if hit:
            out.append(("S1", r["t"], r["u"], build_reply(max(hit, key=len), r["a"], rng, a.echo_p), True))
            used += 1
        else:
            out.append(("S1", r["t"], r["u"], r["a"], False))

    # ---- S4 无名词：随机拼噪声，避免固定短语被去重吃掉 ----
    noise_chars = "？。！…～啊嗯哦呃咦哈嘿喂在吗这个那个什么你说然后所以一二三六八十"
    # 量不能太小：伪名词把「回声」抬到 82% 之后，无名词样本是唯一压住它的东西。
    for _ in range(1600):
        n = rng.randint(1, 9)
        u = "".join(rng.choice(noise_chars) for _ in range(n))
        out.append(("S4", "_noise", u, rng.choice(LEAD) + rng.choice(ATTITUDES), False))

    # ---- 质检：字表 ----
    # 掉 UNK 的行**整条丢弃**：UNK 在板上编码成 <|endoftext|>，会把整句语义毁掉，
    # 留着等于教模型在句子中间出现结束符。
    bad_chars = collections.Counter()
    kept = []
    for item in out:
        miss = [ch for ch in (item[2] + item[3]) if ch not in vocab]
        if miss:
            bad_chars.update(miss)
            continue
        kept.append(item)
    out = kept
    if bad_chars:
        print("！字表外字符 %d 种（相关行已丢弃）：%s" % (
            len(bad_chars), " ".join(bad_chars)))

    # 输入去重（同一句只留一次，答案随机保留一条）
    seen, ded = set(), []
    for item in out:
        key = item[2]
        if key in seen:
            continue
        seen.add(key)
        ded.append(item)
    out = ded

    ul = [len(x[2]) for x in out]
    al = [len(x[3]) for x in out]
    src = collections.Counter(x[0] for x in out)
    echo_rate = sum(1 for x in out if x[4]) / len(out)
    ans_set = {x[3] for x in out}
    print("\n合计 %d 条" % len(out))
    print("  来源：%s" % dict(src))
    print("  输入长度 p50 %d / p90 %d / max %d" % (
        statistics.median(ul), sorted(ul)[int(len(ul) * .9)], max(ul)))
    print("  回复长度 p50 %d / p90 %d / max %d" % (
        statistics.median(al), sorted(al)[int(len(al) * .9)], max(al)))
    print("  带回声 %.0f%%" % (100 * echo_rate))
    # 真正要背的只有「核心答案池」；回声是抄输入，不占容量。
    # 把总字符串数当成容量指标会误判（同一答案配 168 个名词就有 168 种写法）。
    core = sum(len(v) for v in pool_of_topic.values()) + len(ATTITUDES)
    print("  核心答案池 %d 条（23 概念 × 3 + %d 句回避）" % (core, len(ATTITUDES)))
    print("  全部不同回复串 %d 条（= 答案池 × 不同回声，不是要背的量）" % len(ans_set))

    if a.dry_run:
        print("\n-- 样例 --")
        for x in out[:6] + out[-4:]:
            print("  [%s] U:%s\n       B:%s" % (x[0], x[2], x[3]))
        return 0

    with open(a.out, "w", encoding="utf-8") as f:
        for s, c, u, ans, _ in out:
            f.write(json.dumps({"src": s, "t": c, "u": u, "a": ans},
                               ensure_ascii=False) + "\n")
    print("\n已写 %s（%.1f KB）" % (a.out, os.path.getsize(a.out) / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
