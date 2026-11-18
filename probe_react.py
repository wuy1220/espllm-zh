"""验收「反应式」配方：模型到底学没学会抄对方的名词。

四个测试集，互相之间不能混着看，否则数字没法解释：

  A 域内新句式   23 个概念 × 2 句**训练里没有**的说法
                 期望：回声正确 + 答案落在该概念的 3 条池子里
  B 域外已见     训练过的陌生名词（熊猫/区块链…）× 新句式
                 期望：回声正确 + 答案落在 20 条回避池里
  C 域外未见     训练里**一次都没出现**的名词（运行时从候选表里筛）
                 期望：同上。这一组才是「接住任意输入」的真凭据
  D 无名词       "？" / "12345" / "嗯嗯嗯"
                 期望：不回 echo（没有名词可抄）

指标：
  回声正确   回复开头 1~4 字 == 该句里的名词
  回声错误   回复开头不是名词，但从输入里抄了别的片段（抄错位置）
  无回声     开头是语气词
  池归属     去掉回声后的主体是否落在期望池里

CLI:
    python probe_react.py --ckpt model/model_react.pt.best
    python probe_react.py --ckpt model/model_react.pt.best --n 4      # 每句采 4 次
"""
import argparse
import collections
import json
import os
import random
import sys

os.environ.setdefault("ESPGPT_CHAR_VOCAB", "persona/vocab_zh.json")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_long  # noqa: E402

STOP = set("\n")
VOCAB = os.path.join(HERE, "persona", "vocab_zh.json")

# A：域内新句式。刻意不用 gen_react_corpus.TEMPLATES 里的任何一条。
NEW_IN = ["{N}我从来没见过。", "{N}你怕不怕？", "昨天梦见{N}了。",
          "{N}，你说说看。", "我朋友说{N}很厉害。", "{N}是活的吗？",
          "{N}能不能借我？", "快看，那边有{N}！"]
# B/C 用同一批新句式（域外）
NEW_OOD = ["{N}我从来没见过。", "{N}你怕不怕？", "昨天梦见{N}了。",
           "{N}，你说说看。", "我朋友说{N}很厉害。", "{N}是活的吗？"]

# C 候选：词表内、且**没有**出现在 gen_react_corpus.OOD_NOUNS 里
HELD_CAND = """蜗牛 螳螂 壁虎 蚯蚓 水母 海星 云朵 冰雹 露水 泉水 石头 泥巴 沙子 树叶
花瓣 种子 树根 竹竿 藤条 稻草 棉花 羊毛 皮革 木头 铁钉 铜钱 玻璃 陶罐 石板 瓦片
蜂蜜 酸奶 果酱 咸菜 米粥 馒头 饺子 汤圆 粽子 月饼 冰棍 汽水 凉茶 药片 针管 绷带
创口 灯罩 电线 开关 插头 电池 遥控 手柄 耳机 音箱 话筒 天线 硬盘 鼠标 键盘 主机
屏幕 打印机 扫描仪 复印机 传真机 电话卡 信封 邮票 包裹 单据 发票 账本 合同 印章
抽屉 柜子 书桌 椅子 沙发 床垫 被子 窗帘 地毯 挂钟 台灯 花盆 鱼缸 鸟笼 狗窝 围栏
篱笆 台阶 门帘 信箱 水井 水沟 水管 水槽 水桶 盆子 漏斗 天平 砝码 卷尺 圆规 墨汁
毛笔 砚台 纸张 卡片 便签 标签 胶带 图钉 回形针 订书机""".split()


def load_spec():
    import gen_react_corpus as g
    return g


def build_sets(spec):
    vocab = set(json.load(open(VOCAB, encoding="utf-8"))["chars"])
    ood_trained = set(w for w in spec.OOD_NOUNS if all(c in vocab for c in w))
    held = [w for w in HELD_CAND if all(c in vocab for c in w) and w not in ood_trained]
    rng = random.Random(4242)
    rng.shuffle(held)

    A = []
    for cname, topic, nouns, _ in spec.CONCEPTS:
        for noun in nouns[:2]:
            t = rng.choice(NEW_IN)
            A.append((cname, noun, t.replace("{N}", noun)))
    B = [(n, n, NEW_OOD[i % len(NEW_OOD)].replace("{N}", n))
         for i, n in enumerate(sorted(ood_trained)[:16])]
    C = [(n, n, NEW_OOD[i % len(NEW_OOD)].replace("{N}", n))
         for i, n in enumerate(held[:16])]
    D = [("", "", u) for u in ["？", "12345", "嗯嗯嗯", "asdf", "。。。", "哈哈哈哈哈"]]
    return A, B, C, D, held


def classify(reply, noun, src, pool, attitude_set):
    """返回 (回声状态, 是否落池)。"""
    if not noun:
        echo = "无名词"
        rest = reply
    elif reply.startswith(noun):
        echo = "正确"
        rest = reply[len(noun):].lstrip("？啊，！嘛")
    else:
        # 从输入里抄了别的片段？取输入的前 2 字比对
        frag = src[:2]
        echo = "错误" if frag and reply.startswith(frag) else "无回声"
        rest = reply
        for lead in ("哼，", "哈，", "诶，", "喂，", "啊，", "唔，", "哼！"):
            if rest.startswith(lead):
                rest = rest[len(lead):]
                break
    hit = any(rest.startswith(a[:6]) or a[:6] in rest for a in pool)
    if attitude_set and not hit:
        hit = any(rest.startswith(a[:6]) or a[:6] in rest for a in attitude_set)
    return echo, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/model_react.pt.best")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--rep", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--show", type=int, default=10, help="每个集打印几条")
    a = ap.parse_args()

    spec = load_spec()
    A, B, C, D, held = build_sets(spec)
    attitudes = set(spec.ATTITUDES)
    # 池子必须和生成器**同源**：直接调它的 load_answers + 同一条去重规则，
    # 不要从 raw_react 反推（反推要去掉回声前缀，正则一含糊就把命中率算成 0）。
    _, ans_of = spec.load_answers()
    pool_of = {}
    for cname, topic, nouns, _ in spec.CONCEPTS:
        pool = spec.OVERRIDE.get(cname) or ans_of.get(topic) or []
        pool_of[cname] = list(dict.fromkeys(pool))

    gen_long.load(a.ckpt)
    print("checkpoint: %s   每句 %d 次采样   温度 %.2f\n" % (a.ckpt, a.n, a.temp))

    for tag, items, is_ood in (("A 域内新句式", A, False), ("B 域外已见", B, True),
                               ("C 域外未见", C, True), ("D 无名词", D, True)):
        echo_c, n, pool_hit, outs = collections.Counter(), 0, 0, []
        for cname, noun, u in items:
            pool = [] if is_ood else pool_of.get(cname, [])
            for i in range(a.n):
                r = reply(u, a, i)
                echo, hit = classify(r, noun, u, pool, attitudes if is_ood else None)
                echo_c[echo] += 1
                n += 1
                pool_hit += 1 if hit else 0
                outs.append((cname or "-", u, r, echo, hit))
        print("== %s  共 %d 句" % (tag, n))
        print("   回声 %s   池归属 %.0f%%   不同回复 %d 种" % (
            dict(echo_c), 100 * pool_hit / n, len({o[2] for o in outs})))
        for o in outs[: a.show]:
            print("   %-8s U:%s\n            B:%s   [%s%s]" % (
                o[0], o[1], o[2][:34], o[3], " 池" if o[4] else " 外"))
        print()
    print("C 组用的名词（训练中未出现）：%s" % " ".join(held[:16]))
    return 0


def rng_answers(spec):
    """从 raw_react.jsonl 反推每个概念实际用到的答案池。"""
    path = os.path.join(r"D:\无", "")
    path = r"E:\espllm_data\persona_corpus\raw_react.jsonl"
    pools = collections.defaultdict(set)
    if not os.path.exists(path):
        return dict(pools)
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        if r.get("src") == "S2" and not r["t"].startswith("_"):
            body = r["a"]
            for lead in ("哼，", "哈，", "诶，", "喂，", "啊，", "唔，", "哼！"):
                if body.startswith(lead):
                    body = body[len(lead):]
            pools[r["t"]].add(body)
    return {k: list(v) for k, v in pools.items()}


def reply(prompt, a, seed):
    _, out, _ = gen_long.run("User: %s\nBot:" % prompt, win=64, maxnew=40,
                             seed=seed, temp=a.temp, topk=a.topk, rep=a.rep, repwin=20)
    text = gen_long.dec(out)
    for i, ch in enumerate(text):
        if ch in STOP:
            return text[:i].strip()
    return text.strip()


if __name__ == "__main__":
    sys.exit(main())
