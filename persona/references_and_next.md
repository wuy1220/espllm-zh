# 同类项目能借鉴什么（2026-10-21 调研）

问题：**完美入戏的语言模型 + 仅必需规模的规则，而不是让规则层干大部分的活。**
调研了 5 个方向，每条都标了「我们能用/不能用、为什么」。

---

## 1 GuppyLM —— 和我们最像的成功案例（9M 参数，从零训练）

[github.com/tanwubin/guppylm](https://github.com/tanwubin/guppylm)

一只名叫 Guppy 的鱼，60K 条合成单轮对话、60 个话题、T4 训练 5 分钟。

| 它做的 | 数据 | 我们 |

|---|---|---|
| 人格烤进权重，**不用 system prompt** | 每个样本都用同一个 system prompt，9M 模型做不到条件遵循，索性去掉，还省 60 token | ✅ 已做 |
| **只做单轮** | 多轮在第 3~4 轮就崩（128 token 窗口）。原作者原话：鱼会忘事符合人设，但输出乱码不行 | ✅ 已做 |
| **模板组合造数据** | 60 话题 × 30 个缸内物件 × 17 种食物 × 25 个动作 → 约 16K 种不重复输出 | ⚠️ 用错了 |
| **vanilla transformer** | 刻意不用 GQA / SwiGLU / RoPE，"9M 参数下这些不 help" | ❌ 我们需要 RoPE（KV 滑动）和 MoE（换参数效率），这条不适用 |

**最有价值的一句**：
> at a small scale, instruction-following doesn't work. The model learns its personality
> from the statistical patterns in training data.

**我们踩的坑正好在「模板组合」这一栏**：它组合的是**语义合理的部件**（鱼缸 × 食物 × 动作），
我们组合的是**概念别名 × 33 条万能模板**，于是造出「路有几个？」「我想摸摸个子。」这种句子，
模型学到的是「怪输入 → 随便给一个答案」。实测概念路由 41%，误判里 32/70 条全塌到「冰」。
**结论：组合法成立，但部件必须是语义自洽的。** 这一条已经落成 `stance_lexicon.QUESTIONS`
（手写问法，每概念 6 条）。

---

## 2 Dogberry（ESP32-S3 上的 LSTM）—— 最好的产品思路

[ataary.com/?p=912](https://ataary.com?p=912/)

160 万参数词级 LSTM，跑在 ESP32-S3 上，训莎士比亚喜剧，**重点是 Dogberry 这个角色的声音**，
还给他开了 Bluesky 账号，真的能聊。

**它的核心洞察，比任何技术选择都值钱**：

> His character tends to mess up words, has a bit of over confidence, and a use of
> malapropisms. The hope was that these characteristics might lend themselves to a
> small language model like this **so that the errors or foibles of our model might
> feel more like a character mistake than a misunderstanding or limitation of the AI.**

翻译：**挑一个本来就该犯错的角色，模型的缺陷就自动变成角色特征。**

琪露诺（⑨）是这个思路的最佳人选 —— 笨蛋、数不清数、自大、记不住事。我们的
`DEFLECT` 回避池（"咱没听说过这玩意" / "这个……咱数不清"）已经是这个逻辑，
但还不够系统：**现在只是「答不上来时甩一句」，还没有「错得理直气壮」的层次感。**

可做的具体事（低成本）：
- 让模型答错时**不要犹豫**，越肯定越像她（现在的回避句已经偏这个方向）
- 数学题答错要错得有特色（"一加一？三！"已经是了）
- 数字类问题一律数不清 —— 这是她的标志，不是 bug

---

## 3 Fixed-Persona SLMs with Modular Memory（arxiv 2511.10277）

固定人设小模型 + 模块化记忆，在消费级硬件上跑游戏 NPC。

流水线：**10~20 条手写种子 → LoRA 微调一个中间模型当数据生成器 → 生成约 150 条 →
人工审 → 再生成更大一批 → 全量微调**。

记忆模块：对话记忆 + 世界知识记忆，检索 top-k 拼进 prompt。

**能用的**：那个「手写种子 → 合成 → 人工审」的循环。我们已经等价做过了
（canon 语料 → 模板 → 人工审 92 条台词本），但**没有做「人工审」这一步的系统化**——
现在的 `stance_lexicon.py` 就是审过的版本，这是对的。

**不能用的**：检索式记忆。64 token 窗口装不下 top-k 拼接。

**但可以降级实现**：固件层面记住「最近 1~2 轮说了什么」，用于
① 禁重复（已有禁首字，可以扩到禁答句）② 简单的追问承接。
这属于「必需规模的规则」，不是复杂规则层。

---

## 4 Roleplay AI 技术拆解（C.AI / Talkie 这类产品）

[yam.gift 的观察文章](https://yam.gift/2026/03/09/Product/2026-03-09-From-Cai-and-Talkie-to-Roleplay/)

> 简单来说，World 和 Story 是下限，用户偏好是上限。

数据合成：情境重演（NPC/User prompt 库）→ 两个模型对演 → 动态 planning prompt 控走向 →
reward model 多维打分过滤 → BoN 选最优 → 每 M 轮 LLM 评估改写。

多样性四轴：**场景 / Prompt / 风格 / 结构**。

**能用的**：那四个轴里，我们只有「答案池」一个轴。**结构多样性**（连续发言、独白、
追问、沉默）在 2.17M 上做不了，但**固件可以**——比如偶尔连说两句、偶尔先反问。

**不能用的**：reward model、BoN、多模型对演。全部超出算力。

---

## 5 量化与 induction head —— 「抄不下两个字」的科学解释

这是本次调研最重要的一条，它把我们的实测结果从「玄学」变成了「已知现象」。

**[does-quantization-kill-interpretability](https://github.com/lciric/does-quantization-kill-interpretability)**
（5 个模型，124M~2.8B）：

| 模型 | RTN 量化后 induction head 存活 | GPTQ |
|---|---|---|
| GPT-2 124M | **0 / 10** | 10 / 10 |
| Pythia-410M | 13 / 16 | 16 / 16 |
| GPT-2-XL 1.5B | 37 / 54 | 54 / 54 |
| Pythia-2.8B | 38 / 38 | 38 / 38 |

原文结论：
> For small models deployed on edge devices, GPTQ or equivalent calibrated methods are
> essential **not just for performance but for interpretability preservation.**

我们：**2.17M 参数、比 GPT-2 小 57 倍、ternary（1.58 bit）**。
按这张表外推，induction head 存活的概率是 0 —— 与实测「前 2 字命中 0/16」完全一致。

**[Linger and Lose: Knowledge Collapse in Low-Bit Language Models](https://arxiv.org/abs/2609.32902)**
（2.5M~50M，五种精度）：

- ternary 模型的**知识容量只有 fp16 的 6%**，而困惑度只升 1.4~1.6 倍 —— 损失被指标掩盖了
- 崩塌机制是**学习率驻留不稳定**：LR 停在 1~2×10⁻⁴ 不衰减，几百次曝光内就塌，
  而且**调回安全 LR 也救不回来**
- 崩塌**定位在输出头**（那里的权重无界增长）
- 修法：warmup-stable-decay 调度 + 10% cooldown → 容量 ×2.7（25M）/ ×4.3（50M）；
  **单独降低输出头的 LR** 也能防止崩塌

**这条给我们的三个具体动作**（按性价比排序）：

1. **输出头单独降 LR**（lm_head）。我们现在全参数同一个 LR（8e-4），输出头很可能就在
   论文描述的危险区。改动很小，可能直接抬升知识容量（= 概念路由准确率）。
2. **QK 投影提精度**。现在 `qkv` 是一个 `BitLinear`，全是 ternary。
   把 q/k 那部分改成 int8：每层多 `64×64` 个参数从 1.58 bit 抬到 8 bit
   ≈ +3.2 KB/层 × 5 层 = **+16 KB flash**，预算剩 53 KB，装得下。
   这是**唯一有可能让模型自己学会抄名词**的路 —— 但工程量大（训练/导出/固件三处都要改）。
3. **LR 调度改 WSD**。现在是 cosine，论文说 WSD + cooldown 更好。改动最小，收益不确定。

---

## 汇总：下一步该做什么

| 优先级 | 动作 | 依据 | 成本 | 期望 |
|---|---|---|---|---|
| P0 | 固件回声（已完成） | 实测抄不动 = 架构硬边界 | 已做 | 高：绕过硬边界 |
| P0 | 语料换成人话问法（已完成） | GuppyLM 的组合法 + 我们 41% 的实测 | 已做 | 高：路由准确率 |
| P1 | 输出头单独降 LR | Linger and Lose 论文 | 小 | 中高：知识容量 |
| P1 | `n_head` 2→4 | 无强依据，只是排除变量 | 中（40 min 重训） | 低 |
| P2 | QK 投影 int8 | 量化毁 induction head + 我们 0/16 | 大（3 处代码） | 高但不确定 |
| P2 | 缺陷角色化（Dogberry） | 产品思路 | 小 | 中：观感 |
| P3 | 固件层「最近两轮」记忆 | Fixed-Persona SLM 的降级版 | 中 | 中 |

**一句话结论**：让模型干大部分的活，**话术这一半已经做到了**（112 条台词本 + 20 条回避，
全部进权重）；**路由这一半受限于 ternary + 2.17M，短期只能靠语料质量逼近上限**。
真正可能突破的是 QK 精度，那是一个明确但昂贵的实验。
