# HF 微型模型的语料配方，以及我们能借到什么

调研对象：`Picolo-BananaMind`、`MicroSupra-1k`、`CMA-1M-Mini`、`GPT-S-1.4M`、
`TernaryBananaMind-10M`、`BananaMind-2-Nano`、`BananaMind-2-SLMoE`。
数据全部取自各自 model card 原文。

---

## 一、他们各用什么

| 模型 | 参数 | 词表 | 训练 token | 语料 |
|---|---|---|---|---|
| Picolo-BananaMind | 100 | 20（半字节） | 10M | **FineWeb-Edu** |
| MicroSupra-1k | 极小 | 1024 | 300M × 3 epoch | **FineWeb-Edu**（sample-10BT 前 3 亿 token） |
| CMA-1M-Mini | 958,692 | 260（纯字节） | 未披露 | 45% FineWeb-Edu 100BT + 25% DCLM-Baseline + 10% DCLM-Edu + 10% Cosmopedia v2 + 10% FineMath-4+ |
| GPT-S-1.4M | 1.43M | 4096 | 6B | 15% FineWeb-Edu + 30% Cosmopedia v2 + 20% FineWeb-HQ + 20% Ultra-FineWeb L3 QA + 15% Ultra-FineWeb L3 multi-style |
| TernaryBananaMind-10M | 8.43M | 2048 | 2B | **FineWeb-Edu**（单一来源） |
| BananaMind-2-Nano | 9.97M | 8192 | 30B | 55% FineWeb-Edu + 30% DCLM + 10% Cosmopedia v2 + 5% FineMath-4+ |
| BananaMind-2-SLMoE | 25.4M | 8192 | 60B | FineWeb-HQ + FineWeb-Edu + DCLM + Cosmopedia v2 + FineMath + NPSet2 |

## 二、共同配方

三点，七分之七一致：

1. **FineWeb-Edu 是唯一的地基。** 七个模型全部使用，其中三个只用它一个。
2. **Cosmopedia v2 是唯一的合成数据来源**（合成教科书风格，覆盖 4 个模型）。
3. **语料规模从 10M 到 60B token，跨度 6000 倍。**

**没有人用 TinyStories。** 这一点值得单独说：TinyStories 是专门为极小模型
设计的受限语言分布，而这批模型全部选择"用真实网页文本 + 用更多 token 硬扛"。
两条完全相反的路线。

---

## 三、直接把语料换过去要付多少：实测

用我们当前已上线的检查点（`model/model_mixed3_128.pt`，104 窗口固件同源），
同一模型、同样 128 token 窗口、纯交叉熵，只换文本：

| 语料 | 窗口数 | CE | 困惑度 |
|---|---|---|---|
| TinyStories（同分布） | 86 | **2.90** | 18.3 |
| FineWeb-Edu | 6 | **5.28** | 197 |
| Cosmopedia v2 | 7 | **6.09** | 441 |

`eval_corpus_loss.py` 输出。FineWeb/Cosmopedia 只有 6-7 个窗口（样本 2 KB），
数值会抖，但方向没有疑问：**差 2.4 nats，困惑度差 10.7 倍**。

换语料不是"多喂一点数据"，而是把模型放到一个它完全没有先验的分布上重新学。
对一个 1.17M 参数的模型，这意味着从头训练，而不是微调。

### 更硬的约束是算力，不是数据

这批语料的下载是免费的，把它们变成能力不是：

| 模型 | token / 参数 |
|---|---|
| 我们 | **17.5**（20.5M / 1.17M） |
| TernaryBananaMind-10M | 237 |
| BananaMind-2-Nano | 3,009 |
| SLMoE | 2,362 |

我们现在的训练速度约 0.8 s/步（32×128），即 5,120 token/s。
照这个速度追上 Nano 的 30B token 需要 **连续训练约 68 天**。
所以"参考他们的语料"在本机上的真实成本是两个月，不是几小时。

---

## 四、分词器不是障碍（这条和我预判相反）

我原以为 TinyStories 训的 2048 BPE 拿到网页文本上会严重碎片化。实测不是：

| 语料 | chars/token | 落回原始字节的比例 | 词表里被用到的 merge |
|---|---|---|---|
| TinyStories | 2.67 | 29.1% | 1108 / 1791（61.9%） |
| FineWeb-Edu | 2.52 | 25.8% | **290 / 1791（16.2%）** |
| Cosmopedia v2 | 2.68 | 24.6% | 327 / 1791（18.3%） |

`corpus_probe.py` 输出。两个结论：

- **压缩率几乎没退化**（2.52 vs 2.67，只差 6%）。因为英语的高频三字母组合在
  TinyStories 里也高频，256 个字节 id + 前几百条 merge 就能撑住通用英文。
- **但词表 84% 是死的。** 1500 条 merge 只在 TinyStories 的专有名词和句式上触发。
  这些 token 在 flash 里占着 128 KB 的 embedding 行，在网页文本上一辈子不会被选中。

→ 若真要走网页文本路线，正确做法是**按目标分布重训分词器**，顺便把词表缩到
1024，省下 64 KB flash（正好是 5 个专家）。而不是沿用现在的词表硬塞。

---

## 五、真正值得借的是"数据形状"，不是数据集

顺着这条线我去查了自己的 chat 语料，发现的不是"风格不对"，而是**生成目标被截断**。

`dataset_chat.txt` 头部写的是 `summary-conditioned records only; reply capped to fill the window`。
从 3000 个窗口解码头统计：

| 指标 | 结果 |
|---|---|
| User 长度中位数 | 14 词（6-20） |
| Bot 长度中位数 | **21 词**（12-28） |
| Bot 结尾没有句末标点 | **92.7%** |
| Bot 断在**词中间** | **76.8%** |
| Summary 的词也出现在回答里 | 中位数 38% |

也就是说，训练对是「摘要 → 故事的前 35 个 token，而且常常切在半个单词上」。
模型从来没被要求把一句话写完，它学到的是**把窗口填满**。

这解释了两件一直在困扰我们的现象：

- 回复经常半句结束（不是采样问题，是训练目标就长这样）
- 任何问题都以 "Once upon a time" 开头——因为语料里 100% 的 Bot 侧都是故事开头，
  而 User 侧是摘要（一种"条件"，不是"问题"）

另外语料构成：stories 1,356,722 窗口 + chat 199,995 窗口 **×10 复制** = 2,000,000。
chat 占 60% 的窗口，但唯一内容只有 20 万条。这一轮只训练了 9.8M token（语料的 1.7%），
复制还没真正咬人；一旦训练量上去，×10 会先变成记忆。

---

## 六、建议

### 立刻做（半天，收益最大）
**重建 chat 训练对**，不动任何模型结构：

1. 答案在**句末**截断，绝不切在词中间
2. 用满 104 窗口：prompt ~20 token + 完整答案 ~70 token
3. 把源文件里被丢掉的 `Words:` 字段用起来，做成第二类 prompt
   （"Write a story using these words: quit, oak, gloomy"），
   让模型见到"回答问题"和"讲故事"两种形态，而不是只有摘要→故事
4. 复制倍数从 ×10 降到 ×2-3

预期改善：回复完整性、不再一句 "Once upon a time" 打天下。成本约 30 分钟训练。

### 中期（可试）
按他们的**配方结构**（不是语料）做小比例混合：从 Ultra-FineWeb 的 L3 QA 切片
或 Cosmopedia v2 抽 5-10%，混进我们自己的语料。GPT-S-1.4M 用 20% QA 切片
做对话风格，这一点与我们当前最缺的能力直接对齐。

### 不建议
整体切到 FineWeb-Edu。三点叠加：CE 5.28 vs 2.90、需要 68 天算力、
词表 84% 的 merge 失效。这批模型能这么做，是因为他们的算力是我们的一百倍以上。
