# HF 百万参数级小模型调研（2026-10-18）

起因：用户给出三个仓库 `User01110/CMA-1M-Mini`、`AxiomicLabs/GPT-S-1.4M`、
`BananaMind/Picolo-BananaMind`。查证后确认三者都真实存在，并且分属一个更大的社群。
本文件只记录与 ESP8266 项目决策相关的内容。

## 一、这个社群比想象的大

通过 HF API 的 `author=` 查询得到的完整名单（按仓库数）：

| 组织 | 仓库数 | 与本项目相关的 |
|---|---|---|
| AxiomicLabs | 10 | GPT-S-1.4M、GPT-S3-3M、GPT-S-5M、GPT-S2-5M、GPT-X3-150M |
| User01110 | 5 | CMA-1M-Mini、CMA-8M、tinyLM-8M-exp、Linear-50M-exp |
| BananaMind | 60+ | Picolo-BananaMind、TernaryBananaMind-10M、BananaMind-2-SLMoE、BananaMind-2-Nano |
| SupraLabs | 2 | MicroSupra-1k、MicroSupra-10k |

它们共用一套评测口径：`lm_eval` 归一化准确率 + AxiomicLabs 自制的 ArithMark
（2.0 / 3.0）+ "Open SLM Leaderboard" 风格四任务均值。

## 二、横向对照表

| 模型 | 参数 | 词表 | 层/宽 | 训练 token | token/参数 | 四任务均值 |
|---|---|---|---|---|---|---|
| Picolo-BananaMind | **100** | 20（4-bit 半字节） | 1 / 2 | 10M | 100,000 | 31.97 |
| MicroSupra-1k | 1,046 | 1024 | 1 / 1 | 900M | 860,000 | 未报（train loss 6.046） |
| CMA-1M-Mini | 958,692 | 260（字节） | 6 / 128 | 未披露 | — | 34.23 |
| **本项目 esp8266** | **1,169,000** | **2048** | **4 / 64** | **20.5M** | **17.5** | 未测 |
| GPT-S-1.4M | 1.43M | 4096 | 5 / 128 | 6B | 4,196 | 33.49 |
| TernaryBananaMind-10M | 8.43M | 2048 | 10 / 256 | 2B | 237 | 33.49 |
| BananaMind-2-Nano | 9.97M | 8192 | — | 30B | 3,009 | 35.77 |
| BananaMind-2-SLMoE | 25.4M（7.9M 激活） | 8192 | 8 / 256 | 60B | 2,360 | 34.91 |

## 三、结论一：这些基准在这个尺度上是噪声

四任务均值 = mean(ARC-Easy, ARC-Challenge, HellaSwag, PIQA)，都是四选一或二选一，
所以随机基线 = (25 + 25 + 25 + 50) / 4 = **31.25**。

Picolo-BananaMind 只有 **100 个参数**（1 层、隐藏维 2），实测 31.97。
用它做标定，就能算出每个模型"真正学到的部分"有多少：

| 模型 | 四任务均值 | 高出随机基线 |
|---|---|---|
| Picolo（100 参数） | 31.97 | **+0.72** |
| GPT-S-1.4M（1.43M） | 33.49 | **+2.24** |
| TernaryBananaMind-10M（8.43M） | 33.49 | **+2.24** |
| SLMoE（25.4M / 7.9M 激活） | 34.91 | **+3.66** |
| BananaMind-2-Nano（9.97M） | 35.77 | **+4.52** |

从 100 参数到 2540 万参数、从 1000 万 token 到 600 亿 token，全部跨度只有 **3.8 个点**。
而且两个完全不同规模的模型（1.43M 参数 6B token / 8.43M 参数 2B token）拿到了
**完全相同的 33.49**。

**这意味着：任何基于这套四任务均值的架构选择，都是在拟合一堆噪声。**
本项目不该、也不需要向这套口径靠拢。

`BananaMind/Overfitter-1.0` 是同一社群的另一个讽刺性对照：49.89M 参数，直接拿四个
软件工程 benchmark 的标准答案训练，SWE-bench Verified 拿到 100.0。作者在能力一栏
只写了 "Memorization"，限制一栏只写了 "It has no capabilities."。

## 四、结论二：外部独立验证了我们的 head/emb 决策

TernaryBananaMind-10M（8.43M，vocab 2048 —— 和我们的词表完全相同）的量化方案：

| 对象 | 精度 |
|---|---|
| 投影权重、输出头 | ternary {-1, 0, +1}，**5 值/字节** |
| 输入嵌入 | **int8，每行一个 scale** |
| 线性层激活 | 推理时量化到 8 bit |
| 归一化权重 | 浮点 |

即 **"投影+头 ternary、嵌入 int8"** —— 和我们上一轮独立得出的
`int8 emb + ternary head`（val 2.5676）是同一套方案。这是很有价值的旁证：
解绑 + ternary head 不是我们的孤例选择。

另一个数字：他们报告 **2.07 bits / 逻辑权重**（含 scales、浮点归一化权重和文件开销）。
这确认了 base-3 打包（5 值/字节）就是这一档的标准存储方式。

但他们同时明确写了一句话：

> "The packed weights are decoded to temporary tensors for matrix multiplication.
> 2.07 bits per weight describes checkpoint storage, not arithmetic precision
> or peak inference memory."

也就是说，**打包只省存储，计算前要先解包成临时张量**。这正好解释了为什么我们上次
把解包塞进内层点积循环（`base3_lut` 逐行查表）会掉到 978 ms/token —— 他们在 GPU 上
解包，带宽和 SIMD 都是免费的；我们把它放进了每 token 执行 2048 次的最热路径里。

## 五、结论三：我们的训练量是这批里最低的

| | token / 参数 |
|---|---|
| 本项目 | **17.5** |
| TernaryBananaMind-10M | 237 |
| SLMoE | 2,360 |
| GPT-S-1.4M | 4,196 |
| BananaMind-2-Nano | 3,009 |

我们只见过 **20.5M token**（10000 步 × 32 × 64），而语料有 **268M token**
（4,197,318 个窗口 × 64）—— **一个 epoch 的 7.6%**。

补到整整一个 epoch 需要 131,000 步。按实测 78.5 ms/步算是 **2.9 小时**，
正好把 token/参数推到 229，落在 TernaryBananaMind-10M 的 237 附近。

但要注意，我们自己的曲线不支持"多训就一定更好"：

```
iter   500 | val 3.1269
iter 10000 | val 3.0416        （最好 3.0214 @ iter 8000）
```

val 在 9500 步里只降了 0.085，且 train 3.04 ≈ val 3.04（零过拟合）。
训练学习率此时还有 1.04e-3（峰值的 52%），说明**不是 schedule 走完了学不动，
而是这个 lr 下的容量天花板**。所以"多训"要配上设置调整才有意义。

## 六、明确不适用的东西

- **字节级 / 半字节 tokenizer**（CMA 的 260 词表、Picolo 的 20 词表）。
  延迟按 token 计费。我们实测的输出是 **1.9 字符/token**（prof_hot1:
  84 字符 / 45 token），所以同样 100 个字符：BPE-2048 约 53 token，
  字节级约 100 token，半字节约 200 token。就算词表变小让每 token 从
  78 ms 降到 54 ms，**净延迟仍然是 2.4 倍和 4.8 倍**。直接否决。
- **CMA（Channel-Mixing Attention）**。8 个通道槽 × 16 通道、2 个路由头、
  90% 对角先验。它做的是跨隐藏维混合，而 MLP 已经在做这件事；
  CMA-1M-Mini 的 34.23 相对 GPT-S-1.4M 的 33.49 只差 0.74 点 —— 落在上面
  算出的噪声范围内，没有可测收益。
- **64 专家 top-13 MoE**。我们 20 专家 top-1 已经占 81% flash，
  44 专家是物理上限。
- **把词表做得更大**。按实测 1.9 字符/token 折算：词表翻到 4096 会让
  head 从 128 KB 变 256 KB，每 token 从 78 涨到约 105 ms（+34%）。
  要打平需要 chars/token 提升 1.34 倍，而 BPE 词表翻倍通常只给 1.3–1.4 倍，
  属于刚好打平、不构成理由。反方向缩到 1024 会让 token 数变多，同样不划算。
  **2048 就在最优区间附近，head 占 87% 流量是这个词表大小下的必然结果。**

## 七、两个值得试的新想法

### 1. 序列级路由（来自 SLMoE）

SLMoE 用**前 32 个有效 token 的因果前缀**算一次路由，整条消息复用 top-13 组合，
而不是每个 token 重算。我们目前每个 token 都跑一次 int8 router（1.61 ms）。

改成"生成第一个 token 时算一次、整段回复复用"：
- 直接收益：省 1.61 ms/token ≈ 整 token 2%（约 97 ms/条回复）
- 潜在收益：同一组专家权重在连续 60 个 token 里被反复读取，
  若 ESP8266 的 flash 映射区有可用 cache，专家权重可能驻留
- 风险：一条回复只用一个专家组合，表达力下降。我们 top-1、20 专家，
  这个风险比他们 top-13 更大，**必须先测再定**

### 2. 解包到栈上再点积（TernaryBananaMind 的做法在我们板上的对应）

不要在内层点积循环里解包。改成：
1. 从 flash 读 13 个字节（64 个 ternary）
2. 解到 `int8 row[64]`（栈上，64 字节）
3. 对 `row` 调用**现成的** `dot_emb_q_i8`

这样把 64 次 flash 读换成 13 次 flash 读 + 64 次 RAM 写 + 64 次 RAM 读，
点积内核一行不用改。预计 head 从 26.0 ms/token 降到 7~9 ms/token。

**但上次 base-3 曾掉到 978 ms/token，所以必须先做单行微基准**
（在板上循环 10 万次"解包 64 个 ternary + 点积"，测 cycles），
确认没问题再接线。不要再拿整机生成当调试器。

## 八、参考链接

- https://huggingface.co/User01110/CMA-1M-Mini
- https://huggingface.co/AxiomicLabs/GPT-S-1.4M
- https://huggingface.co/BananaMind/Picolo-BananaMind
- https://huggingface.co/BananaMind/TernaryBananaMind-10M
- https://huggingface.co/BananaMind/BananaMind-2-SLMoE
- https://huggingface.co/BananaMind/Overfitter-1.0
- https://huggingface.co/SupraLabs/MicroSupra-1k
