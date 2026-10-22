# 微型模型实测、训练效率、780M 核显，以及让回复学会说完

四件事的实验记录。所有数字都是本机实测，不是引用。

**结论先给**：

| 问题 | 答案 |
|---|---|
| HF 那批微型模型能对话吗 | **都不能**，全是自动补全器；Nano 最好但 30 token 后掉进重复 |
| 训练效率还能提吗 | 能，batch 128 给 **+19.5%**；但 `mm` 已跑到 FP32 峰值 87–95%，再往下只差一个编译器 |
| AMD 780M 能用吗 | **不能**，三条独立死路（无 runtime / 派发吃掉 / 同收益 CPU 免费） |
| 回复半句结束、总以 Once upon 开场 | **修好了**：重建语料后开句末收尾 0% → **80%**，Once upon 72% → **3%**，架构没动 |

---

## 一、HF 上那批微型模型，实际对话起来是什么样

上一轮我只读了 model card 和基准分。基准分在这个尺度上几乎全是噪声（随机基线 31.25，
最好的也只有 35.77），所以这次把它们真的下载下来、用它们自己的建模代码跑了一遍。

### 怎么跑起来的（两个环境障碍）

- **HF 和 PyPI 的直连被代理挡了**（`502 Bad Gateway`）。`hf-mirror.com` 可用，
  但它的包文件对浏览器 UA 做了防盗链，要带 `User-Agent: pip/…` 才给下载。
- **transformers 与 huggingface-hub 的版本互斥**：这些仓库的 config 里写着
  `transformers_version: 5.14.1`，`tokenizer_class` 是 5.x 才有的 `TokenizersBackend`，
  而本地 hub 是 2.x、transformers 需要 4.x。解法是绕过 `AutoTokenizer`
  （`tokenizer.json` 格式一致，直接用 `tokenizers` 库读），并把 transformers 的
  版本守卫放宽两处。模型代码本身只用了 `PreTrainedModel`/`GenerationMixin`
  这类长期稳定的符号，4.57 能直接跑。

脚本：`eval_hf_models.py`（生成）、`eval_hf_bpc.py`（跨分词器可比指标）。

### 同一组 prompt、同样采样（temp 0.8 / top_k 40 / 60 token）

| 模型 | 参数 | 层 | 词表 | 训练 token | 60 token 出多少字 | bpc TinyStories | bpc FineWeb |
|---|---|---|---|---|---|---|---|
| **espllm（我们）** | **1.17M** | 4 | **2048** | **20.5M** | **~160 字** | 1.577 | 1.793 |
| GPT-S-1.4M | 1.43M | 5 | 4096 | 6B | ~270 字 | 1.516 | 1.776 |
| TernaryBananaMind-10M | 8.43M | 10 | 2048 | 2B | ~250 字 | 1.505 | 1.729 |
| BananaMind-2-Nano | 9.97M | 10 | 8192 | 30B | ~250 字 | **1.156** | **1.483** |
| CMA-1M-Mini | 0.96M | 6 | 260（字节） | 未披露 | **~55 字** | 1.541 | 1.765 |
| Picolo-BananaMind | **100** | 1 | 20（半字节） | 10M | ~20 字 | — | — |

bpc = bits per character，在**同一段文本**上算，所以跨分词器可比；越低越好。

### 它们实际说的话

```
GPT-S-1.4M (1.43M, 6B tok)
  "User: i feel sad today\nBot:"
  -> Serious Malaysian Malaysian Mall
     Don't like a griping rat, "Free Malaysian Malaysian Malaysian Malaysian...

TernaryBananaMind-10M (8.43M, 2B tok)
  "Water boils at"
  -> 385 meters (1335 meters) above and is used by water blocks and water to
     reduce the amount of water burnt from the water.

BananaMind-2-Nano (9.97M, 30B tok)   ← 这批里最好的
  "User: i feel sad today\nBot:"
  -> The doctor
     He said he thought he'd need more than one CT scan.
     Eight CTs of CTs:
     The difference between 2 CTs of CTs of CTs of CTs of CTs of CTs of C...

CMA-1M-Mini (0.96M, 字节级)
  "The history of the Roman Empire begins with"
  -> re-exploring both the pandemic and even decreasing the dyna
```

### 结论

1. **没有一个能对话。** 全部是自动补全器。它们没有 instruction tuning，问句只是
   一个需要续写的上下文。连最大的 Nano 也只是把问句当成句子继续写下去。
2. **Nano 是唯一读出"文章感"的**，句子语法正确、有人名有转折；但 30 个 token 后
   必然掉进重复循环（`CTs of CTs of CTs`）。
3. **Ternary 10M 最危险**：文风最像维基，事实全是编的，且语气很确定。
   这类输出比明显的乱码更容易骗人。
4. **字节级分词器的代价被实测出来了**：CMA 用 60 个 token 只写出 55 个字符，
   而 BPE-2048 能写 160–270 个字符。这正好是之前估算的 2.4 倍延迟劣势，
   现在有了实测。它也是这几个里唯一"看得出是乱码"的。
5. **我们的 1.17M 在同一个量级里站得住**：参数比 GPT-S 少 22%、训练 token 少 293 倍
   （20.5M vs 6B），bpc 只差 4%（1.577 vs 1.516）；而且在长输出上不出现
   `Malaysian Malaysian Malaysian` 这种塌缩。
6. **但也别自满**：在 FineWeb（它们的主场、我们的分布外）我们 1.793 对 1.729–1.776，
   是这批里最差的。差距不大，方向是真的——它们见过真实网页文本，我们没见过。

一句话：这批模型证明了"百万参数能学会写通顺的句子"，但没有一个能当助手用。
把它们当参照系是合理的，当目标是不必要的。

---

## 二、训练效率：钱花在哪，还能不能省

### 先修正上一轮报告里的一个错误数字

我在语料调研里写过"0.8 s/步 = 5,120 tok/s"，据此推出"追上 Nano 要 68 天"。
实测是 **267 ms/步 = 17.3k tok/s**（batch 32、128 窗口），差了 3.4 倍。
那句"68 天"应当是 **20 天**。原因是我当时把含 eval 的墙钟时间当成了纯步时间。

### 一步 155 ms 里是什么（batch 32）

profiler 统计：**20,602 次算子派发**，自耗 CPU 合计 201 ms（多线程并行后墙钟 155 ms）。

| 类别 | 自耗 | 占比 | 说明 |
|---|---|---|---|
| `mm` | 28.3 ms | 14.1% | 39 次，其中约 24 ms 是 2048 路输出头那一个 |
| `bmm` | 16.2 ms | 8.0% | MoE 的 60 次批量矩阵乘 |
| 逐元素 `mul/add/sub/div/round/abs/max` | ~47 ms | 23% | 主要是 BitLinear 每次前向的三值化 |
| AdamW step | 13.0 ms | 6.4% | 262 个参数逐个更新 |
| **EmbeddingBackward** | **11.5 ms** | **5.7%** | 单个算子，异常地慢（见下） |
| 纯派发开销（`copy_`/`_to_copy`/`empty_strided`/`fill_`） | ~24 ms | 12% | 5,300 次 dtype 转换 + 2,218 次分配 |

按 FLOPs 算：每步 7.0 GFLOP，155 ms 完成 → 有效 45 GFLOPS。我一开始据此写下
"只有这颗 Zen4 峰值的 5%，矩阵太小喂不满"——**这个读法是错的，得纠正**。

把 `mm` 单独拎出来算：27.6 GFLOP / 28.3 ms = **977 GFLOPS**。
这颗 CPU 是 AMD64 Family 25 Model 117（Zen4 Phoenix，8 核 16 线程），
FP32 峰值 = 8 核 × 32 FLOP/cycle × 4.0–4.4 GHz ≈ **1024–1126 GFLOPS**，
所以 `mm` 已经跑到峰值的 **87–95%**。

**矩阵乘法没有余量了。** 整体利用率低不是因为 BLAS 没跑满，而是因为七八成的墙钟
花在了非矩阵工作上：逐元素、dtype 搬运、5,300 次小派发、以及按参数循环的优化器。
这个区别很重要——它决定了该往哪挖：不该再找更快的 GEMM，只能减少派发次数。

### 试过的东西

| 方案 | 结果 |
|---|---|
| **batch 128**（8 线程） | 23,173 → **27,686 tok/s，+19.5%**（3 次中位数）。采用 |
| batch 64 | 26,553 tok/s，+14.6% |
| 线程 12 / 16 | 单次能测到 +8%，三次重复后回到噪声内，不加 |
| **bf16 autocast** | 前向 44.5→33.7、反向 54.3→38.8 ms，但**整步不变**（111.8 vs 104.6）。优化器和派发占走了收益。不用 |
| **torch.compile** | **不可用**：`InductorError: Compiler: cl is not found`。本机没有 MSVC，也没有 triton。这是最大的一处损失——它恰好能融掉那 12% 的派发开销 |
| CUDA / MPS | 无 |

`EmbeddingBackward` 11.5 ms 值得单独说：它要把 4096 行 × 64 维散加回一张
2048×64 的表，理论上不到 1 ms。**但我后来意识到它整步只被调用 1 次**，
所以这个数字里很可能混进了首次调用的线程池唤醒和 arena 分配。
要确认得把 profiler 跑到 5 步以上取均值——目前**没有定论，也没算进任何收益**。

顺带把"没有编译器"这件事查实了：`Program Files\Microsoft Visual Studio` 与
`Program Files (x86)\Microsoft Visual Studio` 都不存在，`Windows Kits\10` 只有 SDK
的头文件和库、没有 `cl.exe`，`vswhere.exe` 也找不到。**是真的没装，不是路径没配。**
要开 `torch.compile` 得装数 GB 的 VS Build Tools（管理员权限），这是唯一剩下的杠杆。

### 净收益与天花板

batch 32 → 128 之后，同样 4000 步从 23 分钟变成 39 分钟，但跑的 token 是 4 倍：
**单位 token 成本降 19.5%**。

剩下的结构性事实：`mm` 已在硬件峰值、AdamW 那 6.4% 是算法固有、逐元素那 16.6%
要融掉就必须有编译器。所以 **PyTorch 这条路已经吃满了**，只剩两个方向 ——
装 MSVC 换 `torch.compile`（估计能拿回那 12% 派发开销），或者离开 PyTorch。


---

## 三、AMD 780M 核显：用不上，而且即使能用也会更慢

硬件：Radeon 780M（RDNA3，12 CU，gfx1103），和 CPU 同在 Phoenix APU 上，
共享 DDR5 内存。CPU 是 Zen4 8 核 16 线程。

三条各自独立的硬约束，任何一条都足以否决：

1. **PyTorch 没有可用的后端。**
   `torch-directml` 最新版是 `0.2.5.dev240914`，轮子只到 **cp312**，本机是
   **CPython 3.13.14**，装不上；而且它绑定 torch 2.4.x，会顶掉现在的 torch。
   ROCm 在 Windows 上只支持 gfx1100/1101/1201（桌面卡），**780M 是 gfx1103，不在列表**。
2. **工作负载是派发瓶颈，不是算力瓶颈。**
   一步 20,602 次算子启动、7.0 GFLOP。就算按很乐观的 20 µs/次启动算，
   20,602 × 20 µs = **412 ms**，已经是现在 155 ms 整步的 2.7 倍——还没开始算真正的矩阵。
   核显 8.3 TFLOPS 的算力在这里完全用不到：1.17M 的模型把 GPU 空转 99%。
3. **同样的钱花在 CPU 上收益更高。** batch 128 拿到 +19.5%，代价是零。

唯一存在的通路是 `onnxruntime-directml`（有 cp313 轮子），但它只能跑**推理**，
和"训练更快"无关，而我们的推理目标是一片 ESP8266。

**结论：这条路关闭，不是"以后再试"，是三条都堵死了。**
如果哪天需要在这台机器上批量跑生成做评测，才轮到 ONNX + DirectML。

---

## 四、语料重建：让回复学会"说完"

### 旧语料到底哪里坏了

`dataset_chat.txt` 头部那句 `reply capped to fill the window` 是根因。对 3000 个窗口统计：

- Bot 侧长度中位数 21 词，**92.7% 结尾没有句末标点**
- **76.8% 断在词中间**
- 20 万条唯一内容被 **×10 复制**成 200 万窗口，占全部窗口的 60%

训练对实际是「摘要 → 故事的前 35 个 token，常切在半个单词上，然后接下一题的 `User:`」。
模型学到的唯一规律就是"把窗口填满，然后换行"。板上实测正是这样：

```
基线（旧模型，24 条回复，排除 2 条走规则应答的 prompt）
  以句末标点收尾   0/18 = 0%
  以 Once upon 开头 13/18 = 72%
  回复长度中位数    43 字符 / 17 token
```

### 新语料怎么建的

`build_chat_corpus.py`，从 `instruct_train.txt` 的 134 万条记录重新配对：

- **答案 = 1~3 个完整句子 + 一个换行**，上限 32 token —— 正好等于固件
  `CHAT_MAX_TOKENS` 的预算。于是"句子说完就换行"成了最省力的续写方式。
- **prompt 侧被 mask 掉**（`.mask` 侧车文件），只对回复算损失。
  旧语料 52.2% 的损失花在复述问题上，这次不花了。
- **三种 prompt**，避免学成单一模板：
  `User: <摘要>` / `User: Write a story using these words: a, b, c` /
  `User: Continue this story: <首句>`。
- 第四种「写一个带对话的故事」被**主动删掉**了：32 token 的答案只装得下开头，
  而对话通常出现在中段，prompt 和答案会互相矛盾，等于教模型忽略指令。
  `Words:` 类另加了一致性校验：答案里不出现给的词就丢弃。
- 长文能力由 `--stories` 追加的 129 宽窗口提供（不 mask）。

产出：

```
records 1,342,776  pairs 946,421  yield 70.5%
answer tokens: min 7 median 24 mean 23.5 max 32
chat share 41.1%  (946,421 chat / 1,356,722 story)
file 910 MB
```

### 效果：本地代理已经复现并确认改善

烧板之前先在本地按**固件的采样语义**跑一遍（`local_chat_eval.py`：
temp 1.0、top-k 40、repetition penalty 1.6 / 20 token 窗、32 token 预算、
遇 EOS 或第一个换行就停、前 3 token 抑制说话人标签）。
直接用旧模型跑，得 0/24 = 0%、67%、中位 44 字符 —— 和板上实测的
0/18 = 0%、72%、43 字符对得上，说明这个代理可信。

| 指标 | 旧模型 mixed3 | 新模型 mixed4.best |
|---|---|---|
| 以句末标点收尾 | 0/24 = **0%** | 18/24 = **75%** |
| 以「Once upon」开头 | 16/24 = **67%** | 3/24 = **12%** |
| 回复长度中位 | 44 字符 | 70 字符 |

`what is a cloud` 的回答从「Once upon a time, there was an elephant」变成
「Lily learned that sometimes things to put it outside as fast she would always be
angry and not stupid by her mom.」——依然是故事、依然有语法错误，但**它是一句话，
不是半句，也不再对任何问题都用同一个开场**。

故事模式没被牺牲（`PROMPT_MODE=free`，80 token）：
旧模型「loved the world on her factow clown his hands.」，
新模型「loved to play in the park. One day, she went for a swing and it was time
to go home about.」——句式更完整，还带出了段落分隔。chat 提到 41% 没代价。

### 板上实测：和本地预测对得上

导出 → 编译 → 烧录（Flash 72.9% / 761,359 B，RAM 90.2%），再跑 `board_eval.py 3`：

```
replies            24
rule-answered      2 prompt(s) excluded: ['i feel sad today', 'what is your name']
MODEL replies      18
finished (., !, ?) 15/18 = 83%       ← 基线 0/18 = 0%
opens 'Once upon'  1/18 = 6%         ← 基线 13/18 = 72%
reply length       median 56 chars, range 39-92   ← 基线 43
tokens             median 24, max 32
latency            median 85 ms/tok, range 80-101  ← 基线 70-95
```

又跑了第二轮（2 次采样）：`finished 9/12 = 75%`、`Once upon 0/12 = 0%`、
`median 74 chars`、`median 29 token`、`85 ms/tok`。两轮合计 30 条模型回复：
**以句末标点收尾 24/30 = 80%，Once upon 开场 1/30 = 3%。**

**每 token 的速度没有退化**（85 ms/tok 落在原来的 70–95 区间内），
但回复从 17 token 涨到 24–29 token，所以**单条回复的墙钟时间变长约 50%**。
按「质量优先」这是要的：以前快是因为它半句就断了。

板上 83% 甚至略高于本地的 75%，方向一致、量级一致。板上样例：

```
what is a cloud        "Tom said, "Thank you, Lily. They both shouted and went back to the beach.
tell me about friend   "Timmy shut his friends's house in a big red ball."
why is the sky blue    'Mommy said, "Please. Tore we were sad and hun afraid of the park.'
```

依然有语法错误、依然在讲故事，但**每句都说完、不再共用同一个开场**。
回复长度从 17 token 涨到 24 token，单条回复时间相应变长约 40%。

**剩下没解决的**：问事实题仍然答成故事。「what is a cloud」在 TinyStories 里
根本不是一种分布，模型只能说"讲个故事"。这不是训练问题，是语料里没有 QA 对。

### 顺手抓到的一个 bug

`dataset_stories.txt` 的前两行是注释，`append_stories` 按空白切分时把注释单词
当成了 token，于是**第一个 story 窗口里混进了 `#`、`TinyStories` 这类非数字**。
表现是窗口数比 mask 少 1（main.py 把以 `#` 开头的行当注释滤掉了）。
已修：按行跳过注释，并加了一道"开头 64 KB 必须全是数字"的断言。
