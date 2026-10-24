# espllm 项目现状：所有已证实的事实与待决事项

截至 2026-10-19 15:40。所有数字都是本机/本板实测，不是引用。
分三部分：**已定论**（不要再查）、**待决**（需要一次决策或一次实验）、**新发现**（本轮改变判断的）。

---

## 一、硬件边界：已定论

### flash
- 颗粒实测 **4 MB**（厂家 5e / 器件 4016）。但 CPU 可直接寻址的窗口只有
  **1,044,464 B**（`irom0_0_seg` 0x40201010 → 0x40300000），
  `static const PROGMEM` 只能放这里，换任何 ldscript 都不会变大。
- 已切到 `eagle.flash.4m3m.ld`：多出 **3,121,152 B** 的 LittleFS @0x40300000，
  代码区/EEPROM/rfcal 偏移与 4m.ld 完全相同 → 实测切换后
  **Flash 761,359 / RAM 73,912 逐字节不变**，已烧录验证启动正常。
  旧配置在 `platformio.ini.bak-4m`。
- 3 MB 只能经 LittleFS / `ESP.flashRead` 访问，不能当 PROGMEM 指针。
- 1.6 KB（=一个专家的权重）随机读实测：mmap **112 µs** / 裸 flashRead **198 µs** /
  LittleFS 常驻句柄 **435 µs** / 每次 open+close **783 µs**。
  顺序读 4.5 MB/s，**写只有 54 KB/s**。
- 整 token 模拟（每层拉一个专家）：4 层 **1.74 ms = 85 ms 的 2.0%**。
- `flash_mode = qio` 已启用：lm_head 桶 −21%、整 token −8%。

### RAM（本轮修正，重要）
`ARENA_NEED = 11168 + 7072 × N_LAYER`（INFER_CTX=104，n_kv_head=1，head_dim=32）。
arena 40,960 B：

| 层数 | 需求 | 结果 |
|---|---|---|
| 4L | 39,456 | OK（现役） |
| 5L | 46,528 | **OVER** |
| 6L | 53,600 | **OVER** |

把 6L/26E 权重换进去编译，`static_assert(ARENA_NEED <= ARENA_SIZE)` 当场失败
（`src/main.cpp:130`）。**之前"6 层塞得下"的记录是错的**（只数了 KV，漏了固定部分）。
→ **层数硬顶 4 层**。要加层只有两条路：把 INFER_CTX 降到 ≤73（5L），
或把 logits 缓冲 8,192 B 改成分块算（省约 7 KB，也只够 5L，且要动采样热路径）。

### 延迟
延迟 ≈ **每 token 触到的权重字节数**（~3 MB/s 有效带宽），不是 MAC 数。
→ 加专家 ≈ 0 成本（每 token 仍只读 1 个），加宽/加深线性变慢，改位宽直接省时间。

---

## 二、架构：深度路线已封闭，只剩专家轴

固定验证集（100×32×64 token，同种子同批次，消除训练日志 ±0.05 噪声）：

| 臂 | 参数 | Flash | 验证 CE | 相对 A | 每 token 流量 |
|---|---|---|---|---|---|
| A 4L/20E（现役） | 1.17M | 68.0% | 2.1740 | — | 159,392 B |
| B 5L/32E | 2.17M | 90.9% | **2.0572** | **−0.117** | 170,432 B (1.07×) |
| C 4L/40E | 2.16M | 90.6% | 2.0976 | −0.076 | 162,560 B (1.02×) |

- **同样参数预算，花在深度上多赚 0.040 nats**（1.7 倍效率）。
- 训练日志会骗人：那里显示 C≈A（2.8808 vs 2.8788），据此会判"40 专家完全无用"，
  实际是 −0.076。**以后架构对比一律用 `eval_arch_ab.py`，不看训练日志。**
- 但深度已封顶（见上），所以**剩余容量轴只有专家**。
- 专家轴的 flash 上限：4L/40E 实测 **95.5%**（997,039/1,044,464，真实编译测得，
  与 `arch_ab`+代码区 383,087 B 的预测一致）；4L/43E≈99.5%；4L/44E 装不下。
- 突破专家上限要靠 3 MB FS 放专家（每 token 仍只读 1 个，延迟不变，
  增量约 +1.3 ms/token = 1.5%）。

---

## 三、语料：定稿 mixed4，不再找对话数据

### mixed4（现役，910 MB / 230 万窗口）
- 答案按句末截断、cap 32 = 固件 `CHAT_MAX_TOKENS`、`.mask` 只训回复 token
  （prompt 掩 −100，`y[prompt_len-1]` 保留）、三种 prompt（summary / words /
  continue）、chat 占 41%、block 128。
- **效果**：chat ppl 22.39 → **7.69**；story 12.09 → **10.84**（两项都降，
  没有"用一个换另一个"）。板上：句末标点收尾 **0% → 80%**、
  "Once upon" 开场 **72% → 3%**、回复 17 → 24-29 token、延迟仍 85 ms/token。
- 总 val 停在 3.09-3.14 是**口径问题**：story 窗口占 60% 且绝对困惑度更高，
  把加权和抬住了。必须用 `eval_split.py` 看拆分。

### 外部对话语料：全部否决（实测，答案-only 口径，参照 control 2.396）
| 语料 | 能装进窗口 | 答案 CE | 判断 |
|---|---|---|---|
| 抽取式问答（who/what → 故事第一句） | 100% | 2.643 | +0.24，唯一在否决线内 |
| **自有 Dialogue 子集**（引号对话，"上一句当 User / 下一句当 Bot"） | 91-100% | **3.144 / 3.238 / 3.385** | +0.76~+1.00，**超线，否决** |
| summary 当答案 | 100% | 3.512 | 答案 40 token，超 32 预算 |
| SODA（质量最好的一批社交对话） | **0.3%** | 4.230 | 结构性不匹配 |

- SODA 的答案长度其实完全合适（中位 19 token，76% 在 32 以内），
  但 **prompt 中位 253 token**，只有 0.3% 能装进 96 token 的 prompt 预算；
  强行截到"最后三句"再测 CE 仍 4.23。→ smoltalk/OASST/PersonaChat 同类，不用再评估。
- 自有 Dialogue 子集：给 prompt 加上下文（1 句叙述 → +0.09；summary → +0.24）
  **反而更差**，说明问题不是缺上下文，是引号对白的语体本身和 Bot 回复不同分布。
- **"让模型产出给定词表"不可学**：CE 4.66、ppl 106，信息论上没信息。
- **事实题答成故事是语料里没有 QA 对**，不是训练问题；117 万参数装不下世界知识
  （FineWeb 5.20 / Cosmopedia 6.09）。事实题只能靠扩意图规则。

---

## 四、训练：本轮发现一个比 batch 调参大得多的瓶颈

### 已固化的优化
- **batch 32→128 = +19.5%**（23,173 → 27,686 tok/s，三次中位数）。
- MoE 批量化 1.94×、dropout 关掉。
- **线程数：真实循环 8 最优，bench 说 12（两者相反，以真实循环为准）**。
  同样 4L/40E + mixed4 + batch128×block128，iter0→500 的 500 步：
  12 线程 **313 s（0.626 s/step）**、8 线程 **282 s（0.564 s/step）** → 8 快 10%。
  bench（`bench_moe_speed.py`，只含 fwd/bwd/opt）给的是 12 比 8 快 6%——
  差异应该在数据加载与线程同步上，纯算子基准看不见这块。
  另外：`main.py` 此前**不读 `THREADS`**，所以历史测量实际都跑在 torch 默认 8 线程；
  现已接上（`THREADS` 环境变量生效），默认未设时仍是 8。
- 20k 步按 0.564 s/step ≈ **3.1 h**（比最初 1.02 s/step 的 5.7 h 少一半）。
- `aten::mm` 已到 **977 GFLOPS = Zen4 FP32 峰值 87-95%**，矩阵没余量。
- `torch.compile` 不可用：本机**没有 MSVC**（只有 Windows SDK，无 `cl.exe`）。
- **AMD 780M 三条死路**：torch-directml 只到 cp312（本机 3.13）；ROCm on Windows
  不支持 gfx1103；派发开销 412 ms 已是整步 2.7 倍。

### 真实语料比随机 token 慢 80% —— 已定位并修复（2.1×）

基准脚本喂随机 token：**0.57 s/step**；原始真实训练循环：**1.02 s/step**。
差的 0.45 s 全部来自 MoE 的**路由倾斜**，而这条只有真实数据才暴露：

| 每层桶占用 | 真实语料 | 随机 token |
|---|---|---|
| 每专家平均 | 409.6 | 409.6 |
| 最满专家 | 3170 / 4115 / 4697 / 3354 | 695-835 |
| **padding 倍数** | **7.74 / 10.05 / 11.47 / 8.19×** | 1.70-2.04× |

旧实现把每个专家的桶都补到**最满专家**的长度，于是批量 GEMM 白算 8-11 倍；
散回 token 缓冲又是 `index_put` 到**重复索引**（padding 槽全指向 token 0），
反向要在 16 万行上做累加。实测 `_index_put_impl_` **253 ms**、`bmm` 183 ms。

修复：**定宽块分桶**（`MOE_BLOCK=128`）。token 按专家排序后，每个专家每 128 个
切一块；因为 `(a + j//cap)*cap + j%cap == a*cap + j`，槽位是仿射函数，不需要除法；
每个槽都是不同 token，所以进出都是无重复索引的 `index_select`。
浪费从「跟最忙专家成正比」变成「每专家最多一块」。

| | 旧 | 新 |
|---|---|---|
| `_index_put_impl_` | 253 ms | **0.3 ms** |
| `bmm` | 183 ms | **50 ms**（≈100 GFLOPS，已接近实用峰值） |
| 步时（bench, 8 线程） | 1276 ms | **473 ms（2.70×）** |
| 步时（真实循环, 8 线程） | 1.02 s | **0.564 s（1.81×）** |
| 步时（真实循环, 12 线程） | 1.02 s | 0.626 s（1.63×） |

早先记的「真实循环 0.48 s」是 60 步短测的乐观值（无 eval、缓存全热）；
跑满 500 步的稳态是 **0.564 s**，以这个为准。

数值等价已验：前向 **max|diff| = 0.000e+00**（逐位相同），
梯度最差相对差 **6.6e-07**（`verify_moe_bucket.py`）。
块宽扫描 64/96/128/192/256 → 128 最优（473 ms），批大小 128 与 256 吞吐持平（36.3k tok/s）。
`MOE_BLOCK=0` 可退回旧路径做 A/B。

**剩下的都不在 MoE 了**：`aten::mm` 92 ms（稠密层，≈210 GFLOPS，吃满）、
bmm 50 ms（≈100 GFLOPS）、量化逐元素 ~120 ms。
下一个结构杠杆是**批次里 24.2% 是填充 token**（短窗口用 token 0 补到 128，
长度中位 129 但 41% 窗口短于 128）。按长度分桶再裁 T 可再省约 25%，
但要改数据加载与 rope/mask 缓冲，风险高于收益，暂不做。

### 学习率调度
`CosineAnnealingLR(T_max=20000)` 硬编码。之前所有 run 都远没走到退火
（4000 步时 lr 还剩峰值 91.5%）。**跑满 20000 步是第一次真正退火**，
这本身就应该带来一截收益，和扩容的收益混在一起，需要 A/B 才能分开。

---

## 五、板上指标：4L/40E 跑完 20k 步，loss 更好但对话行为退步

`model_mixed5_128.pt`：4L/40E、mixed4、batch128×block128、**20k 步从零训**
（2h58m，8 线程 0.564 s/step）。最佳 **val 2.9497 @ iter19500**，lr 已退到 1.6e-5 —— 
第一次真正把 cosine 退火走完，**最后 3k 步一直在改善**（2.99 → 2.95）。

| | 4L/20E（mixed4, 4k, 热启动） | 4L/40E（mixed4, 20k, 从零） |
|---|---|---|
| chat CE / ppl | 2.09 / 7.69 | **1.86 / 6.41** |
| story CE / ppl | 2.39 / 10.84 | **2.20 / 9.00** |
| 板上句末收尾 | 24/30 = 80% | 23/30 = 77%（平手） |
| 板上 Once upon 开场 | 1/30 = 3% | **6/30 = 20%（退步）** |
| 回复长度 / token | 56-74 字符 / 24-29 | 71-74 字符 / 26 |
| 每 token 延迟 | 85 ms | 87 ms（+2%） |
| Flash / RAM | 68.0% | 95.5%（997,039 B）/ 90.2% |

**确定性证据**：采样 30 条有误差，所以直接量首 token 分布
（`probe_first_token.py`，不采样）。8 条 chat prompt 上首词是 "Once" 的概率：

| prompt | 20E | 40E |
|---|---|---|
| what is a cloud | 1.5% | 5.5% |
| what do you like to do | 1.3% | 11.7% |
| tell me about your friend | 3.9% | 8.3% |
| why is the sky blue | 1.3% | 12.5% |
| i am happy today | 19.2% | 18.8% |
| what do you want to eat | 2.3% | 7.0% |
| hello | 3.5% | 7.7% |
| how are you | 5.7% | 14.5% |
| **平均** | **4.82%** | **10.75%** |

退步是真的：40E 的故事开场倾向是 20E 的 **2.2 倍**。

### 口径结论：masked 答案 CE 不等于对话行为

chat 的验证答案本身就是 TinyStories 句子，**一个故事先验更强的模型在 chat CE 上反而得分更高**——
这个指标在奖励我们不喜欢的行为。以后判对话行为只认两样：
①板上 30 条采样统计；②`probe_first_token.py` 的首 token 概率（确定性、零噪声）。

机制猜测：40 个专家让 router 能把"故事续写"专门分给一批专家，chat prompt 是 OOD，
route 过去就被按故事续写。20 专家容量紧，专家内部被迫混用，反而不容易掉进去。

本地代理 `local_chat_eval.py`（复刻固件采样：temp 1.0 / top-k 40 / rep 1.6 /
32 token 预算 / 遇换行停）在旧模型上复现 0% / 67%，与板上 0% / 72% 对得上。
但 40E 上它给 83% / 8%，板上是 77% / 20% —— temp 1.0 采样方差大，
**判断行为差异要用 `probe_first_token.py`，不要靠 24 条采样对比。**

---

## 六、待决事项（按建议顺序）

1. **4L/40E 已训完并上板，但行为退步，需要选择**（见第五节）：
   loss 全面更好（chat ppl 7.69→6.41、story 10.84→9.00），flash 95.5% 可装，
   延迟只 +2%，但"Once upon 开场"从 3% 涨到 20%（首 token 概率 4.8%→10.8%）。
   三条路：
   - **A. 退回 4L/20E**：已知最好行为，头文件备份在
     `src/model_weights.hpp.bak-20261019-202530`，重编译 + 烧录约 2 分钟。
   - **B. 给 40E 补一段 chat 加权的短微调**（推荐，约 20–30 分钟）：
     从 `model_mixed5_128.pt.best` 续训，把 chat 窗口在采样里加权 3×、
     lr 从 2e-4 余弦到 0，跑 1500–2000 步。不动 val 切分（只改训练侧采样），
     所以指标仍可比。判据用 `probe_first_token.py` 的 P(Once) 降到 <5%。
   - **C. 重训 20k**（3 h）：只有在 B 无效、且确认是"从零训"而非"专家数"造成时才做。
2. ~~**MoE 重写是否先做**~~ —— **已完成**，2.1×（见第四节），数值逐位等价。
3. **3 MB FS 的用法要重排**：原计划的第一价值（支撑 10-12 层）已死，
   剩下的是 ① 突破 43 专家上限（FS 放专家，+1.5% 延迟）；
   ② 存多套权重免重烧；③ OTA 更新权重（400 KB 约 7 秒）。
   注意 flash 已用到 95.5%，**再加专家必须先把一部分搬进 FS**（只剩 47 KB）。
4. `EmbeddingBackward` 11.5 ms 未查因（只 1 次调用，可能含首次初始化）。
5. 事实题只能靠扩 `intent_rules.py`（20 条 → 更多），语料无解。

---

## 七、上下文能不能继续扩？（本轮新分析）

### 现状
104 位置 = arena 允许的 109，被 RoPE 表（128 行）和训练 block size 卡住。
每位置 272 B（4 层 × 68 B：K,V 各 32 值 int8 + 2 个 fp16 scale）。

### 三条扩路，逐条看过
| 路子 | 能到多少 | 代价 | 判断 |
|---|---|---|---|
| KV int8 → int4 | 约 200 位置 | 精度需实测 | **白存**：模型没在 >80 位置训练过 |
| 3 层换位置 | 约 146 位置 | 深度已证更值钱 | 不换 |
| **溢出到 3 MB FS** | 任意 | 写 272 B/token ≈ 5 ms（+6%）；读回 272·N B/token，N=104 约 6.3 ms（+7.4%），N=200 约 12 ms（+14%） | 技术上可行，仍受下面两条限制 |
| 堆里剩的 6.5 KB | +24 位置 | 无 | 边际 |

### 真正的瓶颈不是存储，是训练
- 实测 **超过约 80 个 key 就退化**：96-111 位置 CE 3.20 vs 64-79 的 2.48。
- 模型按 block 128 训练，RoPE 表只有 128 行，位置 >128 连表都没有。
- 所以：**不重训，存更多 KV 没有意义**。要真用上 256 位置，得按 block 256
  重建语料重训，而 256 位置的 KV 要 69,632 B，芯片上任何地方都放不下
  （除非 int4 KV + 3 层，或 FS 溢出 + 6% 写 + 7-14% 读）。

### 上下文压缩对我们有用吗？
- **已经做了最值钱的一种**：KV int8 + per-position fp16 scale = 相对 fp32 压缩 3.8×。
- 再往下压（int4 KV、token 驱逐、滑窗）买的是"能存更多"，
  而存储不是瓶颈 → **边际价值低**。
- 真正实用的"压缩"在 **prompt 侧**：chat 的 prompt 预算只有
  `INFER_CTX − MAX_GEN = 72` token，多轮对话必然要丢历史。
  现在的做法是直接砍 prompt 前半（等价于滑窗）。替代方案是只保留
  "首句 + 最近一轮"，属于固件策略，不改模型，可以零成本试。
- 结论：**KV 压缩对我们的边际价值低；该管的是 72 token 的 prompt 预算怎么花。**

---

## 八、"Once 开场"到底是不是缺陷：本轮把机制查清了（含一处自我更正）

第五节说"40E 行为退步、退步是真的"。本轮先用语料和同口径矩阵复核，结论要改一半。

### 8.1 先查语料：模型是不是在忠实复现数据？

`analyze_chat_openers2.py` / `analyze_once_by_kind.py`（chat 答案首 token）：

| chat 窗口类型 | 占 chat | 答案以 "Once" 开头的比例 |
|---|---|---|
| continue（`Continue this story: <首句>`） | 75.0% | **0.00%** |
| summary（整段剧情摘要） | 25.0% | **17.07%** |
| 合计 | 100% | 4.27% |

机制清楚了：`build_chat_corpus.py` 里 summary/words 两类的 `answer_sents = sents`，
即答案从故事**第一句**开始；而 TinyStories 的第一句有 17% 是 "Once upon a time"。
**这是"描述性 prompt → 故事开头"这个映射在语料里唯一的来源。**

### 8.2 再查口径：之前的 4.82% vs 10.75% 混入了 prompt 选择偏差

`probe_first_token.py` 的 8 条 prompt 里含 "hello" / "how are you"，这两条的
P(Once) 本来就低，把平均值拉下来了。换成按类型分组的
`probe_opener_matrix.py`（同一套确定性算法，P(O)×P(nce|O)）：

| 模型 | Q（问句，6 条） | TOPIC（题材，5 条） | CONT（续写，3 条） |
|---|---|---|---|
| 4L/20E | 4.91% | 4.07% | 0.04% |
| 4L/40E | 10.62% | 4.88% | 0.01% |
| 4L/40E + 微调 | 11.25% | 7.62% | 0.00% |

**关键更正：在模型的本职（TOPIC → 写故事）上，20E 和 40E 几乎一样（4.07% vs 4.88%）。**
所谓"退步"只发生在 Q 类（分布外问句）上，而且 20E 的 Q 值本身是双峰的——
6 条里 5 条 1–4%、"i am happy today" 19.2%；40E 是每条都 5–19%。这更像
"40E 的 router 把所有 OOD 输入都送进同一批专家"，而不是"故事先验整体变强"。

### 8.3 加权微调：CE 真赢，P(Once) 没动

`model_mixed6_ft.pt`：从 `model_mixed5_128.pt.best` 热启动，
chat 窗口采样 3×、summary 类再 ×0.5（实际分布 story 35.3% / continue 55.5% /
summary 9.2%），lr 2.5e-4 余弦到 1e-6，**1500 步 15m46s**，8 线程。

| | 4L/40E | +微调 |
|---|---|---|
| val（总） | 2.9450 | **2.9337** |
| chat CE / ppl | 1.8582 / 6.41 | **1.7861 / 5.97** |
| story CE / ppl | 2.1976 / 9.00 | **2.1591 / 8.66** |
| Q / TOPIC P(Once) | 10.62% / 4.88% | 11.25% / 7.62% |
| 板上收尾 | 15/18 = 83%¹ | 15/18 = 83% |
| 板上 Once 开场 | 6/30 = 20% | 1/18 = 6%（n 小，噪声 ±10pp） |

¹ 第一次板上验收（temp 1.0）就是微调后的模型。

**结论一：数据混合比例控制不了 OOD 路由。** 采样分布已经把 summary 从 10.3% 压到 9.2%、
continue 从 30.8% 提到 55.5%，P(Once) 反而 +0.6pp。原因是这几条 prompt 根本不在训练分布里，
"更多同类数据"不会让它们进入分布——**P(Once) 高其实是"模型不知道自己在分布外"的表现**，
只能靠 ①让短问句进入分布（意图语料的 2710 条 `dataset_intents.txt` 可以混 2–3%），
或 ②意图规则把它们拦住。这条路径本轮**未验证**。

**结论二：40E 20k 步其实没收敛。** lr 已退到 1e-5 时 val 仍在改善，再给一段低 lr 的
cosine 循环（1500 步）就能再拿 0.011 nats，且两个半边同时改善。
**"第二段 cosine 循环"本身是免费的收益，与扩容无关。**

### 8.4 真短板是生造词，温度是免费杠杆

板上回复最刺眼的问题不是开场，是 `ansh` / `hoome` / `pinals` / `rabb` 这类生造词。
新指标 **OOV 率**：用 TinyStories 验证集词表（11,745 词）统计回复里不在词表中的词占比。
温度扫描（`:temp <v>` 是运行时命令，不用重烧）：

| temp | 板上收尾 | 板上 Once | 板上 OOV | 本地 OOV（24 条） |
|---|---|---|---|---|
| 0.70 | 15/18 | 7/18 (39%) | **0.8%** | 2.1% |
| 0.85 | 13/18 | 2/18 (11%) | 1.9% | **0.9%** |
| 1.00 | 13/18 | 3/18 (17%) | **4.4%** | 1.8% |

OOV 随温度单调下降是数学必然（低温压长尾），不用统计就能定方向；0.7 的代价是
把 OOD 问句的故事开场率推到 39%（低温 = 选众数 = 选故事开场）。
**固件默认已改 0.85**（`src/main.cpp` ESP8266 分支，`:temp` 仍可运行时覆盖）。

### 8.5 板上路由实测：10 条验收 prompt 里 6 条真的走模型

`probe_rule_path.py`（按有没有 `[N ms total]` 脚注判定）：
sad / name 走规则，其余 8 条（含 "tell me a story about a robot"）走模型。
所以 `board_eval.py` 的 Q 类指标是**真的在量模型**，不是量规则。

### 8.6 本轮新增脚本

| 脚本 | 用途 |
|---|---|
| `analyze_chat_openers.py` / `_openers2.py` | chat 答案首 token / "Once" bigram 占比 |
| `analyze_once_by_kind.py` | 按 prompt 类型拆 P(Once)（continue 0% / summary 17%） |
| `dump_chat_samples.py` | 解码并打印 chat prompt→answer 原文 |
| `probe_opener_matrix.py` | 按 Q/TOPIC/CONT 三组量 P(Once)，替代 `probe_first_token.py` 做判据 |
| `probe_rule_path.py` | 判定哪条输入走规则、哪条走模型 |
| `sweep_temp.py` / `board_temp_sweep.py` | 本地/板上温度扫描 + OOV 率 |

**踩坑记录**：`python main.py --train > log` 全程日志 0 字节——stdout 重定向到文件时
是块缓冲，1500 步跑完才一次写出。要看进度必须加 `-u`，否则只能等进程退出。

---

## 九、把短问句塞进训练分布：机制成立，但没有能上线的配方

上一节留下的问题是「OOD 问句 P(Once)=11.25%，加权微调压不下去」。这一节把
`dataset_intents.txt`（2,710 条短问句 → 20 条固定回复）真正混进采样，结论是
**四轮配方全部否决，板子留在 `mixed6_ft`**。

### 9.1 先修的两个语料问题（都是静默出错的那类）

| 问题 | 事实 | 修法 |
|---|---|---|
| 序列化不一致 | `dataset_intents.txt` 的 prompt 以 `"Bot: "`（含尾空格）结束、mask 多算 1 token、答案以「空格+词」开头；`dataset_mixed4` 是 `"Bot:"` + 裸词 token | `fix_intent_format.py` → `dataset_intents_fmt.txt` |
| 裸题材在语料里不存在 | chat 语料只有 summary / 词表 / 故事首句三种 prompt，`a cat and a dog` 是纯 OOD | `build_topic_corpus.py`：从 "there was X" 开头抽主语短语（产出率 12.4%，去重 7,237 条），答案用故事自己的开篇 |

训练侧新增通用开关 `ESPGPT_EXTRA="file:share,file:share"`，份额按**窗口**算；
短窗口（~35 token）只占约 1/4 的 token，所以日志会同时打印 token 份额。

### 9.2 剂量-反应（全部从 `mixed6_ft` 热启动，chat 3× / summary 0.5×）

| 配方 | Q P(Once) | Q P(canned) | TOPIC P(Once) | TOPIC P(canned) | chat CE | story CE |
|---|---|---|---|---|---|---|
| 基线 `mixed6_ft` | 11.25% | 1.00% | 7.62% | 0.73% | 1.7861 | 2.1591 |
| 意图 6%，1,200 步 | **0.48%** | 49.97% | 4.06% | **32.23%** | 1.7852 | 2.1741 |
| 意图 5% + 题材 5%，800 步 | 7.61% | 39.34% | 57.46% | 8.59% | 1.7910 | 2.1731 |
| 意图 6% + 题材 8%，1,000 步 | 1.77% | 43.81% | 59.95% | 9.27% | — | — |
| 意图 2%，600 步 | **0.53%** | 23.24% | 4.31% | 10.22% | 1.8002 | 2.1714 |

`P(canned)` = 回复以 20 条固定回复之一开头的概率（首两 token 联合），确定性、零噪声。

### 9.3 数值好看但文本不行——三次读文本才发现

- **意图 6%**：题材被罐头回复污染。`the moon` → `Sorry you're feeling low. Emily had
  hurt his knee, so she ran away the firework` —— 罐头前缀接词沙拉。
- **加题材对照**：TOPIC P(Once) 冲到 57%，但**塌缩成同一句**。
  `a robot who wants a friend` / `a cat and a dog` / `the moon` /
  `a dragon who is afraid of fire` 全答 `Once upon a time, there was an elephant.`
  原因：题材语料只从 "there was" 开头抽取，答案模板单一，模型学会的是模板不是映射。
- **意图 2%**：最好的一档，不塌缩、题材文本与基线同水平，但题材仍有
  **10.2% 罐头泄漏**（基线 0.7%）。

### 9.4 决定

10.2% 的题材泄漏意味着「每 10 个写故事请求就有 1 个被回一句固定话」，而写故事是
主路径、问句是次路径。**不换模型**，板子刷回 `mixed6_ft`（`Flash 95.5% / RAM 90.2%`，
`:params` 已确认 temp=0.85 rules=1）。问句侧交给规则层——那里是确定性的，且已经覆盖
14 类意图。

### 9.5 两条方法学记录（比结论更重要）

1. **P(Once) 联合概率 ≠ 采样率**。板上 top-k 40 会重新归一化，尾部被放大，所以
   0.5% 的联合概率不等于 0.5% 的出现率。**方向可信，绝对值不可换算。**
2. **`board_eval.py` 的 "Once upon" 是 n=18 的混合口径**（±10pp），判不了 11% vs 17%。
   要判必须按 prompt 类分开、n≥60。它的 timing 正则是 `[N ms total] N tok`，
   新脚本照抄成 `[N ms, N tok]` 会把所有模型回复误判成规则命中。

### 9.6 新增脚本

| 脚本 | 用途 |
|---|---|
| `probe_intent_effect.py` | Q/TOPIC/CONT × P(Once)/P(canned) 四格，确定性判据 |
| `show_reply.py` | 复刻固件采样（temp/top-k/rep），直接读文本；`ESPGPT_PROMPTS` 覆盖用例 |
| `build_topic_corpus.py` | 从故事开头抽裸题材短语，答案用同一句 |
| `fix_intent_format.py` | 意图语料重编码到 `dataset_mixed4` 的序列化 |
| `board_q_check.py` | 单条约句在板上连发 N 次，看首词分布 |

---

## 十、故事质量：两个语料缺陷已定量并修复，但 Once 开场反而升高

用户要求「把精力放回故事质量，让句子更稳定通顺」。这一节先把「不通顺」拆成可测的
分量，再做修复。

### 10.1 三个客观指标（`eval_quality.py`）

以 TinyStories 验证集（11,745 词、410,718 bigram）为参照：

| 指标 | 含义 | 抓什么 |
|---|---|---|
| OOV% | 回复中不在参考词表的词占比 | 生造词（`ansh`/`hoome`/`distiedy`） |
| bigram% | 回复 bigram 出现在参考文本的比例 | 词沙拉（`the dog land started`） |
| uniq% | 30 条回复里不同首 6 词的比例 | 模板塌缩 |

### 10.2 重复惩罚是减法、且按出现次数累计（固件既有行为）

`src/main.cpp:460` 是 `g_logits[id] -= g_rep_penalty`（默认 1.6），对 20 token 窗口
**每个位置各减一次**。同一个 `the` 在窗口里出现 3 次就是 −4.8 logits，在 T=0.85 下
等于概率除以 **285 倍**。高频功能词因此被压到长尾，而长尾正是生造词的产地。

实测（旧模型 `mixed6_ft`，30 条样本）：

| 惩罚形式 | 完句% | OOV% | bigram% |
|---|---|---|---|
| `sub 1.6 w20`（现役） | 73 | 2.2 | 93.6 |
| `uniq 0.6 w20` | 93 | 0.4 | 96.4 |
| **关闭惩罚** | **97** | **0.5** | **97.6** |

**注意**：这是旧模型的结论。换成修复后的新模型后，`sub 1.6` 反而是最优（完句 87%、
OOV 1.1%），关闭惩罚掉到 73%。说明惩罚强度必须**逐模型重测**，不能沿用。

### 10.3 两个语料缺陷（946,421 个 chat 窗口全量统计）

`analyze_answer_shape.py` 的结果，两条都是结构性错误而不是概率偏好：

1. **100.0% 的答案首 token 粘在 `Bot:` 上**。答案用 `enc(s)` 编码而 prompt 以
   `Bot:` 结尾，ByteLevel BPE 于是在一个真实文本里不存在的边界切词：`They` 变成
   `T|h|ey`，`She` 变成 `S|he`。每一个 chat 窗口都在教模型「回复从词中间开始」。
2. **12.5% 的答案以裸引号开头**，最常见的开头「句子」是 `" she asked.` /
   `" asked lily.`。旧的 `split_sentences` 把 `?"` 当成句末，再把孤立的 `"`
   当作下一句的开头。

修复（`build_chat_corpus2.py`）：答案统一 `enc(" " + s)`；句子切分改为引号感知，
`"Stop!" said Tom.` 保持一句；并且**丢弃目标句之前的所有引号开头句**，让回复永远
不以对话标点起头。新的 `dataset_mixed6.txt` = 828,728 chat（引号开头 0.00%）+
1,356,722 story（原样复制，逐字节不变）。

### 10.4 修复效果：首 token 形态被彻底纠正

`probe_first_shape.py`，确定性、零噪声（对 `Bot:` 之后的整个分布求和）：

| | word-start% | quote% | fragment% | top-1 |
|---|---|---|---|---|
| `mixed6_ft` | 7.3 / 7.9 | 10.8 / 12.5 | **81.7 / 79.3** | `'O'` |
| `mixed11_shape` | **91.5 / 92.0** | **0.1 / 0.1** | 8.0 / 7.6 | `' O'` |

（前一个数是 Q 组，后一个是 TOPIC 组。）模型原先有 82% 的概率质量落在无前导空格的
碎片上，现在 92% 落在带空格的词首。

`mixed11_shape` = 从 `mixed6_ft` 热启动、用 `dataset_mixed6` 跑 1500 步，
最佳 val 2.9416 @ iter 1200（新验证集起点 2.9749）。

### 10.5 代价：Once 开场升高，模板塌缩加重

板上 `board_eval.py 3 COM5`（18 条模型回复）：

| | 收尾 | Once upon 开场 | 中位 token | 延迟 |
|---|---|---|---|---|
| `mixed6_ft` | 78% | 11% | 28 | 86 ms/tok |
| `mixed11_shape` | **83%** | **39%** | 28 | 87 ms/tok |

必须放在一起看的是：旧模型的「奇怪开头」= **27% 引号碎片 + 11% Once**，新模型 =
**0% 引号 + 39% Once**。总比例相当，但旧的那 27% 是 `" asked the cat.` 这种不可读
片段，新的是 `Once upon a time, there was an old man who needed it.` 这类语法完整
但答非所问的句子。**通顺度上升，相关性下降。**

### 10.6 诊断：Once 高不是「学过头」，而是 OOD + 采样放大

用**语料里真实存在的 prompt** 测（`probe_corpus_fit.py`，每类 60 条）：

| prompt 类型 | 语料中 P(Once) | 模型 P(Once) | 比例 |
|---|---|---|---|
| summary | **26.7%** | 11.7% | **0.4×（低估）** |
| words | 16.7% | 7.5% | 0.4× |
| continue | 0.0% | 0.0% | — |

所以在训练分布内，模型**比语料更少**用 Once。语料本身 summary 类的 Once 就有 26.7%
（跳过引号开头句之后，Once/One 的相对占比更高）。

真正偏高的是**分布外输入**：裸题材 prompt 上 P(Once)=16.9%，板上 top-k 采样后
实测 2/3（`a little girl named Lily found a box`）。原因是 ` O` 这个首 token 有
34.7% 的质量，采样重新归一化会把这个分支反复选中。

**P(联合概率) ≠ 采样率**，这条在本轮再次被证实：17% 的联合概率在 18 条板上样本里
表现为 39%。

### 10.7 否决的方向

- **把题材改写成 `Write a story using these words: ...`**（语料里真实存在的形状）：
  完句率只有 58%，且完全不使用给定的词（`robot, friend` → `Lulugies came to the
  beach with his mom.`）。与「给定词表在信息论上不可学」的旧结论一致。
- **关闭重复惩罚**：在新模型上完句率从 87% 掉到 73%，不如保留。
- **`uniq` / `div` / `freq` 变体**：在新模型上都不优于 `sub 1.6`。

### 10.8 结论

首 token 修复是**结构性、确定性**的改进（7%→92%），但它把「碎片开头」换成了
「模板开头」，没有解决「回复与输入无关」。26.7% 的语料 Once 率是根源：TinyStories
每篇故事只贡献一个开头，Once/One 的自然占比就是四分之一。

下一步只有两条路，都指向**语料的目标分布**而不是解码参数：

1. 在 `build_chat_corpus2.py` 里给 Once/One 开头加配额（如保留 30%），把 summary 类
   降到约 10%，再重训 1500 步（约 35 分钟）。风险是开头多样性的来源同时被削弱。
2. 接受模型只在「描述型长 prompt」上可靠，把问句与短题材交给规则层，模型只处理
   summary/continue 形状。

### 10.9 本轮新增脚本

| 脚本 | 用途 |
|---|---|
| `eval_quality.py` | OOV/bigram/自重复/多样性四指标，`SPEC="mode:pen:win,..."` 扫惩罚形态 |
| `analyze_answer_shape.py` | 全量统计答案首字符、首词、首句、首 token 是否粘连 |
| `inspect_mask_boundary.py` | 打印 mask 边界异常窗口的原文（已用于排除「mask 错位」误判） |
| `build_chat_corpus2.py` | 引号感知切分 + 前导空格，重建 chat 半部 |
| `splice_corpus.py` | 新 chat + 旧 story 拼接，story 半部逐字节不变 |
| `probe_first_shape.py` | 首 token 三分类（word-start/quote/fragment）+ 真实 P(Once)/P(One) |
| `probe_corpus_fit.py` | 用语料真实 prompt 对比「语料分布 vs 模型分布」 |

**两个必须记住的坑**：

- **`.mask` 侧车没有注释行**。用文件物理行号索引 mask 会整体错位 3 行，现象是
  「prompt 缺尾巴、答案多开头」，看起来像语料坏了，其实是脚本错了。`main.py` 是先把
  注释过滤进 `body` 再索引，所以训练侧一直是对的。
- **` Once` 不是单个 token**，是 `[' O', 'nce']`。用 `encode(" Once")[0]` 测 P(Once)
  实际测的是 P(' O')，会把 One/On/Outside 全算进去。必须走前缀概率乘积。


---

## 11. 开头多样性：Once 不是来源，是压在最上的一块（2026-10-20）

### 11.1 先量再调：`probe_opener_diversity.py`

opener = 答案前 4 个词（去标点、小写），在 `_chat5.txt` 828,728 窗口全量统计：

| 集合 | 窗口 | Once% | 不同开头 | H | exp(H) | top20% |
|---|---|---|---|---|---|---|
| ALL | 828,728 | 10.3% | 213,944 | 9.96 | 21,241 | 12.8% |
| summary | 127,536 | 33.9% | 36,001 | 7.74 | 2,305 | 29.4% |
| words | 78,313 | 32.4% | 29,341 | 8.16 | 3,497 | 23.8% |
| continue | 622,879 | 2.6% | 174,306 | 10.00 | 22,119 | 10.7% |

continue 占 75% 且几乎没有 Once，所以全局只有 10.3%——「Once 占三分之一」只在
summary/words 里成立。最高频单模板是 `once upon a time`（全局 4.01%、summary 内 16.93%）。

**关键裁决**：砍 Once 不会削弱多样性，因为它是头部的盖板而不是长尾的来源。

summary 内各方案（`--once-quota` × `--cap`）：

| 方案 | 窗口 | Once% | exp(H) | top20% |
|---|---|---|---|---|
| 现状 | 127,536 | 33.9 | 2,305 | 29.4 |
| 配额 keep 30% | 96,040 | 12.2 | 5,212 | 19.3 |
| 配额 keep 10% | 88,030 | 4.2 | 6,925 | 15.1 |
| cap 1% | 104,153 | 19.4 | 7,173 | 13.5 |
| **配额 30% + cap 2%** | 91,484 | **7.8** | **6,819** | 15.3 |

两种做法都让 exp(H) 涨 2.3~3.7 倍、top20 集中度腰斩。唯一真实损失是「以 Once 开头的
答案」绝对条数：summary 43.2k→7.1k、words 25.4k→4.3k。

### 11.2 故事半部不用动（`probe_story_openers.py`）

`dataset_mixed6` 故事半部 1,356,722 窗口是随机切片，**只有 1.7% 首词是 Once/One**
（`once upon a time` 0.72%，其余 `nce upon a time` / `upon a time there` 是词中切片）。
→ Once 先验完全由 chat 半部的 summary/words 答案制造，只修那一半就够。

### 11.3 落地：配额 + 模板封顶

`build_chat_corpus2.py --once-quota 0.30`：Once/One 开头的故事 70% 概率改抽成
`continue`（Once 落进 prompt、不进答案），30% 仍出 summary/words。
主 rng 流不动，配额硬币用独立 `random.Random(seed ^ 0x5EED)`——**旧语料 92.7% 的窗口
在新语料里逐字节相同**（768,592/828,728），112,136 条新 continue 窗口来自改抽。

`resample_openers.py`：按 kind 内统计 opener，Once 类保留 30%，其余 openers 单模板
封顶 2%（**保留概率制**，硬截断会跟着文件顺序走）。

| | 窗口 | Once% | exp(H) | top20% |
|---|---|---|---|---|
| summary 前 | 97,604 | 13.6 | 5,354 | 19.3 |
| summary 后 | 88,138 | **4.3** | **7,201** | 14.7 |
| words 前 | 60,576 | 12.7 | 7,968 | 13.4 |
| words 后 | 55,241 | **4.2** | **10,368** | 9.2 |

`dataset_mixed7.txt` = 878,913 chat + 1,356,722 story = 2,235,635 窗口（chat 39.3%）。

### 11.4 训练与结果

`mixed12_openers`：从 `mixed11_shape` 热启动、`dataset_mixed7`、lr 2.5e-4、
T_MAX 1500、warmup 50、batch 128；**第 1000 步被 patience 早停**（新语料上总 val
再没低于 iter 0，`ESPGPT_PATIENCE` 默认 10）。这是指标问题不是模型问题，但代价是
余弦没跑完（停时 lr 8e-5）。权重取 iter 1000。

| 判据 | mixed11_shape | mixed12_openers |
|---|---|---|
| P(Once) Q 组（确定性） | 20.6% | **1.2%** |
| P(Once) TOPIC 组 | 16.9% | **0.7%** |
| word-start Q/TOPIC | 91.5 / 92.0 | 88.1 / 88.8 |
| fragment Q/TOPIC | 8.0 / 7.6 | 11.3 / 10.9 |
| 板上 Once（18 条） | 39% | **6%** |
| 板上收尾 | 83% | 78~89%（多轮合计 43/54=80%） |
| 板上 OOV（rep 1.0） | — | 1.2% |
| 板上延迟 | 87 ms/tok | 87 ms/tok |
| Flash / RAM | 95.5% / 90.2% | 95.5% / 90.2%（逐字节未变） |

`eval_quality.py`（本地 30 样本，sub1.6/topk40/temp0.85）：`uniq%` 67→**100**、
fin 87→87、OOV 1.1→1.8、bigram 94.7→92.8。**多样性是拿模板换的，不是白来的**：
Once 那套高概率开场本身贡献了 bigram 命中率。

### 11.5 惩罚强度重测（板上，`board_rep_sweep.py`）

| rep | finished | OOV | chars |
|---|---|---|---|
| 1.6 | 29/36 = 81% | 2.0% | 75 |
| **1.0（已写入固件）** | 29/36 = 81% | **1.2%** | 61 |
| 0.6 | 15/18 | 2.3% | 68 |
| 0.0 | 15/18 | 0.8% | 70 |

完成率相同、OOV 更低；`g_rep_penalty` 已改 1.0f 并重烧。

**两个待办**：① 余弦没跑完，续 500 步可能值 0.02~0.03 nats；
② word-start 从 92% 退到 88.8%，碎片回升 3.3pp，是这次多训 1000 步的副作用，
可以用低 lr 短续训找回来。

**新增脚本**：`probe_opener_diversity.py`（开头分布 + 方案模拟）、
`probe_story_openers.py`（故事半部开头）、`resample_openers.py`（配额 + 封顶）、
`board_rep_sweep.py`（板上惩罚扫描）。

---

## 12. 早停的真凶、碎片的假警报、温度的翻身（2026-10-20 上午）

### 12.1 上一轮判断错了一半：早停不是「指标失效」，是余弦没跑完

`mixed12` 在第 1000 步被 patience 停掉时 lr 还有 8.12e-05，余弦只走了一半。第十一节写的
「这是指标问题不是模型问题」方向对，但归因不完整——**真正的损失是退火没跑完**。

`mixed13_lowlr`：同一份 `dataset_mixed7`，从 `mixed11_shape` 热启动，lr **1e-4**（原来 2.5e-4）、
T_MAX 1500、warmup 30、`ESPGPT_PATIENCE=999` 关掉早停、`ESPGPT_EVAL_INTERVAL=250`
（223,564 个 val 窗口一次 eval 要 4 分钟，100 步一次等于一半训练时间在做评估）。

| iter | 0 | 250 | 500 | 750 | 1000 | 1250 | 1500 |
|---|---|---|---|---|---|---|---|
| val | 2.9425 | 2.9905 | 2.9937 | 2.9808 | 2.9876 | 2.9582 | **2.9386** |

先涨 0.05 再随 lr 退火回落，**终点低于起点 0.0039**。lr 大小不改变形状（2.5e-4 也一样先涨），
所以那是「换分布要付的学费 + lr 扰动」，不是退化。

**教训：热启动训练不要开早停。** 余弦的收益全在最后 20% 步数，patience 卡在中段等于白训。

### 12.2 但 0.0039 也是噪声级：按 prompt 族拆开看

`eval_by_kind.py`，同一进程同一划分，每族 400 条：

| 模型 | cont | cont-once | story | summary | words | pooled(2000) |
|---|---|---|---|---|---|---|
| mixed11 | 1.9078 | 2.0166 | 2.1314 | 1.6952 | 1.9177 | **2.0266** |
| mixed12 | 1.9637 | 2.0648 | 2.1812 | 1.7364 | 1.9664 | 2.0759 |
| mixed13 | 1.9189 | 2.0372 | 2.1526 | 1.7111 | 1.9336 | 2.0456 |

五族全部是 mixed11 最低、mixed13 居中、mixed12 最高，方向与全量 val（mixed13 略低于起点）
相反，两个估计都贴着噪声带。**诚实的结论：这轮训练对 loss 的影响不可分辨。**
真实收益只有行为：P(Once) 20.6%→1.6%。这个交换是划算的，但别再拿 val 当论据。

### 12.3 碎片率上升是假警报（`probe_fragment_tokens.py`）

`probe_first_shape` 报 word-start 92.0 → 88.8 → 84.2，看着像退化。按 token 类型拆：

| 模型 | word | shard-long | shard-short | other | punct | quote |
|---|---|---|---|---|---|---|
| mixed11 | 91.83 | 4.28 | 1.59 | 1.62 | 0.10 | 0.06 |
| mixed12 | 88.51 | 6.15 | 2.14 | 2.38 | 0.26 | 0.18 |
| mixed13 | 84.17 | 8.13 | 3.00 | 3.11 | 0.65 | 0.16 |

碎片是 `'ient'` `'rave'` `'ore'` `'ney'` `'leas'` `'gry'` 这类词中切片，**质量摊在 890 个
fragment token 上，单个最高 0.6%**。top-k 40 + 温度采样几乎采不到：mixed13 板上 36 条里
0 条以碎片开头，完成率反而 86%（mixed12 81%）。

→ **fragment 列只在「某个碎片 token 概率很高」时有意义。砍掉强先验后分布自然摊平，
不预测板上行为。板上判据仍以 `board_eval.py` 收尾率为准。**

顺带两个自己的脚本 bug：
① `.mask` 存的是 **prompt 长度**（`_pad_window(seq, prompt_len)`），按 `len-m` 切会切在
prompt 里，看起来像「答案从词中间开始」——抽查过 4 条改抽窗口，全部句子对齐、干净，
配额逻辑没坏；
② `import probe_first_shape` 会执行它的 argv report 循环，白跑一遍全量探针。

### 12.4 温度：0.7 从「被否决」翻成「最优」

上一轮否决 0.7 的理由是「Once 开场 17%→39%」，那是 mixed11（Once 先验 20.6%）。语料改造后
板上 Once 已经 0/78，低温不再选众数 Once，这条理由消失。重扫（`board_temp_sweep.py`，
两轮共 78 条）：

| temp | 完成率（合并） | OOV（合并） | 单轮 OOV | chars |
|---|---|---|---|---|
| **0.70** | **70/78 = 89.7%** | **7/1030 = 0.68%** | 0.6% / 0.7% | 56~67 |
| 0.85 | 65/78 = 83.3% | 19/1066 = 1.78% | 1.4% / 2.0% | 71~64 |
| 1.00 | 25/30 | 1.9% | — | 64 |

OOV 差 2.6 倍且两轮同向，完成率不被牺牲。`TEMPERATURE` 已改 `0.7f` 重烧。
**温度和重复惩罚一样，是随模型走的参数，模型一换必须重测**——上一轮的结论不能继承。

### 12.5 现役

`model/model_mixed13_lowlr.pt`，`dataset_mixed7`，板上 temp 0.7 / topk 40 / rep 1.0。
hpp 备份 `src/model_weights.hpp.bak-mixed12`。

| 判据 | mixed12 | **mixed13（现役）** |
|---|---|---|
| val 全量（mixed7 同划分） | 2.9811（早停，余弦未完） | **2.9386** |
| 板上收尾（36 条模型回复） | 81% | **86%** |
| 板上 Once | 6% | 6% |
| 板上 OOV（78 条） | 1.2%（rep 扫描口径） | **0.68%** |
| 延迟 | 87 ms/tok | 88 ms/tok |
| Flash / RAM | 95.5% / 90.2% | 逐字节未变 |

板上文本（temp 0.7）：`'Nemo went to the park and saw that it was full of colorful things.'`
`'Tom was very good at playing in the park with his friends.'`
`'One day, Timmy saw a boy sitting in the park.'`
未完成的 5 条都是被 `CHAT_MAX_TOKENS=32` 截断，不是生成崩坏。

新增脚本：`probe_fragment_tokens.py`（按 token 类型拆首 token 质量）、
`eval_by_kind.py`（按 prompt 族拆 val）、`inspect_quota_windows.py`（抽查改抽窗口）。

### 12.6 收尾率 86% → 97%：`CHAT_MAX_TOKENS` 32 → 40

36 条模型回复里 5 条没完成，逐条看全是**被预算截断的正常句子**（`"...and said, \"Do you want"`），
不是生成崩坏。`CHAT_MAX_TOKENS` 32 → 40：

| 轮次 | 完成率 | Once | 中位长度 | 延迟 |
|---|---|---|---|---|
| 32（旧） | 31/36 = 86% | 6% | 64 字符 | 88 ms/tok |
| 40 第 1 轮 | 34/36 = 94% | 0% | 73 字符 | 86 ms/tok |
| 40 第 2 轮 | **36/36 = 100%** | 3% | 58 字符 | 87 ms/tok |
| **合并** | **70/72 = 97.2%** | 1/72 | — | 不变 |

安全：prefill 的 prompt 裁剪（`main.cpp:992`）用的是 `MAX_GEN_TOKENS`(80) 不是 chat 预算，
所以 chat prompt 的 24 token 预算不变；24+40=64 远低于 `INFER_CTX=104`，窗口填不满，
`ctx_push` 的 memmove 路径不会触发。代价只是墙钟时间（40×87ms≈3.5s），质量优先。

**乱码是截断的伴生现象，不是独立的坏 token**：词表里 128 个 token 的 `decode([i])` 是
U+FFFD，它们是 TinyStories 原文里 `'`(U+2019) 这类多字节字符被 BPE 按字节切开的碎片。
单独生成碎片 → 串口按 GBK 显示成 `â€`；完整生成 2~3 个碎片则正常。所以 `'...I\'m sorry.â€'`
那条是**生成被截断在多字节字符中间**。要彻底消除，需要在输出末尾回退不完整的 UTF-8 序列
（未做，影响约 1~2% 的回复）。

---

## 13. 上第五层：先解决 RAM，再谈训练（2026-10-20 下午）

### 13.1 硬约束是 arena，不是 Flash

`ARENA_NEED = 4,000 + 7,072·N_LAYER + logits`，`ARENA_SIZE = 40 * 1024 = 40,960`：

| 部分 | 字节 |
|---|---|
| KV cache（int8 K+V） | 104 × 1 × 32 × 2 = 6,656 / 层 |
| KV scales（fp16 ×2） | 104 × 1 × 2 × 2 = 416 / 层 |
| **每层小计** | **7,072** |
| logits（2048 float） | 8,192 |
| x/xnorm/attn/proj/mlp_out | 64×4×5 = 1,280 |
| qkv_out | (64 + 2×1×32)×4 = 512 |
| gate/up/hidden | 64×4×3 = 768 |
| attention scores | 104×4 = 416 |

4L = 39,456（余 1,504）、5L = 46,528（**缺 5,568**）。RAM 已经 90.2%，把 arena 加 5.5 KB 会到 97%，栈没有空间。

### 13.2 解法：LM head 分块，logits 缓冲 8 KB → 1 KB

`static constexpr int LOGIT_CHUNK = 256;`，`s_logits` 改为 BSS 里的 1 KB 静态数组。

**为什么分块不损失任何东西**：top-k 是流式计算，只需要「当前第 k 优」，从来不需要整行。
2048 个词按 256 切片，插入排序的状态跨块延续，第 i 块的第 k 优并上之前的结果就是全局第 k 优。
argmax 同理。两条路径都仍是**单遍**。

`:topk 0`（精确全表采样）确实需要整行，改成三遍：求 max → 求 sum(exp) → 累积 CDF。
代价是 3× head 时间，所以它只能是诊断路径（默认 `g_topk = 40`）。

**重复惩罚不再能索引 logits 数组**，改为把 recent 窗口折成按 id 升序的 `(id, count)` 小表
（`REP_MAX = 64`，192 B BSS），chunk 扫描用单调游标推进：O(vocab + window)，算术完全等价
（旧代码每个出现减一次 = 现在减 count 次）。

顺带一个净收益：**prefill 不再算 head**。`transformer_forward` 现在止于 final norm，
`lm_head_begin()` / `lm_head_chunk()` 只在采样时调用。prompt 里每个 token 原本白算 15.6 ms
的 head，12 token 的 prompt 白烧 187 ms。

### 13.3 实测（4L mixed13 权重，验证分块没有破坏数值）

| | 分块前 | 分块后 |
|---|---|---|
| 板上一轮收尾 | 34/36 = 94% | 33/36 = 92% |
| 板上 Once | 0/36 | 2/36 = 6% |
| **延迟** | 88 ms/tok | **83 ms/tok** |
| Flash | 95.5% (997,463) | 95.5% (997,631) |
| RAM | 90.2% (73,891) | 91.7% (75,136) |

RAM +1,245 B = `s_logits` 1,024 + `s_lm_x` 132 + 惩罚表 192 —— 与预测逐字节吻合。
延迟反降 5 ms 是 prefill 省下 head 的直接结果。arena 大小不变（40,960），所以 **5L 的 RAM
也是 91.7%**，与层数无关。

`ARENA_NEED(5L) = 5×7,072 + 4,000 = 39,360 ≤ 40,960`，余 1,600 B。

### 13.4 Flash 预测

`arch_ab.py` 报 5L/32E = 949,227 B（90.9%），但实测 4L/40E 比预测高 51,284 B（vocab 表、
RoPE、ternary_lut 这些没进模型的部分）→ **5L/32E 实测预期 ≈ 1,000,500 B = 95.8%**。
4L/40E 现在就是 95.5%，所以余量会从 47 KB 缩到约 44 KB，可接受。

---

## 14. 5L/32E 上线：同参数量本该给深度（2026-10-20 晚）

### 14.1 训练

`model/model_5l32e.pt`，从零 30k 步、`dataset_mixed7`、lr 2e-3 余弦、batch 128、
block 128、`ESPGPT_N_LAYER=5 ESPGPT_N_EXPERT=32`、eval 间隔 250、patience 999。
**0.6~0.72 s/step**（4L/40E 是 0.564），总耗时约 5.5 小时。
参数 2.17M（4L/40E 是 2.16M，基本等价）。

日志 val：7.6656 → 3.7107(500) → 3.2236(2500) → 2.9055(29500，最佳) → 2.9185(30000)。
**日志 val 不可跨运行比较**：`main.py` 的 `random.shuffle(order)` 没有种子，每次划分不同。

### 14.2 判据一：固定验证集 CE（`eval_5l_vs_4l.py`）

同一进程、同一划分、同一批 100 batch × 128 序列，两臂走完全相同的前向：

| 臂 | val CE | ppl |
|---|---|---|
| mixed13 4L/40E（原现役） | 2.1238 | 8.36 |
| **5l32e 5L/32E（从零）** | **2.0614** | **7.86** |

**−0.0624 nats**，是噪声带（±0.01）的六倍。

### 14.3 判据二：按 prompt 族拆（`eval_by_kind.py`，各族 400 条）

`eval_by_kind.py` 已支持 `path@LxE` 语法，可在同一进程里比不同深度的臂。

| 模型 | cont | cont-once | story | summary | words | pooled |
|---|---|---|---|---|---|---|
| mixed13 4L/40E | 1.9189 | 2.0372 | 2.1526 | 1.7111 | 1.9336 | 2.0456 |
| **5l32e 5L/32E** | **1.8471** | **1.9538** | **2.0877** | **1.6214** | **1.8916** | **1.9780** |
| 差 | −0.072 | −0.083 | −0.065 | −0.090 | −0.042 | **−0.068** |

**五族全部下降，无一例外**。这就排除了「收益只来自某类窗口的分布偏移」——
与第十二节的教训（4L 那次训完五族方向互相矛盾）形成鲜明对照。这是纯容量收益。

### 14.4 判据三：首 token 形态（`probe_first_shape.py`，零噪声）

| | mixed13 | **5l32e** |
|---|---|---|
| word-start Q / TOPIC | 84.2 / — | **92.6 / 92.5** |
| fragment Q / TOPIC | 12.0 / — | **7.1 / 7.1** |
| quote | 0.2 | 0.1 |
| P(Once) Q / TOPIC | 1.6 / — | 2.2 / 1.2 |

碎片率从 12% 回到 7.1%，**回到 mixed11 之前的水平**。第十二节判过「fragment 只是
top-k 采不到的尾部」，但这轮它是随容量一起改善的，属于附带收益。

### 14.5 判据四：板上（`board_eval.py` 36 条 + `board_temp_sweep.py` 48 条）

| 判据 | 4L/40E mixed13 | **5L/32E 5l32e** |
|---|---|---|
| 板上收尾（合并全部轮次） | 70/72 = 97.2% | **116/120 = 96.7%** |
| 板上 Once | 3% | **2/120 = 1.7%** |
| 板上 OOV | 0.68% | **0.5%**（4/768） |
| 延迟 | 83~86 ms/tok | **97~98 ms/tok** |
| 平均字符 | 75 | 70~75 |

**收尾率是平的（96.7% vs 97.2%）**——40 token 预算下已经贴到天花板，CE 的 0.062 换不出
板上的可分辨差异。收益体现在别处：Once 1.7% vs 3%、OOV 0.5% vs 0.68%、碎片 7.1% vs 12%。
**代价只有延迟：+17%**。这正好是「质量优先，速度让步」该付的价。

文本样本（这轮）：
`'They went to the beach with their friends. They played together for hours. After a while, they saw a big storm'`
`'When they got to the park, Tim found a little bird. The bird was scared of the frog.'`
`'As the sun went down, a kind elderly old man came to help.'`

后两条是 mixed13 从没产出过的跨句因果结构。但仍有 `'Sam and Sam were playing in the park.'`
`'Joe was a good girl who loved to take the bathtub.'` 这类语义硬伤——**这是 2.17M 参数
（约 2048 词表 × 4 层因果注意）的容量天花板，不是配置问题**。

### 14.6 资源

| | 4L/40E | **5L/32E** |
|---|---|---|
| Flash | 997,631 (95.5%) | **1,001,079 (95.8%)**，余 42.4 KB |
| RAM | 75,136 (91.7%) | **75,200 (91.8%)** |
| arena 需求 | 39,456 | **39,360**（≤40,960） |

RAM 只涨 64 B（少 8 个专家的路由表抵消了一层的 KV），**arena 需求反而比 4L 少 96 B**，
因为 32 专家的 MoE 缓冲比 40 专家小。第五层的 RAM 成本在 13.2 的分块改造里已经预付过了。

### 14.7 结论

同参数预算给深度确实更值：4L/20E→5L/32E 早前在 mixed2 上是 −0.117，这轮在 mixed7 上
从 4L/40E 换成 5L/32E 是 **−0.068**（更强的对手，收益小一些但仍显著）。
**架构轴该往深度走，专家数在 32 附近已经饱和。**

### 14.8 这是当前尺寸的上限

两个轴都到边界了，别再往下试：

| 候选 | 卡在哪 |
|---|---|
| 5L/36E | Flash 1,048,215 > 1,044,464（差 3.7 KB） |
| 6L/32E | Flash 1,062,879 超窗口；arena 46,432 > 40,960（要砍 `INFER_CTX` 到 88） |
| 6L/24E | 参数反而降到约 1.98M（8 个专家 ≈ 477k 参数 > 一层 0.29M），不划算 |

**5L/32E 是 1MB 地址窗口 + 40KB arena 下的最优点。** 再往前只有三条路：
① 用掉那 3MB spiffs 把专家放 FS（收益递减，每 token +1.3 ms）；
② 砍 `INFER_CTX` 换第 6 层（会牺牲上下文，RoPE 超 80 key 本来就退化）；
③ 换芯片。

新增脚本：`eval_5l_vs_4l.py`（固定验证集总 CE）、`eval_5l_families.py`（未采用，
被 `eval_by_kind.py` 的 `@LxE` 语法取代）。

---

## 15. 诊断：它为什么不"听话"（2026-10-20 晚）

用户观察：「感觉它是一个完全不听输入、只会自己造句的模型。」这个判断一半对一半错，
错的那一半正好指向一个可以免费修的地方。

### 15.1 它确实在读输入（`probe_prompt_attention.py`）

**名字绑定测试**：`User: Continue this story: {名字} had a little dog.\nBot:`，
8 个名字 × 6 seed × 40 token，看回复里出现哪个名字：

| 指标 | 值 |
|---|---|
| 回复包含**prompt 里那个**名字 | **35/48 = 73%** |
| 回复包含**别的**名字 | 16/48 = 33% |
| 回复不含任何名字 | 8/48 = 17% |

混淆矩阵对角线独大（Lily 2/6、Sam 6/6、Tim 6/6、Sue 5/6），说明它把 prompt 里的实体
真的带出来了。**这不可能是"完全不听"的模型。**

**内容敏感性**：固定 `Continue this story:` 模板，只换故事内容（8 条）→
首 token 的 pairwise 归一化 KL **中位数 0.862**、top-1 在 `' She'` / `' H'` / `' The'`
`' It'` 之间切换。KL/熵 ≈ 0.86 意味着**换内容几乎等于换整个分布**。

### 15.2 但它只在 continue 模板下读输入（`probe_template_effect.py`）

同一批主题（`a dragon` / `a small cat` / …）分别套两种模板：

| 模板 | pairwise 归一化 KL | 不同 top-1 | 最常见 top-1 |
|---|---|---|---|
| `User: <topic>`（summary） | **0.017** | 2/5 | `' The'` 3/5 |
| `Continue this story: Once upon a time, there was <topic>.` | **0.169** | 3/5 | `' It'` 3/5 |

**continue 模板的输入敏感度是 summary 的 10 倍。** 而 summary 下的 0.017 等于说
「输入换了，下一 token 分布基本没动」。

更直接的一刀：把 `what is a cloud` / `a dragon` / `qwerty zzz` / `123 456` /
`hello there my friend` 五条**语义上毫无关系**的短输入都走 summary 模板，两两归一化 KL
**中位数 0.037** —— 它们对模型来说几乎是同一个输入。这就是「不听」的机器口径。

### 15.3 原因：短输入 100% 落在训练分布之外

`_chat7.txt` 三种模板的 prompt 长度（token，`.mask` 即 prompt 长度）：

| kind | 窗口 | p1 | p10 | 中位 | p90 |
|---|---|---|---|---|---|
| cont | 424,221 | 18 | 21 | 30 | 53 |
| cont-once | 311,313 | 21 | 25 | 31 | 38 |
| **summary** | **88,138** | **28** | **39** | **57** | **79** |
| words | 55,241 | 20 | 21 | 23 | 26 |

**88,138 条 summary prompt 里，长度 ≤6 token 的有 0 条（0.000%）。** 而板上聊天模式把
任何输入包成 `User: <输入>\nBot:`（`src/main.cpp:1118`），用户输入 `a dog`（2 token）、
`what is a cloud`（4 token）——模型在训练里从没见过这么短的 `User:` 前缀，
只能回退到该模板的边缘分布：H≈4.2 nats，top-3 各占 8~10%，即"随便挑个故事开头"。

**次要因素**：词表 2048，` cloud`→`[' cl','oud']`、` sky`→`[' sk','y']`、
` dragon`→`[' dra','g','on']`、` computer`→`[' comp','ut','er']`。问句内容本来就是碎片。

### 15.4 不是「对话训练没生效」，是它学的不是你以为的东西

生效了：`Bot:` 边界、单行回复、句子形态、收尾率 96.7% 全是这轮训练的产物。
但语料里**一个问答对都没有**——三种模板全是 `文本 → 故事散文`：

| 模板 | prompt 是 | 答案 是 | 学到的映射 |
|---|---|---|---|
| summary | 一段故事摘要 | 故事的**第一句** | 给定任意散文 → 吐一个故事开头 |
| continue | 故事第一句 | 后续句子 | 给定开头 → 接着写（**真条件**） |
| words | 词表 | 一个故事 | 给定词表 → 吐一个故事 |

summary 那条本身就是**低信息量映射**：57 token 的摘要决定不了第一句用哪个词，
所以它在这条模板上学到的正是"忽略输入、吐故事开头"。而板上所有输入都走这条。

### 15.5 内置模式确实是有的，而且立刻有用

`:params` 会打印 `mode=CHAT|STORY`。`:mode 0|1` 切换，另有 `:fewshot` / `:temp` /
`:topk` / `:rep`。**但 mode 不是"听话/不听话"的开关，它是 prompt 模板的开关。**

板上实测（`board_mode_demo.py`，同一权重、同一 temp0.7/topk40/rep1.0）：

| 输入 | mode 0（summary 模板，现默认） | mode 1（原样当故事开头） |
|---|---|---|
| `Once upon a time, there was a little dog named Pip.` | `Max was very excited. He wanted to take the home and play with his friends.` ← **丢掉 Pip，换成人名 Max**，`take the home` 破句 | `He lived in trees and many friendly things. One day, he saw a little boy named Tom. Tom was very excited. He wanted to go on an adventure!` ← **代词 He 指 Pip，跨句因果** |
| `a dog` | `The dog was not fair. It wanted to play with the dog. The dog felt angry...` ← 自我指代打转 | `They were happy and proud of their car. Sara liked to play with her toys...` ← 漂移，但句子通顺 |
| `what is a cloud` | `Ben and Lily was excited. They looked around the park each other.` ← 与 cloud 无关 | `and that was not an. It is meat!" Lily and Ben laughed.` ← 更差（错位续写） |

**结论：给故事开头时 mode 1 明显更强（跨句连贯、代词绑定正确）；问句两者都不行。**

两个附带缺陷，都源于语料：
① mode 0 的回复末尾会吐出 `User:` —— 模型在尝试把对话翻页，训练里没有这个约束；
② `take the home`、`was excited` 的主谓一致这类错误在 2.17M 参数下是常态。

### 15.6 裁决

- **不是"只识别很窄的问题"**：它根本不认问题。问句和 `qwerty zzz` 走同一条回退路径
  （KL 0.037），可互换。
- **不是"对话训练没生效"**：生效的是格式，没生效的是"听话"，而后者从来没被训练过。
- **是预期的**：一个 2048 词表 / 2.17M 参数 / 5 层 / 零问答对的模型只能是条件故事生成器。
- **免费改善**：`mode 1` 用于续写场景。真要"回答问题"必须换任务重训（加指令-回复对），
  那是项目级工作量，不是调参能到的。
- **问句继续交规则层**（现已有 `what is your name`、`i feel sad today` 等命中）。

新增脚本：`probe_prompt_attention.py`（内容/模板/名字绑定/词表复用）、
`probe_template_effect.py`（同内容两模板 + 语料 prompt 长度分布）、
`probe_framing_ab.py`（生成侧 A/B）、`board_mode_demo.py`（板上两模式对比）。

---

## 16. 滑动窗口：连续生成任意长度（2026-10-20 深夜）

目标：一次生成几百 token，而不是 40/80 被预算截断。

### 16.1 先在 PC 上把质量问题问清楚（`gen_long.py`）

`main.Transformer.forward` 是 `rope_cos[:T]`，位置每次从 0 起算，所以 PC 上
`ids[-win:]` **就是滑动窗口的精确等价**，不需要任何 KV 手术。结论：

- **不循环**：400 token 的 8-gram 重复率 0.00、distinct-4 0.96。这条必须先过，
  否则固件做多少 KV 手术都白费。
- **窗口大小几乎不影响表面指标**：32 / 64 / 104 / 128 的 rep8、distinct-4 全部
  在 0.93~0.99 之间。限制不在窗口。
- **OOV 随长度单调上升，PC 上也是如此**：200 token 里第 1 块 0.0% → 第 4 块
  5.6%（3 个种子一致）。所以"越写越糊"是模型漂移，**不是滑动的锅**——这条基准
  后来成了判定固件 bug 的标尺。
- 模型的行为其实很合理：**写到 ~120 token 就「The end.」收尾、再起一个新故事**
  （TinyStories 平均长度 ~130）。`--pin N`（钉住开头当 attention sink）不阻止
  重启，只是让重启变得合法（先出 "The end."）。所以"一篇长文"的真实形态是
  **一串短故事**，这已经是这个尺寸能做到的最好结果。

### 16.2 固件设计：slot 与 pos 解耦

原来的 `ctx_push` 一满就 `ctx_pos = 0`，逼着整窗重算（104 次前向 ≈ 8.4 s/token）。
根因是 `transformer_forward(token, pos)` 里 **slot 和 RoPE 位置是同一个变量**。

- 拆成 `transformer_forward(token, slot, pos)`：slot 决定 K/V 写在第几格、注意力
  扫到第几格；pos 只喂 RoPE。
- 滑动一个 slot 时行的旋转**不用重算**，只要 `ctx_shift++` 让
  `pos = ctx_shift + slot` 与原写入位置保持相等。相对偏移因此恒定。
- 位置会随滑动往上漂（`rope_cos` 只有 128 行，窗口 104，只能漂 24），漂满就
  `ctx_renumber(delta)` 把所有 K 行原地旋回 −delta。**每 ~25 个 token 一次**，
  不是每个 token。
- `ctx_pos` 在滑动后置为 `ctx_len`（行都还有效）；但**推 prompt 期间
  `ctx_pos == 0` 时必须只搬 id、不动 KV**，否则会把从没算过的行标成已 prefill。

### 16.3 踩到的真 bug：`apply_rope_row` 不是旋转

第一版用 `apply_rope_row(row, cos_d, -sin_d)` 当逆旋转，文本在 ~120 token 后
逐步塌成生造词沙拉，末段 OOV 18.5 / 23.1 / 26.9%（PC 同条件是 3.6~6.7%）。

根因：PyTorch 的 `apply_rope` 是
`cat([x[0::2]*c - x[1::2]*s, x[0::2]*s + x[1::2]*c])`——**读相邻对、写前后半**，
等于 `R(θ)∘E`（E 是奇偶解交织置换），**不是纯旋转**。两个后果：

1. 连用两次不是旋转 2θ，所以"同 row、正弦取负"**不是它的逆**（偏差 O(1)，实测
   3.23），每次重编号都在污染 K，越滚越坏。
2. 但**注意力仍然是纯相对的**：`(Eq)·R(pq−pk)(Ek)` 只依赖位置差。所以截断窗口
   本身没错，错的只是"怎么把已存的行换个位置"。

修法：`rope_rotate_back()` 直接作用在**已存的布局**上（配对是前后半，跳过 E），
角度 −delta：

```c
t[i]     =  a*c + b*s;
t[h + i] = -a*s + b*c;
```

`verify_rope_shift.py` 把三件事都钉住：重编号 vs 重新编码 2.1e-6（float32 表
精度）、朴素逆旋转偏差 3.23（必须大，否则这个测试就没在测东西）、分数只依赖
位置差 2.9e-7。**教训：验证 RoPE 一定要照抄仓库里的那个函数，不能凭"标准
RoPE 长什么样"去假设。**

### 16.4 板上实测

| 项 | 值 |
|---|---|
| 400 token 总耗时 | **59.8 s = 149.6 ms/tok**（6.7 tok/s） |
| 滑动 314 次 + 重编号 12 次 | **954 ms，占 1.6%** |
| 200 token | ~140 ms/tok |
| 400 token 分 8 块 OOV | **0% / 0% / … / 0%** |
| 200 token 三次 | 末块 3.4 / 0.0 / 0.0% |
| Flash / RAM | 96.0%（1,002,475 B，余 41,989 B）/ 91.9% |

- **memmove 几乎免费**：314 次共 72 ms（0.23 ms/次）。所以 `SLIDE_BLOCK` 定为
  **1**（窗口恒为 104）；按 8 个一批滑只省 0.4% 却要丢掉 8% 上下文，不划算。
- **真正变慢的是注意力**：97 ms/tok（短回复、约 32 格）→ 150 ms/tok（满窗 104
  格）。多出的 ~48 ms 基本就是多扫 72 格 × 2 head × 5 层的 float MAC。
- 命令面：`:len <n>`（1..600，默认 80）、`:slide 0|1`（0 = 旧的全量重算，
  仅作 A/B 基线）、**ESC 中断**（`Serial.peek()` 只看不取，其它排队字节留给下一轮）。
  一轮结束会打 `[slide] N moves X ms, M renum Y ms`，两个桶分开计时才有得调。
- 聊天路径不受影响：prefill 预留 `INFER_CTX - MAX_GEN_TOKENS` 保证 24+40=64 < 104，
  永不滑动，也不会打 `[slide]`。

### 16.5 下一步（未做）

- **窗口可以从 104 砍到 64**：省 13.6 KB arena（每层每格 68 B × 40 格 × 5 层），
  注意力也快 ~20%，而 16.1 已证明 64 和 104 的质量无差别。省下的 arena 足够
  把 6L/32E 装进去（6 层 × 64 格 ≈ 30.1 KB，此前 6L 在 88 格时超限）。
  这是"同参数预算给深度"那条线的直接延续。
- UTF-8 回退仍未修（输出末尾的不完整多字节序列）。

---

## 17. 下一步候选（2026-10-20，外部调研 + 中文可行性）

详细论证见 **`research_2026-10.md`**，这里只留结论指针。

### 17.1 方法侧：真正剩下的三条杠杆

| 排序 | 事项 | 为什么值 | 成本 |
|---|---|---|---|
| ~~1~~ | ~~原生 ternary 训练（QAT）~~ | **已封顶，见 18.1：训练本来就是 QAT，激活 int8 免费，没有 PTQ 损失可回收** | — |
| 2 | **窗口 104→64** | 已证质量无差别；**已上线，−21% 生成时间**（18.2） | 完成 |
| 3 | **跨层共享 KV（CLA）** | KV 减半 → arena 墙彻底消失，深度与窗口脱钩 | 半天 PC + 1 天固件 |
| 4 | 共享专家（MobileMoE） | 专家数在 32 饱和后，唯一还能提利用率的结构改动 | +14,730 B flash |
| 5 | 权重放 LittleFS | 容量 1 MB → 3.1 MB（英文/中文双模型共存） | 先 spike FS 流式读 |
| 6 | 结构化改写语料（FinePhrase） | 治"小输入不读"；**改写模型 1B 就够**，格式比模型重要 | 1~2 天 |

**不做**：KV 量化 / TriAttention（KV 只 21,720 B，窗口就是 eviction 旋钮）、
投机解码、Mamba 混合（要重写所有 kernel，只为换"窗口无限"，而滑窗只占 1.6%）。

### 17.2 中文角色 chatbot：可行，但目标是 GuppyLM 式封闭域角色

- **词表**：汉字覆盖 2000 字 = 98%+；TinyStories-Zh-2M 只有 2,492 个不同汉字。
  **2048 = 2000 汉字 + 48 控制 → Flash 零增量**（每条目 88 B：int8 64 + scale 4
  + head 2-bit 16 + head scale 4）。配语料裁剪。
- **字符级分词顺手修掉乱码**（现有 128 个 BPE token 的 `decode([i])` 是 U+FFFD），
  C 端也更简单（二分查表，无 merges）。
- **路线 A**（1~2 天）：架构不动，只换 tokenizer 表 + 规则，先证可行。话题压到 15~20。
- **路线 B**（3~5 天）：`dim 128 / n_head 4（head_dim 仍 32 → KV arena 不变）/ mlp 128`
  → 8.35M 参数，权重 2.8 MB，必须放 LittleFS，估 250~400 ms/token。
  `main.cpp` 无 dim 硬编码、converter 自动推导 n_embd/n_head，新工作只有 FS 加载器。
- **生死线**：短输入 pairwise 归一化 KL 从 **0.017 → >0.5**（`probe_first_shape.py`）。
  达不到就还是故事机。根因是语料形状（无问答对），不是容量。

---

## 18. QAT 封顶（负结果）+ 窗口 64 上线（2026-10-20 深夜）

### 18.1 量化方向到此为止：训练侧早就没有余量了

17.1 里我把"原生 ternary 训练"列为第一杠杆，理由是"我们现在是 PTQ"。**这个判断是错的**，
`eval_qat_ablation.py` 把它一次证伪。

`main.py:744` 的 `BitLinear` **就是训练路径本身**：absmean 分组量化到 {−1,0,+1}、
per-token int8 absmax 激活、STE；qkv / proj / MLP 三路 / lm_head 全部是 `BitLinear`。
`convert_to_bitlinear()` 只是导出时把同一套量化冻结成 int8 + fp16 scale。从来没有
"先训 fp32 再压坏"的阶段。

`model_5l32e.pt.best`，固定验证集 60 batch × 128 × 128 = 983,040 token，四臂共用同一批数据：

| 臂 | 权重 | 激活 | val CE | 相对臂 0 |
|---|---|---|---|---|
| 0 | ternary | int8 | **2.0627** | — |
| 1 | ternary | fp32 | 2.0626 | −0.0000 ± 0.0001 |
| 2 | fp32 | int8 | 7.4874 | +5.4248 |
| 3 | fp32 | fp32 | 7.4829 | +5.4203 |

- **激活 int8 钳位是免费的**：0.0001 nats。板上跑的是 fp32 激活（`g_x`/`g_xnorm`/
  `g_qkv_out` 全是 `float`），训练时假装 int8 —— 我原以为这里有免费收益，**没有**。
  127 级 absmax 对这个模型完全不紧，这条"误配"不用修。
- **fp32 权重崩到 7.49** 是"潜在权重"不可用（早已知道），臂 2/3 的作用不是量 PTQ 代价，
  而是确认**没有 PTQ 阶段的损失可以回收**。

**结论：不要再在 QAT / PTQ / 比特分配上花时间。** 剩下的只有架构（深度 / KV 共享 /
共享专家）和语料。

### 18.2 窗口 104 → 64：−21% 生成时间，质量无变化

改了两处：`INFER_CTX` 104→64；**prefill 预留从 `MAX_GEN_TOKENS` 改成 `CHAT_MAX_TOKENS`**
（`64 − 80` 是负数，不改会把整个 prompt 丢掉 —— 这是本次唯一真 bug）。

| 项 | 104 窗口 | 64 窗口 |
|---|---|---|
| 400 token 总耗时 | 59.8 s / 149.6 ms/tok | **47.2 s / 118.1 ms/tok** |
| 滑动次数 / 耗时 | 314 / 72 ms | 354 / 52 ms |
| 重编号次数 / 耗时 | 12 / 882 ms | **5 / 225 ms** |
| 分块 OOV（4 块） | 0/0/…/2.9% | 1.8/1.8/0.0/1.8% |
| Flash / RAM | 1,002,475 B | 1,002,459 B / 75,220 B |

- 省下的 ~31 ms/tok 与"每格 0.72 ms"的斜率对得上（32 格 97 ms/tok、104 格 150 ms/tok）。
- 重编号次数 12→5 是因为漂移余量 24→64（`ctx_shift + 63 >= 128` 才触发）。
- 聊天路径回归通过：4 条规则命中（0 ms，无 `[slide]`），2 条走模型
  （`what is a cloud` 95.6 ms/tok、`...shiny key...` 151.1 ms/tok），均无 `[slide]`。
- **arena 从 39,360 降到 25,760**（每层 4,352 × 5）；`s_arena_mem` 仍是固定 40 KB，
  所以 RAM 只从 75,300 降到 75,220 —— **省出来的 13.6 KB 是给第 6 层的，不是还给堆的**。
  6L 需要 4,000 + 6×4,352 = 30,112 < 40,960 ✓。
- 注意：**6L/32E 仍然装不进 flash**（1,002,459 + 61,800 = 1,064,259 > 1,044,464），
  6L 必须降到 **30 专家**（每个专家 2,946 B × 6 层 = 17,676 B，砍 2 个够）。

## 19. 代码审查发现（2026-10-22）

以下问题已由当前工作树的静态对比确认，修复前不要把当前固件视为可复现发布物。

### 19.1 P1：固件权重头文件与训练 checkpoint 架构漂移

- `src/model_weights.hpp` 当前声明：`vocab=1850`、`n_layer=5`、`block_size=128`、
  `n_experts=32`。
- `main.py` 的 ESP8266 默认配置及 `model/model_esp8266.pt` 的架构元数据是：
  `vocab=2048`、`n_layer=4`、`block_size=64`、`n_experts=20`。
- `src/main.cpp` 从头文件直接取得这些常量，因此 PlatformIO 可以成功编译，
  但烧录后的固件实际运行的是另一套模型和词表，行为不能由当前 checkpoint 复现。

修复：从明确指定的 checkpoint 重新运行导出器；在构建或 CI 中比较 checkpoint 的
`__arch__` 与 `model_weights.hpp` 的元数据，任何不一致都拒绝构建。

### 19.2 P1：换行 token ID 被硬编码，词表切换后生成终止失效

- `src/main.cpp` 将 `TOK_NEWLINE` 固定为 `1810`，这是当前中文字符词表的局部事实。
- `main.py` 默认仍使用 BPE，`convert_model_to_c.py` 也支持 BPE 和字符词表，
  但导出的头文件没有记录换行 token ID。
- 使用 BPE 模型时，生成循环不会在真正的换行 token 处停止，结束时还会把 ID 1810
  作为额外 token 写回上下文。

修复：由导出器根据实际 tokenizer 导出 `model_newline_id`，固件统一读取该值，
并为 BPE 与字符词表各做一次生成终止回归测试。

### 19.3 P2：默认 cosine 调度周期超过默认训练步数

- ESP8266 默认 `max_iters=10000`，而 `T_MAX` 默认是 `20000`。
- 训练循环在 10000 步结束时，学习率仍约为 `1.005e-3`，没有降到配置的
  `lr_min=1e-5`。这与“短 fine-tune 在预算内完成退火”的注释不一致。

修复：默认令 `T_MAX` 跟随 `max_iters`，或在每次训练启动时打印并校验最终计划的
学习率；为短 warm-start 运行增加一个覆盖退火终点的 smoke test。

## 20. 封闭域角色扮演优化基线与验收口径（2026-10-22）

本阶段目标已收敛为：**封闭域角色扮演，对话质量和角色能力优先**；允许推理速度放宽，
但固件端平均生成速度必须保持在 **250 ms/token 以内**。不再以通用知识或开放域问答作为主要目标。

### 20.1 当前基线

基线 checkpoint：`model/model_stance26.pt.best`；架构检查对象：
`model/model_stance26.pt`。固定评测命令：

```text
python check_arch_sync.py model/model_stance26.pt
python probe_stance.py --ckpt model/model_stance26.pt.best --set unseen --n 2 --show 8
```

本次记录的结果：

| 指标 | 结果 | 解释 |
|---|---:|---|
| 未见问法概念路由命中 | 35/52 = 67% | 角色概念能部分迁移到新问法 |
| 前 4 字措辞命中 | 35/52 = 67% | 路由命中时措辞基本稳定 |
| 跨概念串味 | 12 | 主要质量缺陷，优先压低 |
| 完全落空 | 5 | 需要扩大问法覆盖并改善边界样本 |
| 陌生词回避 | 12/14 = 86% | 封闭域外输入已有较好防守 |
| 输入敏感性归一化 KL | 4.937 | 高于 0.5 生死线，模型确实读取输入 |

架构检查通过；当前头文件与 `model/model_stance26.pt` 一致。ESP8266 构建资源为
RAM 75,580/81,920（92.3%），Flash 984,915/1,044,464（94.3%）。

### 20.2 统一验收集

每个候选模型都必须跑同一组指标，避免只看训练 loss：

1. `probe_stance.py`：seen/unseen 概念路由、前 4 字措辞、跨概念串味、落空。
2. `probe_react.py`：域内新句式、域外已见词、域外未见词、无名词输入。
3. `probe_zh_kl.py`：同模板换内容的 pairwise 归一化 KL，目标大于 0.5。
4. `board_stance4.py`：量化固件上的贪心纯模型路由、陌生词回避和默认采样体验。
5. 角色一致性、多轮连续性、重复率，以及板上平均 token 延迟；速度硬门槛为 250 ms/token。

### 20.3 下一步实验顺序

先做低学习率角色对话 warm start，再做架构 A/B：

- 以现有 stance/react checkpoint warm start，学习率先试 `1e-4` 到 `3e-4`，
  `ESPGPT_T_MAX=max_iters`，小步 smoke test 通过后再跑完整短微调。
- 训练侧控制对话/故事比例，避免故事开场先验重新压过角色回复；不直接引入已判定结构不匹配的外部对话语料。
- 架构候选优先比较当前部署配置与 `4L/20E`、`5L/32E`、共享专家；任何候选先过固定验收集，
  再决定是否导出和烧录。
- 只接受同时满足“角色指标不退化”和“板上平均速度 ≤250 ms/token”的候选。

本节是后续实验的基线记录。若训练结果与现有模型冲突，以固定评测脚本和板上实测为准，
不以单一训练 loss 作为行为质量结论。

### 20.4 Warm start smoke test（已通过）

2026-10-22 用以下配置验证完整链路：5L/32E、字符词表 1850、block 128、
`model/model_stance26.pt.best` warm start、`lr=1e-4`、4 步 cosine、独立输出
`model/_smoke_stance_ft.pt`。真实数据 `E:\espllm_data\persona_corpus\dataset_stance26.txt`
是带 mask 的预切分 ID 窗口，必须使用 `ESPGPT_FORMAT=ids`；误设为 `qa` 时会读到 0 个 exchange，
并在第一次 eval 时报空训练集错误。

修正格式后链路完整通过：4,841 个窗口（4,356 train / 485 val），warm start 架构校验通过，
best checkpoint、full checkpoint 和 quantized checkpoint 均成功写出，最终学习率按计划降到
`1e-5`。4 步后的探针结果为路由 16/26（62%）、前 4 字 16/26（62%）、串概念 10、
陌生词回避 7/14（50%）、KL 1.856。由于步数仅用于冒烟验证，这个 checkpoint 不进入候选排名，
正式微调仍需使用完整步数并跑完 20.2 的统一验收集。

### 20.5 Warm start 1,000 步结果：不进入部署候选

以 `model/model_stance26.pt.best` 为起点，在同一份 `dataset_stance26.txt` 上运行
1,000 步，`lr=2e-4`、warmup 50、`T_MAX=1000`，输出为
`model/model_stance27_lowlr.pt`。最佳验证 loss 从约 1.329 降到 1.298，训练、架构校验和量化导出均通过。

固定探针对比：

| 指标 | stance26 基线 | stance27 low-lr | 结论 |
|---|---:|---:|---|
| 未见问法路由命中 | 67% | 67% | 持平 |
| 前 4 字措辞命中 | 67% | 67% | 持平 |
| 跨概念串味 | 12 | 16 | 退步 |
| 完全落空 | 5 | 1 | 改善 |
| 陌生词回避 | 86% | 79% | 退步 |
| 输入敏感性归一化 KL | 4.937 | 9.722 | 改善，但可能过敏 |

已知概念集补测为路由 52/52、前 4 字 52/52、串味 0；因此退化集中在未见问法和 OOD 边界，
不是已学角色台词本身被破坏。

结论：这次训练增强了输入区分能力、减少了落空，但同时扩大了错误概念之间的串味，
并削弱 OOD 回避。因此 `model_stance27_lowlr.pt.best` 保留作实验产物，不烧录、不替换 stance26。
下一轮应优先调整正负边界样本的采样权重或加入更窄的 OOD 约束，再重复同一验收集；继续单纯延长
同一语料的微调没有证据能解决串味问题。

### 20.6 窄 OOD 加权实验：仍不进入部署候选

从 `raw_stance26.jsonl` 只抽取 S3（陌生真词）和 S4（无名词噪声），排除已知会压垮路由的
S5 伪名词，生成 600 条 extra corpus：`dataset_ood_narrow_stance26.txt`。新增工具为
`make_ood_extra.py`，打包后以 `ESPGPT_EXTRA=...:0.08` 采样，实际约占训练 token 的 1.9%。
以 stance26 warm start 运行 300 步，输出 `model/model_stance28_ood.pt`。

| 指标 | stance26 | stance28 OOD | 结论 |
|---|---:|---:|---|
| 已知概念路由 | 100% | 94% | 退步，不能接受 |
| 未见问法路由 | 67% | 69% | 小幅改善 |
| 未见串概念 | 12 | 14 | 仍退步 |
| 陌生词回避 | 86% | 64% | 明显退步 |
| 输入敏感性归一化 KL | 4.937 | 5.582 | 小幅改善 |

结论：即使只增加窄 S3/S4 负样本，正空间和 OOD 回避仍同时受损，说明模型容量/归因能力不足，
不能靠继续提高 OOD 采样比例解决。该模型不烧录、不替换 stance26。后续优先级调整为：
规则层处理已知 OOD 边界，模型只学习正空间角色表达；架构侧再评估共享专家或更高 head 数，
每次仍以已知 100%、未见不退化和 250 ms/token 为门槛。

### 20.7 共享专家 A/B：负结果

为验证 MobileMoE 式通用专家，在 `SparseMoEBlock` 中加入一个零门控共享 SwiGLU 专家，
通过 `ESPGPT_SHARED_EXPERT=1` 开启；默认值保持关闭，旧 checkpoint 可 warm start，
`gen_long.py` 已能从 checkpoint 的 `__arch__` 自动恢复该开关。5L/32E、stance26 数据、
500 步 warm start 的 PC-only 结果：

| 指标 | stance26 | shared expert | 结论 |
|---|---:|---:|---|
| 已知概念路由 | 100% | 94% | 退步 |
| 未见问法路由 | 67% | 65% | 退步 |
| 未见串概念 | 12 | 16 | 退步 |
| 陌生词回避 | 86% | 64% | 退步 |
| 输入敏感性归一化 KL | 4.937 | 7.620 | 过敏，未转化为能力 |

最佳验证 loss 约 1.306，但行为指标全面不通过，因此 `model/model_stance_shared.pt` 不进入部署，
也不改固件结构。当前证据表明，在这个 1 MB/40 KB 约束下，增加一个通用专家不能替代更强的
输入归因能力；后续优先把已知 OOD 边界交给规则层，并只用模型承担封闭域正空间表达。

### 20.8 已知 OOD 规则帧：通过构建验收

新增 `zh_ood` 规则帧，显式拦截 8 个固定域外词：天平、剪镜毯、冰淇淋、区块链、摩天轮、
火箭、邮票、比特币。规则回复使用 4 条通用角色化回避，不加入训练样本，也不泛化到所有未知中文。
`coverage_zh_rules.py` 从 60/60 扩展到 **63/63**；8 个词逐一匹配均命中 `zh_ood`。

已运行 `python gen_rules_cpp.py` 生成 `src/intent_rules_gen.h`，并通过 `pio run -e esp8266`：

- RAM：75,580 / 81,920（92.3%）
- Flash：985,275 / 1,044,464（94.3%）
- 构建成功，无新增错误

这条规则路径是 0 ms 的固定前置处理，预计不会影响 250 ms/token 的模型路径预算；新模型仍不烧录，
当前部署候选继续使用 stance26。

### 20.9 对比式训练 A/B：当前版本无效

实现训练期 supervised contrastive loss：每个预切分窗口从 `.labels` 侧文件读取概念 ID，
对 batch 内同概念窗口的 pooled hidden 做 cosine 对比；推理和固件结构不变，
`ESPGPT_CONTRASTIVE_WEIGHT=0` 时完全关闭。新增工具 `make_contrastive_labels.py`，
已生成 4,841 个窗口、27 个类的标签文件。4 步 smoke test 和 500 步正式 warm start 均通过，
输出 `model/model_stance_contrastive.pt`。

500 步、权重 0.02 的固定探针对比：

| 指标 | stance26 | contrastive | 结论 |
|---|---:|---:|---|
| 已知概念路由 | 100% | 96% | 退步 |
| 未见问法路由 | 67% | 67% | 无改善 |
| 未见串概念 | 12 | 16 | 退步 |
| 陌生词回避 | 86% | 57% | 明显退步 |
| 输入敏感性归一化 KL | 4.937 | 6.184 | 敏感性增加但未形成正确归因 |

结论：把整段 pooled hidden 拉近会强化输入差异，却没有教会模型区分“正确概念”和“相邻错误概念”，
反而损伤 OOD 边界。因此该 checkpoint 不烧录、不替换 stance26。下一次若继续对比学习，必须改成
明确的 hard-negative pair（同一问法配错概念回答）并只约束答案前的窄表示；在此之前不继续调大权重。

### 20.10 Hard-negative preference training：部分有效，未部署

新增 `make_hard_negative.py`，为每个 S1/S2 正空间问法随机选择另一个概念的答案，保持 prompt、
窗口顺序和 mask 对齐；S3/S4/S5 行复制自身答案，仅作为无效 preference 行。新增训练开关：
`ESPGPT_PREF_DATASET`、`ESPGPT_PREFERENCE_WEIGHT`、`ESPGPT_PREFERENCE_MARGIN`。
偏好损失比较正负答案 token 的平均 log-prob，推理阶段不增加结构或延迟。

4 步 smoke test 和 500 步正式 warm start 均通过，正式输出为 `model/model_stance_preference.pt`，
权重 0.02、margin 0.5，最佳验证 loss 约 1.3055。

| 指标 | stance26 | preference | 结论 |
|---|---:|---:|---|
| 已知概念路由 | 100% | 98% | 轻微退步 |
| 未见问法路由 | 67% | 71% | 改善 |
| 未见串概念 | 12 | 14 | 改善但仍高 |
| 陌生词回避 | 86% | 57% | 明显退步 |
| 输入敏感性归一化 KL | 4.937 | 5.060 | 基本持平 |

结论：hard-negative 确实改善了正空间泛化和未见问法，但随机错误概念答案同时教坏了 OOD 边界。
该模型不烧录、不替换 stance26。下一步只在 S3/S4/S5 行上构造“回避答案优于任意概念答案”的
偏好对，保持模型自主生成，不引入词表规则；若仍损伤正空间，则停止继续调损失，转向 CLA/6L 或
LittleFS 更大模型路线。
### 20.11 OOD-aware preference：回避回升但整体仍失败

修正 `make_hard_negative.py`：S1/S2 仍以错概念答案作负样本；S3/S4/S5 则以任意概念答案作负样本，
使原始回避回答成为正样本。重新生成并打包 4,841 行对齐窗口，运行 500 步 warm start，
`PREFERENCE_WEIGHT=0.01`、margin 0.5，输出 `model/model_stance_preference_ood.pt`。

| 指标 | stance26 | preference-ood | 结论 |
|---|---:|---:|---|
| 已知概念路由 | 100% | 92% | 明显退步 |
| 未见问法路由 | 67% | 71% | 改善 |
| 未见串概念 | 12 | 14 | 仍高 |
| 陌生词回避 | 86% | 71% | 比普通 preference 好，但仍退步 |
| 输入敏感性归一化 KL | 4.937 | 11.954 | 过敏 |

结论：偏好损失能移动行为分布，但无法同时保持正空间、未见泛化和 OOD 边界；继续调权重只是在
不同指标之间交换退化。该 checkpoint 不烧录、不替换 stance26。至此，当前 2.17M/5L/32E
训练目标路线已基本封闭，下一步应停止叠加损失，转向真正增加容量或归因电路的架构实验（CLA 解锁
6 层，或 LittleFS 上的 dim 128 模型）。

## 20.12 跨层 K/V 共享（CLA）PC-only A/B

实现 `ESPGPT_CLA=1` 的 PC 实验开关：每对相邻层中，后一层仍计算自己的 Q、输出投影和 MoE，
但注意力直接复用前一层的 RoPE 后 K/V 激活。当前版本保留后一层 K/V 投影参数，只验证共享激活
对行为的影响；不代表已实现参数/Flash 缩减，也未改固件。

以 `model/model_stance26.pt.best` warm start，在同一份 stance26 语料上训练 500 步，
`lr=1e-4`，输出 `model/_cla_pc.pt.best`。训练验证 loss 从 3.440 降到 1.379；这只说明模型
可适应该前向图，不能代替角色行为验收。

| 指标 | stance26 | CLA 500 步 | 结论 |
|---|---:|---:|---|
| seen 概念路由（完整 203 问，前 2 字） | 100% | 89% | 退化 |
| seen 措辞命中（完整 203 问，前 4 字） | 100% | 87% | 退化 |
| unseen 问法（每概念 2 问，52 问） | 67% | 58% | 退化 |
| 未见串概念（同上） | 12 | 22 | 明显退化 |
| OOD 回避（14 条） | 86% | 71% | 退化 |
| 输入敏感性归一化 KL | 4.937 | 2.022 | 下降但仍高于 0.5 门槛 |

结论：最简单的相邻层 K/V 激活复用会牺牲概念路由与域外边界，不能作为当前 CLA 方案继续推进到
固件。训练 loss 的大幅下降没有预测行为质量。保留 `_cla_pc` 作为实验产物，不烧录、不替换 stance26；
`ESPGPT_CLA` 默认关闭。若以后重试，应从共享 K/V 的联合训练/初始化方式重新设计，并先证明各角色指标
不掉，再考虑真正删除重复 K/V 参数、改变固件 KV cache 布局或增加层数。

## 20.13 6L/20E、窗口 64 角色模型 A/B

按当前 Flash/RAM 预算，6L/30E 的量化导出包约 1.24 MB，超出可寻址 Flash；改测能够放入现有
Flash 的 6L/20E。先做 20 步从头训练 smoke test，随后在同一 stance26 预切分数据上从头训练
3,000 步，block size 64、batch 32、lr 0.002、warmup 300，输出 `model/model_6l20_stance.pt`。

训练验证 loss 最低为 1.3049。角色行为固定探针：

| 指标 | stance26 | 6L/20E | 结论 |
|---|---:|---:|---|
| seen 概念路由（完整 26 概念） | 100% | 100% | 持平 |
| seen 前 4 字措辞命中 | 100% | 100% | 持平 |
| unseen 问法路由（52 问） | 67% | 65% | 略降 |
| unseen 跨概念串味 | 12 | 18 | 退化 |
| 完全落空 | 5 | 0 | 改善 |
| OOD 回避（14 条） | 86% | 93% | 改善 |
| 输入敏感性归一化 KL | 4.937 | 7.616 | 高于 0.5，但变得更敏感 |

6L/20E 没有同时改善未见问法和串概念，暂不进入部署候选。量化 checkpoint 为 985.1 KB；
临时头文件 `src/model_weights_6l20_stance.hpp` 的架构同步检查通过。预算脚本估算其权重 + 代码
约 834 KB，KV/arena 需求 30,112 B（40,960 B 上限内），每 token 权重流量约 173,552 B，按
当前 118.1 ms/token 基线线性估算约 120 ms/token。尚未使用临时头文件做 PlatformIO 板端构建或
实测延迟，因此这个速度只是估算。6L/30E 的短 smoke 量化包超 Flash，未继续训练。

## 20.14 失败驱动多轮补集与可选历史上下文

6L/20E 说明单纯增加深度不能消除未见问法的串概念，因此转向失败驱动的多轮承接。新增
`make_failure_multiturn.py`，从 26 个概念的现有问法/答案组合出一轮明确主题、一次角色回答、
一个不含主题词的追问；追问要求模型利用上一轮上下文，而不是靠域外词表或推理规则。

生成 `E:\espllm_data\persona_corpus\raw_failure_multiturn_stance26.jsonl` 共 748 条，64 条因
字符或 64-token 窗口超限丢弃。`pack_zh_corpus.py` 打包为
`dataset_failure_multiturn_stance26.txt`：748 窗口，长度 p50=60、p90=64、max=65，UNK=0，
prompt mask 只训练最后一轮回答。

以 stance26 warm start、补集占 12% 窗口（约 6% token）、lr=1e-4 训练 500 步，输出
`model/_stance_mt12.pt.best`。固定单轮探针（每概念 1 个 unseen 问法）为：seen 26/26，
unseen 20/26=77%，OOD 12/14=86%，归一化 KL=3.535；完整多轮 PC 探针中，基线
`model/model_stance26.pt.best` 只命中 8/26=31%，补集模型命中 22/26=85%。这证明补集确实
教会了模型在给定历史下承接追问，但单轮 OOD 没有提高，不能单凭这个结果替换部署模型。

此前固件 `run_chat()` 每轮都会 `clear_context()`，因此训练出的历史无法落地。现在加入默认关闭的
`:history 0|1` 实验开关：开启后保留上一轮 User/Bot token，已有 KV 时用 `ctx_slide()` 裁剪，
没有 KV 时重置 prefill；规则和算数回复也会写回历史。为给 64-token 窗口留下上下文，历史模式
的回复预算降为 20 token，默认单轮仍为 40 token。编译验证通过：RAM 75,600/81,920，Flash
985,715/1,044,464。尚未烧录，尚未在板上用 `_stance_mt12` 权重验收；当前板上头文件仍是
stance26，下一步应导出补集模型并做 `:history 1` 的双轮串口测试，再测平均 token 延迟。

## 20.15 多轮候选导出与临时固件构建

将 `model/_stance_mt12.pt.quantized` 导出为临时头文件
`src/model_weights_mt12.hpp`。导出时发现 `convert_model_to_c.py` 使用系统默认
Windows 编码写文件，注释中的长横线被写成 `A1 AA`，导致 UTF-8 的
`check_arch_sync.py` 无法读取；已将生成文件固定为 UTF-8（并使用 LF 换行），重新导出后问题消失。

`check_arch_sync.py` 已确认临时头文件与 `_stance_mt12.pt` 完全匹配：1850/64/5L/2-head/1-KV/128
窗口/64 group/64 hidden/32 experts。将该头文件临时替换现役 `src/model_weights.hpp` 后，
`pio run -e esp8266` 构建成功：RAM 75,600/81,920（92.3%），Flash 985,715/1,044,464（94.4%）。
构建完成后已恢复现役 stance26 头文件；没有烧录，也没有把 `_stance_mt12` 标成部署模型。

本次构建仍有已有的 `snprintf` 截断警告（历史/规则回复写回缓冲区），不影响编译成功，但板端
双轮测试前应继续确认历史长度和回复长度不会在真实输入下破坏上下文。下一步是烧录临时固件后用
`:history 1` 做青蛙双轮承接测试，并实测平均 token 延迟是否满足 250 ms/token 门槛。

## 20.16 `_stance_mt12` 板端验收：不替换正式固件

按 20.15 的计划，将 `src/model_weights_mt12.hpp` 临时装入并烧录到 COM5，使用
按 bytes 累积的 `probe_board_history.py` 复测，避免串口分块解码中文造成误判。第一轮
“青蛙怎么不动了”正常回答，17 token、127.4 ms/token；第二轮“它还活着吗”只有回显和
换行，1 token、19.0 ms/token，没有形成有效角色回答。多轮承接验收失败。

补充单轮板测（`:history 0`、`:rules 0`、`:greedy`）结果：青蛙 17 token、127.4 ms/token；
“你会飞吗”11 token、140.3 ms/token；“讲个冷笑话”14 token、132.8 ms/token。说明候选
的速度满足 250 ms/token，但不能把单轮速度或 PC 多轮探针 85% 直接当作板端历史能力。

因此已重新烧回 `stance26` 正式固件，并用青蛙单轮复测确认启动正常：14 token、142.7 ms/token，
回答“冻的，结成一整块绿石头”。`_stance_mt12` 保留为实验产物，不替换现役模型；失败原因是
板端历史上下文下第二轮提前结束，而不是速度或 Flash/RAM 超限。烧录期间按授权停止的
`qq_bridge.py` 已从 `qqbot/` 目录重新启动，当前进程正常运行。

## 20.17 修复历史滑窗后重新验收：`_stance_mt12` 留在板上

复盘 20.16 的失败后确认，问题不在模型：第二轮追加的新 prompt token 尚未经过模型，
但原逻辑在预算裁剪时把它们和已有 KV 一起送入 `ctx_slide()`，再把未初始化的行标成
已缓存。修复 `src/main.cpp`：只有 `ctx_pos == ctx_len` 时走完整 `ctx_slide()`；已有
缓存前缀、未缓存 prompt 尾部时走新的 `ctx_slide_pending()`，只移动已缓存行并保留尾部
待 prefill。PlatformIO 构建通过，Flash 986,067/1,044,464，RAM 75,600/81,920。

修复后先在 stance26 上验证：历史双轮从 1 token 立即结束恢复为完整回答，第二轮
170.7 ms/token。随后将 `_stance_mt12` 重新烧录并做三轮同一场景验收：

| 轮次 | 结果 | 速度 |
|---|---|---:|
| 青蛙怎么不动了 | 完整回答 | 127.4 ms/token |
| 它还活着吗 | 完整承接回答 | 178.9 ms/token |
| 你还会再冻它吗 | 完整承接回答 | 95.7 ms/token |

三轮均未提前结束，全部低于 250 ms/token。结合此前 PC 多轮探针 22/26=85%、单轮
unseen 20/26=77%（stance26 分别为 8/26 与 67%），确认 `_stance_mt12` 在当前
固件修复后优于 stance26 的多轮承接能力，现已作为正式板上模型保留。`src/model_weights.hpp`
当前与 `_stance_mt12` 架构同步；`qq_bridge.py` 已重新启动。

## 20.18 GRU + attention 架构候选：盲化单轮偏好待评

PC-only 训练了 1.94M 参数的双向 GRU 编码器、加性注意力和 GRU 解码器，字符词表 1,850。
在 stance26 与失败驱动多轮语料上训练 3,000 步，最佳验证 loss 为 step 750 的 0.3091，
后续验证 loss 上升，存在过拟合。固定探针中，GRU seen 路由 26/26、unseen 21/26、OOD
12/14、多轮前缀命中 26/26；当前正式 Transformer `_stance_mt12` 对应为 26/26、20/26、
12/14、22/26。全量 GRU 探针另得 seen 196/203、unseen 59/78、OOD 12/14。

12 条固定单轮问题的贪心输出已做 A/B 隐藏模型身份评测，见 `docs/gru_architecture_blind_eval.md`；
揭盲表单独保存在 `docs/gru_architecture_blind_eval_key.md`。这批问题有 3 对输出完全相同，
其余 9 对用户选择 GRU 2 次、Transformer 1 次、平手 6 次。样本规模小且平手占多数，
没有显示明确的单轮体验赢家；GRU 的 unseen 路由多命中 1/26，不足以证明用户体验改善。
因此 `_stance_mt12` 继续作为正式模型，GRU 保留为 PC-only 架构实验。
GRU 仍是 PC-only 实验，未测 ESP8266 延迟、未移植、未烧录。若后续出现更强的体验证据，
也需先验证板端可部署性和 ≤250 ms/token，再决定是否替换 `_stance_mt12`。

## 20.19 GRU 多轮体验盲评：完成，暂不替换

使用 `raw_failure_multiturn_stance26.jsonl` 中每个概念按文件顺序遇到的第一条样本，共 26 组。
给两个模型输入完全相同的前文与追问，均用贪心解码、最多 24 token；Transformer 使用窗口 128，
GRU 使用其 PC 推理实现。A/B 身份由固定种子随机分配，揭盲表与问卷分开保存：
`docs/gru_architecture_multiturn_blind_eval.md` 和
`docs/gru_architecture_multiturn_blind_eval_key.md`。用户评分揭盲后，26 组中 GRU 获选 6 次、
Transformer 5 次、平手 15 次；在有明确偏好的 11 组里仅以 6:5 微弱领先，主观体验没有明确赢家。
结合 GRU 前缀路由探针 26/26 对 22/26 的客观优势，说明可测的命中提升尚未转化为稳定的用户体验
优势。GRU 不进入固件移植；`_stance_mt12` 继续作为正式模型。此轮只生成和评估了 PC 输出，
没有板端移植、烧录或速度结论。

## 20.20 困难样本偏好集：已生成，尚未训练

新增 `build_hard_preference_set.py`，从 `raw_stance26.jsonl` 和
`raw_failure_multiturn_stance26.jsonl` 按概念抽取样本，再分别运行 `_stance_mt12` 与
`_gru_attn_stance26.pt` 的贪心解码。按每个概念、每种样本类型保留 5 条高难度样本，优先级为
模型分歧、Transformer 概念路由失败、GRU 路由失败和参考答案前缀不匹配。

本次生成 275 条候选：单轮 145 条、多轮 130 条，覆盖当前语料中的 29 个概念。完整 JSONL
保存在 `docs/gru_transformer_hard_cases.jsonl`，可人工填写的标注表为
`docs/gru_transformer_hard_cases.md`。该数据集仍是候选材料，尚未把参考答案直接当作偏好标签，
也尚未训练新 checkpoint；这样可以避免把语料中的偶然答案或模型错误自动固化。

## 20.21 困难样本加权 warm start：多轮改善，单轮不进候选

将完整 stance26 与失败驱动多轮语料保留，并把 275 条困难样本重复 4 次，形成
`E:\espllm_data\persona_corpus\dataset_hard_weighted.txt`（6,689 窗口，UNK=0）。
从 `_stance_mt12.pt.best` warm start，以匹配的 5L/32E/1850 架构、学习率 `5e-5` 训练 400 步，
输出 `model/_hard_weighted_400.pt`。训练验证 loss 由约 1.3091 降到 1.3020，改善很小。

固定行为对比：

| 指标 | `_stance_mt12` | `_hard_weighted_400` | 判断 |
|---|---:|---:|---|
| unseen 路由（52 条） | 36/52 | 36/52 | 持平 |
| unseen 串概念 | 14 | 14 | 持平 |
| OOD 回避 | 12/14 | 13/14 | 改善 1 条 |
| 多轮前缀命中 | 22/26 | 26/26 | 改善 4 条 |

困难集参考答案前缀命中为 262/275（95.3%），其中单轮 133/145、多轮 129/130；这个口径
只能说明更贴近现有参考回复，不能代替人工偏好。由于单轮 unseen 和串概念没有改善，且本阶段
单轮质量优先，`_hard_weighted_400` 不进入正式候选、不导出、不烧录；保留为多轮训练方向的
实验产物。后续若继续，应先对困难集做人工优选，再进行真正的偏好微调。

为降低标注成本，新增 `make_preference_review_sheet.py`，从 275 条候选中按每个概念和样本类型
选取最高难度的一条，生成 55 条 A/B 隐藏身份的人工评测：
`docs/gru_transformer_preference_review.md`；揭盲表为
`docs/gru_transformer_preference_review_key.md`。这 55 条覆盖单轮与多轮困难样本，评测时不展示
参考答案，避免把人工判断退化成字面匹配。标注完成后才进入偏好微调。

为避免手写 55 个序号，新增 `docs/gru_transformer_preference_review.html` 交互页面。
本地服务地址为 `http://127.0.0.1:8765/docs/gru_transformer_preference_review.html`；
点击 A/B/平手后自动进入下一题，支持键盘 `1/2/3`、上一题回改、浏览器本地保存进度，完成后
可复制结果字符串。页面已实际打开并验证自动换题。

## 20.22 困难样本人工偏好评测：Transformer 轻微领先

用户完成交互式 55 题评测，并明确将“两个都还行”和“两个都不好”都记为平手。统计为：
Transformer 20、GRU 14、平手 21。完整原始序列和说明见
`docs/gru_transformer_preference_review_result.md`。

平手占 21/55，明确偏好的 34 题中 Transformer 仅以 20:14 领先，不能证明稳定的体验差异。
结合此前单轮、多轮盲评都未显示 GRU 优势，GRU 不进入固件候选，`_stance_mt12` 继续作为正式
模型。该结果可作为后续偏好微调的人工基线。

## 20.23 四分类质量评测页：已完成部署

上一轮将“两个都还行”和“两个都不好”合并为平手，只能比较架构，不能直接形成训练标签。
因此新增 `docs/gru_transformer_quality_review.html`，沿用同一批 55 条困难样本，但改为四分类：
`A 更好`、`B 更好`、`都还行`、`都不好`。页面使用独立的本地进度键，不覆盖上一轮结果；键盘
快捷键为 `1/2/3/4`，选择后自动进入下一题，完成后显示可复制的结果序列。

这轮标注完成后：A/B 选择用于相对偏好，`都还行` 作为无偏好样本，`都不好` 进入重写/负例池，
三者不再混合统计。

## 20.24 四分类质量评测：双差多于架构偏好

用户完成全部 55 题。原始选择为 A 20 次、B 12 次、`都还行` 4 次、`都不好` 19 次；按揭盲表
还原模型后，GRU 获选 17 次、Transformer 15 次。详细结果和原始序列见
`docs/gru_transformer_quality_review_result.md`；逐题结构化标注为
`docs/gru_transformer_quality_labels.jsonl`。
只看 32 个明确偏好样本，GRU 以 17:15 微弱领先；该规模不足以确认架构优劣。19/55 的双差
说明困难集中在候选回复本身质量不足，支持把后续资源投入更好的角色回复数据，而不是替换架构。
`都还行` 不作为偏好训练标签；`都不好` 不能直接当作负样本惩罚，因为没有经过验证的优选答案。
逐条人工改写这 19 个样本暂缓：其边际收益局限于少数问题，人工成本与当前目标不匹配。除非批量
数据、训练目标、tokenizer/表征和推理策略等更高杠杆路线都已验证无效，否则不进入逐题雕花。

后续优先评估能覆盖整个封闭域的方案：批量生成并筛选角色对话进行教师蒸馏，或在参数/Flash预算
内对照中文子词 tokenizer 与字符 tokenizer。评估必须使用未参与训练的单轮、多轮盲测和 OOD 集，
避免只验证生成语料本身。

## 20.25 批量教师蒸馏首轮验证：接口暂不可用

按 8 条小批次分别调用已配置的七牛接口和 Gemini 接口，执行 `probe_api_batch.py 8 QN`
与 `probe_api_batch.py 8 GEMINI`。两次请求均返回 HTTP 502，未取得教师回复，因此本轮没有
可统计的 JSON 合法率、角色一致性、句式多样性或重复率；这不是对蒸馏路线或教师质量的否定。

下一步应先恢复任一教师接口，再用 8-16 条验证结构化输出与角色质量；只有小批输出通过质量门槛，
才评估扩大量和去重筛选。生成样本须与训练集及固定单轮/多轮/OOD 留出评测隔离，未经筛选的
教师输出不直接进入训练。本轮未训练模型、未导出或烧录固件，正式模型仍为 `_stance_mt12`。

后续按用户建议，以进程级 `GEMINI_MODEL` 覆盖值追加探测同一 GEMINI 接口：
`glm-5.3-flash` 返回 HTTP 502，`deepseek-v4.1-flash` 返回 HTTP 404；均未取得生成内容。
持久化 API 配置未修改。当前证据更指向中转服务故障或模型名/路由未配置，尚不能比较两者的
蒸馏质量；需先从服务侧确认可用模型 ID 与路由状态，再重试小批验证。

## 20.26 DeepSeek 官方 API 小样本：格式与角色锚点通过，事实质量需筛选

用户提供 DeepSeek API 凭据后，直接调用官方端点 `https://api.deepseek.com/chat/completions`，模型
`deepseek-flash`，生成 8 条单轮候选。首个请求虽 HTTP 200，但 PowerShell 管道编码损坏中文，作废；
修正为 UTF-8 后重试成功。有效请求耗时约 10.0 秒，prompt 189 tokens、completion 1,680 tokens，
其中 reasoning 1,409 tokens。

有效样本 JSON 可解析，8/8 字段齐全、回复均包含“咱”、回复互不重复，平均回复长度 16.1 字，均在
10-24 字区间。话题覆盖青蛙、算数、灵梦、夏天、外号、大妖精、冰屋、迷路。算数题“三加五”答成
“七”符合琪露诺的⑨人设，应计为角色一致性的正向表现，而非事实错误或筛选缺陷。后续质量筛选需
区分有意的人设化表现与无意的设定冲突，不能用通用事实正确性规则抹掉角色特征。该结果支持继续做
小规模采样与筛选器验证，尚不足以扩到大规模生成或训练。
凭据未写入项目文件；本轮仍未训练、导出或烧录，正式模型保持 `_stance_mt12`。

## 20.27 DeepSeek Flash 16 条批量与自动质检：管线可用，质检尚未证明有筛选力

修正 UTF-8 后再次调用官方 `deepseek-flash`，一次生成 16 条，关闭 thinking。请求耗时 4.29 秒，
721 tokens（prompt 190、completion 531），未产生 reasoning tokens。JSON 可解析，字段齐全；8 个主题
各 2 条，回复 16/16 不重复、长度 14-16 字（均值 14.8），开头前三字也都不同。14/16 回复含“咱”，
显示即使提示避免机械插入，模型仍偏向重复显式自称；后续应关注自然度，不能只用锚点命中率评估。

随后用同一模型、独立质检提示审阅 16 条，耗时 2.73 秒、1,248 tokens，16/16 放行，算术答错也被
正确识别为符合人设。这个结果说明提示能避免把“⑨”特征误杀，但质检器与生成器同源且本批没有拒绝项，
不能证明它有足够筛选力或能代替独立质量评测。当前结论：小批生成、JSON 解析和批量调用可行；扩大前
仍要优化降低自称重复，并用固定单轮/多轮盲测验证训练后的体验增益。尚未扩充训练语料、训练或烧录，
正式模型保持 `_stance_mt12`。

## 20.28 显式自称提示 A/B：数量略降，质量无可辨差异

固定同一组 16 个问题，使用 `deepseek-flash` 分别生成两组。A 沿用“自称咱、避免机械重复”的提示；
B 明确说明中文对话常省略主语、不要求每条显式说“咱”，其余角色设定、温度和长度限制相同。
两组均为有效 JSON，回复都 16/16 唯一、前三字无重复，长度均 16/16 符合 10-24 字。A 显式“咱”
16/16，B 为 13/16；平均回复长度分别 19.6 和 17.2 字。

将每题两条回复随机左右排列后交给 `deepseek-flash` 盲比，A 获选 6、B 获选 6、平手 4。小样本及同源
评审不能证明统计等价，但没有看到 B 的质量优势；提示只带来 3 条显式自称减少。结论是不再围绕自称
频率做细调，采用允许自然省略主语的宽松表述即可。后续重点应是扩大多样化封闭域对话数据、完成自动
结构筛查，再以固定单轮/多轮盲测评估训练后体验；此轮未将任何样本加入训练、未训练或烧录。

## 20.29 DeepSeek Flash 批量蒸馏 A/B：当前配方未验证出收益

将隔离生成的 1,024 条候选打包为 `E:\espllm_data\persona_corpus\dataset_distill_deepseek_flash_1024.txt`，
全部窗口通过词表打包检查（UNK=0），保留 prompt mask。它作为额外采样池训练，不改动正式训练集或
`model/_stance_mt12.pt.best`。对照组和实验组均从同一正式 checkpoint warm start，使用同一 stance26 主集、
失败驱动多轮集 12% 补采样、随机种子、学习率 `5e-5` 和 400 步；实验组额外加入蒸馏集 8% 窗口采样
（约 2.81% token）。产物、量化副本和日志隔离保存在
`E:\espllm_data\persona_corpus\distill_ab_20261023\`。第一次控制组运行因 `PATIENCE=0` 在 100 步停止，
未用于比较；随后以 patience=10 重跑完整 400 步。正式 checkpoint 与固件均未覆盖。

训练集验证 loss：正式模型 1.3082；完整控制组最低 1.3010；蒸馏组最低 1.3063。loss 仅作训练诊断，
不能代表对话质量。固定贪心探针（unseen 每概念 1 问、OOD 14 问、多轮每概念 1 条）结果如下：

| 模型 | 单轮 unseen 路由 | OOD 回避 | 多轮前缀命中 |
|---|---:|---:|---:|
| `_stance_mt12` 正式模型 | 20/26 | 12/14 | 22/26 |
| 同步训练控制组 | 19/26 | 12/14 | 24/26 |
| 蒸馏实验组 | 20/26 | 12/14 | 23/26 |

蒸馏组相对同步控制组单轮多 1 条、多轮少 1 条，OOD 持平；这组小型自动探针没有显示一致收益，
不足以支持人工盲评或固件替换。结论：本次 DeepSeek Flash 语料规模与 8% 窗口采样、400 步训练组合
暂不进入候选。无需继续逐题审查这批结果；后续优先检验更高杠杆的数据覆盖或训练方法，蒸馏只有在能
先显示稳定的自动行为增益时才值得投入更大评测成本。

## 20.30 中文子词 tokenizer 可行性 pilot：收益不足以迁移

新增 `probe_zh_bpe_feasibility.py`，用现有 TinyStories-Zh 字符窗口和 stance26 对话训练 1,850 项
ByteLevel BPE，在未参与 tokenizer 训练的 1,024 条 DeepSeek 对话上测 token 数与解码完整性。留出集
字符级基线为 43,519 tokens，ByteLevel BPE 为 39,466，减少 9.3%；1024/1024 条完整往返还原。
但 1,850 个 BPE 词表项中，393 个单 token 解码会出现 U+FFFD（21.2%），板端若采用此 tokenizer
还需增加字节片段缓冲与输入编码路径。9.3% 的序列压缩不足以抵消部署复杂度，且本次没有训练 BPE
语言模型，未证明对回复质量有提升。因此不迁移词表、不训练模型；pilot tokenizer 与统计保存在
`E:\espllm_data\persona_corpus\subword_pilot_1850\`。这只关闭了 ByteLevel BPE 这一实现，不代表
所有 Unicode 子词方案都已验证。

## 20.31 Unicode 字符片段 tokenizer 预算扫描：压缩与覆盖无法兼得

新增 `probe_zh_unicode_subword.py`，保留训练角色语料与模板中的全部字符，在 1,850 项固定词表内，
扫描中文故事字覆盖和最长匹配汉字片段之间的预算分配。片段只从 TinyStories-Zh 和 stance26 训练来源
统计；留出仍为 1,024 条 DeepSeek 对话。所有词表项均为完整 Unicode 字符串，不存在 ByteLevel BPE 的
单 token UTF-8 碎片问题。

| 新增故事单字 | 片段 token | token 压缩率 | 留出字符 UNK 率 | 可逐字还原条数 |
|---:|---:|---:|---:|---:|
| 160 | 711 | 21.18% | 0.882% | 740/1,024 |
| 320 | 551 | 20.73% | 0.637% | 802/1,024 |
| 480 | 391 | 20.10% | 0.519% | 838/1,024 |
| 640 | 231 | 18.29% | 0.216% | 941/1,024 |
| 800 | 71 | 10.13% | 0.051% | 1,005/1,024 |

没有配置同时达到至少 15% 压缩与低于 0.1% UNK。高压缩配置会丢失留出对话字符；覆盖更稳时，
压缩收益降至 10.13%。因此不做词表迁移或模型训练，Unicode 片段方案暂不进入模型级 A/B。完整词表
与统计保存在 `E:\espllm_data\persona_corpus\unicode_subword_pilot_1850\`；这轮结果覆盖了当前提议的
最长匹配字符片段方案，不代表任何可能的子词学习算法都已排除。

## 20.32 LittleFS 稀疏专家扩容 A/B：48E 未显示稳定体验收益

从正式模型 `_stance_mt12.pt.best` 克隆路由与专家，扩为 48E，并确认扩容前后初始 logits 最大差值为 0。
32E 对照组和 48E 候选组均使用同一 stance26 主集、失败驱动多轮语料 12% 窗口采样、相同随机种子与
400 步 warm-start 配方；产物和日志隔离在
`E:\espllm_data\persona_corpus\expert_capacity_ab_20261023\`。两组最低验证 loss 分别为 1.3010 和
1.3008，差值仅 0.0002，不能据此判断回复质量。

固定贪心探针结果如下。单轮未见概念每概念 1 题、seed=7；OOD 为固定 14 词；多轮为同一份 26 概念
失败驱动样本，每概念 1 条：

| 模型 | 单轮 unseen 路由 | OOD 回避 | 多轮前缀命中 |
|---|---:|---:|---:|
| 同步训练 32E 对照 | 20/26 | 13/14 | 24/26 |
| 扩容训练 48E 候选 | 21/26 | 11/14 | 24/26 |

48E 单轮多命中 1/26，但 OOD 少回避 2/14，多轮完全持平。这个方向没有形成一致行为增益，因此不继续做
LittleFS 权重读取、板端移植或速度测试，也不导出或烧录 48E；当前正式 checkpoint 和 `_stance_mt12` 固件
保持不变。结论只覆盖本次 48E 克隆扩容和 400 步训练配方，不排除其他稀疏模型设计。

## 20.33 长历史训练试验：数据格式错误，结果作废

初版 `make_history_distractor.py` 在旧对话后拼接焦点问句时漏了 `User: `，把
`Bot: <旧回复>\n<焦点问句>` 写入训练和测试上下文，偏离正式对话格式。虽然打包时 UNK 与窗口长度检查
通过，角色结构检查未覆盖此问题；该版生成的训练结果、单轮/OOD 指标及其比较全部无效，不作为实验结论。
修正后的生成器会在焦点问句前保留完整 `User:` 标记。此作废试验的候选 checkpoint 只留作隔离产物，未烧录。

## 20.34 推理上下文扩展与最近轮次保留：板端通过实验验收

固件模型 block size 为 128，但 ESP8266 原运行窗口固定为 64；历史模式再为回复预留 20 token，导致提示最多
保留 44 token。用正式 `_stance_mt12.pt.best` 和 52 条按概念留出的三轮样本比较 64 与 104 窗口，窗口
均按板端回复预算预留规则模拟。提示格式为标准 `User:/Bot:`；另加一条旧话题作为干扰，焦点对话与省略追问
保持不变：

| 设置 | 短历史承接 | 保留完整旧话题时的承接 |
|---|---:|---:|
| 64 窗口（提示上限 44） | 42/52 | 42/52 |
| 104 窗口（提示上限 84） | 46/52 | 34/52 |

更大的窗口单独使用会把旧话题一并保留，反而干扰焦点指代；104 窗口下淘汰最早完整对话轮次后，提示与
短历史条件相同，承接为 46/52。于是 `src/main.cpp` 将 ESP8266 缓存窗口提高到 104，并在 `:history 1`
路径下新一轮到来时删除最早的完整 `\nUser: ` 轮次，只保留最近一轮。该策略不识别话题或域外词，也不改模型权重；
短历史分布与训练格式保持一致。默认关闭的单轮模式不受历史裁剪逻辑影响。

`pio run -e esp8266` 与 `pio run -e esp8266 -t upload` 均成功；构建 RAM 75,680/81,920，Flash
986,355/1,044,464，104 窗口 arena 静态检查通过。COM5 板测中，连续青蛙三轮为 127.4、197.2、200.9 ms/token；
“食物（旧话题）→ 青蛙 → ‘它还活着吗’”末轮保留青蛙语义，三轮分别为 139.5、190.9、181.8 ms/token。
所有生成速度均低于 250 ms/token。`qq_bridge.py` 已在板测后重新启动；正式模型权重未变，当前固件包含此
历史模式实验实现，仍需实际长时间对话体验确认后再决定是否作为长期默认策略。

## 20.35 七轮历史模式板测：缓存与速度通过，长程承接仍弱

使用正式 `_stance_mt12` 固件，在 COM5 以 `:history 1`、`:slide 1`、关闭意图规则和贪心解码连续运行
七轮青蛙对话，验证重复淘汰旧轮后 KV 缓存仍可继续生成。七轮均有完整提示符和非空回复，没有缓存失步或
立即终止；每轮速度为 127.4、197.2、235.8、222.8、235.3、226.0、203.6 ms/token，全部低于
250 ms/token，但两轮已接近上限。

质量上，前两轮仍提及青蛙及冻结设定；第三轮开始逐渐转为通用角色台词，后续不能稳定保留用户刚提供的
“放在太阳下”等情节细节。它说明历史裁剪和缓存可以连续工作，却不足以证明长程对话体验改善；单轮质量和
角色口吻没有明显回归，但此实验不构成长期默认开启历史模式的依据。复现实验脚本为
`probe_board_history_long.py`。板测期间临时停止的 QQ 桥接已用原配置从 `qqbot/` 目录重新启动，进程
85824 为启动器、24704 为实际桥接进程。

## 20.36 问题对齐蒸馏 A/B：未显示体验收益（2026-10-24）

针对“模型会复用熟悉回复，但不善于按用户问题重新组织回答”的现象，新增
`generate_alignment_distill.py`，让教师按事实、解释、办法、情绪回应、质疑、邀请、纠错、选择、细节承接和
域外坦率回应等互动任务生成角色对话。完成 576 条候选；全部问句和回复唯一，10 类均有覆盖。词表、长度、
标点、自称和禁用表达的机械筛查累计拒绝 58 条候选。该筛查只保证格式与基本约束，不等于语义质量审核。

用固定 seed=20261024 分层切出 512 条训练样本和 64 条独立留出问题。训练文件 UNK=0、最大窗口 81 token；
留出问题未进入训练池。控制组和实验组都从正式 `_stance_mt12.pt.best` 起步，采用相同 stance26 主集、
失败驱动多轮集 12% 补采样、随机种子、学习率 `5e-5`、warmup 20、400 步和 5L/32E 架构。实验组额外
加入新蒸馏池 8% 窗口采样（约 3.47% batch token）。产物保存在
`E:\\espllm_data\\persona_corpus\\alignment_distill_ab_20261024\\`；正式权重、固件头文件和板端状态均未改动。

64 条留出问题用同一 board-style greedy 解码生成回复，随后将 A/B 身份随机互换交给 Kimi K3 做盲评。候选组
获选 10，控制组 8，平手 29，两者都不好 17。四项 0–2 分均值（候选/控制）为：切题 `0.547/0.547`、
利用用户细节 `0.328/0.250`、回答形式 `1.000/1.047`、角色口吻 `1.422/1.469`。两组都仍大量复用通用回复；
非空率均为 64/64，独立回复数为 31/30。细节分差只有 0.078，切题完全持平，且多数题为平手或两者都差。

既有固定贪心回归（正式模型 / 同步控制 / 对齐实验）为：unseen 概念路由 `20/26 / 20/26 / 20/26`，OOD
回避 `12/14 / 13/14 / 12/14`，多轮前缀命中 `22/26 / 24/26 / 22/26`。对齐实验相对同步控制没有单轮路由收益，
多轮少 2/26；主验证 loss 最佳为初始 `1.3063`，训练至 400 步后为 `1.3334`。本轮没有形成足以抵消回归的
体验提升，因此不继续扩大该配方，不做板端延迟测试，也不替换正式模型。评测仍有局限：生成数据和盲评来自同一
Kimi K3 系列，留出问题虽未参加训练但仍由同一生成管线构造；当前证据可以否决“已证明有收益”，不能排除所有
其他蒸馏方法。

复现/产物：`pack_alignment_distill.py`、`eval_alignment_holdout.py`、`judge_alignment_holdout.py`；训练权重、
量化副本、64 条预测及逐题盲评结果均在上述隔离目录中。

## 20.37 对齐蒸馏采样剂量 A/B：提高到 24% 仍无收益（2026-10-24）

为区分 20.36 的失败是否只因训练暴露不足，复用同一 512 条训练池、64 条留出集和完全相同的正式 checkpoint、
stance26 主集、多轮集 12%、seed、学习率与 400 步配方，只把蒸馏窗口采样从 8% 提高到 24%。此时蒸馏样本
约占 batch token 的 11.56%，400 步预计重复抽取约 6 遍训练池。产物保存在同一隔离目录的 `alignment24.*`，
没有覆盖 8% 组或正式模型。

同一 64 条留出题的盲评结果（24% 候选 / 同步控制）为：候选获选 9、控制获选 10、平手 34、两者都不好 11。
切题均分为 `0.438/0.453`，细节均分 `0.172/0.172`，回答形式 `1.125/1.141`，角色口吻 `1.266/1.312`；
候选回复仍为 64/64 非空、31 种不同回复。相较 8% 组，增加采样没有带来偏好或细节利用提升。

24% 组验证 loss 在 step 0 为 `1.3066`，400 步后升至 `1.3696`。固定回归为 unseen `20/26`、OOD 回避
`12/14`、多轮 `22/26`：与正式模型持平，较同步控制组分别少 1 个 OOD 命中和 2 个多轮命中。提高剂量没有
解决切题问题且扩大了旧任务上的损失，故停止这条“仅提高教师蒸馏采样比例”的路线；不做板端测试或固件替换。
下一轮应改变训练信号本身，先验证模型能否根据用户细节区分回应，而不是继续堆同类单轮样本。额外预测与盲评文件为
`holdout_predictions_24.jsonl`、`holdout_judgments_24.jsonl`。

## 20.38 同主题反事实 preference pilot：否决该训练信号（2026-10-24）

为让模型在相同主题下学习区分更合适的回答，新增 `build_detail_preference.py`，从 20.36 的 512 条训练样本构造
512 对正负窗口：优先保持同一任务，负例从其他回复中选择字符重合度最低者。两组都从正式 `_stance_mt12.pt.best`
开始，使用同一训练池、学习率 `5e-5`、400 步和 5L/32E 架构；control 不加 preference loss，实验组使用
`ESPGPT_PREFERENCE_WEIGHT=0.01`、`ESPGPT_PREFERENCE_MARGIN=0.5`。两个 checkpoint 均只保存在
`E:\\espllm_data\\persona_corpus\\alignment_distill_ab_20261024\\`，未覆盖正式模型。

64 条留出题使用相同 board-style greedy 解码并盲评。preference 候选获选 13，control 获选 19，平手 8，两者都不好
24。切题均分为 `0.281/0.359`，利用用户细节 `0.109/0.156`，回答形式 `0.672/0.672`，角色口吻
`0.938/0.906`（候选/控制）。候选产生 61 种不同回复，control 为 59 种，但多样性没有转化为更好的对齐。

固定回归同样没有形成一致收益：unseen 概念路由为 control `4/26`、preference `5/26`，多轮前缀命中为
control `3/26`、preference `5/26`；两项均远低于正式模型/同步控制的历史水平，且不能解释盲评中的质量下降。该
pilot 说明“同主题、低重合负例 + 小权重 preference loss”没有解决模型只复用熟悉回复的问题，反而损害切题和
细节利用，因此正式否决这条训练信号，不扩大数据、不做板端延迟测试、不进入正式固件。

下一步停止继续堆叠同类单轮 preference 或蒸馏样本，转向更高杠杆的结构性实验：让训练样本显式要求组合用户
细节并重新组织答案，或评估更强容量/架构；每个候选仍需先通过固定回归和小规模盲评，再决定是否板测。

## 20.39 结构化细节组合 pilot：明确退化，否决当前数据形状（2026-10-24）

为直接测试“回答能否同时组合用户给出的两个条件”，新增 `build_composition_pilot.py`，调用官方
`deepseek-flash` 生成包含两个具体细节的用户问题和回答，并要求回答分别处理两个细节。自动筛选允许常见
同义改写，同时检查词表、长度、句末标点和自称。教师请求 25 次后得到 16 条通过样本；通过样本本身大多
自然，但大量候选因为词表或细节改写被拒，不能把该筛选率当作教师质量指标。样本保存在
`E:\\espllm_data\\persona_corpus\\raw_detail_composition_64.jsonl`。

用 `pack_composition_pilot.py` 将 16 条组合样本重复到 128 个窗口，与 384 条原始对齐样本混成 512 窗口，
保持与 20.38 相同的 `_stance_mt12` warm start、5L/32E/1850、`5e-5`、400 步和回复 mask。候选
checkpoint 为 `E:\\espllm_data\\persona_corpus\\alignment_distill_ab_20261024\\composition.pt.best`；
正式模型和固件未改动。候选验证 loss 最佳为 4.7921，但这是小池重复和随机验证切分的结果，不能替代行为评测。

固定回归显示组合数据没有迁移到未见问题：unseen 概念路由 control `4/26`、composition `2/26`；OOD 回避
control `6/14`、composition `0/14`；多轮前缀 control `3/26`、composition `2/26`。同一 64 条留出题的
盲评中，composition 获选 13、control 获选 24、平手 1、两者都不好 26。四项均值（composition/control）为：
切题 `0.281/0.422`、利用细节 `0.172/0.219`、回答形式 `0.281/0.594`、角色口吻 `0.781/0.969`。
候选还出现更多语句不通和答非所问，说明少量重复的“双细节”窗口会破坏原有角色和域内边界，不能靠这种
数据形状获得自由重组能力。

结论：否决本轮结构化细节组合配方，不扩大生成、不做量化或板测，也不把组合样本加入正式语料。到此，
单轮蒸馏、同主题 preference 和小比例组合数据都未解决“从给定回复拼句子”的根因；下一轮应停止在当前
2.17M 参数模型上继续叠加相似训练信号，转向真正增加可用容量或改变推理任务分解的方案，并先做 PC-only
可证伪实验。

## 20.40 高 head 数注意力 A/B：同预算下明确失败（2026-10-24）

当前 2.17M 模型路线的剩余架构候选是增加 query head。先测了 8L/16E/128 大容量配置：CPU 训练约
`2.09 s/step`，12,000 步约 7 小时，参数约 10.02M；按现有板端权重流量比例估算会超过 250 ms/token，
因此不投入完整训练，也不作为本板候选。

随后验证同预算的 5L/32E、4 query heads/2 KV heads。通过 `convert_head_split_init.py` 从正式
`_stance_mt12` 生成了重新计算 RoPE 的 warm-start 初始化，20 步 smoke、量化和架构同步均通过。400 步
warm-start 候选 `head4.pt.best` 的 unseen 路由为 `0/26`、OOD 回避 `0/14`；从头训练 1000 步的
`head4_scratch.pt.best` 仍为 unseen `0/26`、OOD `0/14`、多轮前缀 `0/26`，输出还出现大量语句
破碎和重复。对照 detail-control 在相同口径为 unseen `4/26`、OOD `6/14`、多轮 `3/26`。

这不是训练 loss 的问题：from-scratch 候选验证 loss 最佳约 4.8586，继续训练后训练 loss 下降但行为没有
恢复，说明 4-head/2-KV 的表示初始化或该小数据配方不适合当前角色任务。候选未做留出盲评、未导出板端
固件、未替换正式模型。高 head 数和 8L/16E 大容量路线均暂时关闭。

截至本条，当前硬件预算内的深度、专家数、CLA、共享专家、tokenizer、蒸馏、preference、组合数据和高 head
数均没有提供可部署的单轮质量增益。后续若继续，只应转向外部容量/更换硬件，或设计真正不同的推理任务分解；
不再在现有 5L/32E 权重上做相似微调。

## 20.41 自生成回答动作标记 pilot：没有转化为体验收益（2026-10-24）

为验证“先决定回答方式，再组织语言”这一不同于 preference/组合数据的任务分解，新增
`build_plan_prefix_pilot.py`。从 20.36 的 512 条非留出样本构造带单字符动作标记的窗口：事实、解释、办法、
情绪、质疑、邀请、纠错、选择、细节、域外分别映射到一个词表内字符，目标变为
`User: ...\nBot: <动作字符><回复>`。推理评测只剥离模型自己生成的首个动作字符，未加入关键词或域外规则；
相关剥离逻辑仅通过 `ESPGPT_PLAN_TAGS` 环境变量启用。训练池为 512 条原始对齐窗口 + 512 条标记窗口，
从正式 `_stance_mt12.pt.best` warm start，5L/32E、`5e-5`、400 步。候选为
`E:\\espllm_data\\persona_corpus\\alignment_distill_ab_20261024\\plan_prefix.pt.best`，正式权重和固件未改动。

固定回归中，候选 unseen 路由 `5/26`、OOD 回避 `8/14`、多轮前缀 `3/26`；同步 control 为 `4/26`、`6/14`、
`3/26`。但同一 64 条留出题盲评明显偏向 control：候选获选 10、control 获选 25、平手 6、两者都不好 23。
四项均值（候选/control）为切题 `0.312/0.547`、利用细节 `0.109/0.219`、回答形式 `0.766/0.828`、角色口吻
`0.906/1.156`。动作标记没有减少复用或提高回答重组，反而增加了输出损失和角色退化。

结论：否决自生成动作标记任务分解，不做动作标签扩充、不做板端解析或烧录。它说明在当前容量和数据规模下，
把一个隐式回答动作显式化会分走本已有限的生成能力；后续不再继续添加隐变量、分类头或相似辅助损失。
当前可行路径只剩更大外部模型/硬件，或完全不同的非自回归推理设计；正式模型继续保持 `_stance_mt12`。
