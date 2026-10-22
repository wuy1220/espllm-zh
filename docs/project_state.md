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

