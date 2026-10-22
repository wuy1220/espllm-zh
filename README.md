# ESP-LLM 中文人设分支 · 在 ESP8266 上跑一个会说中文的角色模型

一块 **4 MB Flash / 80 KB DRAM** 的 ESP8266（NodeMCU，80 MHz 单核），跑一个 **2.17 M 参数**的
MoE Transformer，字符级中文词表 **1850** 字，人设是《东方 Project》的琪露诺。
权重全部塞进 Flash：ternary 1.58-bit（5 个三进制权重/字节）+ int8 词表 + int8 router，
量化后 **1163 KB**，固件占 Flash 94.3%、DRAM 92.3%。

没有操作系统、没有 PSRAM、没有堆分配 —— 一个裸机 C++ 推理引擎加一层规则表，
就是全部。

---

## 1. 关键结果

| 指标 | 数值 | 口径 |
|---|---|---|
| 逐概念命中率 | **191 / 203 = 94%** | 26 概念 × 混合问法，PC 侧 fp32 贪心解码 |
| 板上日常三概念 | **61 / 66 = 92%** | 吃饭 17/22 · 睡觉 22/22 · 玩耍 22/22，`:greedy` 关规则 |
| 言语行为（规则层） | **12 / 12** | 威胁 / 第三方比较，全部不经过模型 |
| 自比问句放行 | **4 / 4** | 「你打得过灵梦吗」必须交给模型，且答对 |
| 规则覆盖回归 | **60 / 60** | `coverage_zh_rules.py` |
| 训练集 / 验证集 loss | 1.8757 / 3.0334 | 7,019 窗，1398 步，约 15 分钟（8 线程 CPU） |

> 判行为只认**逐概念探针**。本项目里 val loss 相近而行为天差地别的例子出现过两次
> （val 2.96 → 73%，val 3.03 → 94%），见 `docs/project_state.md`。

## 2. 架构

| 参数 | 值 |
|---|---|
| 层数 / 专家数 | 5 层 / 每层 32 专家（Top-1 硬路由） |
| 隐藏维 / 头数 | 64 / 2（KV 头 1，MQA） |
| MoE 隐层 | 64（SwiGLU） |
| 训练窗 / 板上窗 | 128 / 64（滑动窗口，`slot` 与 `pos` 解耦） |
| 词表 | 1850，字符级（ASCII 仅 27 个） |
| 量化 | 权重 ternary 1.58 · tok_emb int8 · router int8 · scale fp16 |
| 参数量 | 2.17 M（每层 32 专家 = 97,152 B 打包后） |

## 3. 核心设计：三层分工

这是踩了几轮坑之后定下来的划界，也是这个项目最值钱的一条经验。

| 层 | 负责什么 | 例子 | 在哪实现 |
|---|---|---|---|
| **模型** | **名词可直接触发**的话题概念 | 青蛙 / 冰 / 湖 / 大妖精 … 26 概念 × 4 表态 | 权重 |
| **规则** | **信号在句法里**的言语行为 | 威胁「让灵梦收拾你」、第三方比较「A 和 B 谁厉害」、问候、告别 | `intent_rules.py` → `src/intent_rules_gen.h` |
| **固件** | 元信息与兜底 | 垃圾输入拦截、防背稿（签名环）、回声 | `src/main.cpp` |

**为什么这么切**：2 头 / 64 维 / ternary 的 QK 电路只能按**名词**路由。实测把「被挑衅」
「比试」放进模型，它会答成句子里那个名词的概念 ——「让灵梦收拾你」答灵梦的台词、
「让魔理沙打你」答魔理沙的台词，概念一多整类塌（6/6 → 1/6）。所以这两类交给规则帧
`zh_taunt` / `zh_duel`，模型腾出的容量给日常三概念（问法从 6 条扩到 22 条）。

**例外**：自比问句（「你打得过灵梦吗」「你俩谁强」）必须放行给模型，它答得对 ——
规则层用 `ZH_UNLESS` 显式排除「你 / 咱 / 比」。

## 4. 目录

```
src/                    固件：main.cpp / inference.hpp / intent_rules_gen.h / model_weights.hpp(生成)
include/ lib/           PlatformIO 占位
persona/                人设：stance_lexicon.py(词典唯一真相) / vocab_zh.json / 设定卡 / 方案文档
main.py                 PyTorch 训练（训练时即 QAT）
convert_model_to_c.py   导出 C++ 权重头文件（ternary 位打包 + int8 词表）
check_arch_sync.py      构建闸门：头文件常量 vs checkpoint 架构，不一致拒绝构建
intent_rules.py         意图规则唯一入口 → gen_rules_cpp.py 生成 C++
coverage_zh_rules.py    规则覆盖回归（60 条）
check_lexicon.py        词典三条硬约束：答案不以概念名开头 / 跨概念开头 2 字唯一 / 字表内
probe_stance.py         模型侧概念探针（seen / unseen 分开报）
per_concept.py          逐概念命中率（203 条）
board_stance4.py        板上概念探针
probe_board_acceptance.py  板上验收三段：日常概念 / 言语行为 / 自比放行
gen_*_corpus.py         语料工厂（反应语料 / 表态语料 / 人设语料）
docs/                   设计文档与实验记录（project_state.md 是完整实验日志）
logs/ archive/          本地归档，已 gitignore
```

> 顶层保留扁平布局是有意的：这些脚本互相 `import`，且大量以 `cwd` 相对路径读
> `persona/`、`dataset_*`。拆进子目录会成片断掉。

## 5. 跑起来

```bash
# 1) 训练（四个环境变量少一个就废模型，见下表）
export ESPGPT_CHAR_VOCAB=persona/vocab_zh.json
ESPGPT_CHAT_WEIGHT=10 ESPGPT_WARMUP_ITERS=50 ESPGPT_T_MAX=1500 \
ESPGPT_BATCH_SIZE=128 ESPGPT_PATIENCE=999 \
python main.py --train

# 2) 导出权重头文件（必须带字符词表，否则写进去的是英文 BPE 表，板上只吐英文碎片）
ESPGPT_CHAR_VOCAB=persona/vocab_zh.json python convert_model_to_c.py stance26

# 3) 静态检查与回归
python check_arch_sync.py      # 头文件常量 == checkpoint 架构
python test_newline_token.py   # 两种词表的换行 token id 都能终止生成
python check_lexicon.py        # 词典约束
python coverage_zh_rules.py    # 规则覆盖 60/60

# 4) 编译烧录
pio run -e esp8266 -t upload

# 5) 板上验收（需先说 COM 口号）
python probe_board_acceptance.py COM5
python per_concept.py --ckpt model/model_stance26.pt
```

### 训练配方的四个环境变量

| 变量 | 值 | 不设会怎样 |
|---|---|---|
| `ESPGPT_CHAT_WEIGHT` | `10` | 故事窗淹没对话窗，逐概念从 94% 掉到 **17%** |
| `ESPGPT_WARMUP_ITERS` | `50` | 默认 200 会把余弦周期吃掉，学习率爬不到峰值 |
| `ESPGPT_T_MAX` | `1500` | 余弦退火的收益全在最后 20%，周期短了拿不到 |
| `ESPGPT_PATIENCE` | `999` | 热启动时早停会砍掉收敛的后半段 |

日志里应出现 `sampling: story … x1.00 | chat-summary … x10.00 -> 95.6%`，否则配比没生效。

## 6. 已知边界

- **抄不动**：2.17 M / 2 头 / ternary 学不会「从上下文复制一段字」，随机伪名词前 2 字命中 0/16。
  所以「复述用户的话」交给固件回声，不进权重。
- **短输入天然分布外**：语料里零问答对，summary 模板下 prompt 中位 57 token，而板上输入
  只有几个字。事实型问答一律走规则层，不指望模型。
- **防背稿不能管贪心**：一个概念只有 4 条表态，`:greedy` 连问必然复读；检测只在
  `temp > 0` 时启用，否则重掷＝同一句，二次撞环后整段转回避。
- **QK 提精度是唯一可能的突破**（q/k 抬到 int8/4bit，约 +16 KB），工程量太大，暂缓。
- 上下文 64 token，靠滑窗支撑多轮；真成本是注意力扫满窗，不是滑窗本身。

## 7. 文档

| 文件 | 内容 |
|---|---|
| [docs/project_state.md](docs/project_state.md) | 完整实验日志：每个数字怎么来的、哪些路已否决 |
| [docs/ENGINE_SPEC.md](docs/ENGINE_SPEC.md) | 上游引擎规格书（内存布局、量化、看门狗切片） |
| [persona/model_first_plan.md](persona/model_first_plan.md) | 人设方案与三层分工定稿（第 13 节） |
| [docs/research_2026-10.md](docs/research_2026-10.md) | 文献侧调研：小模型量化、induction head、语料可行性 |
| [docs/arch_scaling_and_flash.md](docs/arch_scaling_and_flash.md) | 架构预算：为什么同参数给深度不给专家 |
| [docs/micro_models_and_speed.md](docs/micro_models_and_speed.md) | 微模型与速度实测 |
| [docs/head_packing_2bit.md](docs/head_packing_2bit.md) | 输出头 2-bit 位打包 |

## 8. 来源与许可

推理引擎的骨架（`src/inference.hpp`、`main.cpp` 原有结构、`platformio.ini`、打包格式）源自
[ahmedbarakat207/espllm](https://github.com/ahmedbarakat207/espllm)。**上游仓库未声明开源许可证**，
二次分发请自行确认上游要求。

本项目新增的部分 —— 中文词表与人设词典、规则层、语料工厂、训练配方、全部文档 ——
以 MIT 发布，见 [LICENSE](LICENSE)；逐项归属与同人声明见 [NOTICE.md](NOTICE.md)。
