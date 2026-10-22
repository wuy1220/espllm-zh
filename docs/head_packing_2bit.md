# 输出头 2-bit 打包：微基准与上线结果

日期 2026-10-18。承接同日的"输出头精度修复"（ternary head 替换 int8）。

## 为什么先做微基准

上一次 base-3 打包在板上实测 978 ms/token（int8 版是 77），结论被记成
"打包太慢"。但那个内核同时犯了两个错：逐字节读 flash、把解包塞进每 token
执行 2048 次的内层循环。在不知道哪一项致命之前，不该再改一版。

所以先建独立工程 `espllm/bench_unpack/`（不塞进主固件，那里 RAM 已 90%）：
2048 行 × 64 列，全部数据放 PROGMEM，每变体重复 8 次并逐次打印，
另有一段逐行校验保证解包结果与 int8 参考点积**完全相同**才认账。

## 结果（ms / 2048 行，160 MHz，bench_unpack/bench4.log）

| 变体 | 时间 | 说明 |
|---|---|---|
| A int8 from flash + MAC | 22.3 | 生产现状 |
| B int8 from flash，不乘 | 15.5 | 128 KB 的纯 flash 开销 |
| C 2-bit 逐字节 LUT + MAC | 17.7 | 上一版失败的做法 |
| D 2-bit from flash，不乘 | 2.2 | 32 KB 的纯 flash 开销 |
| E 2-bit 移位解包 + MAC | 19.3 | 不查表更慢 |
| F 2-bit dword 解包进缓冲 + MAC | 16.3 | 仍慢 |
| **H 2-bit dword 解包，融合，不缓冲** | **10.1** | 采用 |

成本模型逐项对得上：

```
A = 15.5 (128 KB flash) + 6.8 (MAC+取字节)            = 22.3
H =  2.2 ( 32 KB flash) + 6.8 (MAC+取字节) + 1.1 (LUT) = 10.1
```

## 两条硬件定律

1. **`pgm_read_byte` 与 `pgm_read_dword` 代价几乎相同。**
   16 字节拆成 16 次字节读，flash 事务数就是 4 倍。这是 978 ms 的真正来源。
   凡从 flash 取压缩权重，必须按 dword 读。
2. **不要"解包到缓冲再点积"。** LX106 没有 store buffer，先写 64 字节再读回来
   要比"解包后立刻相乘"多付 60%（16.3 vs 10.1）。必须在寄存器里解包、立刻消费。

## 上线

导出侧（`convert_model_to_c.py`）：

- `lm_head_q[131072]`（int8 存 {-1,0,1}）→ `lm_head_p2[32768]`，4 值/字节、16 B/行
- `lm_head_scales[2048]` 不变
- 新增 `ternary_lut256[256]`（1 KB DRAM），**由导出脚本生成**，
  保证打包表与解包表不会各改各的
- 导出时断言：值域落在 2 bit 内、列数是 4 的倍数、解包回去与源矩阵逐值相同

固件侧：`dot_ternary_p2()` 替换 `dot_emb_q_i8()`，`#include` 顺序调整
（先用 model_weights.hpp，因为内核读它生成的表）。删掉上次遗留的死内核
`dot_ternary_i8`（`base3_lut` 保留，专家 matmul 仍在用）。

### 等价性核验（独立于导出脚本）

从新旧两个头文件里逐 token 解析后比对：

```
lm_head_scales: 2048 个值，与旧版完全一致
lm_head_q[131072] -> lm_head_p2[32768]
解包后 131072 个值，与旧 int8 头逐值相同，值域 {-1, 0, 1}
```

配合板上微基准逐行校验（2048/2048 行点积完全相同），可以判定
**板上 logits 逐位不变，这是一次纯速度改动**。

### 资源

| | 之前 | 现在 |
|---|---|---|
| Flash | 849931 B (81.4%) | **753119 B (72.1%)**，省 95 KB |
| RAM | 72788 B (88.9%) | 73832 B (90.1%)，+1 KB 解包表 |
| 输出头耗时 | 26.0 ms/forward | **15.6 ms/forward** |

省下的 95 KB 可以换 8 个专家（每个 11784 B）或 1.5 层，是后续加容量的预算。

剩下的 15.6 ms 里，约 5.5 ms 是每行 `(float)s * xs * lm_head_scales[v]` 的
**软件浮点**（LX106 无 FPU，一次 fp32 乘约 150 cycles）。它现在占输出头的 35%，
是新暴露出来的下一个抓手。

## 顺带修正一个口径错误

给固件加了 token 计数（`[N ms total] %d tok X ms/tok`）之后才发现，此前所有
"78 ms/token" 都是 `elapsed / 前向次数`，而前向次数**包含提示词预填充**，
所以系统性读低了。真实值见下一节。

## 更严重的问题：上下文溢出，每 token 贵 64 倍

同一批测量里，单次生成的每 token 成本从 92.8 一路到 **680.7** ms：

| 提示词 | 生成 token | 总时间 | ms/token |
|---|---|---|---|
| he was so happy that | 10 | 1118 | 111.8 |
| once upon a time there was a little | 46 | 4269 | 92.8 |
| the dragon looked at the | 52 | 9174 | 176.4 |
| she opened the box and | 60 | 40842 | **680.7** |

原因是 `ctx_push()`（main.cpp:332）：ctx 满时它 memmove 掉最老的一个 id，并把
`ctx_pos` 重置为 0，于是下一个 token 要把整个 63 token 前缀**从头前向一遍**
重建 KV cache。

四组数据全部拟合下式：

```
总时间 ≈ 400 ms 预填充 + 70 ms/token + 64 × 70 ms × (超出 64 窗口的 token 数)
```

`he was so happy that`：400 + 10×70 = 1100（实测 1118）
`she opened the box and`：400 + 52×70 + 8×4480 ≈ 39400（实测 40842）

`MAX_GEN_TOKENS` 是 60、窗口只有 64，所以**绝大多数回复都会溢出**，
溢出 8 个 token 就吃掉 35.8 秒。这个量级比刚优化掉的 10 ms 大三个数量级，
是目前最该处理的问题。
