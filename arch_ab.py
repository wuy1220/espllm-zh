"""Architecture budget: exact param count, flash residency and per-token traffic.

Mirrors convert_model_to_c.py's packing so the numbers are what the linker will
actually see, not an estimate:
  - ternary weights: 5 per byte, per output row padded to whole groups of 64,
    then each 64-group padded to 65 so it divides by 5 -> 13 bytes/group/row
  - fp16 group scales: one per (row, 64-group)
  - tok_emb / lm_head: int8, 1 B/weight (tied)
  - router: int8 + fp16 per-row scale (retrofit_router_int8.py)
Code size is held constant at the measured value, since n_layer / n_experts are
compile-time constants but the kernel loops over them.
"""
import math

VOCAB, N_EMBD, N_HEAD, N_KV, GROUP, MLP_H = 2048, 64, 2, 1, 64, 64
FLASH_LIMIT = 1044464  # irom0_0_seg len = 0xfeff0
MEASURED_FLASH = 710667  # shipped build at L=4, EXP=20


def ternary_bytes(out: int, n: int) -> int:
    """Bytes for one ternary matrix, exactly as get_quantized() packs it."""
    n_groups = math.ceil(n / GROUP)
    pad_g = (5 - (GROUP % 5)) % 5
    return out * n_groups * ((GROUP + pad_g) // 5)


def scales_bytes(out: int, n: int) -> int:
    return out * math.ceil(n / GROUP) * 2  # fp16


def budget(n_layer: int, n_expert: int):
    qkv_n = N_EMBD + 2 * N_KV * (N_EMBD // N_HEAD)   # 64 + 64 = 128
    attn = sum(ternary_bytes(o, i) for o, i in ((qkv_n, N_EMBD), (N_EMBD, N_EMBD)))
    attn_sc = sum(scales_bytes(o, i) for o, i in ((qkv_n, N_EMBD), (N_EMBD, N_EMBD)))
    shapes = ((MLP_H, N_EMBD), (MLP_H, N_EMBD), (N_EMBD, MLP_H))
    per_expert = sum(ternary_bytes(o, i) for o, i in shapes)
    per_expert_sc = sum(scales_bytes(o, i) for o, i in shapes)

    body = n_layer * (attn + per_expert * n_expert)
    scales = n_layer * (attn_sc + per_expert_sc * n_expert)
    router_b = n_layer * (n_expert * N_EMBD + n_expert * 2)  # int8 + fp16 row scale
    tok_b = VOCAB * N_EMBD                                   # int8, shared with lm_head

    params = (VOCAB * N_EMBD
              + n_layer * (qkv_n * N_EMBD + N_EMBD * N_EMBD
                           + n_expert * 3 * MLP_H * N_EMBD
                           + n_expert * N_EMBD))
    active = (VOCAB * N_EMBD
              + n_layer * (qkv_n * N_EMBD + N_EMBD * N_EMBD
                           + 3 * MLP_H * N_EMBD + n_expert * N_EMBD))
    traffic = (tok_b + n_layer * (ternary_bytes(qkv_n, N_EMBD) + ternary_bytes(N_EMBD, N_EMBD)
                                  + scales_bytes(qkv_n, N_EMBD) + scales_bytes(N_EMBD, N_EMBD)
                                  + per_expert + per_expert_sc + router_b // n_layer))
    kv_arena = n_layer * (2 * 64 * N_KV * (N_EMBD // N_HEAD) + 2 * 64 * N_KV * 2)
    return dict(params=params, active=active, weights=tok_b + body + scales + router_b,
                traffic=traffic, kv_arena=kv_arena)


CODE_BYTES = MEASURED_FLASH - budget(4, 20)["weights"]


def main():
    print(f"标定：当前版本权重 {budget(4, 20)['weights']:,} B，反推代码区 {CODE_BYTES:,} B "
          f"（总 {MEASURED_FLASH:,}，对上说明模型可信）\n")
    print(f"{'配置':<10}{'总参数':>9}{'激活':>8}{'权重B':>10}{'Flash总':>10}{'占比':>8}"
          f"{'token流B':>10}{'KV缓存B':>9}{'余量B':>9}")
    print("-" * 92)
    for L, E in ((4, 20), (5, 20), (6, 20), (4, 32), (5, 32), (4, 40), (4, 44), (4, 48)):
        b = budget(L, E)
        total = CODE_BYTES + b["weights"]
        room = FLASH_LIMIT - total
        flag = "" if room > 40000 else ("  << 余量不足" if room > 0 else "  << 放不下")
        print(f"{L}L/{E}E{'':<4}{b['params']/1e6:>8.2f}M{b['active']/1e6:>7.2f}M"
              f"{b['weights']:>10,}{total:>10,}{total/FLASH_LIMIT*100:>7.1f}%"
              f"{b['traffic']:>10,}{b['kv_arena']:>9,}{room:>9,}{flag}")
    print(f"\n40KB arena 上限，KV 4 层 {budget(4,20)['kv_arena']:,} B / "
          f"5 层 {budget(5,20)['kv_arena']:,} B / 6 层 {budget(6,20)['kv_arena']:,} B")
    print(f"每加一层权重 {budget(5,20)['weights']-budget(4,20)['weights']:,} B；"
          f"每加一个专家 {budget(4,21)['weights']-budget(4,20)['weights']:,} B × 层数")


if __name__ == "__main__":
    main()
