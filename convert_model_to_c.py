import gzip, math, os, shutil, sys, time
import torch
import torch.nn.functional as F
import main

sys.modules["__main__"] = main
QUANTIZED_PATH = None
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    arg_p = sys.argv[1].lower()
    candidates = [
        arg_p,
        f"model/model_{arg_p}.pt.quantized",
        f"model/{arg_p}.pt.quantized",
        f"model/{arg_p}.pt",
    ]
    if "esp32s3" in arg_p:
        candidates.extend(
            [
                "model/model_esp32s3.pt.quantized",
                "model/model_esp32s3.pt.best",
                "model/model_esp32s3.pt",
            ]
        )
    elif "esp32" in arg_p:
        candidates.extend(
            [
                "model/model.pt.quantized",
                "model/model.pt",
                "model/model_esp32.pt.quantized",
            ]
        )
    elif "esp8266" in arg_p:
        candidates.extend(
            [
                "model/model_esp8266.pt.quantized",
                "model/model_esp8266.pt.best",
                "model/model_esp8266.pt",
            ]
        )
    for c in candidates:
        if os.path.exists(c):
            QUANTIZED_PATH = c
            break

if not QUANTIZED_PATH:
    default_candidates = [
        "model/model_esp32s3.pt.quantized",
        "model/model.pt.quantized",
        "model/model_esp8266.pt.quantized",
        "model/model.pt",
        "model/model_esp8266.pt.best",
    ]
    for c in default_candidates:
        if os.path.exists(c):
            QUANTIZED_PATH = c
            break

if not QUANTIZED_PATH:
    print("Error: No model checkpoint found in model/ directory.")
    sys.exit(1)

OUTPUT_PATH = "src/model_weights.hpp"
# `--out PATH` redirects the header. Without it this script overwrites the
# production header in place, which has already cost us one working firmware
# (the pre-ternary-head export) because there was no copy anywhere else.
for _i, _a in enumerate(sys.argv):
    if _a == "--out" and _i + 1 < len(sys.argv):
        OUTPUT_PATH = sys.argv[_i + 1]
GROUP_SIZE = 64
print(f"Loading {QUANTIZED_PATH} ...")
saved_arch = None
if QUANTIZED_PATH.endswith(".quantized"):
    try:
        with gzip.open(QUANTIZED_PATH, "rb") as f:
            model = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(model, dict) and "__arch__" in model:
            saved_arch = model["__arch__"]
    except Exception as e:
        print(f"Error: could not load {QUANTIZED_PATH} ({e}).")
        print(
            "It may be a stale checkpoint from an older code version — retrain first."
        )
        sys.exit(1)
else:
    # Raw (unquantized) state dict: infer dims from tensors, point main.py's
    # globals at them, rebuild the model, then quantize. The profile is only
    # used to pick a matching checkpoint; dims always come from the file.
    state_dict = torch.load(QUANTIZED_PATH, map_location="cpu", weights_only=False)
    saved_arch = None
    if isinstance(state_dict, dict) and "__arch__" in state_dict:
        saved_arch = state_dict["__arch__"]
    if hasattr(state_dict, "blocks"):
        model = state_dict
    else:
        sd = state_dict
        _tok = sd["tok_emb.weight"]
        _vocab, _embd = _tok.shape
        _nl = len(
            [
                k
                for k in sd
                if k.startswith("blocks.") and k.endswith(".attn.qkv.weight")
            ]
        )
        _qkv_out, _ = sd["blocks.0.attn.qkv.weight"].shape
        _nh = main.n_head
        for cand_nh in (2, 4, 6, 8, 12):
            if _embd % cand_nh == 0:
                _hd = _embd // cand_nh
                if _qkv_out - _embd > 0 and (_qkv_out - _embd) % (2 * _hd) == 0:
                    _nh = cand_nh
                    break
        _hd = _embd // _nh
        _nkv = (_qkv_out - _embd) // (2 * _hd)
        _gate_out, _ = sd["blocks.0.mlp.experts.0.gate_proj.weight"].shape
        _ne = len(
            [
                k
                for k in sd
                if k.startswith("blocks.0.mlp.experts.")
                and k.endswith(".gate_proj.weight")
            ]
        )
        main.n_embd, main.n_head, main.n_kv_head = _embd, _nh, _nkv
        main.n_layer, main.n_experts, main.moe_hidden = _nl, _ne, _gate_out
        main.dropout = 0.0
        if "rope_cos" in sd:
            main.block_size = sd["rope_cos"].shape[0]
        model = main.Transformer(group_size=main.qat_group_size)
        # arch tag is not a parameter, but keep it: it is the only record of
        # which architecture the weights were trained for, and the header must
        # carry it so a rebuild can be verified (audit 19.1).
        saved_arch = sd.pop("__arch__", None)
        model.load_state_dict(sd)
    model.eval()
    model = main.convert_to_bitlinear(model)
print("Model loaded.")
# tok_emb 行数 = 模型真正会的 token 数。词表分支选错时，只有它能发现
# （2026-10-21：没设 ESPGPT_CHAR_VOCAB，英文 2048 项 BPE 表被写进 1850 行
# 的中文模型，板上不报错、只吐英文碎片 —— 见 _vocabcheck.py）。
# 注意必须放在 if/else 之后：量化文件走上面的 try 分支，raw 走 else。
_tok_rows = None
for _name, _mod in model.named_modules():
    if _name.split(".")[-1] == "tok_emb":
        for _attr in ("weight", "qweight"):
            _t = getattr(_mod, _attr, None)
            if _t is not None and getattr(_t, "dim", lambda: 0)() == 2:
                _tok_rows = int(_t.shape[0])
                break
    if _tok_rows is not None:
        break
for _m in model.modules():
    if hasattr(_m, "group_size") and isinstance(getattr(_m, "group_size"), int):
        GROUP_SIZE = int(_m.group_size)
        break
print(f"Using quantization group size: {GROUP_SIZE}")


def unicode_to_bytes():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(cs, bs))


u2b = unicode_to_bytes()

import json

# 两套词表格式，导出的表必须和模型的 tok_emb 行数一致：
#   BPE      bpe-vocab.json，键是 GPT-2 字节级编码后的字符串，要先过 u2b 还原成字节
#   字符级   persona/vocab_zh.json 的 chars 列表，每一项本来就是一个 Unicode 字符，
#            直接 UTF-8 编码即可。走错分支不会报错，只会把英文 2048 项的表写进
#            1850 行的中文模型里 —— 板上每个 token 都解码成乱码，而且体积看着正常。
_CHAR_VOCAB = os.environ.get("ESPGPT_CHAR_VOCAB", "")
vocab_bytes_flat = bytearray()
vocab_offsets = []

if _CHAR_VOCAB:
    with open(_CHAR_VOCAB, "r", encoding="utf-8") as f:
        chars = json.load(f)["chars"]
    vocab_size = len(chars)
    for ch in chars:
        vocab_offsets.append(len(vocab_bytes_flat))
        vocab_bytes_flat.extend(ch.encode("utf-8"))
    print(f"Vocab: char-level, {vocab_size} entries from {_CHAR_VOCAB}")
    # 换行 token id 由词表算出，**不靠固件硬编码**：中文字符表里是 1810，
    # BPE 里是 199。写死任何一个，另一种词表上生成循环都不会在换行处停，
    # 收尾还会把这个 id 当额外 token 写回上下文（2026-10-22 审查 19.2）。
    if "\n" not in chars:
        raise SystemExit(
            "FATAL: char vocab %r has no '\\n' entry — the firmware cannot know\n"
            "        where to stop generating. Add it to the vocab file." % _CHAR_VOCAB
        )
    newline_id = chars.index("\n")
else:
    with open("bpe-vocab.json", "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    vocab_size = len(vocab_dict)
    v2i = {v: k for k, v in vocab_dict.items()}
    newline_id = None
    for i in range(vocab_size):
        vocab_offsets.append(len(vocab_bytes_flat))
        token_str = v2i[i]
        raw = bytes([u2b[c] for c in token_str])
        if raw == b"\n":
            newline_id = i          # GPT-2 字节编码里换行是 'Ċ'，id 199
        vocab_bytes_flat.extend(raw)
    if newline_id is None:
        raise SystemExit(
            "FATAL: bpe-vocab.json has no newline token — the firmware cannot\n"
            "        know where to stop generating."
        )
print(f"Newline token id: {newline_id}")

vocab_offsets.append(len(vocab_bytes_flat))

if _tok_rows is not None and vocab_size != _tok_rows:
    raise SystemExit(
        "FATAL: vocab table has %d entries but tok_emb has %d rows.\n"
        "        Wrong vocab branch — for the Chinese persona models run with\n"
        "        ESPGPT_CHAR_VOCAB=persona/vocab_zh.json (now %r).\n"
        "        Without it the English BPE table gets baked in and every\n"
        "        board token decodes to garbage."
        % (vocab_size, _tok_rows, _CHAR_VOCAB)
    )


def get_quantized(mod):
    if hasattr(mod, "qweight"):
        q = mod.qweight
        scale = mod.scale.float()
        group_size = getattr(mod, "group_size", GROUP_SIZE)
    else:
        w = mod.weight
        group_size = getattr(mod, "group_size", GROUP_SIZE)
        out, n = w.shape
        n_groups = math.ceil(n / group_size)
        padded_n = n_groups * group_size
        w_pad = F.pad(w, (0, padded_n - n)) if padded_n != n else w
        wg = w_pad.view(out, n_groups, group_size)
        scale = wg.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        q = torch.clamp(torch.round(wg / scale), -1, 1).to(torch.int8)
        q = q.view(out, padded_n)[:, :n]
        scale = scale.squeeze(-1)

    # 5 ternary weights per byte
    out, n = q.shape
    n_groups = math.ceil(n / group_size)
    padded_n = n_groups * group_size
    if padded_n != n:
        q = F.pad(q, (0, padded_n - n))

    q_groups = q.view(out, n_groups, group_size)
    pad_g = (5 - (group_size % 5)) % 5
    if pad_g > 0:
        q_groups = F.pad(q_groups, (0, pad_g))

    q_map = torch.where(
        q_groups == -1,
        torch.tensor(2, dtype=torch.uint8, device=q.device),
        q_groups.to(torch.uint8),
    )
    m5 = q_map.view(out, n_groups, -1, 5)
    packed = (
        m5[..., 0].to(torch.int32)
        + m5[..., 1].to(torch.int32) * 3
        + m5[..., 2].to(torch.int32) * 9
        + m5[..., 3].to(torch.int32) * 27
        + m5[..., 4].to(torch.int32) * 81
    ).to(torch.uint8)
    packed_flat = packed.view(out, -1)
    return packed_flat, scale


def emit_quantized(prefix, packed, scale):
    # Scales are stored as FP16 (2 bytes) — firmware converts to float on the fly.
    scale_h = scale.detach().cpu().float().to(torch.float16).view(torch.int16)
    parts = [
        emit_u8(packed, f"{prefix}_weights"),
        emit_u16(scale_h, f"{prefix}_scales"),
    ]
    return "\n".join(parts)


def emit_f32(arr, name):
    flat = arr.detach().cpu().float().numpy().flatten()
    lines = [
        f"static const float {name }[{len (flat )}] PROGMEM __attribute__((aligned(4))) = {{"
    ]
    row = []
    for i, v in enumerate(flat):
        row.append(f"{v :.6f}f")
        if len(row) == 8 or i == len(flat) - 1:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    lines.append("};\n")
    return "\n".join(lines)


def emit_u8(arr, name):
    flat = arr.detach().cpu().numpy().flatten()
    lines = [
        f"static const uint8_t {name }[{len (flat )}] PROGMEM __attribute__((aligned(4))) = {{"
    ]
    row = []
    for i, v in enumerate(flat):
        row.append(f"0x{int (v ):02x}")
        if len(row) == 12 or i == len(flat) - 1:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    lines.append("};\n")
    return "\n".join(lines)


def emit_u16(arr, name):
    flat = arr.detach().cpu().numpy().flatten()
    lines = [
        f"static const uint16_t {name }[{len (flat )}] PROGMEM __attribute__((aligned(4))) = {{"
    ]
    row = []
    for i, v in enumerate(flat):
        row.append(f"0x{int (v ) & 0xffff :04x}")
        if len(row) == 12 or i == len(flat) - 1:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    lines.append("};\n")
    return "\n".join(lines)


tok_emb_w = None

for name, mod in model.named_modules():
    if name == "tok_emb":
        tok_emb_w = mod.weight.detach().float()

n_embd = tok_emb_w.shape[1]
n_layer = len(model.blocks)
n_head = model.blocks[0].attn.n_head
n_kv_head = model.blocks[0].attn.qkv.out_features
n_kv_head = int((n_kv_head - n_embd) / 2 / (n_embd // n_head))
head_dim = n_embd // n_head
block_size = model.rope_cos.shape[0]
mlp = model.blocks[0].mlp
first_gate_q, _ = get_quantized(mlp.experts[0].gate_proj)
mlp_hidden = first_gate_q.shape[0]
n_experts = len(mlp.experts)

print(
    f"  n_embd={n_embd }  n_layer={n_layer }  n_head={n_head }  "
    f"head_dim={head_dim }  mlp_hidden={mlp_hidden }  block_size={block_size }"
)
theta = 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
t = torch.arange(block_size).float()
freqs = torch.outer(t, theta)
rope_cos = freqs.cos()
rope_sin = freqs.sin()
sections = []
total_bytes = 0

# ── 溯源注释（audit 19.1）────────────────────────────────────────────────────
# 固件只认这个头文件里的常量，训练侧只认 checkpoint 里的 __arch__。两者此前
# 没有任何关联，于是 model_esp8266.pt（2048/4L/64）和 5L/32E/1850 的头文件可以
# 相安无事地共存，谁重新导出一次就悄悄换掉板上跑的模型。现在把来源写进注释，
# 并由 check_arch_sync.py 在构建前比对。
_arch_comment = (
    "// arch-source: {src}\n"
    "// arch: vocab={vocab_size} n_embd={n_embd} n_layer={n_layer} n_head={n_head} "
    "n_kv_head={n_kv_head} block_size={block_size} group_size={GROUP_SIZE} "
    "mlp_hidden={mlp_hidden} n_experts={n_experts}\n"
).format(
    src=QUANTIZED_PATH,
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head,
    n_kv_head=n_kv_head,
    block_size=block_size,
    GROUP_SIZE=GROUP_SIZE,
    mlp_hidden=mlp_hidden,
    n_experts=n_experts,
)
if saved_arch is not None:
    _derived = {
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "block_size": block_size,
        "group_size": GROUP_SIZE,
        "mlp_hidden": mlp_hidden,
        "n_experts": n_experts,
    }
    _diffs = [
        "{}: ckpt={} vs derived={}".format(k, saved_arch.get(k), v)
        for k, v in _derived.items()
        if k in saved_arch and saved_arch[k] != v
    ]
    if _diffs:
        print("  WARNING: checkpoint __arch__ disagrees with the exported header:")
        for _d in _diffs:
            print("    " + _d)
        print("  The header follows the tensor shapes. If this was not intended,")
        print("  re-export from the checkpoint that matches the firmware.")

sections.append(
    _arch_comment
    + f"""// Auto-generated by convert_model_to_c.py — DO NOT EDIT
#ifndef MODEL_WEIGHTS_HPP
#define MODEL_WEIGHTS_HPP
#include <stdint.h>
#include <stddef.h>
#if defined(__AVR__)
#  include <avr/pgmspace.h>
#elif defined(ESP8266) || defined(ESP32)
#  include <pgmspace.h>
#else
#  define PROGMEM
#endif
static const uint16_t model_vocab_size = {vocab_size};
static const uint16_t model_n_embd     = {n_embd};
static const uint8_t  model_n_layer    = {n_layer};
static const uint8_t  model_n_head     = {n_head};
static const uint16_t model_block_size = {block_size};
static const uint8_t  model_group_size = {GROUP_SIZE};
static const uint16_t model_mlp_hidden = {mlp_hidden};
static const uint8_t  model_n_experts  = {n_experts};
static const uint8_t  model_n_kv_head  = {n_kv_head};
static const uint8_t  model_weights_per_byte = 5;
// Newline token id for THIS vocab (BPE 199 / char-level 1810). The generation
// loop stops here and uses it to detect the end of a reply.
static const uint16_t model_newline_id = {newline_id};
"""
)
sections.append(
    emit_u8(
        torch.tensor(list(vocab_bytes_flat), dtype=torch.uint8), "model_vocab_bytes"
    )
)
sections.append(
    f"static const uint32_t model_vocab_offsets[{vocab_size + 1}] PROGMEM __attribute__((aligned(4))) = {{"
)
sections.append("    " + ", ".join(str(o) for o in vocab_offsets) + "\n};\n")
total_bytes += len(vocab_bytes_flat) + (vocab_size + 1) * 4
# Token embeddings: per-token INT8 + FP32 row scale (4x smaller than FP32).
emb_scale = tok_emb_w.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5) / 127.0
emb_q = torch.clamp(torch.round(tok_emb_w / emb_scale), -128, 127).to(torch.int16)
emb_q_u8 = (emb_q & 0xFF).to(torch.uint8)
sections.append(emit_u8(emb_q_u8.flatten(), "tok_emb_q"))
sections.append(emit_f32(emb_scale.squeeze(-1), "tok_emb_scales"))
total_bytes += emb_q_u8.numel() + emb_scale.numel() * 4
# Output head: same tied matrix, but quantized the way TRAINING saw it
# (per-group absmean ternary) instead of importing the fp32 latent weights as
# int8. Those are not the same matrix: BitLinear applies _quantize_weight on
# every forward pass, so the fp32 values never reach the output projection
# directly. Measured on model_mixed2_init.pt, 32 val batches:
#   fp32 latent -> int8 head        val 3.0520  ppl 21.16   (what shipped)
#   training-path ternary head     val 2.5674  ppl 13.03   (this export)
# so the old path was handing the board a head the body was never trained
# against.
#
# Stored 2-bit packed: four ternary values per byte, 16 B per row instead of 64.
#
# The earlier base-3 attempt failed on the board (978 ms/token) because it
# unpacked *inside* the dot loop. The on-device microbenchmark in bench_unpack/
# (2048 rows x 64 cols, results in bench_unpack/bench4.log) then showed where
# the money actually is: 128 KB of int8 flash costs 15.5 ms and the MACs 6.8 ms,
# so the head was mostly a flash *transaction* bill, not arithmetic. Packing
# cuts transactions 4x, and a fused kernel that expands four values out of a
# 256-entry word LUT and consumes them immediately runs in 10.1 ms vs 22.3 ms
# for the int8 path -- 2.2x -- with unpacked dot products bit-identical on all
# 2048 rows.
#
# Byte layout: value v sits at bit offset 2k with code = v + 1, so -1/0/+1 maps
# to 0/1/2. Code 3 is never emitted; the firmware LUT maps it to +2, which is
# only reachable if this export breaks, hence the assert.
_hw = tok_emb_w
_hscale = _hw.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
_hq = torch.clamp(torch.round(_hw / _hscale), -1, 1).to(torch.int8)
_code = (_hq.to(torch.int32) + 1)
assert int(_code.min()) >= 0 and int(_code.max()) <= 2, "ternary code does not fit 2 bits"
assert _code.shape[1] % 4 == 0, "2-bit packing needs a multiple of 4 columns"
_p2 = ((_code[..., 0::4] & 3)
       | ((_code[..., 1::4] & 3) << 2)
       | ((_code[..., 2::4] & 3) << 4)
       | ((_code[..., 3::4] & 3) << 6)).to(torch.uint8)
assert torch.equal(((_p2.unsqueeze(-1) >> torch.tensor([0, 2, 4, 6])) & 3).reshape(_code.shape).to(torch.int32) - 1, _hq.to(torch.int32)), "2-bit round-trip mismatch"
sections.append(emit_u8(_p2.flatten(), "lm_head_p2"))
sections.append(emit_f32(_hscale.squeeze(-1), "lm_head_scales"))
total_bytes += _p2.numel() + _hscale.numel() * 4
# Unpack table for the 2-bit head, emitted here so the packing above and the
# decoding in the firmware can never drift apart. Indexed by one packed byte,
# returns its four ternary values as four int8 bytes in a uint32, which lets the
# kernel copy them with a single store or, better, extract them in place.
#
# Deliberately NOT PROGMEM: it is 1 KB read once per group of four weights, and
# on this chip a flash read costs the same whether it returns 1 byte or 4, so
# keeping it in DRAM is what makes the kernel cheap.
_lut_words = []
for _byte in range(256):
    _w = 0
    for _k in range(4):
        _v = ((_byte >> (2 * _k)) & 3) - 1
        _w |= (_v & 0xFF) << (8 * _k)
    _lut_words.append(_w)
sections.append(
    "static const uint32_t ternary_lut256[256] __attribute__((aligned(4))) = {\n  "
    + ",".join(f"0x{v:08x}u" for v in _lut_words) + "\n};\n")
sections.append(emit_f32(rope_cos.flatten(), "rope_cos"))
sections.append(emit_f32(rope_sin.flatten(), "rope_sin"))
total_bytes += rope_cos.numel() * 4 * 2

for li, block in enumerate(model.blocks):
    attn = block.attn
    qkv_q, qkv_s = get_quantized(attn.qkv)
    proj_q, proj_s = get_quantized(attn.proj)
    sections.append(emit_quantized(f"l{li}_attn_qkv", qkv_q, qkv_s))
    sections.append(emit_quantized(f"l{li}_attn_proj", proj_q, proj_s))
    sections.append(emit_f32(block.ln1.weight.detach().float(), f"l{li}_ln1_gamma"))
    sections.append(emit_f32(block.ln2.weight.detach().float(), f"l{li}_ln2_gamma"))
    # Router: int8 weights + one FP16 scale per expert.
    # The router is a float dot product on every (layer, token); on the FPU-less
    # ESP8266 that measured ~7% of per-token time, so it goes to integer MACs.
    # Routing agreement 99.36%, val loss delta -0.0001 nats
    # (see verify_router_int8.py). Also cuts the router's flash by ~3/4.
    router_w = block.mlp.router.weight.detach().float()
    router_s = router_w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    router_q = torch.clamp(torch.round(router_w / router_s), -127, 127).to(torch.int16)
    router_q_u8 = (router_q & 0xFF).to(torch.uint8)
    router_s_h = router_s.squeeze(1).to(torch.float16).view(torch.int16)
    sections.append(emit_u8(router_q_u8.flatten(), f"l{li}_router_q"))
    sections.append(emit_u16(router_s_h, f"l{li}_router_s"))
    total_bytes += router_q_u8.numel() + router_s_h.numel() * 2
    gate_qs, gate_ss = zip(
        *[get_quantized(expert.gate_proj) for expert in block.mlp.experts]
    )
    up_qs, up_ss = zip(*[get_quantized(expert.up_proj) for expert in block.mlp.experts])
    down_qs, down_ss = zip(
        *[get_quantized(expert.down_proj) for expert in block.mlp.experts]
    )
    gate_q_concat = torch.cat(gate_qs, dim=0)
    gate_s_concat = torch.cat(gate_ss, dim=0)
    up_q_concat = torch.cat(up_qs, dim=0)
    up_s_concat = torch.cat(up_ss, dim=0)
    down_q_concat = torch.cat(down_qs, dim=0)
    down_s_concat = torch.cat(down_ss, dim=0)
    sections.append(emit_quantized(f"l{li}_experts_gate", gate_q_concat, gate_s_concat))
    sections.append(emit_quantized(f"l{li}_experts_up", up_q_concat, up_s_concat))
    sections.append(emit_quantized(f"l{li}_experts_down", down_q_concat, down_s_concat))
    layer_bytes = (
        qkv_q.numel()
        + proj_q.numel()
        + gate_q_concat.numel()
        + up_q_concat.numel()
        + down_q_concat.numel()
    )
    layer_bytes += 2 * (
        qkv_s.numel()
        + proj_s.numel()
        + gate_s_concat.numel()
        + up_s_concat.numel()
        + down_s_concat.numel()
    )  # FP16 scales
    layer_bytes += (block.ln1.weight.numel() + block.ln2.weight.numel()) * 4
    total_bytes += layer_bytes
sections.append(emit_f32(model.ln_f.weight.detach().float(), "ln_f_gamma"))
total_bytes += model.ln_f.weight.numel() * 4
first_qkv_q, _ = get_quantized(model.blocks[0].attn.qkv)
sections.append(emit_u8(first_qkv_q, "model_weights"))
sections.append(
    f"static const unsigned int model_weights_len = {first_qkv_q.numel()};\n"
)

layer_inits = []
for li in range(n_layer):
    layer_inits.append(f"""    {{
        l{li}_attn_qkv_weights,      l{li}_attn_qkv_scales,
        l{li}_attn_proj_weights,     l{li}_attn_proj_scales,
        l{li}_router_q,              l{li}_router_s,
        l{li}_experts_gate_weights,  l{li}_experts_gate_scales,
        l{li}_experts_up_weights,    l{li}_experts_up_scales,
        l{li}_experts_down_weights,  l{li}_experts_down_scales,
        l{li}_ln1_gamma,
        l{li}_ln2_gamma,
    }}""")

layers_str = ",\n".join(layer_inits)
sections.append(f"""
struct LayerW {{
    const uint8_t* qkv_w;  const uint16_t* qkv_s; // scales: FP16, half_to_float on load
    const uint8_t* proj_w; const uint16_t* proj_s;
    const uint8_t* router_q;  const uint16_t* router_s;
    const uint8_t* experts_gate_q;
    const uint16_t* experts_gate_s;
    const uint8_t* experts_up_q;
    const uint16_t* experts_up_s;
    const uint8_t* experts_down_q;
    const uint16_t* experts_down_s;
    const float* ln1_g;
    const float* ln2_g;
}};

static const LayerW g_layers[{n_layer}] __attribute__((aligned(4))) = {{
{layers_str}
}};

#endif // MODEL_WEIGHTS_HPP
""")

os.makedirs(os.path.dirname(OUTPUT_PATH) or "src", exist_ok=True)
# Keep the previous header around. Overwriting it destroyed a known-good export
# once; a timestamped sibling costs 8 MB and makes that unrecoverable mistake
# impossible.
if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0:
    _bak = OUTPUT_PATH + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(OUTPUT_PATH, _bak)
    print(f"previous header saved -> {_bak}")
with open(OUTPUT_PATH, "w") as f:
    f.write("\n".join(sections))
file_kb = os.path.getsize(OUTPUT_PATH) / 1024
print(f"\nWrote {OUTPUT_PATH }  ({file_kb :.0f} KB source)")
print(f"Estimated binary flash usage: ~{total_bytes //1024 } KB")
print("\nArrays exported:")
print(
    f"  model_vocab_bytes[{len (vocab_bytes_flat )}], model_vocab_offsets[{vocab_size +1 }]"
)
print(
    f"  tok_emb_q[{vocab_size }×{n_embd }] int8 + scales, rope_cos/sin[{block_size }×{head_dim //2 }]"
)
for li in range(n_layer):
    print(f"  l{li }: qkv, proj, gate_proj, up_proj, down_proj, ln1, ln2")
print("  ln_f_gamma")
