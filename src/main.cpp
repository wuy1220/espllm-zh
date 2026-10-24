#include <Arduino.h>
#include <string.h>
#include <stdlib.h>
#if defined(ESP8266) || defined(ESP8266_BOARD)
#  include <ESP8266WiFi.h>
#endif
// model_weights.hpp first: inference.hpp's 2-bit head kernel reads
// ternary_lut256, which the exporter emits alongside the packed weights.
#include "model_weights.hpp"
#include "inference.hpp"

static constexpr int N_EMBD          = model_n_embd;
static constexpr int N_HEAD          = model_n_head;
static constexpr int N_KV_HEAD       = model_n_kv_head;
static constexpr int N_LAYER         = model_n_layer;
static constexpr int HEAD_DIM        = N_EMBD / N_HEAD;  
static constexpr int MLP_HIDDEN      = model_mlp_hidden;              
static constexpr int N_EXPERTS       = model_n_experts;                
static constexpr int GRP             = model_group_size;                
#if defined(ESP8266) || defined(ESP8266_BOARD)
// Window size is capped by the arena, not by the trained block_size. The int8
// 68 B per position per layer (K + V at HEAD_DIM=32, plus two fp16 scales), so
// 5 layers at 104 positions is 35,360 B of the 40 KB arena.  That headroom was
// the binding constraint on depth: a 6th layer wanted 7,072 more and did not
// fit.
//
// 104 positions use 35,360 B for the 5-layer KV cache and fit the 40 KB arena
// with the inference scratch buffers (ARENA_NEED is checked below).  This lets
// history mode retain one complete prior exchange plus the current prompt while
// reserving 20 positions for the reply. Older turns are evicted at a User boundary
// so a larger cache does not preserve stale topics that compete with the latest one.
static constexpr int INFER_CTX       = (104 < (int)model_block_size) ? 104 : (int)model_block_size;
// Two budgets: a story may fill the window, a chat reply is one short
// paragraph. The chat cap exists because the model does not always emit the
// newline that normally ends a reply -- without it a single chat turn ran for
// 80 tokens / 7.4 s.
static constexpr int MAX_GEN_TOKENS  = 80;
// 40, not 32: on the mixed13 board run 5 of 36 replies were cut mid-clause by
// this budget, every one of them a normally-flowing sentence that just needed a
// few more tokens (e.g. `"...and said, \"Do you want"`).  The prefill trim at
// chat prefill trim reserves CHAT_MAX_TOKENS, leaving up to 64 prompt tokens;
// 64 + 40 = 104 fits INFER_CTX, so a normal chat turn does not slide the cache.
// The separate story prefill path uses MAX_GEN_TOKENS (80).
// Cost is wall time (40*88ms = 3.5s),
// which is the right side of the trade at this quality level.
static constexpr int CHAT_MAX_TOKENS = 40;
// Compact multi-turn rows keep the prior exchange plus prompt within the 104-token window.
static constexpr int HISTORY_CHAT_MAX_TOKENS = 20;
// 0.7, not 0.85: re-measured on hardware after the opener corpus rebuild
// (mixed13, 78 replies per setting, board_temp_sweep.py).  The old argument for
// 0.85 no longer holds -- it was that 0.7 pushed the "Once upon a time" opener
// on out-of-distribution questions from 17% to 39%, but that model had a 20%
// P(Once) prior.  The quota rebuild cut the corpus prior to 4.3% and the board
// now opens on Once 0/78 at BOTH settings, so the low-temperature risk is gone.
// What is left is the invented-word rate, which is monotone in temperature:
// OOV 1.78% at 0.85 -> 0.68% at 0.7, two independent runs agreeing (0.6/0.7 vs
// 1.4/2.0), while the sentence-finish rate does not suffer (89.7% vs 83.3%
// pooled).  ":temp <v>" still overrides at runtime.
//
// 2026-10-21 签名制下重测（_temp_sweep.py，板上 8 输入 x 6 次，按概念签名判分）：
// 0.7 -> 62% | 0.6 -> 54% | 0.5 -> 73%。签名路由靠首 2 字的概率差，
// 温度一高分布被拉平、邻近概念的签名就翻上来。默认降到 0.5。
static constexpr float TEMPERATURE   = 0.5f;
static constexpr size_t ARENA_SIZE   = 40 * 1024;

#if defined(ESP_LLM_PROFILE)
// Per-request LM-head timing. micros() is 32-bit and wraps ~every 71 min, so
// every accumulation uses an explicit unsigned difference.
static uint32_t g_prof_lmhead = 0;   // accumulated us inside the vocab loop
static uint32_t g_prof_b[8] = {0};
uint32_t g_prof_h2f_calls = 0;
uint32_t g_prof_lmrow_calls = 0;
// PB/PE open and close a timing bucket. A single shared slot avoids the
// "redeclaration of _pbN" collision when several buckets are open in the same
// scope; they never nest, so one slot is enough.
static uint32_t g_prof_t0 = 0;
#define PB(i)      (g_prof_t0 = micros())
#define PE(i)      (g_prof_b[i] += (uint32_t)(micros() - g_prof_t0))
#else
#define PB(i)      do {} while (0)
#define PE(i)      do {} while (0)
#endif
static uint32_t g_prof_calls = 0;    // transformer_forward invocations

static uint8_t s_arena_mem[ARENA_SIZE] __attribute__((aligned(4)));
static MemoryArena s_arena(s_arena_mem, ARENA_SIZE);
static MemoryArena* arena = &s_arena;
#elif defined(ESP32S3_BOARD)
// High-capacity profile: ESP32-S3 N16R8 (16MB flash + 8MB PSRAM).
// Arena (4.0MB used) lives in octal PSRAM, supporting up to 512 context tokens.
static constexpr int INFER_CTX       = (512 < (int)model_block_size) ? 512 : (int)model_block_size;
static constexpr int MAX_GEN_TOKENS  = 256;
static constexpr float TEMPERATURE   = 0.85f;
static constexpr size_t ARENA_SIZE   = 4096 * 1024;
static MemoryArena* arena = nullptr;
#else
static constexpr int INFER_CTX       = 64;
static constexpr int MAX_GEN_TOKENS  = 80;
static constexpr float TEMPERATURE   = 0.85f;
static constexpr size_t ARENA_SIZE   = 160 * 1024;
static MemoryArena* arena = nullptr;
#endif

static float* g_x;
// KV cache is int8 with one fp16 scale per (layer, position, kv-head). Float
// storage would exceed the 40 KB arena at the current 104-position window.
static int8_t*  g_kbuf;
static int8_t*  g_vbuf;
static uint16_t* g_kscale;
static uint16_t* g_vscale;
static float* g_xnorm;
static float* g_qkv_out;   
static float* g_attn_out;   
static float* g_proj_out;   
static float* g_att;        
static float* g_mlp_gate;   
static float* g_mlp_up;     
static float* g_mlp_hidden; 
static float* g_mlp_out;    
// No full-vocabulary logits buffer: the head is computed in LOGIT_CHUNK slices
// (see lm_head_chunk).  Dropping it freed 8 KB of arena and is what made
// N_LAYER=5 fit at all.
static uint16_t ctx_ids[INFER_CTX];
static int     ctx_len = 0;    // tokens in the window
static int     ctx_pos = 0;    // how many of them have been through the model
// RoPE position of slot 0.  Positions must stay inside the baked table (128
// rows), so the window slides by drifting this upward and pulling it back to 0
// periodically -- see ctx_slide()/ctx_renumber().  It is deliberately NOT the
// same as the slot index: slot decides where a row lives in the cache, pos
// decides its rotation, and decoupling them is what removes the re-prefill.
static int     ctx_shift = 0;

// Cache geometry, shared by the forward pass and the sliding helpers.
static constexpr int KV_ROW    = N_KV_HEAD * HEAD_DIM;   // int8 values per slot
static constexpr int KV_STRIDE = INFER_CTX * KV_ROW;     // per layer
static constexpr int KS_STRIDE = INFER_CTX * N_KV_HEAD;  // scales per layer
// Slots to free when the window fills.  1 = the window is always exactly
// INFER_CTX long, which is what the PC reference does.  Block-shifting was
// tried (8 at a time, window oscillating 96..104) to cut the copy cost, but the
// board measured the copy at 0.2 ms per slide -- 3 ms over a 200-token story --
// so it bought nothing and cost 8% of the context.
static constexpr int SLIDE_BLOCK = 1;

#if defined(ESP32S3_BOARD)
struct ExpertPSRAMCache {
    uint8_t*  gate_q;
    uint16_t* gate_s;
    uint8_t*  up_q;
    uint16_t* up_s;
    uint8_t*  down_q;
    uint16_t* down_s;
    int       cached_layer;
    int       cached_expert;
};
static ExpertPSRAMCache g_expert_cache = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, -1, -1 };
#endif

// Inference context must not exceed the RoPE tables baked from training.
static_assert(INFER_CTX <= (int)model_block_size, "INFER_CTX exceeds trained block_size: regenerate model_weights.hpp");

// Vocabulary slice width for the LM head. Declared up here because ARENA_NEED
// below has to account for it. 256 keeps the buffer at 1 KB while leaving the
// per-chunk loop overhead under the noise floor of a 15 ms kernel.
static constexpr int LOGIT_CHUNK = 256;

// Everything alloc_buffers() takes out of the arena, in one place. If a future
// export raises block_size (and with it INFER_CTX) past what the 40 KB static
// arena can hold, this fails the build instead of silently handing back null
// pointers at runtime, which on this chip shows up as a boot loop.
//
// The logits term is LOGIT_CHUNK, not model_vocab_size: the head streams in
// slices (lm_head_chunk).  That is the whole reason N_LAYER=5 fits -- see the
// comment above lm_head_chunk for the arithmetic.
static constexpr size_t ARENA_NEED =
      (size_t)N_LAYER * INFER_CTX * N_KV_HEAD * HEAD_DIM * 2   // int8 K and V
    + (size_t)N_LAYER * INFER_CTX * N_KV_HEAD * sizeof(uint16_t) * 2  // scales
    + (size_t)LOGIT_CHUNK * sizeof(float)                      // logits slice (BSS)
    + (size_t)N_EMBD * sizeof(float) * 5                       // x/xnorm/attn/proj/mlp_out
    + (size_t)(N_EMBD + 2 * N_KV_HEAD * HEAD_DIM) * sizeof(float)  // qkv_out
    + (size_t)MLP_HIDDEN * sizeof(float) * 3                   // gate/up/hidden
    + (size_t)INFER_CTX * sizeof(float);                       // attention scores
static_assert(ARENA_NEED <= ARENA_SIZE, "KV cache + scratch exceed the arena: lower INFER_CTX");

// `slot` is where the row is written in the KV cache (and how many rows the
// attention loop walks); `pos` is the RoPE position, which the sliding window
// lets drift away from the slot index.  Keeping them separate is the whole
// trick: a row can be shifted down a slot without being re-encoded.
static void transformer_forward(int token, int slot, int pos) {
    dequant_emb_row(tok_emb_q + (size_t)token * N_EMBD, tok_emb_scales[token], g_x, N_EMBD);
    for (int l = 0; l < N_LAYER; ++l) {
        const LayerW& lw = g_layers[l];
        PB(0);
        rms_norm(g_x, g_xnorm, lw.ln1_g, N_EMBD);
        PE(0);
        PB(1);
        matmul_bitnet_ternary(lw.qkv_w, lw.qkv_s, g_xnorm, g_qkv_out, N_EMBD + 2 * N_KV_HEAD * HEAD_DIM, N_EMBD, GRP);
        PE(1);
        // K/V are produced as float by the qkv matmul, RoPE'd in place, then
        // quantized into the int8 cache. Per (layer, pos, kv-head) absmax keeps
        // the scale local so one outlier cannot flatten the whole cache.
        int8_t*  layer_kq   = g_kbuf   + (size_t)l * KV_STRIDE;
        int8_t*  layer_vq   = g_vbuf   + (size_t)l * KV_STRIDE;
        uint16_t* layer_ks  = g_kscale + (size_t)l * KS_STRIDE;
        uint16_t* layer_vs  = g_vscale + (size_t)l * KS_STRIDE;
        PB(2);
        const float* cos_t = rope_cos + pos * (HEAD_DIM / 2);
        const float* sin_t = rope_sin + pos * (HEAD_DIM / 2);
        for (int h = 0; h < N_HEAD; ++h) {
            apply_rope_row(g_qkv_out + h * HEAD_DIM, cos_t, sin_t, HEAD_DIM);
        }
        for (int h = 0; h < N_KV_HEAD; ++h) {
            apply_rope_row(g_qkv_out + N_EMBD + h * HEAD_DIM, cos_t, sin_t, HEAD_DIM);
        }
        // Quantize the K/V rows for this slot into the cache.  Note `slot`,
        // not `pos`: the row lives at the slot index, its rotation at `pos`.
        for (int h = 0; h < N_KV_HEAD; ++h) {
            quantize_i8_to(g_qkv_out + N_EMBD + h * HEAD_DIM,
                           layer_kq + (size_t)slot * KV_ROW + h * HEAD_DIM,
                           HEAD_DIM);
            layer_ks[slot * N_KV_HEAD + h] = kv_last_scale;
            quantize_i8_to(g_qkv_out + N_EMBD + N_KV_HEAD * HEAD_DIM + h * HEAD_DIM,
                           layer_vq + (size_t)slot * KV_ROW + h * HEAD_DIM,
                           HEAD_DIM);
            layer_vs[slot * N_KV_HEAD + h] = kv_last_scale;
        }
        memset(g_attn_out, 0, N_EMBD * sizeof(float));
        const float scale = 1.0f / sqrtf((float)HEAD_DIM);
        for (int h = 0; h < N_HEAD; ++h) {
            const float* q_h = g_qkv_out + h * HEAD_DIM;
            int kv_h = h / (N_HEAD / N_KV_HEAD);
            for (int s = 0; s <= slot; ++s) {
                const int8_t* krow = layer_kq + (size_t)s * KV_ROW + kv_h * HEAD_DIM;
                float ks = half_to_float(layer_ks[s * N_KV_HEAD + kv_h]);
                g_att[s] = dot_f32_i8scale(q_h, krow, HEAD_DIM, ks) * scale;
            }
            softmax(g_att, slot + 1);
            float* out_h = g_attn_out + h * HEAD_DIM;
            for (int s = 0; s <= slot; ++s) {
                const float a = g_att[s];
                const int8_t* vrow = layer_vq + (size_t)s * KV_ROW + kv_h * HEAD_DIM;
                float vs = half_to_float(layer_vs[s * N_KV_HEAD + kv_h]);
                for (int d = 0; d < HEAD_DIM; ++d) out_h[d] += a * vs * (float)vrow[d];
            }
        }
        PE(2);
        PB(1);
        matmul_bitnet_ternary(lw.proj_w, lw.proj_s, g_attn_out, g_proj_out, N_EMBD, N_EMBD, GRP);
        PE(1);
        for (int d = 0; d < N_EMBD; ++d) g_x[d] += g_proj_out[d];
        PB(0);
        rms_norm(g_x, g_xnorm, lw.ln2_g, N_EMBD);
        PE(0);
        PB(3);
        // Router in int8. It ran N_EXPERTS x N_EMBD sequential FP32 FMAs per
        // (layer, token) -- emulated, since there is no FPU -- and measured
        // ~7% of per-token time. Quantizing the shared input once and doing
        // the per-expert products on the integer ALU removes that.
        // Routing agreement with the fp32 router: 99.36%; end-to-end val loss
        // delta -0.0001 nats (verify_router_int8.py).
        static Int8Vec g_router_x;
        quantize_i8(g_xnorm, N_EMBD, &g_router_x);
        int best_expert = 0;
        float best_score = -1e30f;
        for (int e = 0; e < N_EXPERTS; ++e) {
            int32_t s = dot_emb_q_i8(&g_router_x,
                                     lw.router_q + (size_t)e * N_EMBD, N_EMBD);
            // g_router_x.scale is shared by every expert, so it cancels out of
            // the argmax; only the per-expert weight scale has to be applied.
            float score = (float)s * half_to_float(pgm_read_word(lw.router_s + e));
            if (score > best_score) {
                best_score = score;
                best_expert = e;
            }
        }
        PE(3);
        int gate_n_groups = (N_EMBD + GRP - 1) / GRP;
        int gate_bytes_per_row = gate_n_groups * ((GRP + 4) / 5);
        int gate_q_off = best_expert * MLP_HIDDEN * gate_bytes_per_row;
        int gate_s_off = best_expert * MLP_HIDDEN * gate_n_groups;

        int down_n_groups = (MLP_HIDDEN + GRP - 1) / GRP;
        int down_bytes_per_row = down_n_groups * ((GRP + 4) / 5);
        int down_q_off = best_expert * N_EMBD * down_bytes_per_row;
        int down_s_off = best_expert * N_EMBD * down_n_groups;

        const uint8_t* p_gate_q = lw.experts_gate_q + gate_q_off;
        const uint16_t* p_gate_s = lw.experts_gate_s + gate_s_off;
        const uint8_t* p_up_q = lw.experts_up_q + gate_q_off;
        const uint16_t* p_up_s = lw.experts_up_s + gate_s_off;
        const uint8_t* p_down_q = lw.experts_down_q + down_q_off;
        const uint16_t* p_down_s = lw.experts_down_s + down_s_off;

#if defined(ESP32S3_BOARD)
        if (g_expert_cache.gate_q && (g_expert_cache.cached_layer != l || g_expert_cache.cached_expert != best_expert)) {
            size_t gate_q_sz = (size_t)MLP_HIDDEN * gate_bytes_per_row;
            size_t gate_s_sz = (size_t)MLP_HIDDEN * gate_n_groups * sizeof(uint16_t);
            size_t down_q_sz = (size_t)N_EMBD * down_bytes_per_row;
            size_t down_s_sz = (size_t)N_EMBD * down_n_groups * sizeof(uint16_t);

            memcpy(g_expert_cache.gate_q, p_gate_q, gate_q_sz);
            memcpy(g_expert_cache.gate_s, p_gate_s, gate_s_sz);
            memcpy(g_expert_cache.up_q, p_up_q, gate_q_sz);
            memcpy(g_expert_cache.up_s, p_up_s, gate_s_sz);
            memcpy(g_expert_cache.down_q, p_down_q, down_q_sz);
            memcpy(g_expert_cache.down_s, p_down_s, down_s_sz);

            g_expert_cache.cached_layer = l;
            g_expert_cache.cached_expert = best_expert;
        }
        if (g_expert_cache.gate_q) {
            p_gate_q = g_expert_cache.gate_q;
            p_gate_s = g_expert_cache.gate_s;
            p_up_q   = g_expert_cache.up_q;
            p_up_s   = g_expert_cache.up_s;
            p_down_q = g_expert_cache.down_q;
            p_down_s = g_expert_cache.down_s;
        }
#endif

        PB(4);
        matmul_bitnet_ternary(p_gate_q, p_gate_s,
                        g_xnorm, g_mlp_gate, MLP_HIDDEN, N_EMBD, GRP);
        matmul_bitnet_ternary(p_up_q, p_up_s,
                        g_xnorm, g_mlp_up, MLP_HIDDEN, N_EMBD, GRP);
        PE(4);
        PB(5);
        swiglu(g_mlp_gate, g_mlp_up, g_mlp_hidden, MLP_HIDDEN);
        PE(5);
        PB(4);
        matmul_bitnet_ternary(p_down_q, p_down_s,
                        g_mlp_hidden, g_mlp_out, N_EMBD, MLP_HIDDEN, GRP);
        PE(4);
        for (int d = 0; d < N_EMBD; ++d) g_x[d] += g_mlp_out[d];
        yield();
    }
    PB(0);
    rms_norm(g_x, g_xnorm, ln_f_gamma, N_EMBD);
    PE(0);
    // The head is NOT computed here any more -- see lm_head_chunk().  Prefill
    // steps never needed it (sample_next only reads the logits of the last
    // prompt token), and computing it per step was 15.6 ms of pure waste for
    // every prompt token.  transformer_forward now ends at the final norm.
}

// LM head, one slice of the vocabulary at a time.
//
// Why this is chunked: the head used to write all 2048 logits into an 8 KB
// float buffer inside the arena, and that buffer was the binding constraint on
// model depth.  Each extra layer costs 7,072 B of int8 KV cache, so going from
// 4 to 5 layers needs 5,568 B the 40 KB arena does not have.  Shrinking the
// buffer to LOGIT_CHUNK floats frees 7,168 B, which covers the fifth layer with
// 1,600 B to spare and leaves the arena size itself untouched (so peak static
// RAM does not move).
//
// Top-k selection is a streaming computation -- it only ever needs the running
// k-th best, never the whole row -- so chunking costs nothing in quality.  The
// two-pass paths below (:topk 0) do re-run the head, which is why that escape
// hatch stays a diagnostic and not the default.
static float    s_logits[LOGIT_CHUNK];
static Int8Vec  s_lm_x;          // 128 B + scale, quantized once per sampling

// Quantize the final hidden state once, then run integer dot products against
// vocab rows lo..hi-1, writing s_logits[0..hi-lo-1].  Was: 2048 * N_EMBD
// software float FMAs.
static void lm_head_begin() {
    quantize_i8(g_xnorm, N_EMBD, &s_lm_x);
}

static void lm_head_chunk(int lo, int hi) {
    // Head weights hold the ternary matrix {-1,0,1} that training actually
    // projected through (not an int8 import of the fp32 latent weights),
    // 2-bit packed: 16 B per row instead of 64. Measured on the board this
    // kernel is 10.1 ms/token vs 22.3 ms for the int8 form, and the unpacked
    // dot products are bit-identical. See dot_ternary_p2 for the breakdown.
    const float xs = s_lm_x.scale;
    const int   out = hi - lo;
    for (int i = 0; i < out; ++i) {
        const int v = lo + i;
        int32_t s = dot_ternary_p2(&s_lm_x, lm_head_p2 + (size_t)v * (N_EMBD / 4), N_EMBD);
        s_logits[i] = (float)s * xs * lm_head_scales[v];
    }
}

// Prints the token and reports whether it contained a newline.
//
// Tried and reverted: requiring the newline to follow sentence punctuation, so
// a mid-clause break would not end the reply. On the board that simply let the
// model keep going -- it rarely closes a sentence inside 80 tokens, so chat
// replies turned into 7.4 s rambles and still ended mid-word. Plain "first
// newline wins", bounded by CHAT_MAX_TOKENS, reads better at this model size.
static bool print_token(int id) {
    bool has_newline = false;
    if (id < 0 || id >= (int)model_vocab_size) return false;
    uint32_t start = pgm_read_dword(&model_vocab_offsets[id]);
    uint32_t end   = pgm_read_dword(&model_vocab_offsets[id + 1]);
    for (uint32_t i = start; i < end; ++i) {
        char c = (char)pgm_read_byte(&model_vocab_bytes[i]);
        Serial.print(c);
        if (c == '\n') has_newline = true;
    }
    return has_newline;
}

// ---- 回声 ----
// 把用户的话原样念一遍再加问号，然后由模型表态。
//
// 这件事必须在固件里做，因为它正好踩在模型的硬边界上：在训练集自己的伪名词
// 样本上贪心解码，前两个字命中 0/16（首字常对、次字必错）——5 层 / 2 头 /
// 64 维 / ternary 长不出能精确匹配内容的 QK 电路。固件抄是 100% 准确的，
// 代价是它不像"想出来的"，但用户分辨不出这两者，能分辨的只有"你有没有在听我说话"。
//
// 这是整个方案里唯一还需要判断的地方，而且它不是规则表，只是一个长度窗口：
//   * 1 个字念出来像复读机；
//   * 超过 8 个字念一遍要两秒，而回答根本接不住那么长的前提。
// 训练语料已保证答案不以输入里的名词开头，所以回声和回答不会接成结巴。
static int g_echo = 1;
// 语料里答案前面有一个空格（`prompt + " " + ans`），所以模型第一步总要吐一个空格。
// 回声之后那个空格会变成「青蛙怎么不动了？ 一冻就…」—— 中文里不该有它。
// 只在回声真的打出来之后吞一次，且只吞不打印（仍然 ctx_push，KV 必须和训练一致）。
static int g_swallow_space = 0;

static int utf8_len(const char* s, int len) {
    int n = 0;
    for (int i = 0; i < len; ++i)
        if (((unsigned char)s[i] & 0xC0) != 0x80) ++n;
    return n;
}

// 已经以句末标点收尾的就不再补问号（「今天好热。」不该被念成「今天好热。？」）
static bool ends_with_sentence_punct(const char* s, int len) {
    while (len > 0 && (s[len-1] == ' ' || s[len-1] == '\t' ||
                       s[len-1] == '\r' || s[len-1] == '\n')) --len;
    if (len <= 0) return false;
    char c = s[len-1];
    if (c == '?' || c == '!' || c == '.') return true;
    if (len >= 3) {
        const unsigned char* p = (const unsigned char*)(s + len - 3);
        // ？= EF BC 9F    ！= EF BC 81    。= E3 80 82
        if (p[0] == 0xEF && p[1] == 0xBC && (p[2] == 0x9F || p[2] == 0x81)) return true;
        if (p[0] == 0xE3 && p[1] == 0x80 && p[2] == 0x82) return true;
    }
    return false;
}

static void echo_print(const char* s, int len) {
    g_swallow_space = 0;
    if (!g_echo) return;
    int n = utf8_len(s, len);
    if (n < 2 || n > 8) return;
    for (int i = 0; i < len; ++i) Serial.print(s[i]);
    if (!ends_with_sentence_punct(s, len)) Serial.print(F("？"));
    g_swallow_space = 1;
}

// 1 = slide the KV cache when the window fills (continuous generation),
// 0 = drop a token and re-encode the whole window (exact, ~100x slower).
// Runtime-selectable so the two paths can be A/B'd on the same seed.
static int g_ctx_slide = 1;
// Experimental multi-turn chat. Default stays single-turn so existing
// firmware behavior and latency are unchanged; :history 1 preserves turns.
static int g_chat_history = 0;
// Slide cost accounting, printed at the end of a turn.  The two halves are
// tunable independently (block size vs. renumber interval) and the split is
// the only way to know which one to attack.
static uint32_t g_slide_us = 0, g_renum_us = 0;
static int      g_slide_n = 0, g_renum_n = 0;
static void slide_stats_reset() { g_slide_us = g_renum_us = 0; g_slide_n = g_renum_n = 0; }

// True when the token's text is a leading speaker tag ("Bot", "User", "Bot:",
// ...). The training corpus was written as "Bot: Bot: <answer>" for a third of
// its pairs, so the model likes to emit its own label right after the prompt's
// "Bot:". Detecting it from the vocab table avoids hardcoding token ids, which
// would break if the tokenizer is ever retrained.
static bool token_is_speaker_tag(int id) {
    if (id < 0 || id >= (int)model_vocab_size) return false;
    uint32_t start = pgm_read_dword(&model_vocab_offsets[id]);
    uint32_t end   = pgm_read_dword(&model_vocab_offsets[id + 1]);
    int len = (int)(end - start);
    if (len <= 0 || len > 6) return false;
    static const char* kTags[3] = { "Bot", "User", "bot" };
    for (int t = 0; t < 3; ++t) {
        const char* tag = kTags[t];
        int tl = 0; while (tag[tl]) ++tl;
        if (tl != len) continue;
        bool eq = true;
        for (int j = 0; j < len; ++j) {
            if ((char)pgm_read_byte(&model_vocab_bytes[start + j]) != tag[j]) { eq = false; break; }
        }
        if (eq) return true;
    }
    return false;
}

// Pull every cached position back toward 0 by `delta`, in place.
//
// The cache rows do not need re-encoding, only re-rotating: RoPE is additive
// (verify_rope_shift.py checks this to 1.5e-6, three orders below the int8
// quantization floor), so R(p)k rotated once more by -delta equals R(p-delta)k
// exactly.  That is 5 layers x ctx_len rows of 32 multiplies, versus a full
// re-prefill of the same rows at ~80 ms each.
//
// Cost is charged to the K cache only -- V carries no position.
// Rotate a *stored* K row back by `delta` positions.
//
// apply_rope_row is not a plain rotation: it reads adjacent pairs (x[2i],
// x[2i+1]) but writes the results to the split halves (i, half+i), i.e. it is
// R(theta) . E for a fixed de-interleave E.  A stored row is already past E, so
// rotating it has to skip E -- calling apply_rope_row again would de-interleave
// a second time and scramble the row.  (That mistake cost a whole debug cycle:
// the naive "same row, negated sine" inverse is off by O(1), and it only shows
// up as invented words after the third renumber.)
//
// The angle is -delta*theta, hence the [c s; -s c] block.
// verify_rope_shift.py pins this down: identical to a re-encode to 2e-6, while
// the naive version is wrong by 3.2.
static void rope_rotate_back(float* x, const float* cos_d, const float* sin_d, int dim) {
    const int h = dim / 2;
    float t[HEAD_DIM];
    for (int i = 0; i < h; ++i) {
        const float a = x[i], b = x[h + i];
        const float c = cos_d[i], s = sin_d[i];
        t[i]     = a * c + b * s;
        t[h + i] = -a * s + b * c;
    }
    for (int i = 0; i < dim; ++i) x[i] = t[i];
}

static void ctx_renumber(int delta, int rows) {
    if (delta <= 0 || rows <= 0) return;
    uint32_t t0 = micros();
    const int half = HEAD_DIM / 2;
    const float* cos_d = rope_cos + delta * half;
    const float* sin_d = rope_sin + delta * half;
    float row[HEAD_DIM];
    for (int l = 0; l < N_LAYER; ++l) {
        int8_t*   kq = g_kbuf   + (size_t)l * KV_STRIDE;
        uint16_t* ks = g_kscale + (size_t)l * KS_STRIDE;
        for (int s = 0; s < rows; ++s) {
            for (int h = 0; h < N_KV_HEAD; ++h) {
                int8_t* kp = kq + (size_t)s * KV_ROW + h * HEAD_DIM;
                const float sc = half_to_float(ks[s * N_KV_HEAD + h]);
                for (int d = 0; d < HEAD_DIM; ++d) row[d] = (float)kp[d] * sc;
                rope_rotate_back(row, cos_d, sin_d, HEAD_DIM);
                quantize_i8_to(row, kp, HEAD_DIM);
                ks[s * N_KV_HEAD + h] = kv_last_scale;
            }
        }
    }
    ctx_shift -= delta;
    g_renum_us += (uint32_t)(micros() - t0);
    g_renum_n++;
}

// Slide the window one token to the left, keeping every cached row valid.
//
// Shifting a row down a slot does not change its rotation, so the position of
// slot i has to stay put: ctx_shift++ keeps `ctx_shift + slot` equal to the
// rotation the row was written with.  Positions therefore drift upward with
// each slide and are pulled back when they would leave the baked RoPE table.
static void ctx_slide(int n) {
    if (n <= 0 || ctx_len <= n) return;
    uint32_t t0 = micros();
    for (int l = 0; l < N_LAYER; ++l) {
        memmove(g_kbuf   + (size_t)l * KV_STRIDE, g_kbuf   + (size_t)l * KV_STRIDE + n * KV_ROW,
                (size_t)(ctx_len - n) * KV_ROW);
        memmove(g_vbuf   + (size_t)l * KV_STRIDE, g_vbuf   + (size_t)l * KV_STRIDE + n * KV_ROW,
                (size_t)(ctx_len - n) * KV_ROW);
        memmove(g_kscale + (size_t)l * KS_STRIDE, g_kscale + (size_t)l * KS_STRIDE + n * N_KV_HEAD,
                (size_t)(ctx_len - n) * N_KV_HEAD * sizeof(uint16_t));
        memmove(g_vscale + (size_t)l * KS_STRIDE, g_vscale + (size_t)l * KS_STRIDE + n * N_KV_HEAD,
                (size_t)(ctx_len - n) * N_KV_HEAD * sizeof(uint16_t));
    }
    memmove(ctx_ids, ctx_ids + n, (size_t)(ctx_len - n) * sizeof(uint16_t));
    ctx_len  -= n;
    ctx_shift += n;
    ctx_pos   = ctx_len;   // every surviving row is still prefilled
    // Keep the next query inside the table. With INFER_CTX=104 and 128 rows,
    // renumbering triggers every 25 slides; a 64-position window triggers every
    // 65 slides, but retains less history.
    // Charge only the memmove to the slide bucket; ctx_renumber times itself,
    // and the two are tuned by different knobs.
    g_slide_us += (uint32_t)(micros() - t0);
    g_slide_n++;
    if (ctx_shift + INFER_CTX - 1 >= (int)model_block_size) {
        ctx_renumber(ctx_shift, ctx_len);
    }
}

// Drop prompt tokens while preserving the already-prefilled prefix. The tail
// was appended after the previous turn and has ids but no KV rows yet, so the
// normal ctx_slide() cannot move the whole ctx_len range.
static void ctx_slide_pending(int n) {
    if (n <= 0 || ctx_len <= n) return;
    uint32_t t0 = micros();
    const int old_len = ctx_len;
    const int cached = (ctx_pos < old_len) ? ctx_pos : old_len;
    const int kept = (cached > n) ? (cached - n) : 0;
    for (int l = 0; l < N_LAYER; ++l) {
        memmove(g_kbuf + (size_t)l * KV_STRIDE,
                g_kbuf + (size_t)l * KV_STRIDE + n * KV_ROW,
                (size_t)kept * KV_ROW);
        memmove(g_vbuf + (size_t)l * KV_STRIDE,
                g_vbuf + (size_t)l * KV_STRIDE + n * KV_ROW,
                (size_t)kept * KV_ROW);
        memmove(g_kscale + (size_t)l * KS_STRIDE,
                g_kscale + (size_t)l * KS_STRIDE + n * N_KV_HEAD,
                (size_t)kept * N_KV_HEAD * sizeof(uint16_t));
        memmove(g_vscale + (size_t)l * KS_STRIDE,
                g_vscale + (size_t)l * KS_STRIDE + n * N_KV_HEAD,
                (size_t)kept * N_KV_HEAD * sizeof(uint16_t));
    }
    memmove(ctx_ids, ctx_ids + n, (size_t)(old_len - n) * sizeof(uint16_t));
    ctx_len = old_len - n;
    ctx_pos = kept;
    ctx_shift = kept ? (ctx_shift + n) : 0;
    g_slide_us += (uint32_t)(micros() - t0);
    g_slide_n++;
    if (ctx_shift + INFER_CTX - 1 >= (int)model_block_size) {
        ctx_renumber(ctx_shift, ctx_pos);
    }
}

static void ctx_push(uint16_t id) {
    if (ctx_len >= INFER_CTX) {
        if (ctx_pos == 0) {
            // Nothing has been through the model yet, so the window is still
            // just a token list -- drop from the front without touching the
            // cache.  This is the path a long prompt takes while it is being
            // pushed (ctx_push_str), and it must not pretend the rows are
            // prefilled the way ctx_slide does.
            memmove(ctx_ids, ctx_ids + 1, (size_t)(ctx_len - 1) * sizeof(uint16_t));
            ctx_len  -= 1;
            ctx_shift = 0;
        } else if (g_ctx_slide) {
            ctx_slide(SLIDE_BLOCK);
        } else {
            // Baseline path: drop the oldest token and re-encode the whole
            // window at positions 0..len-1.  Exact, but ~100 forward passes.
            // Kept as the reference side of the `:slide 0` A/B.
            memmove(ctx_ids, ctx_ids + 1, (size_t)(ctx_len - 1) * sizeof(uint16_t));
            ctx_len  -= 1;
            ctx_shift = 0;
            ctx_pos   = 0;
        }
    }
    ctx_ids[ctx_len++] = id;
}

static void ctx_push_str(const char* text) {
    const char* p = text;
    while (*p) {
        int best_id = -1;
        int best_len = 0;
        for (int i = 0; i < (int)model_vocab_size; ++i) {
            uint32_t start = pgm_read_dword(&model_vocab_offsets[i]);
            uint32_t end   = pgm_read_dword(&model_vocab_offsets[i+1]);
            int len = end - start;
            if (len > best_len) {
                bool match = true;
                for (int j = 0; j < len; ++j) {
                    if (p[j] != (char)pgm_read_byte(&model_vocab_bytes[start + j])) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    best_len = len;
                    best_id = i;
                }
            }
        }
        if (best_id != -1) {
            ctx_push((uint16_t)best_id);
            p += best_len;
        } else {
            p++;
        }
    }
}

static bool ctx_text_matches_at(int pos, const char* text) {
    int token = pos;
    while (*text) {
        if (token >= ctx_len) return false;
        int id = (int)ctx_ids[token++];
        if (id >= (int)model_vocab_size) return false;
        uint32_t start = pgm_read_dword(&model_vocab_offsets[id]);
        uint32_t end = pgm_read_dword(&model_vocab_offsets[id + 1]);
        if (end <= start) return false;
        for (uint32_t at = start; at < end; ++at) {
            if (!*text) return true;
            if ((char)pgm_read_byte(&model_vocab_bytes[at]) != *text++) return false;
        }
    }
    return true;
}

static void history_keep_latest_exchange() {
    int drop = 0;
    for (int i = 0; i < ctx_len; ++i) {
        if (ctx_text_matches_at(i, "\nUser: ")) drop = i + 1;
    }
    if (drop <= 0) return;
    if (g_ctx_slide) {
        if (ctx_pos == ctx_len) ctx_slide(drop);
        else ctx_slide_pending(drop);
    } else {
        memmove(ctx_ids, ctx_ids + drop, (size_t)(ctx_len - drop) * sizeof(uint16_t));
        ctx_len -= drop;
        ctx_pos = 0;
        ctx_shift = 0;
    }
}

// ---- Runtime-tunable decoding params (override via serial commands) ----
static float g_temperature = TEMPERATURE;
static int   g_topk        = 40;      // 0 = no top-k (full distribution)
static float g_rep_penalty = 1.0f;
static int   g_rep_window  = 20;
static int   g_mode_story  = 0;       // 0 = chat ("User:/Bot:"), 1 = story continuation
// Story budget, runtime-tunable with ":len <n>".  Unlike the chat budget this
// is not bounded by the window: once the cache slides, generation can run
// indefinitely at ~100 ms/token.  600 tokens is ~1 minute of wall time, which
// is a sane default ceiling for an interactive session.
static int   g_story_len   = MAX_GEN_TOKENS;
static constexpr int STORY_LEN_MAX = 600;
static int   g_intent_rules = 1;      // deterministic replies (":rules 0|1")
static constexpr int TOPK_MAX = 64;
static float    s_topk_val[TOPK_MAX];
static uint16_t s_topk_id[TOPK_MAX];

static inline float rand01() {
#if defined(ESP8266) || defined(ESP8266_BOARD)
    return (float)random(0, 1000000) / 1000000.0f;
#else
    return (float)esp_random() / ((float)UINT32_MAX + 1.0f);
#endif
}

// Sample from the top-k of the logits.
//
// The previous order was: scale all 2048 logits by temperature, run expf over
// all 2048, then find the top-k and renormalise. On ESP8266 expf is emulated,
// so that pass alone cost ~20 ms/token -- the largest non-flash item in the
// profile, showing up as "UNACCOUNTED" because it happens outside the
// transformer.
//
// Top-k commutes with softmax: both are monotone in the logit for temp > 0, so
// selecting first and exponentiating only the k survivors gives the same
// distribution for 2048/k fewer expf calls. The sampled id is now carried
// alongside the value, which also removes the second full-vocabulary scan.
// Repetition penalty without a full-vocabulary table.
//
// The penalty used to be applied by indexing the 2048-float logits array. With
// the head chunked, an arbitrary id needs O(1) lookup, so the recent window is
// folded into a small sorted (id, count) list and the chunk scan walks it with
// a monotone cursor: O(vocab + window) per pass, 192 B of BSS instead of 8 KB
// of arena, and identical arithmetic -- the old code subtracted once per
// occurrence, this one subtracts count times.
static constexpr int REP_MAX = 64;
static uint16_t s_pen_id[REP_MAX];
static uint8_t  s_pen_cnt[REP_MAX];
static int      s_pen_n   = 0;
static int      s_pen_cur = 0;

static void build_penalty() {
    int w = (ctx_len < g_rep_window) ? ctx_len : g_rep_window;
    if (w > REP_MAX) w = REP_MAX;    // :rep p w above 64 is diagnostic only
    s_pen_n = 0;
    for (int c = ctx_len - w; c < ctx_len; ++c) {
        const uint16_t id = ctx_ids[c];
        if (id >= model_vocab_size) continue;
        int j = 0;
        while (j < s_pen_n && s_pen_id[j] != id) ++j;
        if (j == s_pen_n) {
            s_pen_id[s_pen_n]  = id;
            s_pen_cnt[s_pen_n] = 1;
            ++s_pen_n;
        } else if (s_pen_cnt[j] < 255) {
            ++s_pen_cnt[j];
        }
    }
    // Insertion sort by id -- the chunk scan needs ascending order.
    for (int i = 1; i < s_pen_n; ++i) {
        const uint16_t ki = s_pen_id[i];
        const uint8_t  ci = s_pen_cnt[i];
        int j = i - 1;
        while (j >= 0 && s_pen_id[j] > ki) {
            s_pen_id[j + 1]  = s_pen_id[j];
            s_pen_cnt[j + 1] = s_pen_cnt[j];
            --j;
        }
        s_pen_id[j + 1]  = ki;
        s_pen_cnt[j + 1] = ci;
    }
    s_pen_cur = 0;
}

// Penalty for one id, advancing the cursor. Callers must visit ids in ascending
// order within a pass (every path below walks 0..vocab-1) and must reset
// s_pen_cur to 0 at the start of each new pass.
static inline float penalty_of(int id) {
    while (s_pen_cur < s_pen_n && s_pen_id[s_pen_cur] < id) ++s_pen_cur;
    if (s_pen_cur < s_pen_n && s_pen_id[s_pen_cur] == id)
        return g_rep_penalty * (float)s_pen_cnt[s_pen_cur];
    return 0.0f;
}

// ---- 开头禁重 (anti-repeat on the first character) --------------------------
//
// 症状：连着问几句不同的东西，回答全从同一个字起步（"咱……""咱……"），
// 读起来就是车轱辘话，哪怕内容其实不同。369 条规范回复的首字分布本来就集中
// （咱/哼/那/不/当），温度再高也救不了 —— 这是首字层面的塌缩，不是语义问题。
//
// 做法：记住最近 4 条回复的首字，只在本次回复的**第一个可见 token** 上，把命中
// 这些首字的候选概率压到 0。词表是字符级（每个 token 恰好一个汉字），所以
// "禁首字"精确到 token，不需要前缀匹配，也不需要额外内存（4×5 B）。
//
// 为什么只禁首字：字符级词表下一步只出一个字，第二字要到下一步才知道；要禁
// 前两字就得回退已经前向过的 token，而这块板子没有 KV 回滚。首字是"看起来
// 一模一样"的主要来源，够用。
static constexpr int BAN_SLOTS = 4;
static char s_ban_ch[BAN_SLOTS][5];
static int  s_ban_n    = 0;
static int  s_ban_next = 0;
static int  g_ban_first = 0;     // 只在首 token 生效，sample_next 用完即清
static int  g_no_repeat = 1;     // ":norep 0|1"
static bool s_topk_ok[TOPK_MAX];

// 取词表里某个 token 的首个 UTF-8 字符（最多 4 B），返回字节数。
static int token_first_char(int id, char* out) {
    if (id < 0 || id >= (int)model_vocab_size) return 0;
    uint32_t st = pgm_read_dword(&model_vocab_offsets[id]);
    uint32_t en = pgm_read_dword(&model_vocab_offsets[id + 1]);
    if (en <= st) return 0;
    char c0 = (char)pgm_read_byte(&model_vocab_bytes[st]);
    int len = 1;
    if      ((c0 & 0xE0) == 0xC0) len = 2;
    else if ((c0 & 0xF0) == 0xE0) len = 3;
    else if ((c0 & 0xF8) == 0xF0) len = 4;
    if ((uint32_t)len > en - st) len = (int)(en - st);
    if (len > 4) len = 4;
    for (int i = 0; i < len; ++i) out[i] = (char)pgm_read_byte(&model_vocab_bytes[st + i]);
    out[len] = '\0';
    return len;
}

// 换行 token 的 id —— **由 convert_model_to_c.py 按实际词表算出并导出**
// （model_newline_id）。曾经这里写死 199（英文 2048 词表时代 '\n' 的位置），
// 换中文词表后 199 变成「卡」，那个判断就恒真了 —— 每轮都往 ctx 多推一个换行；
// 后来又写死 1810（中文字符表的位置），于是英文 BPE 模型上生成循环不会在换行处停。
// 两种词表现在都正确：BPE 199 / 中文字符表 1810。
static constexpr int TOK_NEWLINE = (int)model_newline_id;

static bool token_is_space(int id) {
    char c[5];
    return token_first_char(id, c) == 1 && c[0] == ' ';
}

static void ban_record_char(const char* ch) {
    if (!g_no_repeat || !ch || !ch[0]) return;
    snprintf(s_ban_ch[s_ban_next], sizeof(s_ban_ch[0]), "%s", ch);
    s_ban_next = (s_ban_next + 1) % BAN_SLOTS;
    if (s_ban_n < BAN_SLOTS) ++s_ban_n;
}

static void ban_record_token(int id) {
    char c[5];
    if (token_first_char(id, c) > 0) ban_record_char(c);
}

// 规则层回复也在 PROGMEM 里，一样记首字 —— 否则规则答完、模型紧接着又用同一个
// 字开头，读起来还是复读。
static void ban_record_flash(const char* p) {
    if (!p) return;
    char c0 = (char)pgm_read_byte(p);
    int len = 1;
    if      ((c0 & 0xE0) == 0xC0) len = 2;
    else if ((c0 & 0xF0) == 0xE0) len = 3;
    else if ((c0 & 0xF8) == 0xF0) len = 4;
    char c[5];
    for (int i = 0; i < len; ++i) c[i] = (char)pgm_read_byte(p + i);
    c[len] = '\0';
    ban_record_char(c);
}

static bool char_banned(const char* c) {
    for (int b = 0; b < s_ban_n; ++b) {
        if (strcmp(s_ban_ch[b], c) == 0) return true;
    }
    return false;
}

static bool token_starts_banned(int id) {
    if (!s_ban_n) return false;
    char c[5];
    if (token_first_char(id, c) <= 0) return false;
    return char_banned(c);
}

// 规则层用：看一条 PROGMEM 回复的首字是否被禁。规则回复是随机抽的，
// 不禁的话"抱抱""捏捏"连着问会抽到同一句，或者连抽两次"喂！摸头会长不高的！"。
static bool flash_starts_banned(const char* p) {
    if (!s_ban_n || !p) return false;
    char c0 = (char)pgm_read_byte(p);
    int len = 1;
    if      ((c0 & 0xE0) == 0xC0) len = 2;
    else if ((c0 & 0xF0) == 0xE0) len = 3;
    else if ((c0 & 0xF8) == 0xF0) len = 4;
    char c[5];
    for (int i = 0; i < len; ++i) c[i] = (char)pgm_read_byte(p + i);
    c[len] = '\0';
    return char_banned(c);
}

// ---- 签名复读环（2026-10-21）------------------------------------------------
// 首字环（s_ban_ch）禁的是"上一个首字"，管不住整句复读。实测：对没学过的
// 问句类型（你饿不饿/你要去哪/你想要什么），模型连续回退到同一个高频安全
// 答案「冰之妖精，幻想乡最快的。」——用户一眼看出"它在背稿"，比答非所问更伤。
//
// 检测量是**答案的前 6 个可见字符**，不是 2 字签名。2 字看起来更灵敏，但
// `stance_lexicon.py` 的设计约束 2 是"每个概念 4 条表态共用同一个 2 字签名"
// —— 用户连问 3 个同一概念的问题（三个大妖精问法）就会误触发，把第 3 个
// 合法答案换成回避池，那正是"规则抢话"的老毛病。6 字前缀下同概念不同表态
// 两两可分（_sigprefix_check.py 全量验证：规则池+词典+DEFLECT 共 408 组
// 零碰撞），只有整句复读才会相撞。
//
// 环记每轮答案的前 6 字 + 该轮输入的前 11 字节：**不同输入**在最近 4 轮给出
// 同前 6 字 >=2 次 = 模型在瞎答。2026-10-22 起处置方式从「擦掉已打印的 6 字 +
// 回避池」升级为**无痕重掷**：检测点前根本不打印（pend 缓冲），第一次撞环就
// 截断 ctx 回 prompt 末尾、提温到 SIG_REROLL_TEMP 整段重抽 —— 语料每概念 4 条
// 表态计数均衡（_stance_balance.py 实测 21~38 次/条），坍缩只是 temp 0.5 太集中，
// 重抽大概率拿到另一条表态；重掷仍撞才走回避池。用户看不见退格擦除过程。
// 同一输入连问两次不触发（同输入同答案是合法的）。纯固件侧，不动模型不动语料。
// **:greedy（temp<=0）整体关闭检测**：贪心没有随机性，一个概念又只有 4 条表态，
// 连问同概念第 3 句必然复读；重掷不提温=重掷同一句，只会把后面每句都推成回避。
// 详见生成循环里的检测点注释。
static constexpr int SIG_SLOTS = 4;
static constexpr int SIG_MAXCHARS = 6;                    // 签名 = 前 6 个字符
static constexpr int SIG_MAXBYTES = SIG_MAXCHARS * 3 + 1; // UTF-8 最坏 3 B/字
static constexpr float SIG_REROLL_TEMP = 0.9f;            // 重掷采样温度
static char s_sig[SIG_SLOTS][SIG_MAXBYTES + 5];           // 多留 5 B 余量
static char s_sig_in[SIG_SLOTS][12];  // 触发轮输入的前 11 字节
static int  s_sig_n = 0, s_sig_next = 0;

static void sig_record(const char* sig, const char* in) {
    if (!sig || !sig[0]) return;
    // 先清零再写：snprintf 只终止不擦尾，残留字节会让 strncmp 比到陈旧数据
    // （两条一模一样的短签名可能因此被判不同）。
    memset(s_sig[s_sig_next], 0, sizeof(s_sig[0]));
    snprintf(s_sig[s_sig_next], sizeof(s_sig[0]), "%s", sig);
    snprintf(s_sig_in[s_sig_next], sizeof(s_sig_in[0]), "%.11s", in ? in : "");
    s_sig_next = (s_sig_next + 1) % SIG_SLOTS;
    if (s_sig_n < SIG_SLOTS) ++s_sig_n;
}

// 规则层回复也入环：规则答完模型接着用同一签名开头，读起来还是复读。
static void sig_record_flash(const char* p, const char* in) {
    if (!p) return;
    char two[SIG_MAXBYTES + 5];
    memset(two, 0, sizeof(two));
    int n = 0, chars = 0;
    while (chars < SIG_MAXCHARS) {
        char c0 = (char)pgm_read_byte(p + n);
        if (c0 == '\0') break;
        int len = 1;
        if      ((c0 & 0xE0) == 0xC0) len = 2;
        else if ((c0 & 0xF0) == 0xE0) len = 3;
        else if ((c0 & 0xF8) == 0xF0) len = 4;
        if (n + len >= (int)sizeof(two) - 1) break;
        for (int i = 0; i < len; ++i) two[n + i] = (char)pgm_read_byte(p + n + i);
        n += len;
        ++chars;
    }
    sig_record(two, in);
}

static bool sig_stuck(const char* sig, const char* in) {
    if (!sig || !sig[0] || s_sig_n < 2) return false;
    int hits = 0;
    for (int b = 0; b < s_sig_n; ++b) {
        // 前 6 字相同才算同一个签名；缓冲区零填充，短签名不会比到陈旧字节
        if (strncmp(s_sig[b], sig, SIG_MAXCHARS * 3) != 0) continue;
        if (strncmp(s_sig_in[b], in ? in : "", 11) == 0) return false;  // 同输入合法
        ++hits;
    }
    return hits >= 2;
}

// 上一条规则回复的指针：用来保证同一个问题连问两次不会原样复读。
static const char* s_last_flash = NULL;

static int sample_next(float temp) {
    if (ctx_len == 0) return 0;
    while (ctx_pos < ctx_len) {
        transformer_forward(ctx_ids[ctx_pos], ctx_pos, ctx_shift + ctx_pos);
        ctx_pos++;
        llm_optimistic_yield(1);
    }
    build_penalty();
    lm_head_begin();
    const int n = (int)model_vocab_size;
    g_prof_calls++;

    // 禁首字只对这一次采样有效，取完就清 —— 否则整段回复都被限制在同一个
    // 首字集合里，越往后越不像人话。
    const bool ban_now = (g_ban_first && s_ban_n > 0);
    g_ban_first = 0;

    if (temp <= 0.0f) {
        // Greedy: one pass, running argmax over penalised logits.
        // 禁首字时另记一个"允许集合里的最优"，以及一个全集合回退值：若某个
        // 输入下整行 top 全是禁用首字（理论上不该发生），宁可复读也不返回空。
        int   best_idx = 0;
        float best     = -3.0e38f;
        int   fb_idx   = 0;
        float fb       = -3.0e38f;
        s_pen_cur = 0;
        for (int lo = 0; lo < n; lo += LOGIT_CHUNK) {
            const int hi = (lo + LOGIT_CHUNK < n) ? lo + LOGIT_CHUNK : n;
            lm_head_chunk(lo, hi);
            for (int i = 0; i < hi - lo; ++i) {
                const int   id = lo + i;
                const float v  = s_logits[i] - penalty_of(id);
                if (v > fb) { fb = v; fb_idx = id; }
                if (ban_now && token_starts_banned(id)) continue;
                if (v > best) { best = v; best_idx = id; }
            }
        }
        return (best > -1.0e37f) ? best_idx : fb_idx;
    }

    if (g_topk <= 0) {
        // Escape hatch (:topk 0) keeps exact full-vocabulary sampling.  It
        // genuinely needs the whole row, so a chunked head costs three passes
        // here: max, sum of exp, then the cumulative walk.  Diagnostic only --
        // 3x the head time, which is why g_topk defaults to 40.
        float mx = -3.0e38f;
        s_pen_cur = 0;
        for (int lo = 0; lo < n; lo += LOGIT_CHUNK) {
            const int hi = (lo + LOGIT_CHUNK < n) ? lo + LOGIT_CHUNK : n;
            lm_head_chunk(lo, hi);
            for (int i = 0; i < hi - lo; ++i) {
                const float v = s_logits[i] - penalty_of(lo + i);
                if (v > mx) mx = v;
            }
        }
        const float inv_t = 1.0f / temp;
        float total = 0.0f;
        s_pen_cur = 0;
        for (int lo = 0; lo < n; lo += LOGIT_CHUNK) {
            const int hi = (lo + LOGIT_CHUNK < n) ? lo + LOGIT_CHUNK : n;
            lm_head_chunk(lo, hi);
            for (int i = 0; i < hi - lo; ++i) {
                const float v = s_logits[i] - penalty_of(lo + i);
                total += fast_exp((v - mx) * inv_t);
            }
        }
        if (!(total > 0.0f)) return 0;
        const float r0 = rand01() * total;
        float c0 = 0.0f;
        s_pen_cur = 0;
        for (int lo = 0; lo < n; lo += LOGIT_CHUNK) {
            const int hi = (lo + LOGIT_CHUNK < n) ? lo + LOGIT_CHUNK : n;
            lm_head_chunk(lo, hi);
            for (int i = 0; i < hi - lo; ++i) {
                const int   id = lo + i;
                const float v  = s_logits[i] - penalty_of(id);
                c0 += fast_exp((v - mx) * inv_t);
                if (r0 <= c0) return id;
            }
        }
        return n - 1;
    }

    int k = g_topk < TOPK_MAX ? g_topk : TOPK_MAX;
    if (k > n) k = n;

    // Insertion-sorted top-k: one compare per vocab entry against the current
    // k-th best, and a shift only when it actually places. k <= 64, so the
    // buffer stays in the static BSS rather than eating arena.  Streaming over
    // chunks is exact -- the k-th best of the whole row is the k-th best of the
    // union, so visiting 256 at a time changes nothing but the buffer size.
    int   best_idx = 0;
    float best     = -3.0e38f;
    int   m        = 0;
    s_pen_cur = 0;
    for (int lo = 0; lo < n; lo += LOGIT_CHUNK) {
        const int hi = (lo + LOGIT_CHUNK < n) ? lo + LOGIT_CHUNK : n;
        lm_head_chunk(lo, hi);
        for (int i = 0; i < hi - lo; ++i) {
            const int   id = lo + i;
            const float v  = s_logits[i] - penalty_of(id);
            if (v > best) { best = v; best_idx = id; }
            if (m >= k) {
                if (!(v > s_topk_val[k - 1])) continue;
                m = k - 1;
            }
            int j = m;
            while (j > 0 && s_topk_val[j - 1] < v) {
                s_topk_val[j] = s_topk_val[j - 1];
                s_topk_id[j]  = s_topk_id[j - 1];
                --j;
            }
            s_topk_val[j] = v;
            s_topk_id[j]  = (uint16_t)id;
            if (m < k) ++m;
        }
    }

    // 禁首字：先标记哪些候选可用，再只对可用的做 softmax。
    // 不动 s_topk_val（后面还要用），另开一个 bool 表，64 B 的 BSS。
    int ok_n = k;
    if (ban_now) {
        ok_n = 0;
        for (int j = 0; j < k; ++j) {
            s_topk_ok[j] = !token_starts_banned(s_topk_id[j]);
            if (s_topk_ok[j]) ++ok_n;
        }
        // 极端情况下 top-k 全被禁（k 很小、候选又高度集中），那就放弃禁重：
        // 复读一句，好过什么都不返回。
        if (ok_n == 0) {
            for (int j = 0; j < k; ++j) s_topk_ok[j] = true;
            ok_n = k;
        }
    } else {
        for (int j = 0; j < k; ++j) s_topk_ok[j] = true;
    }

    const float inv_t = 1.0f / temp;
    const float mx = s_topk_val[0];
    float total = 0.0f;
    for (int j = 0; j < k; ++j) {
        if (!s_topk_ok[j]) { s_topk_val[j] = 0.0f; continue; }
        const float e = fast_exp((s_topk_val[j] - mx) * inv_t);
        s_topk_val[j] = e;
        total += e;
    }
    if (!(total > 0.0f)) return best_idx;
    float r = rand01() * total;
    float cdf = 0.0f;
    for (int j = 0; j < k; ++j) {
        if (!s_topk_ok[j]) continue;
        cdf += s_topk_val[j];
        if (r <= cdf) return s_topk_id[j];
    }
    return best_idx;
}


static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    return p;
}

static bool parse_math_expr(const char*& p, double& val, int& op_count);

static bool parse_math_factor(const char*& p, double& val, int& op_count) {
    p = skip_ws(p);
    if (*p == '+') { p++; return parse_math_factor(p, val, op_count); }
    if (*p == '-') {
        p++;
        double sub = 0.0;
        if (!parse_math_factor(p, sub, op_count)) return false;
        val = -sub;
        return true;
    }
    if (*p == '(') {
        p++;
        if (!parse_math_expr(p, val, op_count)) return false;
        p = skip_ws(p);
        if (*p != ')') return false;
        p++;
        return true;
    }
    char* endp = nullptr;
    val = strtod(p, &endp);
    if (endp == p) return false;
    p = endp;

    p = skip_ws(p);
    if (*p == '^') {
        p++;
        op_count++;
        double exp = 0.0;
        if (!parse_math_factor(p, exp, op_count)) return false;
        val = pow(val, exp);
    }
    return true;
}

static bool parse_math_term(const char*& p, double& val, int& op_count) {
    if (!parse_math_factor(p, val, op_count)) return false;
    while (true) {
        p = skip_ws(p);
        char op = *p;
        if (op == '*' || op == '/' || op == '%') {
            p++;
            op_count++;
            double rhs = 0.0;
            if (!parse_math_factor(p, rhs, op_count)) return false;
            if (op == '*') val *= rhs;
            else if (op == '/') {
                if (fabs(rhs) < 1e-12) return false;
                val /= rhs;
            }
            else if (op == '%') {
                if (fabs(rhs) < 1e-12) return false;
                val = fmod(val, rhs);
            }
        } else {
            break;
        }
    }
    return true;
}

static bool parse_math_expr(const char*& p, double& val, int& op_count) {
    if (!parse_math_term(p, val, op_count)) return false;
    while (true) {
        p = skip_ws(p);
        char op = *p;
        if (op == '+' || op == '-') {
            p++;
            op_count++;
            double rhs = 0.0;
            if (!parse_math_term(p, rhs, op_count)) return false;
            if (op == '+') val += rhs;
            else val -= rhs;
        } else {
            break;
        }
    }
    return true;
}

static bool try_evaluate_math(const char* raw_input, char* output, size_t out_len) {
    if (!raw_input || !*raw_input) return false;

    char clean[128];
    size_t in_len = strlen(raw_input);
    if (in_len >= sizeof(clean)) in_len = sizeof(clean) - 1;
    memcpy(clean, raw_input, in_len);
    clean[in_len] = '\0';

    char lower[128];
    for (size_t i = 0; i <= in_len; ++i) {
        lower[i] = (char)tolower((unsigned char)clean[i]);
    }

    const char* start_ptr = clean;
    const char* lower_ptr = lower;

    const char* prefixes[] = {
        "what is", "what's", "whats", "calculate", "calc", "solve",
        "how much is", "evaluate", "compute", "tell me", "please calculate", "math:"
    };
    for (size_t k = 0; k < sizeof(prefixes)/sizeof(prefixes[0]); ++k) {
        const char* pfx = prefixes[k];
        size_t plen = strlen(pfx);
        if (strncmp(lower_ptr, pfx, plen) == 0) {
            start_ptr += plen;
            lower_ptr += plen;
            break;
        }
    }

    while (*start_ptr == ' ' || *start_ptr == ':' || *start_ptr == '\t') {
        start_ptr++;
    }

    char expr_buf[128];
    size_t elen = strlen(start_ptr);
    if (elen >= sizeof(expr_buf)) elen = sizeof(expr_buf) - 1;
    memcpy(expr_buf, start_ptr, elen);
    expr_buf[elen] = '\0';

    while (elen > 0 && (expr_buf[elen - 1] == '?' || expr_buf[elen - 1] == '=' ||
                        expr_buf[elen - 1] == ' ' || expr_buf[elen - 1] == '.' ||
                        expr_buf[elen - 1] == '!' || expr_buf[elen - 1] == '\r' ||
                        expr_buf[elen - 1] == '\n')) {
        expr_buf[--elen] = '\0';
    }

    if (elen == 0) return false;

    const char* p = expr_buf;
    double result = 0.0;
    int op_count = 0;
    if (!parse_math_expr(p, result, op_count)) return false;

    p = skip_ws(p);
    if (*p != '\0') return false; // Contains trailing non-math characters

    if (op_count == 0) return false; // Must contain at least one operator (+, -, *, /, %, ^)

    if (fabs(result - round(result)) < 1e-7 && fabs(result) < 1e14) {
        snprintf(output, out_len, "%lld", (long long)round(result));
    } else {
        snprintf(output, out_len, "%.6g", result);
    }
    return true;
}

// ---- Intent rules ------------------------------------------------------
//
// Why this layer exists. The checkpoint on the board is 1.30M parameters
// (1.17M excluding embeddings), 4 layers, n_embd=64, trained with a 64-token
// block. Runtime context is now 104 tokens. At that model size,
// size an open-ended "reply to whatever the user says" task is not learnable,
// and the measured behaviour confirms it: every chat pair in the corpus is
// "User: <story summary>\nBot: <story opening>", so `hello` draws a story
// opening rather than a greeting -- the model has only one mapping to apply.
//
// Rather than fight that with more training, deterministic asks are answered
// here and the model is left with the task it is actually good at: writing
// prose. Rule replies cost microseconds instead of the ~3-90 s a forward pass
// takes, so the device also stops feeling broken.
//
// Turn the whole layer off with ":rules 0" to measure the raw model again.

static bool is_word_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
}

// Case-fold, turn punctuation into spaces, collapse runs, trim both ends.
// Matching is done on this form so "Hello!" and "hello" are the same query.
static void normalize_for_match(const char* in, char* out, size_t out_len) {
    size_t o = 0;
    bool prev_space = true;   // also strips leading space
    for (size_t i = 0; in[i] != '\0' && o + 1 < out_len; ++i) {
        char c = in[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c + 32);
        bool sep = (c == ' ' || c == '\t' || c == '\r' || c == '\n' ||
                    c == '?' || c == '!' || c == '.' || c == ',' ||
                    c == ';' || c == ':' || c == '\'' || c == '"' ||
                    c == '(' || c == ')');
        if (sep) {
            if (!prev_space) { out[o++] = ' '; prev_space = true; }
        } else {
            out[o++] = c;
            prev_space = false;
        }
    }
    while (o > 0 && out[o - 1] == ' ') o--;
    out[o] = '\0';
}

// Substring search that requires non-alphanumeric (or string edge) on both
// sides. Without this, "hi" would match "this" and every sentence would look
// like a greeting.
static bool contains_word(const char* hay, const char* needle) {
    size_t nl = strlen_P(needle);
    if (nl == 0) return false;
    const char* p = hay;
    while ((p = strstr_P(p, needle)) != NULL) {
        bool left_ok  = (p == hay) || !is_word_char(p[-1]);
        bool right_ok = !is_word_char(p[nl]);
        if (left_ok && right_ok) return true;
        p += nl;
    }
    return false;
}

// `kws` points into PROGMEM, so each element must be fetched with
// pgm_read_ptr before it can be dereferenced by strstr_P.
static bool matches_any(const char* norm, const char* const* kws, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const char* kw = (const char*)pgm_read_ptr(&kws[i]);
        if (contains_word(norm, kw)) return true;
    }
    return false;
}

// Returns a pointer to the canned reply (in flash) when the input is a
// deterministic request, or NULL to fall through to the model. The reply is
// never copied into RAM -- the caller streams it straight out of flash, so this
// path costs no stack buffer and cannot truncate.
//
// The rules are generated from intent_rules.py (gen_rules_cpp.py): 21 Chinese
// frames, 185 phrases. The English tables and their exact/short/utter match
// modes went out with the English model -- they still fired on English input
// and answered in English, which read as a language switch mid-conversation.
static const char* g_intent_label = "";

static const char* rule_intent_label() { return g_intent_label; }

// Included here, not at the top of the file: the generated code calls
// matches_any above and assigns g_intent_label just below.
#include "intent_rules_gen.h"

static const char* try_intent_rule(const char* raw_input) {
    if (!raw_input || !*raw_input) return NULL;

    char norm[128];
    normalize_for_match(raw_input, norm, sizeof(norm));
    if (norm[0] == '\0') return NULL;
    const char* r = match_intent_rules(norm);
    if (r != NULL) return r;
    // zh_junk：关键词帧全部落空，且输入不含一个非 ASCII 字节（纯英文/数字/
    // 符号/乱按）→ 直接抽回避池，不进模型。词表 ASCII 只有 27 个，这类输入
    // 在模型眼里就是 UNK 序列，它分不清"垃圾"和"没学过的问法"（2026-10-21
    // 实测：temp 0.7 下采样 6/6 串到别的池甚至语言崩坏）；"是不是中文"是
    // 元信息，规则层几行就能判，模型侧再怎么加语料也学不稳。
    bool all_ascii = true;
    for (const unsigned char* p = (const unsigned char*)raw_input; *p; ++p) {
        if (*p >= 0x80) { all_ascii = false; break; }
    }
    if (all_ascii) {
        g_intent_label = "zh_junk";
        return pick_flash(kZhJunkRep, ZH_JUNK_REPS);
    }
    return NULL;
}

static void clear_context() {
    ctx_len = 0;
    ctx_pos = 0;
    ctx_shift = 0;
}

static void append_flash_turn(const char* input, const char* reply) {
    // Intent replies live in PROGMEM. Copy this short line before tokenizing it
    // when history mode is enabled.
    char answer[96];
    int n = 0;
    while (n < (int)sizeof(answer) - 1) {
        char c = (char)pgm_read_byte(reply + n);
        if (!c) break;
        answer[n++] = c;
    }
    answer[n] = '\0';
    char turn[INFER_CTX * 2];
    snprintf(turn, sizeof(turn), "User: %s\nBot: %s\n", input, answer);
    ctx_push_str(turn);
}

// ---- Serial tuning commands: ":temp <v>" ":topk <n>" ":rep <pen> [win]"
//      ":greedy"  ":rand"  ":params"  ":help".  Return true if handled.
static bool handle_tuning_cmd(const char* in) {
    if (in[0] != ':') return false;
    char key[16];
    const char* p = in + 1;
    int ki = 0;
    while (*p && *p != ' ' && ki < 15) { key[ki] = *p; ++p; ++ki; }
    key[ki] = '\0';
    while (*p == ' ') ++p;
    float v1 = atof(p);                 // first numeric arg (skips key)
    while (*p && *p != ' ') ++p;        // advance to second arg
    while (*p == ' ') ++p;
    int   v2 = atoi(p);                 // second numeric arg

    if      (!strcmp(key, "temp"))   { g_temperature = v1; }
    else if (!strcmp(key, "topk"))   { g_topk = (int)v1; if (g_topk < 0) g_topk = 0; }
    else if (!strcmp(key, "rep"))    { g_rep_penalty = v1; if (v2 > 0) g_rep_window = v2; }
    else if (!strcmp(key, "norep"))  { g_no_repeat = (v1 != 0.0f) ? 1 : 0;
                                       if (!g_no_repeat) s_ban_n = 0; }
    else if (!strcmp(key, "greedy")) { g_temperature = 0.0f; }
    else if (!strcmp(key, "rand"))   { g_temperature = TEMPERATURE; }
    else if (!strcmp(key, "rules"))   { g_intent_rules = (v1 != 0.0f) ? 1 : 0; }
    else if (!strcmp(key, "echo"))    { g_echo = (v1 != 0.0f) ? 1 : 0; }
    else if (!strcmp(key, "mode"))   { g_mode_story = (v1 != 0.0f) ? 1 : 0; }
    else if (!strcmp(key, "len"))    {
        g_story_len = (int)v1;
        if (g_story_len < 1) g_story_len = 1;
        if (g_story_len > STORY_LEN_MAX) g_story_len = STORY_LEN_MAX;
    }
    else if (!strcmp(key, "slide"))  { g_ctx_slide = (v1 != 0.0f) ? 1 : 0; }
    else if (!strcmp(key, "history")) {
        g_chat_history = (v1 != 0.0f) ? 1 : 0;
        if (!g_chat_history) clear_context();
    }
    else if (!strcmp(key, "help")) {
        Serial.println(F("  :temp <v>   temperature (0=greedy, 0.6-1.2 sampled)"));
        Serial.println(F("  :topk <n>   top-k cutoff (0 = off)"));
        Serial.println(F("  :rep <p> <w>  repetition penalty / window"));
        Serial.println(F("  :rules <0|1>  intent rules (default 1; 0 = raw model)"));
        Serial.println(F("  :echo <0|1>  repeat user line before the reply (default 1)"));
        Serial.println(F("  :mode <0|1>  0 = chat (User:/Bot:), 1 = story continuation"));
        Serial.println(F("  :len <n>    story budget in tokens (default 80, max 600)"));
        Serial.println(F("  :slide <0|1>  KV-cache slide (1 = continuous, 0 = re-prefill)"));
        Serial.println(F("  :history <0|1>  preserve chat turns in the KV window (default 0)"));
        Serial.println(F("  :greedy / :rand / :params"));
        return true;
    }
    else if (!strcmp(key, "params")) { /* fallthrough to print */ }
    else { Serial.println(F("  unknown command (:help)")); return true; }

    Serial.printf("  temp=%.2f  topk=%d  rep=%.2f/%d  mode=%s  rules=%d  echo=%d  len=%d  slide=%d  history=%d  norep=%d  %s\n",
                  g_temperature, g_topk, g_rep_penalty, g_rep_window,
                  g_mode_story ? "STORY" : "CHAT", g_intent_rules, g_echo,
                  g_mode_story ? g_story_len : (g_chat_history ? HISTORY_CHAT_MAX_TOKENS : CHAT_MAX_TOKENS),
                  g_ctx_slide, g_chat_history,
                  g_no_repeat,
                  g_temperature <= 0.0f ? "[GREEDY]" : "[SAMPLED]");
    return true;
}

// This ESP8266 core (3.x) has no Print::println_P, and Print::print(const
// char*) / printf("%s") both assume the pointer is in RAM -- handing either a
// flash address prints garbage. So the reply is streamed byte by byte out of
// flash with pgm_read_byte. Replies are short (the longest is 95 chars) and go
// out at 115200 baud, so the per-byte cost is irrelevant.
//
// The terminator test MUST also go through pgm_read_byte. A plain `while (*p)`
// dereferences the flash address directly, which is what produced
// "Exception (3) / epc1=0x40202a43 / excvaddr=0x402a8xxx" -- exception 3 is
// LoadStoreError, and 0x402a8xxx is inside the flash segment.
static void print_flash_line(const char* p) {
    char c;
    while ((c = (char)pgm_read_byte(p)) != '\0') {
        Serial.write((uint8_t)c);
        ++p;
    }
    Serial.println();
}

// Labels are plain RAM strings (see INTENT_REPLY), so %s / print() are safe.
// Only the reply needs the flash-aware walk.
static void print_intent_trace(const char* label) {
    Serial.print(F("[0 ms - intent:"));
    Serial.print(label);
    Serial.println(']');
}

// 背稿确认（无痕重掷后仍撞同一个 6 字前缀）：回避池一句话同行接打。
// 背稿签名和回避句都入环 —— 模型继续回退同一句就持续触发，不用重新攒次数。
// 调用时机：pend 缓冲还没打印，所以行面上只有回声，接上去读起来就是正常答案。
static void sig_deflect(const char* sig, const char* in) {
    sig_record(sig, in);
    const char* rep = pick_flash(kZhJunkRep, ZH_JUNK_REPS);
    print_flash_line(rep);   // 同行接打，行尾自带换行
    ban_record_flash(rep);
    sig_record_flash(rep, in);
    print_intent_trace("sig_stuck");
}

// The console sends raw UTF-8, so one Chinese character arrives as three
// bytes. Erasing one byte per backspace left the trailing continuation bytes
// (0x80-0xBF) behind: the terminal still showed a sliver of the old glyph and
// those orphans were fed to the model as input. A backspace must therefore
// step back over a whole code point, and the erasure must repaint every column
// that code point occupied -- CJK and full-width glyphs are two columns wide,
// ASCII and Latin-1 one. Stepping back over anything shorter than three bytes
// and erasing a single column is what made it take three presses and leave
// residue; with the boundary walk below one press erases one character.
static int utf8_prev_start(const char* s, int end) {
    int i = end - 1;
    // 10xxxxxx marks a continuation byte; the code point starts at the first
    // byte that is not one. i > 0 guards the all-continuation degenerate case.
    while (i > 0 && ((uint8_t)s[i] & 0xC0) == 0x80) --i;
    return i;
}

static int utf8_cols(int nbytes) {
    // Three or four bytes covers CJK, kana and full-width punctuation, all of
    // which the console renders double width. Two-byte (accented Latin, Greek)
    // and one-byte stay single width.
    return (nbytes >= 3) ? 2 : 1;
}

static void run_chat() {
    Serial.print("\nUser: ");
    char input[INFER_CTX + 1];
    int  ilen = 0;
    while (true) {
        if (!Serial.available()) continue;
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') { if (ilen > 0) break; continue; }
        if (c == '\b' || c == 0x7F) {
            if (ilen > 0) {
                int start = utf8_prev_start(input, ilen);
                int cols  = utf8_cols(ilen - start);
                ilen = start;
                for (int k = 0; k < cols; ++k) Serial.print("\b \b");
            }
            continue;
        }
        // Arrow keys, Home/End, Delete and function keys all arrive as an ANSI
        // escape sequence (ESC [ ... final, or ESC O final). Without this the
        // ESC byte plus its parameters were echoed as invisible characters and
        // stored in `input`, so pressing Up silently corrupted the prompt. The
        // whole sequence is swallowed here -- no echo, nothing buffered -- and
        // the 60 ms budget keeps a lone Escape press from eating the next key.
        if (c == 0x1b) {
            unsigned long t0 = millis();
            char c2 = 0;
            while (millis() - t0 < 60) {
                if (Serial.available()) { c2 = (char)Serial.read(); break; }
            }
            if (c2 == '[' || c2 == 'O') {
                while (millis() - t0 < 60) {
                    if (!Serial.available()) continue;
                    char c3 = (char)Serial.read();
                    // Final byte of a CSI/SS3 sequence: 0x40-0x7E ends it.
                    if (c3 >= 0x40 && c3 <= 0x7E) break;
                }
            }
            continue;
        }
        // Any other control byte (Tab, Ctrl-*, a stray NUL from the DTR reset)
        // carries no meaning for the model: drop it instead of echoing it.
        if ((uint8_t)c < 0x20) continue;
        if (ilen < (int)sizeof(input) - 1) { input[ilen++] = c; Serial.print(c); }
    }
    input[ilen] = '\0';
    Serial.println();
    if (ilen == 0) return;

    if (!g_chat_history || g_mode_story) clear_context();
    // 禁首字环必须跟着轮次一起清。它记的是**前几轮回复**的开头字，而表态模型
    // 每个概念有固定的 2 字签名 —— 不清的话，上一轮答过「冻的」，这轮再问
    // 青蛙就再也吐不出「冻」开头，被迫串到别的概念。英文时代它是防"复读"的
    // （369 条池子大量共用开头），签名制下它防的正是路由本身：实测 46 连问
    // 路由从 88% 掉到 37%，全耗在这。
    s_ban_n = 0;
    s_ban_next = 0;
    // 注意：s_sig 签名环**不在这里清** —— 它要跨轮记忆才能发现"最近几轮都在
    // 背同一句"（冰之妖精琪露诺连吃 N 个陌生问句）。和 ban 环相反：ban 环每轮
    // 清（签名制下禁首字跨轮会阻断路由），sig 环跨轮累积（检测复读）。

    if (handle_tuning_cmd(input)) return;

    char math_response[64];
    if (try_evaluate_math(input, math_response, sizeof(math_response))) {
        Serial.print("Bot: ");
        Serial.println(math_response);
        Serial.println("[0 ms - Math Harness]");

        char turn[INFER_CTX * 2];
        snprintf(turn, sizeof(turn), "User: %s\nBot: %s\n", input, math_response);
        ctx_push_str(turn);
        return;
    }

    // Intent rules run after math so arithmetic keeps priority (a bare "no"
    // never collides, but "what is 2+2 ok" would), and before the model so a
    // greeting is answered by a string instead of by 60 forward passes.
    //
    // In default single-turn mode, rule replies are not kept in context.
    // History mode appends them as completed turns so the next user turn can
    // refer to them without re-running the rules.
    // 故事模式不走规则层：续写请求里出现的"你是谁"是故事里的一句话，不是
    // 在问设备。中文规则全是子串匹配，不挡就会把叙事句子劫成一句角色台词。
    if (g_intent_rules && !g_mode_story) {
        const char* reply = try_intent_rule(input);
        if (reply != NULL) {
            Serial.print(F("Bot: "));
            print_flash_line(reply);
            ban_record_flash(reply);
            sig_record_flash(reply, input);
            print_intent_trace(rule_intent_label());
            if (g_chat_history) append_flash_turn(input, reply);
            return;
        }
    }

    // Preprocess input: lowercase, strip punctuation, fix "i'm".
    // This normalisation exists because the QA corpus is written in lowercase
    // short questions. Story text must keep its original casing and
    // punctuation, so the step is skipped in story mode -- it also mutates
    // `input` in place, so guarding here is what protects the story prompt
    // built further down.
    char clean_input[INFER_CTX * 2];
    snprintf(clean_input, sizeof(clean_input), "%s", input);
    if (!g_mode_story) {
        int len = strlen(input);
        while (len > 0 && (input[len-1] == '?' || input[len-1] == '!' || input[len-1] == '.' || input[len-1] == ' ')) {
            input[len-1] = '\0';
            len--;
        }
        for (int i = 0; i < len; ++i) {
            if (input[i] >= 'A' && input[i] <= 'Z') {
                input[i] = input[i] + 32;
            }
        }
        if (strncmp(input, "i'm", 3) == 0) {
            snprintf(clean_input, sizeof(clean_input), "I'm%s", input + 3);
        } else if (strncmp(input, "i ", 2) == 0) {
            snprintf(clean_input, sizeof(clean_input), "I %s", input + 2);
        } else {
            snprintf(clean_input, sizeof(clean_input), "%s", input);
        }
    }

    // Prompt mode. The two supported model corpora need different framing:
    //
    //   chat  -- trained on "User: q\nBot: a" pairs, so the question is
    //            wrapped in that template before generation.
    //   story -- trained on raw TinyStories prose with no dialogue markers at
    //            all. Wrapping it in "User:/Bot:" would be an unseen
    //            distribution and the model would answer with dialogue tags
    //            instead of continuing the text, so the line is fed straight
    //            in as the opening of a story.
    //
    // The mode is switchable at runtime because it is a property of the
    // weights, not of the firmware: whichever model is flashed decides which
    // one is correct.
    char prompt[INFER_CTX * 2];
    if (g_mode_story) {
        // No template, no lowercasing, no punctuation stripping -- story text is
        // case- and punctuation-sensitive, and "Once upon a time" lowercased to
        // "once upon a time" is off-distribution.
        snprintf(prompt, sizeof(prompt), "%s", input);
        ctx_push_str(prompt);
        Serial.print("Story>");
    } else {
        // History mode appends the next user turn to the already-prefilled
        // conversation. Keep the latest completed exchange, which matches the
        // training context and avoids letting older topics dominate a follow-up.
        // The previous generation ends in a newline, so do not add another one.
        if (g_chat_history && ctx_len > 0) {
            history_keep_latest_exchange();
            ctx_push_str("User: ");
            ctx_push_str(clean_input);
            ctx_push_str("\nBot:");
        } else {
            snprintf(prompt, sizeof(prompt), "User: %s\nBot:", clean_input);
            ctx_push_str(prompt);
        }
        Serial.print("Bot:");
        // 回声只进显示，不进 prompt —— 模型看不见它，所以它不会去模仿、也不会重复。
        // 用 clean_input（原始输入）而不是 input（已剥尾部标点）：回显要念得像人话。
        echo_print(clean_input, (int)strlen(clean_input));
    }
    // Reserve room for the current chat reply. In single-turn mode this is a
    // cheap token-list trim; in history mode the existing KV rows are trimmed
    // through ctx_slide so the cache remains aligned with ctx_ids.
    //
    // History mode uses a smaller reply budget so the preceding turn remains
    // visible inside the 104-token inference window.
    //
    // Story mode deliberately skips this: :len can be longer than the window,
    // and the whole point of the slide is that the prompt no longer has to be
    // sacrificed to make room for the reply.
    const int chat_budget = g_chat_history ? HISTORY_CHAT_MAX_TOKENS : CHAT_MAX_TOKENS;
    if (!g_mode_story && ctx_len > INFER_CTX - chat_budget) {
        int drop = ctx_len - (INFER_CTX - chat_budget);
        // Newly appended history prompt tokens have ids but no KV rows yet.
        // ctx_slide() only moves already-prefilled rows; using it here would
        // mark those uncached rows as valid and the next sample can terminate
        // immediately. Re-prefill whenever the prompt tail is still pending.
        if (ctx_pos == ctx_len && g_ctx_slide) {
            ctx_slide(drop);
        } else if (ctx_pos > 0 && g_ctx_slide) {
            ctx_slide_pending(drop);
        } else {
            memmove(ctx_ids, ctx_ids + drop, (size_t)(ctx_len - drop) * sizeof(uint16_t));
            ctx_len -= drop;
            ctx_pos = 0;
            ctx_shift = 0;
        }
    }
    unsigned long t_start = millis();
    slide_stats_reset();
#if defined(ESP_LLM_PROFILE)
    g_prof_lmhead = 0; g_prof_calls = 0;
    for (int i = 0; i < 8; ++i) g_prof_b[i] = 0;
    g_prof_h2f_calls = 0; g_prof_lmrow_calls = 0;
#endif
    // Stop rule. Chat replies are one line, so a newline ends them -- but only
    // when it follows sentence punctuation (see print_token): a mid-clause
    // break used to truncate the reply to a fragment. Stories run across
    // paragraphs, so stopping there would clip every one to a single opening
    // paragraph -- the exact opposite of what the story model was trained to
    // produce. In story mode we run to the token budget; with the cache slide
    // that budget is no longer bounded by the window.
    //
    // A long story is ~1 minute of wall time, so ESC must be able to cut it
    // short. peek() is used so any other queued byte (a command typed while it
    // was running) stays in the buffer for the next turn.
    //
    // The `printed > 1` guard exists because a leading "Bot:" tag (learned from
    // the polluted corpus) used to be printed, making replies read
    // "Bot: Bot: ...". Suppression is limited to the first 3 tokens so a
    // legitimate "Bot:" later in the answer still prints.
    int printed = 0;
    int gen_tokens = 0;
    int first_visible_id = -1;
    // 签名复读检测：收集答案前 SIG_MAXCHARS 个可见字符（UTF-8 字节）。
    char vis_sig[SIG_MAXBYTES + 5];
    memset(vis_sig, 0, sizeof(vis_sig));
    int  vis_sig_len = 0;    // 已收集字节数
    int  vis_chars = 0;      // 已收集字符数
    bool sig_handled = false;  // 已走回避池（无痕重掷后仍撞背稿）
    bool sig_checked = false;  // 6 字检测点已过（短答案在循环后补测）
    // 探测期缓冲：6 字检测通过前不打印，第一次撞环才能无痕整段重抽
    // （2026-10-22 升级，取代 \b \b 擦除 —— 用户看得见擦除过程）。
    // print_token 无内部状态（只流式吐字节 + 报是否含换行），延迟调用安全。
    int  pend_id[20];
    int  pend_n = 0;
    const int prompt_len = ctx_len;   // 重掷截断点（prompt 前面已截好余量）
    const int budget = g_mode_story ? g_story_len : chat_budget;
    const float temp_save = g_temperature;
    // 最多两次机会。第一次采样撞背稿签名：截断 ctx 回 prompt 末尾、提温重抽
    // （注意力严格因果 s<=slot，旧 KV 永不被读，不用重算 prompt）；语料每概念
    // 4 条表态计数均衡（_stance_balance.py），坍缩只是 temp 0.5 太集中，0.9
    // 大概率拿到另一条表态。:greedy（temp<=0）不提温 —— 提了就不是贪心，
    // 回归探针依赖确定性，greedy 下直接走回避池。
    for (int attempt = 0; attempt < 2 && !sig_handled; ++attempt) {
        if (attempt == 1) {
            ctx_len = ctx_pos = prompt_len;
            g_swallow_space = 1;          // 重新吞答案前导空格
            g_temperature = SIG_REROLL_TEMP;
            printed = 0;
            first_visible_id = -1;
            memset(vis_sig, 0, sizeof(vis_sig));
            vis_sig_len = 0;
            vis_chars = 0;
            sig_checked = false;
            pend_n = 0;
        }
        bool retry = false;   // 本attempt撞环，丢掉已生成内容进重掷
        for (int i = 0; i < budget; ++i) {
            // ESC aborts a long story.  Serial.available()/peek() never reported
            // the byte from inside this loop on this core, so the read is done
            // directly: -1 means nothing queued.  Any other byte typed mid-story is
            // discarded -- there is nowhere to push it back to.
            if (g_mode_story && Serial.read() == 0x1B) {
                Serial.print(F("\n[stopped at "));
                Serial.print(i);
                Serial.println(F(" tokens]"));
                break;
            }
            // 禁首字只打在第 1 步：第 0 步几乎总是那个前导空格（训练是
            // `prompt + " " + ans`，板端 prefill 停在 "Bot:"），真正的首字在第 1 步。
            g_ban_first = (!g_mode_story && i == 1) ? 1 : 0;
            llm_optimistic_yield(1);
            int next_id  = sample_next(g_temperature);
            gen_tokens++;
            if (next_id == 0) break;
            bool hit_newline = false;
            if (g_swallow_space && printed == 0 && token_is_space(next_id)) {
                g_swallow_space = 0;      // 吞掉回声后那个前导空格（不打印，但仍入 ctx）
            } else if (g_mode_story || i >= 3 || !token_is_speaker_tag(next_id)) {
                if (!g_mode_story && !sig_checked) {
                    // 探测期：只记录不打印
                    if (pend_n < (int)(sizeof(pend_id) / sizeof(pend_id[0]))) {
                        pend_id[pend_n++] = next_id;
                    }
                    // 探测期里 print_token 没被调用，hit_newline 永远是 false ——
                    // 不足 6 字的短答案若不在这里截住，会越过换行继续生成第二行。
                    if (next_id == TOK_NEWLINE) break;
                } else {
                    hit_newline = print_token(next_id);
                    printed++;
                }
                if (!token_is_space(next_id) && next_id != TOK_NEWLINE &&
                    !token_is_speaker_tag(next_id)) {
                    if (first_visible_id < 0) {
                        first_visible_id = next_id;
                    }
                    if (!g_mode_story && vis_chars < SIG_MAXCHARS) {
                        char cb[5];
                        int cl = token_first_char(next_id, cb);
                        if (cl > 0 && vis_sig_len + cl < (int)sizeof(vis_sig) - 1) {
                            memcpy(vis_sig + vis_sig_len, cb, (size_t)cl);
                            vis_sig_len += cl;
                            vis_sig[vis_sig_len] = '\0';
                            ++vis_chars;
                        }
                    }
                    // 签名复读检测点（第 6 个可见字）：前 6 字 = 概念签名级。
                    // 不同输入在最近 4 轮给出同前 6 字 >=2 次 = 模型对这类问句
                    // 没有概念槽、在回退高频安全答案（实测「冰之妖精，幻想乡最快
                    // 的。」连吃陌生问句）。第一次撞：无痕重掷；仍撞：回避池。
                    // 用 6 字而不是 2 字：同概念 4 条表态共用 2 字签名，2 字会误杀
                    // 连问同一概念的合法答案（见签名环注释）。
                    //
                    // **:greedy（temp<=0）整段跳过**：贪心是确定性诊断模式，
                    // 而一个概念只有 4 条表态 —— 连问 5 句同概念的话，第 3 次
                    // 拿到同一句是**必然**而不是"背稿"。而且贪心下重掷不提温
                    // = 重掷出逐字节相同的一句，必然二次撞环 → 直接走回避池，
                    // 于是从第 4 句起全变成「没听过…」。2026-10-22 板上实测：
                    // 睡觉 22 问在贪心下只有 5 句命中，探针完全测不出真实水平
                    // （PC 侧同权重是 22/22）。采样模式（温度 > 0）才启用。
                    if (!g_mode_story && !sig_checked && vis_chars >= SIG_MAXCHARS) {
                        sig_checked = true;
                        if (temp_save > 0.0f && sig_stuck(vis_sig, input)) {
                            if (attempt == 0 && temp_save > 0.0f) {
                                retry = true;
                                break;
                            }
                            sig_deflect(vis_sig, input);
                            sig_handled = true;
                            break;
                        }
                        // 通过：补打印探测期攒的 token，之后恢复流式
                        for (int k = 0; k < pend_n; ++k) {
                            if (print_token(pend_id[k])) hit_newline = true;
                            printed++;
                        }
                        pend_n = 0;
                    }
                }
            }
            ctx_push((uint16_t)next_id);
            if (!g_mode_story && hit_newline && printed > 1) break;
            llm_optimistic_yield(1);
        }
        if (sig_handled) break;
        if (retry) continue;
        // 循环正常结束（换行/预算/终止符）。短答案（<6 字）没到检测点，补测一次。
        // 同样只在采样模式测（贪心下跳过，理由见上面的检测点注释）。
        if (!g_mode_story && !sig_checked) {
            if (temp_save > 0.0f && vis_chars >= 2 && sig_stuck(vis_sig, input)) {
                if (attempt == 0 && temp_save > 0.0f) continue;   // 进重掷
                sig_deflect(vis_sig, input);
                sig_handled = true;
                break;
            }
            for (int k = 0; k < pend_n; ++k) {
                print_token(pend_id[k]);
                printed++;
            }
            pend_n = 0;
        }
        // 模型自己的答案入签名环 —— 否则环里只有规则回复的记录，sig_stuck 永远
        // 看不到历史模型签名、检测失效（2026-10-21 greedy 实测：第 3 个「冰之」
        // 签名未触发，就因为模型答案从未入环）。
        if (!g_mode_story && !sig_handled && vis_chars >= 1) {
            sig_record(vis_sig, input);
        }
        break;   // 答案定稿
    }
    g_temperature = temp_save;
    if (!g_mode_story && first_visible_id >= 0) ban_record_token(first_visible_id);
    if (ctx_len == 0 || ctx_ids[ctx_len - 1] != TOK_NEWLINE) {
        ctx_push_str("\n");
    }
unsigned long elapsed = millis() - t_start;
Serial.println();
#if defined(ESP_LLM_PROFILE)
    if (g_prof_calls) {
        // Buckets hold microseconds; `elapsed` holds milliseconds.
        static const char* kName[8] = {
            "norm", "qkv+proj", "attn+rope", "router",
            "mlp_matmul", "swiglu", "lm_quant", "lm_2048rows"
        };
        const unsigned long tc = g_prof_calls;
        const unsigned long wall_us = elapsed * 1000UL;
        Serial.printf("[prof] %lu fwd, %lu ms wall, %.1f ms/tok\n",
                      tc, elapsed, (float)elapsed / (float)tc);
        unsigned long acc_us = 0;
        for (int i = 0; i < 8; ++i) {
            unsigned long us = g_prof_b[i];
            acc_us += us;
            if (us == 0) continue;
            Serial.printf("[prof]   %-11s %6lu ms  %3lu%%\n", kName[i], us / 1000UL,
                          wall_us ? (us * 100UL / wall_us) : 0UL);
        }
        unsigned long rest = (wall_us > acc_us) ? (wall_us - acc_us) : 0;
        Serial.printf("[prof]   %-11s %6lu ms  %3lu%%\n", "UNACCOUNTED",
                      rest / 1000UL, wall_us ? (rest * 100UL / wall_us) : 0UL);
        Serial.printf("[prof]   calls: h2f=%lu lmrow=%lu\n",
                      (unsigned long)g_prof_h2f_calls,
                      (unsigned long)g_prof_lmrow_calls);
        g_prof_lmhead = 0; g_prof_calls = 0;
        for (int i = 0; i < 8; ++i) g_prof_b[i] = 0;
        g_prof_h2f_calls = 0; g_prof_lmrow_calls = 0;
    }
#endif
    // Token count is printed so latency claims stay measured rather than
    // inferred from reply length: chars-per-token varies with sampling, and a
    // long reply that hit the cap looks identical to one that stopped early.
    Serial.printf("[%lu ms total] %d tok %.1f ms/tok\n", elapsed, gen_tokens,
                  (double)elapsed / (double)(gen_tokens ? gen_tokens : 1));
    // Only printed when the window actually turned over, so the chat path stays
    // quiet.  Split matters: 8 moves per renumber is the current design point,
    // and whichever bucket dominates is the one worth attacking next.
    if (g_slide_n || g_renum_n) {
        Serial.printf("[slide] %d moves %.0f ms total, %d renum %.0f ms total\n",
                      g_slide_n, (double)g_slide_us / 1000.0,
                      g_renum_n, (double)g_renum_us / 1000.0);
    }
}

static void print_model_info() {
    Serial.printf("  Vocab  : %u chars\n",   (unsigned)model_vocab_size);
    Serial.printf("  n_embd : %d\n",          N_EMBD);
    Serial.printf("  n_head : %d\n",          N_HEAD);
    Serial.printf("  n_layer: %d\n",          N_LAYER);
    Serial.printf("  ctx    : %d tokens\n",   INFER_CTX);
    Serial.printf("  mlp_h  : %d\n",          MLP_HIDDEN);
    Serial.printf("  flash  : %u bytes\n",    (unsigned)model_weights_len);
}

static bool alloc_buffers() {
    arena->reset();
    g_x          = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    // INT8 KV cache + per (layer, pos, kv-head) fp16 scales.
    g_kbuf       = (int8_t*) arena->alloc((size_t)N_LAYER * INFER_CTX * N_KV_HEAD * HEAD_DIM);
    g_vbuf       = (int8_t*) arena->alloc((size_t)N_LAYER * INFER_CTX * N_KV_HEAD * HEAD_DIM);
    g_kscale     = (uint16_t*)arena->alloc((size_t)N_LAYER * INFER_CTX * N_KV_HEAD * sizeof(uint16_t));
    g_vscale     = (uint16_t*)arena->alloc((size_t)N_LAYER * INFER_CTX * N_KV_HEAD * sizeof(uint16_t));
    g_xnorm      = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_qkv_out    = (float*)arena->alloc((N_EMBD + 2 * N_KV_HEAD * HEAD_DIM) * sizeof(float));
    g_attn_out   = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_proj_out   = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_att        = (float*)arena->alloc(INFER_CTX              * sizeof(float));
    g_mlp_gate   = (float*)arena->alloc(MLP_HIDDEN             * sizeof(float));
    g_mlp_up     = (float*)arena->alloc(MLP_HIDDEN             * sizeof(float));
    g_mlp_hidden = (float*)arena->alloc(MLP_HIDDEN             * sizeof(float));
    g_mlp_out    = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    // s_logits is a static array of LOGIT_CHUNK floats, not arena: 1 KB in BSS
    // instead of 8 KB in the arena.  Net arena saving 7,168 B.

#if defined(ESP32S3_BOARD)
    int gate_n_groups = (N_EMBD + GRP - 1) / GRP;
    int gate_bytes_per_row = gate_n_groups * ((GRP + 4) / 5);
    size_t gate_q_sz = (size_t)MLP_HIDDEN * gate_bytes_per_row;
    size_t gate_s_sz = (size_t)MLP_HIDDEN * gate_n_groups * sizeof(uint16_t);

    int down_n_groups = (MLP_HIDDEN + GRP - 1) / GRP;
    int down_bytes_per_row = down_n_groups * ((GRP + 4) / 5);
    size_t down_q_sz = (size_t)N_EMBD * down_bytes_per_row;
    size_t down_s_sz = (size_t)N_EMBD * down_n_groups * sizeof(uint16_t);

    g_expert_cache.gate_q = (uint8_t*)arena->alloc(gate_q_sz);
    g_expert_cache.gate_s = (uint16_t*)arena->alloc(gate_s_sz);
    g_expert_cache.up_q   = (uint8_t*)arena->alloc(gate_q_sz);
    g_expert_cache.up_s   = (uint16_t*)arena->alloc(gate_s_sz);
    g_expert_cache.down_q = (uint8_t*)arena->alloc(down_q_sz);
    g_expert_cache.down_s = (uint16_t*)arena->alloc(down_s_sz);
    g_expert_cache.cached_layer = -1;
    g_expert_cache.cached_expert = -1;

    bool expert_ok = g_expert_cache.gate_q && g_expert_cache.gate_s &&
                     g_expert_cache.up_q && g_expert_cache.up_s &&
                     g_expert_cache.down_q && g_expert_cache.down_s;
#else
    bool expert_ok = true;
#endif

    return g_x && g_kbuf && g_vbuf && g_kscale && g_vscale &&
           g_xnorm && g_qkv_out &&
           g_attn_out && g_proj_out && g_att && g_mlp_gate &&
           g_mlp_up && g_mlp_hidden && g_mlp_out && expert_ok;
}

void setup() {
    Serial.begin(115200);
    delay(2000);

#if defined(ESP8266) || defined(ESP8266_BOARD)
    WiFi.mode(WIFI_OFF);
    WiFi.forceSleepBegin();
    delay(1);
    ESP.wdtEnable(5000);
    ESP.wdtFeed();
#elif defined(ESP32S3_BOARD)
    if (!arena) {
        uint8_t* p = (uint8_t*)ps_malloc(ARENA_SIZE); // 8MB octal PSRAM on N16R8
        if (!p) p = (uint8_t*)malloc(ARENA_SIZE);      // last-resort SRAM attempt
        if (p) arena = new MemoryArena(p, ARENA_SIZE);
    }
#else
    if (!arena) {
        arena = new MemoryArena(ARENA_SIZE);
    }
#endif

    Serial.println("\n╔══════════════════════════╗");
    Serial.println("║      ESP-LLM  v1.0       ║");
#if defined(ESP8266) || defined(ESP8266_BOARD)
    Serial.println("║ BitNet 1.58b on ESP8266  ║");
#elif defined(ESP32S3_BOARD)
    Serial.println("║ BitNet 1.58b on ESP32-S3 ║");
#else
    Serial.println("║ BitNet 1.58b on ESP32    ║");
#endif
    Serial.println("╚══════════════════════════╝");
#if defined(ESP32S3_BOARD)
    Serial.printf("PSRAM: %u B | Free heap at boot: %u B\n",
                  (unsigned)ESP.getPsramSize(), (unsigned)ESP.getFreeHeap());
#else
    Serial.printf("Free heap at boot (Wi-Fi OFF): %u B\n", (unsigned)ESP.getFreeHeap());
#endif

    if (!arena || !arena->is_valid() || !alloc_buffers()) {
        Serial.printf("[FAIL] Memory arena allocation failed (need %u KB, free %u B)\n",
                      (unsigned)(ARENA_SIZE / 1024), (unsigned)ESP.getFreeHeap());
        return;
    }
    Serial.printf("[OK] Memory Arena: %u KB (%u / %u B mapped)\n",
                  (unsigned)(ARENA_SIZE / 1024),
                  (unsigned)arena->used(), (unsigned)arena->capacity());
    print_model_info();
    Serial.printf("Free heap remaining: %u B\n\n", (unsigned)ESP.getFreeHeap());
    clear_context();
    Serial.println("Type your message and press Enter.");
    Serial.println("──────────────────────────────────");
}

void loop() {
    if (!arena || !arena->is_valid() || !g_x) {
        delay(5000);
        return;
    }
    run_chat();
}
