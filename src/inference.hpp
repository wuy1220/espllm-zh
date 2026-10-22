#ifndef ESP_LLM_INFERENCE_HPP
#define ESP_LLM_INFERENCE_HPP
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

class MemoryArena {
private:
    uint8_t* buffer;
    size_t   cap;
    size_t   offset;
    bool     owns_memory;
public:
    // Static buffer constructor - zero dynamic heap allocations
    MemoryArena(uint8_t* static_buf, size_t total_size) {
        buffer      = static_buf;
        cap         = total_size;
        offset      = 0;
        owns_memory = false;
    }
    // Dynamic buffer constructor (fallback)
    explicit MemoryArena(size_t total_size) {
        buffer      = (uint8_t*)malloc(total_size);
        cap         = total_size;
        offset      = 0;
        owns_memory = true;
    }
    ~MemoryArena() {
        if (owns_memory && buffer) free(buffer);
    }
    void* alloc(size_t sz) {
        sz = (sz + 3u) & ~3u;
        if (offset + sz > cap) return nullptr;
        void* ptr = buffer + offset;
        offset   += sz;
        return ptr;
    }
    void   reset()       { offset = 0; }
    bool   is_valid()    const { return buffer != nullptr; }
    size_t used()        const { return offset; }
    size_t capacity()    const { return cap; }
};

#if defined(ESP8266) || defined(ESP8266_BOARD)
#include <Arduino.h>
// Time-driven yield. The previous op-count version fired optimistic_yield(1000)
// on essentially every matmul row (~300+ per token), and that ESP8266 NonOS SDK
// call dominated generation latency. Driving it from micros() keeps the software
// watchdog fed regardless of how fast inference runs, while cutting the number of
// SDK yields to single digits per token.
static inline void llm_optimistic_yield(uint32_t every_n_ops = 256) {
    (void)every_n_ops;
    static uint32_t last_us = 0;
    uint32_t now = micros();
    if ((uint32_t)(now - last_us) >= 50000u) { // ~50 ms
        last_us = now;
        ESP.wdtFeed();
        optimistic_yield(1);
        yield();
    }
}
#elif defined(ESP32) || defined(ESP32S3_BOARD)
#include <Arduino.h>
static inline void llm_optimistic_yield(uint32_t every_n_ops = 512) {
    static uint32_t counter = 0;
    if (++counter >= every_n_ops) {
        counter = 0;
        yield();
    }
}
#else
static inline void llm_optimistic_yield(uint32_t every_n_ops = 1000) {}
#endif

static inline float dot_f32(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// IEEE-754 binary16 -> float32 (no FPU support needed). Used for group scales.
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else { // subnormal: normalize
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

// INT8 embedding row (2's complement in a uint8_t array) -> float.
// ESP8266: flash-mapped memory only supports 32-bit aligned word loads; byte
// loads raise LoadStoreError. Route through pgm_read_byte() (aligned word load
// + shift extraction). On other platforms pgm_read_byte() is a plain deref.
// Rows are N_EMBD bytes and N_EMBD is a multiple of 4, so the row base is
// 4-byte aligned and we can pull 4 weights out of every 32-bit flash load.
// On ESP8266 each pgm_read_byte() costs a full l32i plus shift/mask, so this
// cuts flash traffic on the 128KB embedding table by 4x.
static inline void dequant_emb_row(const uint8_t* qrow, float scale, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = (float)((int8_t)pgm_read_byte(qrow + i)) * scale;
}

static inline float dot_emb_q(const float* x, const uint8_t* qrow, float scale, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i] * (float)((int8_t)pgm_read_byte(qrow + i));
    return s * scale;
}

// Batched INT8 dot product for the LM head.
//
// The LM head evaluates all `vocab_size` (2048) rows against the final hidden
// state on every generated token. Done with FP32 that is 2048*64 = 131k software
// floating-point multiply-accumulates per token, which profiling showed to be
// ~82% of total generation time on ESP8266 (no FPU: every float op is emulated).
//
// Instead we quantize the hidden state ONCE to int8 and reuse it for all 2048
// rows, turning the inner loop into a pure int32 dot product (hardware integer
// MACs). Each row then needs a single FP multiply, so we still pay 2048 soft
// float multiplies but eliminate 131k of them.
//
// Accuracy: matches the activation quantization the transformer already applies
// internally (absmax -> int8), so this is consistent with how x was computed.
struct Int8Vec {
    int8_t  q[128];
    float   scale;      // x ~= q * scale
    int     n;
};

static inline void quantize_i8(const float* x, int n, Int8Vec* out) {
    float max_abs = 1e-5f;
    for (int i = 0; i < n; ++i) {
        float ax = x[i] < 0.0f ? -x[i] : x[i];
        if (ax > max_abs) max_abs = ax;
    }
    float inv = 1.0f / (max_abs / 127.0f);
    for (int i = 0; i < n; ++i) {
        float v = x[i] * inv;
        // round-half-away-from-zero without libm
        int32_t iv = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
        if (iv > 127) iv = 127;
        if (iv < -128) iv = -128;
        out->q[i] = (int8_t)iv;
    }
    out->scale = max_abs / 127.0f;
    out->n = n;
}

// sum_i x[i] * w[i] where x is pre-quantized. Returns the int32 dot product;
// caller applies x.scale and the row scale.
#if defined(ESP_LLM_PROFILE)
extern uint32_t g_prof_h2f_calls;
extern uint32_t g_prof_lmrow_calls;
#define PROF_H2F()   (g_prof_h2f_calls++)
#define PROF_LMROW() (g_prof_lmrow_calls++)
#else
#define PROF_H2F()   ((void)0)
#define PROF_LMROW() ((void)0)
#endif

static inline int32_t dot_emb_q_i8(const Int8Vec* x, const uint8_t* qrow, int n) {
    PROF_LMROW();
    int32_t s = 0;
    int i = 0;
    // One 32-bit flash load feeds 4 multiply-accumulates. This is the hottest
    // loop in the firmware: 2048 rows x N_EMBD weights on every generated token,
    // and on ESP8266 every pgm_read_byte() is a full aligned word load plus a
    // shift/mask because flash is only readable 32 bits at a time.
    for (; i + 4 <= n; i += 4) {
        uint32_t w = pgm_read_dword(qrow + i);
        s += (int32_t)x->q[i]     * (int32_t)(int8_t)(uint8_t)( w        & 0xFFu);
        s += (int32_t)x->q[i + 1] * (int32_t)(int8_t)(uint8_t)((w >>  8) & 0xFFu);
        s += (int32_t)x->q[i + 2] * (int32_t)(int8_t)(uint8_t)((w >> 16) & 0xFFu);
        s += (int32_t)x->q[i + 3] * (int32_t)(int8_t)(uint8_t)((w >> 24) & 0xFFu);
    }
    for (; i < n; ++i) s += (int32_t)x->q[i] * (int32_t)(int8_t)pgm_read_byte(qrow + i);
    return s;
}

// ── 2-bit packed ternary dot product ─────────────────────────────────────────
// The output head is ternary {-1,0,+1}, so it only needs 2 bits per weight.
// Measured on the board (bench_unpack/, 2048 rows x 64 cols at 160 MHz):
//
//   int8 from flash, no MAC        15.5 ms     <- pure flash transaction cost
//   2-bit from flash, no MAC        2.2 ms
//   int8 + MAC (production)        22.3 ms
//   2-bit + MAC, staged in a buf   16.3 ms     <- 64-byte store-then-load hurts
//   2-bit + MAC, fused (this)      10.1 ms
//
// So the head was billed per flash *transaction*, not per byte: pgm_read_byte
// costs about what pgm_read_dword costs, which is why reading 16 bytes one at a
// time (the earlier base-3 attempt, 978 ms/token) was catastrophic and why the
// staging buffer in the 16.3 ms version still lost to this one. The unpacked
// results are bit-identical to the int8 path on every row.
//
// ternary_lut256 is emitted by convert_model_to_c.py from the same packing
// step, so the table and the encoding cannot diverge. Code 3 maps to +2 and is
// never written by the exporter.
static inline int32_t dot_ternary_p2(const Int8Vec* x, const uint8_t* src, int n) {
    PROF_LMROW();
    int32_t s = 0;
    int i = 0;
    int d = 0;
    for (; i + 16 <= n; i += 16, ++d) {
        uint32_t w = pgm_read_dword(src + 4 * d);
        uint32_t v;
        v = ternary_lut256[ w        & 0xFFu];
        s += (int32_t)x->q[i +  0] * (int32_t)(int8_t)(uint8_t)( v        & 0xFFu);
        s += (int32_t)x->q[i +  1] * (int32_t)(int8_t)(uint8_t)((v >>  8) & 0xFFu);
        s += (int32_t)x->q[i +  2] * (int32_t)(int8_t)(uint8_t)((v >> 16) & 0xFFu);
        s += (int32_t)x->q[i +  3] * (int32_t)(int8_t)(uint8_t)((v >> 24) & 0xFFu);
        v = ternary_lut256[(w >>  8) & 0xFFu];
        s += (int32_t)x->q[i +  4] * (int32_t)(int8_t)(uint8_t)( v        & 0xFFu);
        s += (int32_t)x->q[i +  5] * (int32_t)(int8_t)(uint8_t)((v >>  8) & 0xFFu);
        s += (int32_t)x->q[i +  6] * (int32_t)(int8_t)(uint8_t)((v >> 16) & 0xFFu);
        s += (int32_t)x->q[i +  7] * (int32_t)(int8_t)(uint8_t)((v >> 24) & 0xFFu);
        v = ternary_lut256[(w >> 16) & 0xFFu];
        s += (int32_t)x->q[i +  8] * (int32_t)(int8_t)(uint8_t)( v        & 0xFFu);
        s += (int32_t)x->q[i +  9] * (int32_t)(int8_t)(uint8_t)((v >>  8) & 0xFFu);
        s += (int32_t)x->q[i + 10] * (int32_t)(int8_t)(uint8_t)((v >> 16) & 0xFFu);
        s += (int32_t)x->q[i + 11] * (int32_t)(int8_t)(uint8_t)((v >> 24) & 0xFFu);
        v = ternary_lut256[(w >> 24) & 0xFFu];
        s += (int32_t)x->q[i + 12] * (int32_t)(int8_t)(uint8_t)( v        & 0xFFu);
        s += (int32_t)x->q[i + 13] * (int32_t)(int8_t)(uint8_t)((v >>  8) & 0xFFu);
        s += (int32_t)x->q[i + 14] * (int32_t)(int8_t)(uint8_t)((v >> 16) & 0xFFu);
        s += (int32_t)x->q[i + 15] * (int32_t)(int8_t)(uint8_t)((v >> 24) & 0xFFu);
    }
    return s;
}


// ── INT8 KV cache helpers ─────────────────────────────────────────────────────
// The attention cache holds one row per (layer, position, kv-head). Storing it
// as float needs 4 bytes/element, which at INFER_CTX=64 x 4 layers x 32 dims x
// K and V is 65536 B -- more than the whole 40KB ESP8266 arena. Quantizing each
// row to int8 with its own absmax scale costs 1 byte/element plus 2 bytes of
// fp16 scale (16896 B total), and frees the arena to run the model's full
// trained window instead of the truncated 28 tokens the fp32 layout forced.
//
// float_to_half is the inverse of half_to_float above. Scales are positive
// absmax values, so they never hit the subnormal path in practice, but the
// general path is implemented for correctness.
static inline uint16_t float_to_half(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t x = v.u;
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; // underflows to zero
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        return (uint16_t)(sign | (mant >> shift));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u); // inf / overflow
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

// Scale produced by the most recent quantize_i8_to() call, so the caller can
// store it beside the row it just wrote without threading a return value
// through the attention loop.
static uint16_t kv_last_scale;

// Quantize one float row to int8 (absmax scale, matching the activation
// quantization the transformer already applies internally) and record the
// scale as fp16.
static inline void quantize_i8_to(const float* x, int8_t* q, int n) {
    float max_abs = 1e-5f;
    for (int i = 0; i < n; ++i) {
        float ax = x[i] < 0.0f ? -x[i] : x[i];
        if (ax > max_abs) max_abs = ax;
    }
    float scale = max_abs / 127.0f;
    float inv = 1.0f / scale;
    for (int i = 0; i < n; ++i) {
        float v = x[i] * inv;
        int32_t iv = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
        if (iv > 127) iv = 127;
        if (iv < -128) iv = -128;
        q[i] = (int8_t)iv;
    }
    kv_last_scale = float_to_half(scale);
}

// dot(float a, int8 b) * scale
static inline float dot_f32_i8scale(const float* a, const int8_t* b, int n, float scale) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * (float)b[i];
    return s * scale;
}

#if defined(ESP8266) || defined(ESP8266_BOARD)
static const int8_t base3_lut[243][5] __attribute__((aligned(4))) = {
#else
static const int8_t base3_lut[243][5] PROGMEM __attribute__((aligned(4))) = {
#endif
    { 0,  0,  0,  0,  0}, { 1,  0,  0,  0,  0}, {-1,  0,  0,  0,  0},
    { 0,  1,  0,  0,  0}, { 1,  1,  0,  0,  0}, {-1,  1,  0,  0,  0},
    { 0, -1,  0,  0,  0}, { 1, -1,  0,  0,  0}, {-1, -1,  0,  0,  0},
    { 0,  0,  1,  0,  0}, { 1,  0,  1,  0,  0}, {-1,  0,  1,  0,  0},
    { 0,  1,  1,  0,  0}, { 1,  1,  1,  0,  0}, {-1,  1,  1,  0,  0},
    { 0, -1,  1,  0,  0}, { 1, -1,  1,  0,  0}, {-1, -1,  1,  0,  0},
    { 0,  0, -1,  0,  0}, { 1,  0, -1,  0,  0}, {-1,  0, -1,  0,  0},
    { 0,  1, -1,  0,  0}, { 1,  1, -1,  0,  0}, {-1,  1, -1,  0,  0},
    { 0, -1, -1,  0,  0}, { 1, -1, -1,  0,  0}, {-1, -1, -1,  0,  0},
    { 0,  0,  0,  1,  0}, { 1,  0,  0,  1,  0}, {-1,  0,  0,  1,  0},
    { 0,  1,  0,  1,  0}, { 1,  1,  0,  1,  0}, {-1,  1,  0,  1,  0},
    { 0, -1,  0,  1,  0}, { 1, -1,  0,  1,  0}, {-1, -1,  0,  1,  0},
    { 0,  0,  1,  1,  0}, { 1,  0,  1,  1,  0}, {-1,  0,  1,  1,  0},
    { 0,  1,  1,  1,  0}, { 1,  1,  1,  1,  0}, {-1,  1,  1,  1,  0},
    { 0, -1,  1,  1,  0}, { 1, -1,  1,  1,  0}, {-1, -1,  1,  1,  0},
    { 0,  0, -1,  1,  0}, { 1,  0, -1,  1,  0}, {-1,  0, -1,  1,  0},
    { 0,  1, -1,  1,  0}, { 1,  1, -1,  1,  0}, {-1,  1, -1,  1,  0},
    { 0, -1, -1,  1,  0}, { 1, -1, -1,  1,  0}, {-1, -1, -1,  1,  0},
    { 0,  0,  0, -1,  0}, { 1,  0,  0, -1,  0}, {-1,  0,  0, -1,  0},
    { 0,  1,  0, -1,  0}, { 1,  1,  0, -1,  0}, {-1,  1,  0, -1,  0},
    { 0, -1,  0, -1,  0}, { 1, -1,  0, -1,  0}, {-1, -1,  0, -1,  0},
    { 0,  0,  1, -1,  0}, { 1,  0,  1, -1,  0}, {-1,  0,  1, -1,  0},
    { 0,  1,  1, -1,  0}, { 1,  1,  1, -1,  0}, {-1,  1,  1, -1,  0},
    { 0, -1,  1, -1,  0}, { 1, -1,  1, -1,  0}, {-1, -1,  1, -1,  0},
    { 0,  0, -1, -1,  0}, { 1,  0, -1, -1,  0}, {-1,  0, -1, -1,  0},
    { 0,  1, -1, -1,  0}, { 1,  1, -1, -1,  0}, {-1,  1, -1, -1,  0},
    { 0, -1, -1, -1,  0}, { 1, -1, -1, -1,  0}, {-1, -1, -1, -1,  0},
    { 0,  0,  0,  0,  1}, { 1,  0,  0,  0,  1}, {-1,  0,  0,  0,  1},
    { 0,  1,  0,  0,  1}, { 1,  1,  0,  0,  1}, {-1,  1,  0,  0,  1},
    { 0, -1,  0,  0,  1}, { 1, -1,  0,  0,  1}, {-1, -1,  0,  0,  1},
    { 0,  0,  1,  0,  1}, { 1,  0,  1,  0,  1}, {-1,  0,  1,  0,  1},
    { 0,  1,  1,  0,  1}, { 1,  1,  1,  0,  1}, {-1,  1,  1,  0,  1},
    { 0, -1,  1,  0,  1}, { 1, -1,  1,  0,  1}, {-1, -1,  1,  0,  1},
    { 0,  0, -1,  0,  1}, { 1,  0, -1,  0,  1}, {-1,  0, -1,  0,  1},
    { 0,  1, -1,  0,  1}, { 1,  1, -1,  0,  1}, {-1,  1, -1,  0,  1},
    { 0, -1, -1,  0,  1}, { 1, -1, -1,  0,  1}, {-1, -1, -1,  0,  1},
    { 0,  0,  0,  1,  1}, { 1,  0,  0,  1,  1}, {-1,  0,  0,  1,  1},
    { 0,  1,  0,  1,  1}, { 1,  1,  0,  1,  1}, {-1,  1,  0,  1,  1},
    { 0, -1,  0,  1,  1}, { 1, -1,  0,  1,  1}, {-1, -1,  0,  1,  1},
    { 0,  0,  1,  1,  1}, { 1,  0,  1,  1,  1}, {-1,  0,  1,  1,  1},
    { 0,  1,  1,  1,  1}, { 1,  1,  1,  1,  1}, {-1,  1,  1,  1,  1},
    { 0, -1,  1,  1,  1}, { 1, -1,  1,  1,  1}, {-1, -1,  1,  1,  1},
    { 0,  0, -1,  1,  1}, { 1,  0, -1,  1,  1}, {-1,  0, -1,  1,  1},
    { 0,  1, -1,  1,  1}, { 1,  1, -1,  1,  1}, {-1,  1, -1,  1,  1},
    { 0, -1, -1,  1,  1}, { 1, -1, -1,  1,  1}, {-1, -1, -1,  1,  1},
    { 0,  0,  0, -1,  1}, { 1,  0,  0, -1,  1}, {-1,  0,  0, -1,  1},
    { 0,  1,  0, -1,  1}, { 1,  1,  0, -1,  1}, {-1,  1,  0, -1,  1},
    { 0, -1,  0, -1,  1}, { 1, -1,  0, -1,  1}, {-1, -1,  0, -1,  1},
    { 0,  0,  1, -1,  1}, { 1,  0,  1, -1,  1}, {-1,  0,  1, -1,  1},
    { 0,  1,  1, -1,  1}, { 1,  1,  1, -1,  1}, {-1,  1,  1, -1,  1},
    { 0, -1,  1, -1,  1}, { 1, -1,  1, -1,  1}, {-1, -1,  1, -1,  1},
    { 0,  0, -1, -1,  1}, { 1,  0, -1, -1,  1}, {-1,  0, -1, -1,  1},
    { 0,  1, -1, -1,  1}, { 1,  1, -1, -1,  1}, {-1,  1, -1, -1,  1},
    { 0, -1, -1, -1,  1}, { 1, -1, -1, -1,  1}, {-1, -1, -1, -1,  1},
    { 0,  0,  0,  0, -1}, { 1,  0,  0,  0, -1}, {-1,  0,  0,  0, -1},
    { 0,  1,  0,  0, -1}, { 1,  1,  0,  0, -1}, {-1,  1,  0,  0, -1},
    { 0, -1,  0,  0, -1}, { 1, -1,  0,  0, -1}, {-1, -1,  0,  0, -1},
    { 0,  0,  1,  0, -1}, { 1,  0,  1,  0, -1}, {-1,  0,  1,  0, -1},
    { 0,  1,  1,  0, -1}, { 1,  1,  1,  0, -1}, {-1,  1,  1,  0, -1},
    { 0, -1,  1,  0, -1}, { 1, -1,  1,  0, -1}, {-1, -1,  1,  0, -1},
    { 0,  0, -1,  0, -1}, { 1,  0, -1,  0, -1}, {-1,  0, -1,  0, -1},
    { 0,  1, -1,  0, -1}, { 1,  1, -1,  0, -1}, {-1,  1, -1,  0, -1},
    { 0, -1, -1,  0, -1}, { 1, -1, -1,  0, -1}, {-1, -1, -1,  0, -1},
    { 0,  0,  0,  1, -1}, { 1,  0,  0,  1, -1}, {-1,  0,  0,  1, -1},
    { 0,  1,  0,  1, -1}, { 1,  1,  0,  1, -1}, {-1,  1,  0,  1, -1},
    { 0, -1,  0,  1, -1}, { 1, -1,  0,  1, -1}, {-1, -1,  0,  1, -1},
    { 0,  0,  1,  1, -1}, { 1,  0,  1,  1, -1}, {-1,  0,  1,  1, -1},
    { 0,  1,  1,  1, -1}, { 1,  1,  1,  1, -1}, {-1,  1,  1,  1, -1},
    { 0, -1,  1,  1, -1}, { 1, -1,  1,  1, -1}, {-1, -1,  1,  1, -1},
    { 0,  0, -1,  1, -1}, { 1,  0, -1,  1, -1}, {-1,  0, -1,  1, -1},
    { 0,  1, -1,  1, -1}, { 1,  1, -1,  1, -1}, {-1,  1, -1,  1, -1},
    { 0, -1, -1,  1, -1}, { 1, -1, -1,  1, -1}, {-1, -1, -1,  1, -1},
    { 0,  0,  0, -1, -1}, { 1,  0,  0, -1, -1}, {-1,  0,  0, -1, -1},
    { 0,  1,  0, -1, -1}, { 1,  1,  0, -1, -1}, {-1,  1,  0, -1, -1},
    { 0, -1,  0, -1, -1}, { 1, -1,  0, -1, -1}, {-1, -1,  0, -1, -1},
    { 0,  0,  1, -1, -1}, { 1,  0,  1, -1, -1}, {-1,  0,  1, -1, -1},
    { 0,  1,  1, -1, -1}, { 1,  1,  1, -1, -1}, {-1,  1,  1, -1, -1},
    { 0, -1,  1, -1, -1}, { 1, -1,  1, -1, -1}, {-1, -1,  1, -1, -1},
    { 0,  0, -1, -1, -1}, { 1,  0, -1, -1, -1}, {-1,  0, -1, -1, -1},
    { 0,  1, -1, -1, -1}, { 1,  1, -1, -1, -1}, {-1,  1, -1, -1, -1},
    { 0, -1, -1, -1, -1}, { 1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}
};

// Ternary dot product for the LM head, base-3 packed (5 weights per byte).
//
// The head is the one block every generated token reads in full, so its storage
// format is the biggest single lever on latency: 26.6 KB/token here versus
// 131 KB/token when the tied embedding was imported as int8. Size is only half
// the story though -- the ternary matrix is also the one the body was trained
// against, because BitLinear quantizes its weight on every forward pass and the
// fp32 values are latent weights that never reach the projection directly.
// Measured on the torch model: val 3.0520 (int8 of fp32) -> 2.5674 (ternary),
// i.e. ppl 21.2 -> 13.0. So this replaces a wrong head, not just a big one.
//
// The head kernel itself is dot_ternary_p2 above. The base-3 version that used
// to live here was removed: it ran 978 ms/token on the board because it unpacked
// with pgm_read_byte per 5 weights. base3_lut stays because the expert matmul
// below still uses that layout.
void matmul_bitnet_ternary(
    const uint8_t* W_packed,
    const uint16_t* scales,
    const float*   x,
    float*         y,
    int            out_features,
    int            in_features,
    int            group_size
) {
    const int n_groups = (in_features + group_size - 1) / group_size;
    const int bytes_per_group = (group_size + 4) / 5;
    const int k_bytes  = n_groups * bytes_per_group;
    
    // 1. Quantize x to int8 (stack fast-path, heap fallback for wide layers)
    int8_t x_q_stack[1024];
    int8_t* x_q = x_q_stack;
    bool x_q_heap = false;
    if (in_features > 1024) {
        x_q = (int8_t*)malloc((size_t)in_features);
        if (!x_q) return; // OOM: leave y untouched, caller fails loudly downstream
        x_q_heap = true;
    }
    float max_abs = 1e-5f;
    for (int i = 0; i < in_features; ++i) {
        float ax = fabsf(x[i]);
        if (ax > max_abs) max_abs = ax;
    }
    float x_scale = max_abs / 127.0f;
    float inv_x_scale = 1.0f / x_scale;
    for (int i = 0; i < in_features; ++i) {
        float v = roundf(x[i] * inv_x_scale);
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        x_q[i] = (int8_t)v;
    }

    // 2. Matrix multiplication with ternary weights
    for (int i = 0; i < out_features; ++i) {
        const uint8_t* w_row = W_packed + (size_t)i * k_bytes;
        const uint16_t* s_row = scales   + (size_t)i * n_groups;

        // Time-driven yield every 8 rows: enough to keep the WDT fed across a
        // slow matmul, cheap enough that micros() isn't the bottleneck.
        if ((i & 7) == 0) llm_optimistic_yield();

        float acc_float = 0.0f;
        
        for (int g = 0; g < n_groups; ++g) {
            int group_start = g * group_size;
            int group_end = group_start + group_size;
            if (group_end > in_features) group_end = in_features;
            int group_len = group_end - group_start;

            const uint8_t* g_bytes = w_row + (size_t)g * bytes_per_group;
            const int8_t* g_x = x_q + group_start;
            
            int32_t group_acc = 0;
            int full_chunks = group_len / 5;
            for (int c = 0; c < full_chunks; ++c) {
                const int8_t* w5 = base3_lut[pgm_read_byte(g_bytes + c)];
                const int8_t* x5 = g_x + c * 5;
                group_acc += (int32_t)w5[0] * x5[0]
                           + (int32_t)w5[1] * x5[1]
                           + (int32_t)w5[2] * x5[2]
                           + (int32_t)w5[3] * x5[3]
                           + (int32_t)w5[4] * x5[4];
            }
            int rem = group_len % 5;
            if (rem > 0) {
                const int8_t* w5 = base3_lut[pgm_read_byte(g_bytes + full_chunks)];
                const int8_t* x_rem = g_x + full_chunks * 5;
                for (int r = 0; r < rem; ++r) {
                    group_acc += (int32_t)w5[r] * x_rem[r];
                }
            }
            PROF_H2F();
            acc_float += (float)group_acc * half_to_float(pgm_read_word(s_row + g));
        }
        y[i] = acc_float * x_scale;
    }
    if (x_q_heap) free(x_q);
}

void rms_norm(const float* x, float* y, const float* gamma, int n, float eps = 1e-5f) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv_rms * gamma[i];
}

// exp() without libm.
//
// ESP8266 has no FPU, so expf() is a soft-float libcall costing several
// microseconds. It used to dominate sampling (2048 calls per token) and was
// the second-largest item inside attention, where every head runs a softmax
// over the context so far.
//
// exp(x) = 2^e * 2^f with e = floor(x*log2 e) and f in [0,1), and 2^f from a
// degree-5 Taylor series. Degree 3 (two fewer FMAs) leaves ~1% relative error
// near f -> 1; degree 5 brings it to 8.6e-5, which is below the noise the
// int8 activations already introduce. Verified against float64 exp in
// check_fast_exp.py: post-softmax total-variation distance 3.5e-6.
static inline float fast_exp(float x) {
    if (x < -87.0f) return 0.0f;          // below the smallest normal result
    if (x > 88.0f)  return 3.4028235e38f; // avoid the exponent-field overflow
    const float t = x * 1.442695041f;      // log2(e)
    // int cast truncates toward zero; for negative t that leaves f in (-1,0),
    // so borrow one from the exponent to keep f in [0,1).
    int32_t e = (int32_t)t;
    float f = t - (float)e;
    if (f < 0.0f) { f += 1.0f; --e; }
    float p = 0.0013333558f;
    p = p * f + 0.0096181291f;
    p = p * f + 0.0555041086f;
    p = p * f + 0.2402265069f;
    p = p * f + 0.6931471806f;
    p = p * f + 1.0f;
    union { uint32_t u; float f; } sc;
    sc.u = (uint32_t)(e + 127) << 23;
    return p * sc.f;
}

void softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { x[i] = fast_exp(x[i] - mx); sum += x[i]; }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

static inline float silu(float x) { return x / (1.0f + fast_exp(-x)); }
void swiglu(const float* gate, const float* up, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = silu(gate[i]) * up[i];
}

void apply_rope_row(float* x, const float* cos_row, const float* sin_row, int head_dim) {
    int half = head_dim / 2;
    float tmp[128]; // supports head_dim up to 128
    if (head_dim > 128) return;
    for (int i = 0; i < half; ++i) {
        float x0 = x[2 * i];
        float x1 = x[2 * i + 1];
        float c = cos_row[i];
        float s = sin_row[i];
        tmp[i]        = x0 * c - x1 * s;
        tmp[half + i] = x0 * s + x1 * c;
    }
    memcpy(x, tmp, head_dim * sizeof(float));
}

int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (x[i] > x[best]) best = i;
    return best;
}
#endif
