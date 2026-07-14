#pragma once

#include <torch/torch.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <unordered_map>
#include <cstdint>

struct GraftGemmProfileStats {
    uint64_t call_count = 0;
    uint64_t rocblas_calls = 0;
    uint64_t hipblaslt_calls = 0;
    uint64_t aten_fallback_calls = 0;
    uint64_t fallback_calls = 0;
    double total_ms = 0.0;
    double host_approx_ms = 0.0;
    bool host_approx = false;
    size_t workspace_bytes = 0;
    int m = 0;
    int n = 0;
    int k = 0;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    int transa = 0;
    int transb = 0;
    int dtype_a = 0;
    int dtype_b = 0;
    int dtype_c = 0;
    int compute_type = 0;
    float alpha = 0.0f;
    float beta = 0.0f;
    std::string backend = "rocblas";
    std::string fallback_reason;
};

struct GraftGemmKey {
    GraftGemmKind kind;
    int m;
    int n;
    int k;
    int transa;
    int transb;
    int lda;
    int ldb;
    int ldc;
    int dtype_a;
    int dtype_b;
    int dtype_c;
    int compute_type;

    bool operator==(const GraftGemmKey& other) const {
        return std::tie(kind, m, n, k, transa, transb, lda, ldb, ldc, dtype_a, dtype_b, dtype_c, compute_type)
            == std::tie(other.kind, other.m, other.n, other.k, other.transa, other.transb,
                        other.lda, other.ldb, other.ldc, other.dtype_a, other.dtype_b,
                        other.dtype_c, other.compute_type);
    }
};

struct GraftGemmKeyHash {
    size_t operator()(const GraftGemmKey& key) const noexcept {
        size_t h = static_cast<size_t>(key.kind);
        auto mix = [&](size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(std::hash<int>{}(key.m));
        mix(std::hash<int>{}(key.n));
        mix(std::hash<int>{}(key.k));
        mix(std::hash<int>{}(key.transa));
        mix(std::hash<int>{}(key.transb));
        mix(std::hash<int>{}(key.lda));
        mix(std::hash<int>{}(key.ldb));
        mix(std::hash<int>{}(key.ldc));
        mix(std::hash<int>{}(key.dtype_a));
        mix(std::hash<int>{}(key.dtype_b));
        mix(std::hash<int>{}(key.dtype_c));
        mix(std::hash<int>{}(key.compute_type));
        return h;
    }
};

struct GraftGemmLtPlan {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t a_layout = nullptr;
    hipblasLtMatrixLayout_t b_layout = nullptr;
    hipblasLtMatrixLayout_t c_layout = nullptr;
    hipblasLtMatrixLayout_t d_layout = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    hipblasLtMatmulHeuristicResult_t algo{};
    size_t workspace_bytes = 0;
    bool valid = false;
};

// ============================================================================
// Data Structure: Residual Stream & Delta Activations (LayerActivations)
// Operation: Stores all per-layer tensors needed for the backward pass
// (checkpoint recompute). Each element holds: post-attention residual,
// FFN input (bf16), RMSNorm variance, all three delta outputs (gate/up/down)
// at float32 precision, and per-token energy norms for silence loss.
// Cleared and repopulated each training step.
// ============================================================================
struct LayerActivations {
    at::Tensor x_pre_attn;            // x before attention residual [batch, seq, hidden]
    at::Tensor x_post_attn;           // x after attention residual, before post-attn norm [batch, seq, hidden]
    at::Tensor x_ffn_input;           // x after post-attn norm (input to FFN) [batch, seq, hidden]
    at::Tensor rms_norm_variance;     // variance from post-attention RMSNorm [batch, seq, 1]
    // PATCH P2-13: intermediate_norm_max removed (was always 0; monitoring was disabled).
    // Per-projection delta outputs (float32, stored for backward pass)
    at::Tensor delta_gate_out;        // delta_gate output [batch, seq, g_cfg.slice_dim]
    at::Tensor delta_up_out;          // delta_up output [batch, seq, g_cfg.slice_dim]
    at::Tensor delta_down_out;        // delta_down output [batch, seq, g_cfg.h_slice_dim]
    // Per-token silence energy: ||delta_out|| / sqrt(V) — TPHS-compatible loss term
    at::Tensor energy_gate;           // [M] energy for gate delta
    at::Tensor energy_up;             // [M] energy for up delta
    at::Tensor energy_down;           // [M] energy for contract delta
};

// ============================================================================
// Data Structure: Base Model Weights (bfloat16, frozen)
// Operation: Holds all frozen SmolLM3-3B weights loaded from safetensors.
// LayerWeights contains the 9 per-layer tensors (layernorms, Q/K/V/O,
// gate/up/down). ModelWeights wraps these with embed_tokens + final_norm
// + lm_head. All tensors are bfloat16 on CUDA and are never modified
// during training (only the delta weights are trained).
// ============================================================================
struct LayerWeights {
    at::Tensor input_layernorm_weight;
    at::Tensor q_proj_weight;
    at::Tensor k_proj_weight;
    at::Tensor v_proj_weight;
    at::Tensor o_proj_weight;
    at::Tensor post_attention_layernorm_weight;
    at::Tensor gate_proj_weight;
    at::Tensor up_proj_weight;
    at::Tensor down_proj_weight;
    // PATCH P2-14: Cached f32 copies of gate/up/down. Avoids per-step cast
    // in backward (~108 f32 copies/step @ ~90 MB each = 9.7 GB/step wasted).
    // Frozen weights, so safe to cache.
    // Shadow Copy note: forward reads the bf16 tensors above, NOT these f32
    // caches. The f32 caches are therefore stale after shadow assembly, but
    // shadow mode forbids backward (see backward_layer guard), so this is safe.
    at::Tensor gate_proj_weight_f32;
    at::Tensor up_proj_weight_f32;
    at::Tensor down_proj_weight_f32;
};

struct ModelWeights {
    at::Tensor embed_tokens_weight;
    std::vector<LayerWeights> layers;
    at::Tensor final_norm_weight;
    at::Tensor lm_head_weight;
};

// ============================================================================
// Phase 1: Monolithic VRAM Arena — Single-allocation GPU Memory Manager
// Operation: Allocates all GPU buffers in one hipMalloc call using a
// byte-offset arena pattern. The arena_offset<T>() helper advances a
// cursor and returns typed pointers, eliminating per-buffer allocation
// overhead and fragmentation. Includes RAII VramArena struct with move
// semantics, plus rocBLAS handle, pipelined AdamW stream, and events.
//
// Buffer reuse pattern (saves ~600 MB VRAM vs per-layer allocation):
// - d_ffn_input: Single-layer bf16 buffer reused across all 36 layers
//   Forward: stores current layer's input. Backward: input for GEMM.
//   Size: B * g_cfg.hidden_dim * sizeof(uint16_t)
// - d_delta_out: Single-layer f32 buffer for expand delta output
//   Forward: delta_out for current layer. Backward: silence blend source.
//   Size: B * g_cfg.slice_dim * sizeof(float)
// - d_dy_slice: Single-layer f32 gradient buffer
//   Backward: stores gradient wrt delta_out. Reused gate + up sequentially.
//   Size: B * g_cfg.slice_dim * sizeof(float)
// - d_delta_grad / d_adam_m / d_adam_v (arena): Persistent gradient + optimizer
//   state for all 108 projections × g_cfg.slice_dim × g_cfg.h_slice_dim.
//   The persistent delta *weights* now live in GraftWeights (d_weight / d_weight_bf16).
//
// Trade-off: requires FFN recompute during backward (acceptable for 36 layers).
// ============================================================================

// Forward declarations for RAII (struct uses these in inline constructors/destructors)
struct VramArena;
void allocate_arena(VramArena& arena, int batch_size, int seq_len);
void free_arena(VramArena& arena);
void free_arena_noexcept(VramArena& arena) noexcept;  // PATCH P0-04: forward decl for destructor

// Owner of the persistent graft parameter buffers (delta weights + optional
// bf16 mirror). Phase 2: GraftWeights is the physical owner, allocated via
// allocate_graft_weights() and freed via free_graft_weights_noexcept().
// VramArena owns only scratch, gradients, optimizer state, and handles.
struct GraftWeights {
    float* d_weight = nullptr;         // [total_projections * slice_dim * h_slice_dim]
    uint16_t* d_weight_bf16 = nullptr; // optional bf16 mirror
    size_t weight_elems = 0;
    int domain_id = 0;
    bool allocated = false;
    bool active = false;  // true once a checkpoint / init-deltas / training load is applied

    GraftWeights() = default;
    GraftWeights(const GraftWeights&) = delete;
    GraftWeights& operator=(const GraftWeights&) = delete;
};

struct VramArena {
    void*    raw_ptr = nullptr;
    int32_t* d_input_ids = nullptr;      // [batch, seq]
    uint16_t* d_ffn_input = nullptr;    // [batch, seq, g_cfg.hidden_dim] (bf16, reused per layer)
    uint16_t* d_intermediate = nullptr;  // [batch, seq, g_cfg.intermediate_dim] (bf16, reused per layer)
    float*   d_delta_out = nullptr;      // [batch, seq, g_cfg.slice_dim] (expand delta output, reused per layer)
    float*   d_delta_out_contract = nullptr; // [batch, seq, g_cfg.h_slice_dim] (contract delta output, reused per layer)
    float*   d_dy_slice = nullptr;       // [batch, seq, g_cfg.slice_dim] (gradient for expand deltas)
    float*   d_dy_slice_contract = nullptr; // [batch, seq, g_cfg.h_slice_dim] (gradient for contract delta)
    int32_t* d_ood_mask = nullptr;       // [batch, seq] (1 for OOD tokens)
    // Persistent graft parameters (delta weights + optional bf16 mirror) are
    // owned by GraftWeights, NOT the arena. The arena owns only scratch,
    // gradients, optimizer state, and GEMM/optimizer handles.
    float*   d_delta_grad = nullptr;     // [g_cfg.total_projections * g_cfg.slice_dim * g_cfg.h_slice_dim]
    float*   d_adam_m = nullptr;         // same shape as d_delta_grad
    float*   d_adam_v = nullptr;         // same shape as d_delta_grad
    uint16_t* d_bf16_temp = nullptr;    // temp buffer for f32→bf16 conversion (g_cfg.slice_dim * g_cfg.h_slice_dim)

    // Gradient clipping scratch (pre-allocated to avoid hipMalloc in hot path)
    float* d_grad_clip_partial = nullptr; // [1024] partial L2 sums
    float* d_grad_clip_scale = nullptr;   // [1] scale factor
    float* h_grad_clip_scale_ping = nullptr; // pinned host copy, step parity 0
    float* h_grad_clip_scale_pong = nullptr; // pinned host copy, step parity 1

    rocblas_handle rocblas_handle = nullptr;
    hipStream_t rocblas_stream = nullptr;
    hipblasLtHandle_t hipblaslt_handle = nullptr;
    void* d_hipblaslt_workspace = nullptr;
    size_t hipblaslt_workspace_bytes = 0;

    // Separate stream for AdamW to overlap with next step's data loading
    hipStream_t adamw_stream = nullptr;
    hipEvent_t backward_event = nullptr;  // signaled when backward completes on the current stream
    hipEvent_t adamw_event = nullptr;     // signaled when AdamW completes (adamw_stream)
    hipEvent_t grad_clip_copy_event = nullptr; // async grad_clip_scale copy completion

    size_t total_bytes = 0;
    bool allocated = false;

    // RAII: Default constructor (no allocation)
    VramArena() = default;

    // RAII: Constructor with allocation
    VramArena(int batch_size, int seq_len) {
        allocate_arena(*this, batch_size, seq_len);
    }

    // RAII: Destructor ensures cleanup
    // PATCH P0-04: Destructor must be noexcept. Calling free_arena (which
    // throws on HIP error) from a destructor during stack unwinding causes
    // std::terminate. Use the noexcept variant.
    ~VramArena() noexcept {
        free_arena_noexcept(*this);
    }

    // RAII: Delete copy operations (unique ownership)
    VramArena(const VramArena&) = delete;
    VramArena& operator=(const VramArena&) = delete;
    // PATCH P2-12: Move ctor/assign removed. VramArena is constructed in-place
    // at line ~2712 and never moved. The ~92 lines of pointer-by-pointer
    // transfer added maintenance surface (and missed fields per P1-06) for
    // functionality that was never used. If a future caller needs moves, add
    // them back — but cover ALL 22 fields and null-out lists.
    VramArena(VramArena&&) = delete;
    VramArena& operator=(VramArena&&) = delete;
};
