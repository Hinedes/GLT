#pragma once

#include <cmath>
#include <string>
#include <vector>

// ============================================================================
// GraftConfig: all model dimensions + training hyperparameters in one box.
//
// Operation: formerly a pile of free `int HIDDEN_DIM = ...` globals scattered
// across config.hpp and silently mutated deep inside kernel files
// (layer_recompute_impl.inc rewrote SLICE_DIM, model_io_impl.inc rewrote
// VOCAB_SIZE, etc.) — those hidden writes were root-cause black holes.
//
// Now there is ONE instance, `g_cfg`, created once and mutated only at the
// CLI site (grafting.hip). Every read is `g_cfg.<field>`, so an agent can
// grep `g_cfg.` to see every consumer and find the single write site.
//
// SmolLM3-3B defaults: 2048 hidden, 11008 intermediate, 36 layers, 16 heads,
// 128K vocab. All overridable via CLI (grafting.hip).
// ============================================================================

// Compile-time constant (never changes at runtime) — safe to keep free.
constexpr int PROJECTIONS_PER_LAYER = 3;    // gate, up, down per FFN block

enum class GraftGemmBackend {
    Rocblas,
    HipblasltSelective,
};

enum class GraftGemmKind {
    ExpandForward = 0,
    ContractForward = 1,
    ExpandWeightGrad = 2,
    ContractWeightGrad = 3,
    ExpandInputGrad = 4,
    ContractInputGrad = 5,
};

struct GraftConfig {
    // --- Model dimensions ---
    int hidden_dim = 2048;
    int intermediate_dim = 11008;
    int slice_dim = 2752;
    int vocab_size = 128256;
    int num_layers = 36;
    int num_heads = 16;
    int num_kv_heads = 4;
    int head_dim = 0;            // auto-computed from hidden_dim / num_heads if 0
    int h_slice_dim = 0;         // auto-computed: hidden_dim / max_domains
    int total_projections = 108; // num_layers * PROJECTIONS_PER_LAYER (recomputed at CLI)

    // --- Norm / loss ---
    float rms_norm_eps = 1e-6f;
    float lm_loss_scale = 1.0f;
    float lambda_silence = 5.0f;
    float rope_theta = 5000000.0f;
    std::vector<int> no_rope_layers; // empty = apply RoPE to all layers

    // --- Session / diagnostics ---
    std::string session_id;          // shared by sibling grafts in a stack
    std::string parity_dump_dir;
    int parity_dump_layer = 35;      // NUM_LAYERS - 1 by default
    bool contract_probe = false;
    bool delta_bf16_mirror = false;
    bool use_aten_delta_input_grad = false;
    bool profile_graft_gemms = false;

    // --- Backend selection ---
    GraftGemmBackend graft_gemm_backend = GraftGemmBackend::Rocblas;

    // --- Layer / projection ablation (CLI) ---
    int graft_layer_start = 0;
    int graft_layer_end = 35;    // inclusive
    int graft_proj_gate = 0;
    int graft_proj_up = 0;
    int graft_proj_down = 1;
    bool allow_expansion_trap = false;

    // --- AdamW optimizer ---
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_eps = 1e-8f;
    float adam_weight_decay = 0.01f;
    float gradient_clip_norm = 1.0f;  // global L2 norm clip threshold
    int kernel_threads = 256;         // threads per block for HIP kernels

    // --- LR scheduler: linear warmup -> cosine decay ---
    int lr_warmup_steps = 20;
    float lr_peak = 0.0f;             // set from CLI --lr
};

// The single instance. Created once; mutated only at the CLI site.
inline GraftConfig g_cfg;

// ----------------------------------------------------------------------------
// Ablation helpers (read g_cfg; kept as free functions so call sites are
// unchanged — no signature churn across the 600 reference sites).
// ----------------------------------------------------------------------------
inline bool should_graft_layer(int layer_idx) {
    return layer_idx >= g_cfg.graft_layer_start && layer_idx <= g_cfg.graft_layer_end;
}

inline bool should_graft_proj(int layer_idx, int proj_idx) {
    if (!should_graft_layer(layer_idx)) return false;
    if (proj_idx == 0) return g_cfg.graft_proj_gate;
    if (proj_idx == 1) return g_cfg.graft_proj_up;
    if (proj_idx == 2) return g_cfg.graft_proj_down;
    return false;
}

inline int count_active_grafts() {
    int active = 0;
    for (int layer_idx = 0; layer_idx < g_cfg.num_layers; ++layer_idx) {
        if (should_graft_proj(layer_idx, 0)) ++active;
        if (should_graft_proj(layer_idx, 1)) ++active;
        if (should_graft_proj(layer_idx, 2)) ++active;
    }
    return active;
}

inline float get_lr(int step, int total_steps) {
    if (step < g_cfg.lr_warmup_steps) {
        return g_cfg.lr_peak * (step + 1) / g_cfg.lr_warmup_steps;
    }
    float denominator = (float)(total_steps - g_cfg.lr_warmup_steps);
    float progress = (float)(step - g_cfg.lr_warmup_steps) / fmaxf(denominator, 1.0f);
    return g_cfg.lr_peak * (1.0f + cosf(M_PI * progress)) * 0.5f;
}

inline const char* graft_gemm_backend_name(GraftGemmBackend backend) {
    switch (backend) {
        case GraftGemmBackend::Rocblas: return "rocblas";
        case GraftGemmBackend::HipblasltSelective: return "hipblaslt-selective";
    }
    return "unknown";
}

inline const char* graft_gemm_kind_name(GraftGemmKind kind) {
    switch (kind) {
        case GraftGemmKind::ExpandForward: return "ExpandForward";
        case GraftGemmKind::ContractForward: return "ContractForward";
        case GraftGemmKind::ExpandWeightGrad: return "ExpandWeightGrad";
        case GraftGemmKind::ContractWeightGrad: return "ContractWeightGrad";
        case GraftGemmKind::ExpandInputGrad: return "ExpandInputGrad";
        case GraftGemmKind::ContractInputGrad: return "ContractInputGrad";
    }
    return "Unknown";
}
