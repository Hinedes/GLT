// ============================================================================
// grafting.hip — Multi-Domain Grafting Training for SmolLM3-3B
//
// Operation: This file implements the full training pipeline for
// "grafting" — a multi-domain fine-tuning technique where each domain
// learns specialized delta weight matrices (gate/up/down projections)
// that are added into the frozen base model's FFN activations.
//
// Architecture (6 Phases, executed in order each training step):
//   Phase 1: VRAM Arena — monolithic GPU memory manager (buffer reuse)
//   Phase 2: Weight Loading — load frozen base model from safetensors
//   Phase 3: HIP Kernels — fused RMSNorm, silence blend, gate*up bwd,
//            gradient clipping, energy norm, AdamW
//   Phase 4: Data Loading — binary .bin token files (domain + OOD)
//   Phase 5: Checkpoint I/O — save/load delta weights as safetensors
//   Phase 6: CLI + Main Loop — arg parsing, forward, loss, backward,
//            gradient clipping, pipelined AdamW, logging, save
//
// Key Design Decisions:
//   - Single monolithic file for tight kernel-ATen integration
//   - bfloat16 base model + float32 delta weights/gradients/optimizer states
//   - pipelined AdamW on separate stream to overlap with next backward
//   - fused HIP kernels replace 15+ ATen launches per layer each direction
//   - silence blend regularizes OOD token activations toward zero
//   - checkpoint recompute saves ~600 MB VRAM at recompute cost
// ============================================================================
// TABLE OF CONTENTS
// 197    Phase 1:  VRAM Arena
// 506    Phase 2:  Weight Loading
// 697    Phase 3:  HIP Kernels & GEMM
// 1692   Phase 1a: Forward Pass
// 1884   Recompute (Checkpoint)
// 1957   Phase 2b: Backward Pass
// 2145   Phase 4:  Data Loading
// 2184   Phase 5:  Checkpoint I/O
// 2357   Phase 6:  CLI + Main Loop
// ============================================================================
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <tuple>
#include <c10/hip/HIPStream.h>
#include <hipblaslt/hipblaslt.h>
#include <memory>
#include <unordered_map>
#include <cassert>
#include <iomanip>

#include "src/kernels_rmsnorm.hpp"
// PATCH P2-11: Persistent RMSNorm kernels removed (dead code).
#include <cstdint>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// PATCH P3-22: Experimental / implemented kernel inventory moved to
// docs/kernel_inventory.md. Keep this source lean.

#include <fstream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <filesystem>
namespace fs = std::filesystem;

#define SAFETENSORS_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-designator"
#include "safetensors.h"
#pragma clang diagnostic pop

#include "src/checks.hpp"
#include "src/config.hpp"
#include "src/types.hpp"
#include "src/gemm.hpp"
#include "src/kernels_elementwise.hpp"
#include "src/kernels_optimizer.hpp"
#include "src/checkpoint.hpp"
#include "src/data.hpp"
#include "src/diagnostics.hpp"
#include "src/arena.hpp"
#include "src/graft_weights.hpp"
#include "src/graft_ops.hpp"
#include "src/model_io.hpp"
#include "src/rope.hpp"
#include "src/attention_autograd.hpp"
#include "src/cli.hpp"
#include "src/layer_forward.hpp"
#include "src/layer_recompute.hpp"
#include "src/layer_backward.hpp"
#include "src/shadow_bake.hpp"


#include "src/arena_impl.inc"

#include "src/graft_weights_impl.inc"

#include "src/diagnostics_impl.inc"

// ============================================================================
// Helper: Fused RMSNorm Forward (HIP Kernel Wrapper)
// Operation: Allocates output tensors (bf16 y, f32 variance) and launches
// the fused rms_norm_fwd_kernel with shared memory for warp reduction.
// Single kernel replaces: x.pow(2).mean(-1).rsqrt() → y * w → bf16 cast.
// Returns pair (y, variance) for use in backward pass.
// ============================================================================
// ponytail: g_rmsnorm_check_done gates a one-shot ATen-vs-HIP parity self-check.
// First call: run both paths, assert cosine>0.9999 on y and tolerance on variance,
// then set to true so subsequent calls skip the check.
static bool g_rmsnorm_check_done = false;

std::pair<at::Tensor, at::Tensor> rms_norm(const at::Tensor& x, const at::Tensor& weight) {
    int M = x.numel() / g_cfg.hidden_dim;
    auto y = torch::empty_like(x);
    auto variance = torch::empty({x.size(0), x.size(1)}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    int blocks = M;
    int smem = ((g_cfg.kernel_threads + 63) / 64) * sizeof(float);
    rms_norm_fwd_kernel<<<blocks, g_cfg.kernel_threads, smem, current_hip_stream()>>>(
        reinterpret_cast<uint16_t*>(y.data_ptr()),
        variance.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(x.data_ptr()),
        reinterpret_cast<const uint16_t*>(weight.data_ptr()),
        g_cfg.hidden_dim, g_cfg.rms_norm_eps
    );
    HIP_CHECK(hipGetLastError());

    if (!g_rmsnorm_check_done) {
        g_rmsnorm_check_done = true;
        auto x_f32 = x.to(torch::kFloat32);
        auto rstd = torch::rsqrt(x_f32.pow(2).mean(-1, true) + g_cfg.rms_norm_eps);
        auto y_aten = (x_f32 * weight.to(torch::kFloat32) * rstd).to(torch::kBFloat16);
        auto cosine = torch::nn::functional::cosine_similarity(
            y.view({-1}).to(torch::kFloat32), y_aten.view({-1}).to(torch::kFloat32), torch::nn::functional::CosineSimilarityFuncOptions().dim(0)).item<float>();
        auto var_aten = x_f32.pow(2).mean(-1);
        auto var_maxdiff = (variance - var_aten).abs().max().item<float>();
        std::cerr << "[diag] rms_norm parity check: cosine=" << cosine
                  << " var_maxdiff=" << var_maxdiff << std::endl;
        TORCH_CHECK(cosine > 0.9999f, "rms_norm parity check failed");
    }
    return {y, variance};
}

// ============================================================================
// Helper: Fused RMSNorm Backward (HIP Kernel Wrapper)
// Operation: Launches rms_norm_bwd_kernel with shared memory for warp
// reduction. Computes dx from dy, x, weight, and precomputed variance.
// ============================================================================
at::Tensor rms_norm_bwd(const at::Tensor& dy, const at::Tensor& x,
                         const at::Tensor& weight, const at::Tensor& variance) {
    int M = dy.numel() / g_cfg.hidden_dim;
    auto dy_f32 = dy.to(torch::kFloat32).contiguous();
    auto dx = torch::empty({M, g_cfg.hidden_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(dy.device()));
    if (dy.dim() == 3) dx = dx.view({dy.size(0), dy.size(1), g_cfg.hidden_dim});

    int blocks = M;
    int smem = ((g_cfg.kernel_threads + 63) / 64) * sizeof(float);
    rms_norm_bwd_kernel<<<blocks, g_cfg.kernel_threads, smem, current_hip_stream()>>>(
        dx.data_ptr<float>(),
        dy_f32.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(x.data_ptr()),
        reinterpret_cast<const uint16_t*>(weight.data_ptr()),
        variance.data_ptr<float>(),
        g_cfg.hidden_dim, g_cfg.rms_norm_eps
    );
    HIP_CHECK(hipGetLastError());
    return dx;
}

// ============================================================================
// Global Training State
// ============================================================================
ModelWeights g_model_weights;
std::vector<LayerActivations> g_layer_activations;
bool g_verbose = false;

#include "src/model_io_impl.inc"

#include "src/kernels_elementwise_impl.inc"

#include "src/kernels_rmsnorm_kernels.inc"

#include "src/kernels_optimizer_impl.inc"

#include "src/gemm_impl.inc"

#include "src/graft_ops_impl.inc"


#include "src/rope_impl.inc"

#include "src/attention_autograd_impl.inc"


#include "src/layer_forward_impl.inc"
#include "src/layer_recompute_impl.inc"

#include "src/layer_backward_impl.inc"

#include "src/data_impl.inc"

#include "src/checkpoint_impl.inc"

#include "src/shadow_bake_impl.inc"

// ============================================================================
// Phase 6: CLI Interface & Main Training Loop
// ============================================================================

#include "src/cli_impl.inc"

// ============================================================================
// forward_model — shared full forward pass (base model + optional graft).
// Sequence: token embedding → forward_layer() per layer → final RMSNorm →
// LM head. Used by BOTH training and eval-only so transformer math lives in
// exactly one place. Returns (logits, x_before_final_norm, final_norm_variance).
// Does NOT compute loss, backward, or AdamW. Parity debug dumps (if enabled)
// happen here once.
// ============================================================================
std::tuple<at::Tensor, at::Tensor, at::Tensor> forward_model(const at::Tensor& input_ids,
                                                             VramArena& arena,
                                                             const GraftWeights& graft,
                                                             int batch_size,
                                                             int seq_len,
                                                             bool use_explicit_graft = true) {
    at::Tensor x;
    {
        torch::NoGradGuard guard;
        x = at::embedding(g_model_weights.embed_tokens_weight, input_ids);
    }
    if (g_verbose) std::cerr << "[diag] x after embedding: " << x.sizes()
               << " dtype=" << x.dtype() << std::endl;

    for (int l = 0; l < g_cfg.num_layers; ++l) {
        forward_layer(l, x, arena, graft, batch_size, seq_len, use_explicit_graft);
        if (g_verbose && (l == 0 || l == 35)) std::cerr << "[diag] x after layer " << l << ": " << x.sizes() << std::endl;
        if (!g_cfg.parity_dump_dir.empty() && (l == 0 || l == 1 || l == 5 || l == 10 || l == 20 || l == 35)) {
            auto x0 = x[0][0].slice(0, 0, 8).to(torch::kCPU).to(torch::kFloat32);
            std::cout << "[Parity] x after L" << l << " = ";
            for (int i = 0; i < 8; i++) std::cout << x0.data_ptr<float>()[i] << " ";
            std::cout << "\n";
        }
    }

    at::Tensor x_before_final_norm = x;
    at::Tensor final_norm_variance;
    at::Tensor logits;
    {
        torch::NoGradGuard guard;
        auto nr = rms_norm(x, g_model_weights.final_norm_weight);
        x = nr.first;
        final_norm_variance = nr.second;
        logits = at::linear(x, g_model_weights.lm_head_weight);
    }
    if (!g_cfg.parity_dump_dir.empty()) {
        auto x_cpu = x_before_final_norm[0][0].slice(0, 0, 16).to(torch::kCPU).to(torch::kFloat32);
        std::cout << "[Parity] Base hidden[0,0,:16] = ";
        for (int i = 0; i < 16; i++) {
            if (i > 0) std::cout << " ";
            std::cout << x_cpu.data_ptr<float>()[i];
        }
        std::cout << "\n";
        dump_f32_tensor_raw(x_before_final_norm, parity_dump_path("final_hidden.raw"));
        dump_f32_tensor_raw(logits[0], parity_dump_path("logits.raw"));
    }
    return {logits, x_before_final_norm, final_norm_variance};
}

int main(int argc, char** argv) try {
    std::string safetensors_path;
    std::string domain_data_path;
    std::vector<std::string> ood_data_paths;
    int domain_id = 0;
    int max_domains = 4;
    int steps = 200;
    float lr = 2e-4f;
    int batch_size = 16;
    int max_len = 512;
    std::string output_path;
    std::string checkpoint_path;
    std::string parity_dump_dir;
    int parity_dump_layer = g_cfg.num_layers - 1;
    std::string init_deltas_path;
    std::string graft_gemm_backend_str = "rocblas";
    std::string eval_data_path;
    bool eval_only = false;
    bool shadow_eval = false;
    int seed = 42;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--safetensors" && i + 1 < argc) safetensors_path = argv[++i];
        else if (arg == "--domain-data" && i + 1 < argc) domain_data_path = argv[++i];
        else if (arg == "--ood-data") {
            while (i + 1 < argc && argv[i + 1][0] != '-') ood_data_paths.push_back(argv[++i]);
        }
        else if (arg == "--eval-only") eval_only = true;
        else if (arg == "--shadow-eval") shadow_eval = true;
        else if (arg == "--eval-data" && i + 1 < argc) eval_data_path = argv[++i];
        else if (arg == "--domain-id" && i + 1 < argc) domain_id = std::stoi(argv[++i]);
        else if (arg == "--max-domains" && i + 1 < argc) max_domains = std::stoi(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
        else if (arg == "--lm-loss-scale" && i + 1 < argc) g_cfg.lm_loss_scale = std::stof(argv[++i]);
        else if (arg == "--lambda-silence" && i + 1 < argc) g_cfg.lambda_silence = std::stof(argv[++i]);
        else if (arg == "--batch-size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--max-len" && i + 1 < argc) max_len = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--checkpoint" && i + 1 < argc) checkpoint_path = argv[++i];
        else if (arg == "--session-id" && i + 1 < argc) g_cfg.session_id = argv[++i];
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoi(argv[++i]);
        else if (arg == "--hidden-dim" && i + 1 < argc) g_cfg.hidden_dim = std::stoi(argv[++i]);
        else if (arg == "--intermediate-dim" && i + 1 < argc) g_cfg.intermediate_dim = std::stoi(argv[++i]);
        else if (arg == "--vocab-size" && i + 1 < argc) g_cfg.vocab_size = std::stoi(argv[++i]);
        else if (arg == "--num-layers" && i + 1 < argc) g_cfg.num_layers = std::stoi(argv[++i]);
        else if (arg == "--num-heads" && i + 1 < argc) g_cfg.num_heads = std::stoi(argv[++i]);
        else if (arg == "--num-kv-heads" && i + 1 < argc) g_cfg.num_kv_heads = std::stoi(argv[++i]);
        else if (arg == "--head-dim" && i + 1 < argc) g_cfg.head_dim = std::stoi(argv[++i]);
        else if (arg == "--slice-dim" && i + 1 < argc) g_cfg.slice_dim = std::stoi(argv[++i]);
        else if (arg == "--rope-theta" && i + 1 < argc) g_cfg.rope_theta = std::stof(argv[++i]);
        else if (arg == "--no-rope-layers" && i + 1 < argc) {
            std::string val = argv[++i];
            size_t pos = 0;
            while (pos < val.size()) {
                size_t comma = val.find(',', pos);
                std::string token = val.substr(pos, comma - pos);
                if (!token.empty()) g_cfg.no_rope_layers.push_back(std::stoi(token));
                pos = (comma == std::string::npos) ? val.size() : comma + 1;
            }
        }
        else if (arg == "--beta1" && i + 1 < argc) g_cfg.adam_beta1 = std::stof(argv[++i]);
        else if (arg == "--beta2" && i + 1 < argc) g_cfg.adam_beta2 = std::stof(argv[++i]);
        else if (arg == "--adam-eps" && i + 1 < argc) g_cfg.adam_eps = std::stof(argv[++i]);
        else if (arg == "--weight-decay" && i + 1 < argc) g_cfg.adam_weight_decay = std::stof(argv[++i]);
        else if (arg == "--grad-clip" && i + 1 < argc) g_cfg.gradient_clip_norm = std::stof(argv[++i]);  // PATCH P2-08
        else if (arg == "--warmup-steps" && i + 1 < argc) g_cfg.lr_warmup_steps = std::stoi(argv[++i]);  // PATCH P2-09
        else if (arg == "--threads" && i + 1 < argc) g_cfg.kernel_threads = std::stoi(argv[++i]);
        else if (arg == "--graft-layer-start" && i + 1 < argc) g_cfg.graft_layer_start = std::stoi(argv[++i]);
        else if (arg == "--graft-layer-end" && i + 1 < argc) g_cfg.graft_layer_end = std::stoi(argv[++i]);
        else if (arg == "--graft-proj-gate" && i + 1 < argc) g_cfg.graft_proj_gate = std::stoi(argv[++i]);
        else if (arg == "--graft-proj-up" && i + 1 < argc) g_cfg.graft_proj_up = std::stoi(argv[++i]);
        else if (arg == "--graft-proj-down" && i + 1 < argc) g_cfg.graft_proj_down = std::stoi(argv[++i]);
        else if (arg == "--allow-expansion-trap") { g_cfg.allow_expansion_trap = true; }
        else if (arg == "--graft-gemm-backend" && i + 1 < argc) graft_gemm_backend_str = argv[++i];
        else if (arg == "--delta-bf16-mirror") { g_cfg.delta_bf16_mirror = true; }
        else if (arg == "--use-aten-delta-input-grad") { g_cfg.use_aten_delta_input_grad = true; }
        else if (arg == "--profile-graft-gemms") { g_cfg.profile_graft_gemms = true; }
        else if (arg == "--parity-dump" && i + 1 < argc) parity_dump_dir = argv[++i];
        else if (arg == "--parity-dump-layer" && i + 1 < argc) parity_dump_layer = std::stoi(argv[++i]);
        else if (arg == "--init-deltas" && i + 1 < argc) init_deltas_path = argv[++i];
        else if (arg == "--contract-probe") { g_cfg.contract_probe = true; }
        else if (arg == "--verbose") { g_verbose = true; }
        else if (arg == "--help") { print_usage(argv[0]); return 0; }
    }
    g_cfg.lr_peak = lr;

    // Shared validation -- must run for ALL modes (incl. --shadow-eval, which
    // returns before the training-only checks below) and BEFORE the head_dim /
    // h_slice_dim divisions to avoid divide-by-zero and out-of-range indices.
    if (g_cfg.num_heads <= 0) {
        std::cerr << "Error: --num-heads must be positive.\n";
        return EXIT_FAILURE;
    }
    if (max_domains <= 0) {
        std::cerr << "Error: --max-domains must be positive.\n";
        return EXIT_FAILURE;
    }

    // Auto-compute head_dim from hidden / heads unless explicitly set
    if (g_cfg.head_dim == 0) {
        g_cfg.head_dim = g_cfg.hidden_dim / g_cfg.num_heads;
    }
    if (graft_gemm_backend_str == "rocblas") {
        g_cfg.graft_gemm_backend = GraftGemmBackend::Rocblas;
    } else if (graft_gemm_backend_str == "hipblaslt-selective") {
        g_cfg.graft_gemm_backend = GraftGemmBackend::HipblasltSelective;
    } else {
        std::cerr << "Error: invalid --graft-gemm-backend '" << graft_gemm_backend_str
                  << "' (expected rocblas or hipblaslt-selective)\n";
        return EXIT_FAILURE;
    }
    if (g_cfg.hidden_dim % g_cfg.num_heads != 0) {
        std::cerr << "Error: hidden-dim (" << g_cfg.hidden_dim << ") not divisible by num-heads (" << g_cfg.num_heads << ")\n";
        return EXIT_FAILURE;
    }
    g_cfg.parity_dump_dir = parity_dump_dir;
    g_cfg.parity_dump_layer = parity_dump_layer;
    if (!g_cfg.parity_dump_dir.empty()) {
        std::filesystem::create_directories(g_cfg.parity_dump_dir);
    }
    g_cfg.h_slice_dim = g_cfg.hidden_dim / max_domains;
    if (g_cfg.hidden_dim % max_domains != 0) {
        std::cerr << "Error: hidden-dim (" << g_cfg.hidden_dim << ") not divisible by max-domains (" << max_domains << ")\n";
        return EXIT_FAILURE;
    }
    g_cfg.total_projections = g_cfg.num_layers * PROJECTIONS_PER_LAYER;

    // Remaining shared checks (apply to every mode). Duplicated from the
    // training-only block below so --shadow-eval / --eval-only also enforce
    // them (e.g. RoPE-mask length is indexed inside apply_rope).
    if (domain_id < 0 || domain_id >= max_domains) {
        std::cerr << "Error: --domain-id (" << domain_id << ") out of range [0, " << max_domains << ").\n";
        return EXIT_FAILURE;
    }
    if (!g_cfg.no_rope_layers.empty() && static_cast<int>(g_cfg.no_rope_layers.size()) != g_cfg.num_layers) {
        std::cerr << "Error: --rope-layer-mask expects " << g_cfg.num_layers
                  << " values, got " << g_cfg.no_rope_layers.size() << ".\n";
        return EXIT_FAILURE;
    }
    if (g_cfg.intermediate_dim % g_cfg.slice_dim != 0) {
        std::cerr << "Error: intermediate-dim (" << g_cfg.intermediate_dim << ") not divisible by slice-dim ("
                  << g_cfg.slice_dim << ").\n";
        return EXIT_FAILURE;
    }
    // Vector kernels assume these alignments: silence/energy need slice%4,
    // gate*up silu needs intermediate%8. Reject instead of going OOB.
    if (g_cfg.slice_dim % 4 != 0) {
        std::cerr << "Error: --slice-dim (" << g_cfg.slice_dim << ") must be a multiple of 4.\n";
        return EXIT_FAILURE;
    }
    if (g_cfg.intermediate_dim % 8 != 0) {
        std::cerr << "Error: --intermediate-dim (" << g_cfg.intermediate_dim << ") must be a multiple of 8.\n";
        return EXIT_FAILURE;
    }
    if ((g_cfg.graft_proj_gate != 0 && g_cfg.graft_proj_gate != 1) ||
        (g_cfg.graft_proj_up != 0 && g_cfg.graft_proj_up != 1) ||
        (g_cfg.graft_proj_down != 0 && g_cfg.graft_proj_down != 1)) {
        std::cerr << "Error: --graft-proj-* flags must be 0 or 1.\n";
        return EXIT_FAILURE;
    }
    if (!g_cfg.graft_proj_gate && !g_cfg.graft_proj_up && !g_cfg.graft_proj_down) {
        std::cerr << "Error: no graft projection enabled.\n";
        return EXIT_FAILURE;
    }
    if (g_cfg.graft_proj_gate && g_cfg.graft_proj_up && !g_cfg.allow_expansion_trap) {
        std::cerr << "Error: gate+up enabled: Expansion Quadratic Trap. "
                  << "This trains Δgate and Δup together, creating Δgate*Δup in SwiGLU. "
                  << "Use contract-only by default, or pass --allow-expansion-trap for explicit ablation.\n";
        return EXIT_FAILURE;
    }

    if (shadow_eval) {
        // ---- Shadow-eval mode: startup bake, then base-only inference ----
        if (!domain_data_path.empty()) {
            std::cerr << "Error: --shadow-eval is incompatible with --domain-data (training-only).\n"; return EXIT_FAILURE;
        }
        if (!ood_data_paths.empty()) {
            std::cerr << "Error: --shadow-eval is incompatible with --ood-data (training-only).\n"; return EXIT_FAILURE;
        }
        if (!output_path.empty()) {
            std::cerr << "Error: --shadow-eval does not save a checkpoint; remove --output.\n"; return EXIT_FAILURE;
        }
        if (!init_deltas_path.empty()) {
            std::cerr << "Error: --shadow-eval is incompatible with --init-deltas (training-only).\n"; return EXIT_FAILURE;
        }
        if (g_cfg.contract_probe) {
            std::cerr << "Error: --shadow-eval is incompatible with --contract-probe.\n"; return EXIT_FAILURE;
        }
        if (safetensors_path.empty() || eval_data_path.empty()) {
            std::cerr << "Error: --shadow-eval requires --safetensors and --eval-data.\n";
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        if (!load_base_model_safetensors(safetensors_path)) {
            return EXIT_FAILURE;
        }

        TokenBuffer eval;
        if (!load_bin(eval_data_path, eval)) return EXIT_FAILURE;
        const int seq_len = eval.seq_len;

        VramArena arena{};
        allocate_arena(arena, batch_size, seq_len);
        std::cout << "[Arena] Allocated " << (arena.total_bytes / 1024 / 1024) << " MB of VRAM.\n";

        GraftWeights graft{};
        bool has_checkpoint = !checkpoint_path.empty();
        if (has_checkpoint) {
            allocate_graft_weights(graft, domain_id, false);
            std::cout << "[Checkpoint] Loading graft from " << checkpoint_path << "\n";
            if (!load_safetensors_checkpoint(checkpoint_path, domain_id, arena, graft))
                return EXIT_FAILURE;
            // Bake graft into shadow copy
            apply_graft_to_shadow(g_model_weights, graft);
            // Free graft parameter memory before inference
            free_graft_weights_noexcept(graft);
            std::cout << "[Shadow] Graft freed before first forward: yes\n";
        } else {
            std::cout << "[Shadow] Base-only mode (no checkpoint).\n";
        }

        // Real signal: g_graft_fwd_calls is incremented in forward_layer;
        // for explicit-eval (use_explicit_graft=true) it is nonzero by design.
        g_graft_fwd_calls = 0;

        // Deterministic forward over all eval sequences, batched by batch_size
        double total_nll = 0.0;
        int64_t token_count = 0;
        double logits_sum = 0.0, logits_sumsq = 0.0;
        uint64_t logits_hash = 1469598103934665603ULL;  // FNV-1a 64-bit offset basis
        const int total_seq = eval.num_sequences;
        for (int s0 = 0; s0 < total_seq; s0 += batch_size) {
            int cur = std::min(batch_size, total_seq - s0);
            int total_tokens = cur * seq_len;
            std::vector<int32_t> host_ids(total_tokens);
            std::memcpy(host_ids.data(), &eval.tokens[static_cast<size_t>(s0) * seq_len],
                        total_tokens * sizeof(int32_t));
            HIP_CHECK(hipMemcpy(arena.d_input_ids, host_ids.data(),
                                total_tokens * sizeof(int32_t), hipMemcpyHostToDevice));

            auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
            auto input_ids = torch::from_blob(arena.d_input_ids, {cur, seq_len}, options);

            at::Tensor logits;
            {
                torch::NoGradGuard guard;
                // use_explicit_graft=false: no graft GEMMs during shadow inference
                logits = std::get<0>(forward_model(input_ids, arena, graft, cur, seq_len, false));
            }

            auto logits_cpu = logits.to(torch::kCPU).to(torch::kFloat32).contiguous();
            auto targets = input_ids.to(torch::kCPU).to(torch::kInt64).slice(1, 1).contiguous();
            auto logits_shift = logits_cpu.slice(1, 0, seq_len - 1).contiguous();
            auto ce = torch::nn::functional::cross_entropy(
                logits_shift.permute({0, 2, 1}).contiguous(), targets,
                torch::nn::functional::CrossEntropyFuncOptions{}.ignore_index(-100).reduction(torch::kMean));
            int64_t predicted_tokens = static_cast<int64_t>(cur) * (seq_len - 1);
            total_nll += ce.item<double>() * predicted_tokens;
            token_count += predicted_tokens;

            auto acc = logits_cpu.accessor<float, 3>();
            for (int b = 0; b < cur; ++b)
                for (int t = 0; t < seq_len; ++t)
                    for (int v = 0; v < acc.size(2); ++v) {
                        float f = acc[b][t][v];
                        logits_sum += f;
                        logits_sumsq += static_cast<double>(f) * f;
                        const unsigned char* p = reinterpret_cast<const unsigned char*>(&f);
                        for (int k = 0; k < 4; ++k) {
                            logits_hash ^= p[k];
                            logits_hash *= 1099511628211ULL;
                        }
                    }
        }
        HIP_CHECK(hipStreamSynchronize(current_hip_stream()));

        double mean_ce = total_nll / token_count;
        double ppl = std::exp(mean_ce);
        double logits_l2 = std::sqrt(logits_sumsq);

        std::cout << "mode=" << (has_checkpoint ? "shadow" : "base") << "\n";
        std::cout << "checkpoint=" << (checkpoint_path.empty() ? "<none>" : checkpoint_path) << "\n";
        std::cout << "tokens=" << token_count << "\n";
        std::cout << "mean_ce=" << std::setprecision(10) << mean_ce << "\n";
        std::cout << "ppl=" << std::setprecision(10) << ppl << "\n";
        std::cout << "logits_sum=" << std::setprecision(10) << logits_sum << "\n";
        std::cout << "logits_l2=" << std::setprecision(10) << logits_l2 << "\n";
        std::cout << "logits_hash=" << std::hex << logits_hash << std::dec << "\n";
        std::cout << "shadow_baked=" << (has_checkpoint ? "true" : "false") << "\n";
        bool graft_freed_before_forward =
            graft.d_weight == nullptr &&
            graft.d_weight_bf16 == nullptr &&
            graft.weight_elems == 0 &&
            !graft.allocated &&
            !graft.active;

        if (has_checkpoint && !graft_freed_before_forward)
            throw std::runtime_error("Graft storage remained resident before shadow forward");

        std::cout << "graft_freed_before_forward=" << (has_checkpoint ? (graft_freed_before_forward ? "true" : "false") : "n/a") << "\n";
        std::cout << "[eval] graft forward calls=" << g_graft_fwd_calls
                  << " (nonzero = explicit graft GEMMs active)\n";

        return EXIT_SUCCESS;
    }

    if (eval_only) {
        bool bad = false;
        if (!domain_data_path.empty()) {
            std::cerr << "Error: --eval-only is incompatible with --domain-data (training-only).\n"; bad = true;
        }
        if (!ood_data_paths.empty()) {
            std::cerr << "Error: --eval-only is incompatible with --ood-data (training-only).\n"; bad = true;
        }
        if (!output_path.empty()) {
            std::cerr << "Error: --eval-only does not save a checkpoint; remove --output.\n"; bad = true;
        }
        if (!init_deltas_path.empty()) {
            std::cerr << "Error: --eval-only is incompatible with --init-deltas (training-only).\n"; bad = true;
        }
        if (g_cfg.contract_probe) {
            std::cerr << "Error: --eval-only is incompatible with --contract-probe.\n"; bad = true;
        }
        if (bad) return EXIT_FAILURE;
        if (safetensors_path.empty() || eval_data_path.empty()) {
            std::cerr << "Error: --eval-only requires --safetensors and --eval-data.\n";
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    if (eval_only) {
        // ---- Eval-only inference: immutable base + optional external graft ----
        if (!load_base_model_safetensors(safetensors_path)) {
            return EXIT_FAILURE;
        }
        TokenBuffer eval;
        if (!load_bin(eval_data_path, eval)) return EXIT_FAILURE;
        const int seq_len = eval.seq_len;

        VramArena arena{};
        allocate_arena(arena, batch_size, seq_len);
        std::cout << "[Arena] Allocated " << (arena.total_bytes / 1024 / 1024) << " MB of VRAM.\n";

        GraftWeights graft{};
        allocate_graft_weights(graft, domain_id, g_cfg.delta_bf16_mirror);

        // Ownership proof retained for eval (graft outside arena range).
        {
            char* arena_base = static_cast<char*>(arena.raw_ptr);
            char* arena_end = arena_base + arena.total_bytes;
            assert(reinterpret_cast<char*>(graft.d_weight) < arena_base ||
                   reinterpret_cast<char*>(graft.d_weight) >= arena_end);
            std::cerr << "[ownership] arena.raw_ptr=" << arena.raw_ptr
                      << " arena.total_bytes=" << arena.total_bytes
                      << " graft.d_weight=" << graft.d_weight
                      << " graft.d_weight_bf16=" << graft.d_weight_bf16 << std::endl;
        }

        if (!checkpoint_path.empty()) {
            std::cout << "[Checkpoint] Loading graft from " << checkpoint_path << "\n";
            if (!load_safetensors_checkpoint(checkpoint_path, domain_id, arena, graft))
                return EXIT_FAILURE;
            graft.active = true;
        }
        // base-only: graft.active remains false; forward_layer skips graft GEMMs.

        std::cout << "graft.active=" << (graft.active ? "true" : "false") << std::endl;

        // Deterministic forward over all eval sequences, batched by batch_size.
        double total_nll = 0.0;
        int64_t token_count = 0;
        double logits_sum = 0.0, logits_sumsq = 0.0;
        uint64_t logits_hash = 1469598103934665603ULL;  // FNV-1a 64-bit offset basis
        g_graft_fwd_calls = 0;
        const int total_seq = eval.num_sequences;
        for (int s0 = 0; s0 < total_seq; s0 += batch_size) {
            int cur = std::min(batch_size, total_seq - s0);
            int total_tokens = cur * seq_len;
            std::vector<int32_t> host_ids(total_tokens);
            std::memcpy(host_ids.data(), &eval.tokens[static_cast<size_t>(s0) * seq_len],
                        total_tokens * sizeof(int32_t));
            HIP_CHECK(hipMemcpy(arena.d_input_ids, host_ids.data(),
                                total_tokens * sizeof(int32_t), hipMemcpyHostToDevice));

            auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
            auto input_ids = torch::from_blob(arena.d_input_ids, {cur, seq_len}, options);

            at::Tensor logits;
            {
                torch::NoGradGuard guard;
                logits = std::get<0>(forward_model(input_ids, arena, graft, cur, seq_len));
            }

            auto logits_cpu = logits.to(torch::kCPU).to(torch::kFloat32).contiguous();
            auto targets = input_ids.to(torch::kCPU).to(torch::kInt64).slice(1, 1).contiguous();
            auto logits_shift = logits_cpu.slice(1, 0, seq_len - 1).contiguous();
            auto ce = torch::nn::functional::cross_entropy(
                logits_shift.permute({0, 2, 1}).contiguous(), targets,
                torch::nn::functional::CrossEntropyFuncOptions{}.ignore_index(-100).reduction(torch::kMean));
            int64_t predicted_tokens = static_cast<int64_t>(cur) * (seq_len - 1);
            total_nll += ce.item<double>() * predicted_tokens;
            token_count += predicted_tokens;

            // Deterministic fingerprint + aggregates over float32 LE bytes.
            // dtype: float32, byte order: host native little-endian (x86-64),
            // layout: [batch, seq_len, vocab] row-major.
            auto acc = logits_cpu.accessor<float, 3>();
            for (int b = 0; b < cur; ++b)
                for (int t = 0; t < seq_len; ++t)
                    for (int v = 0; v < acc.size(2); ++v) {
                        float f = acc[b][t][v];
                        logits_sum += f;
                        logits_sumsq += static_cast<double>(f) * f;
                        const unsigned char* p = reinterpret_cast<const unsigned char*>(&f);
                        for (int k = 0; k < 4; ++k) {
                            logits_hash ^= p[k];
                            logits_hash *= 1099511628211ULL;
                        }
                    }
        }
        HIP_CHECK(hipStreamSynchronize(current_hip_stream()));

        double mean_ce = total_nll / token_count;
        double ppl = std::exp(mean_ce);
        double logits_l2 = std::sqrt(logits_sumsq);

        std::cout << "mode=" << (graft.active ? "grafted" : "base") << "\n";
        std::cout << "checkpoint=" << (checkpoint_path.empty() ? "<none>" : checkpoint_path) << "\n";
        std::cout << "tokens=" << token_count << "\n";
        std::cout << "mean_ce=" << std::setprecision(10) << mean_ce << "\n";
        std::cout << "ppl=" << std::setprecision(10) << ppl << "\n";
        std::cout << "logits_sum=" << std::setprecision(10) << logits_sum << "\n";
        std::cout << "logits_l2=" << std::setprecision(10) << logits_l2 << "\n";
        std::cout << "logits_hash=" << std::hex << logits_hash << std::dec << "\n";
        std::cout << "[eval] graft forward calls=" << g_graft_fwd_calls
                  << " (expected 0 for base-only)\n";

        free_graft_weights_noexcept(graft);
        return EXIT_SUCCESS;
    }
    } else {
        if (safetensors_path.empty() || domain_data_path.empty() || ood_data_paths.empty() || output_path.empty()) {
            std::cerr << "Error: --safetensors, --domain-data, --ood-data (at least one), and --output are required.\n";
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    if (fs::exists(output_path) && fs::is_directory(output_path)) {
        std::cerr << "Error: --output path '" << output_path << "' is a directory. "
                  << "Specify a file path (e.g., /path/to/checkpoint.safetensors).\n";
        return EXIT_FAILURE;
    }
    {
        fs::path parent = fs::path(output_path).parent_path();
        if (!parent.empty() && !fs::exists(parent)) {
            std::cerr << "Error: parent directory '" << parent.string() << "' does not exist.\n";
            return EXIT_FAILURE;
        }
    }
    if (batch_size % 2 != 0) {
        std::cerr << "Error: --batch-size must be even.\n";
        return EXIT_FAILURE;
    }
    if (!g_cfg.no_rope_layers.empty() && static_cast<int>(g_cfg.no_rope_layers.size()) != g_cfg.num_layers) {
        std::cerr << "Error: --rope-layer-mask expects " << g_cfg.num_layers << " values (one per layer), got "
                  << g_cfg.no_rope_layers.size() << ".\n";
        return EXIT_FAILURE;
    }
    if (g_cfg.intermediate_dim % g_cfg.slice_dim != 0) {
        std::cerr << "Error: intermediate-dim (" << g_cfg.intermediate_dim << ") not divisible by slice-dim ("
                  << g_cfg.slice_dim << ").\n";
        return EXIT_FAILURE;
    }

    TokenBuffer domain, ood;
    if (!load_bin(domain_data_path, domain)) return EXIT_FAILURE;
    if (domain.num_sequences == 0) {
        std::cerr << "Error: domain data file '" << domain_data_path << "' has 0 sequences.\n";
        return EXIT_FAILURE;
    }
    for (const auto& p : ood_data_paths) {
        TokenBuffer tmp;
        if (!load_bin(p, tmp)) return EXIT_FAILURE;
        if (tmp.num_sequences == 0) {
            std::cerr << "Warning: OOD file '" << p << "' has 0 sequences ” skipping.\n";
            continue;
        }
        if (tmp.seq_len != domain.seq_len) {
            std::cerr << "Error: seq_len mismatch. Domain has " << domain.seq_len
                      << ", OOD file " << p << " has " << tmp.seq_len << "\n";
            return EXIT_FAILURE;
        }
        ood.tokens.insert(ood.tokens.end(), tmp.tokens.begin(), tmp.tokens.end());
        ood.num_sequences += tmp.num_sequences;
    }
    if (ood.num_sequences == 0) {
        std::cerr << "Error: no valid OOD sequences loaded.\n";
        return EXIT_FAILURE;
    }
    ood.seq_len = domain.seq_len;
    if (max_len != domain.seq_len) {
        std::cerr << "Warning: --max-len (" << max_len << ") != file seq_len (" << domain.seq_len << "). Using file value.\n";
    }

    if (domain_id * g_cfg.slice_dim + g_cfg.slice_dim > g_cfg.intermediate_dim) {
        std::cerr << "Error: domain_id=" << domain_id << " slice [" << domain_id * g_cfg.slice_dim
                  << ", " << (domain_id + 1) * g_cfg.slice_dim << ") exceeds g_cfg.intermediate_dim=" << g_cfg.intermediate_dim << "\n";
        return EXIT_FAILURE;
    }
    if (domain_id * g_cfg.h_slice_dim + g_cfg.h_slice_dim > g_cfg.hidden_dim) {
        std::cerr << "Error: domain_id=" << domain_id << " H-slice [" << domain_id * g_cfg.h_slice_dim
                  << ", " << (domain_id + 1) * g_cfg.h_slice_dim << ") exceeds g_cfg.hidden_dim=" << g_cfg.hidden_dim << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "=== Grafting Training Binary ===\n";
    std::cout << "Model: " << g_cfg.hidden_dim << "hd " << g_cfg.num_layers << "L "
              << g_cfg.num_heads << "h " << g_cfg.num_kv_heads << "kvh "
              << g_cfg.head_dim << "d " << g_cfg.vocab_size << "voc\n";
    std::cout << "Domain ID: " << domain_id << " / " << max_domains
              << " (I-slice: " << domain_id * g_cfg.slice_dim << " to " << (domain_id + 1) * g_cfg.slice_dim
              << ", H-slice: " << domain_id * g_cfg.h_slice_dim << " to " << (domain_id + 1) * g_cfg.h_slice_dim << ")\n";
    std::cout << "Graft projections: gate=" << g_cfg.graft_proj_gate
              << ", up=" << g_cfg.graft_proj_up
              << ", down=" << g_cfg.graft_proj_down
              << ", active_grafts=" << count_active_grafts() << "\n";
    std::cout << "Domain sequences: " << domain.num_sequences
              << ", OOD sequences: " << ood.num_sequences << "\n";
    std::cout << "Seq len: " << domain.seq_len << ", Batch size: " << batch_size
              << ", Steps: " << steps << ", LR: " << lr
              << ", LM loss scale: " << g_cfg.lm_loss_scale
              << ", Lambda silence: " << g_cfg.lambda_silence << "\n";
    std::cout << "Seed: " << seed << "\n";

    // ---- PATCH GLT: steady-state timing / peak-VRAM instrumentation ----
    // Peak VRAM is sampled from hipMemGetInfo after allocation and after every
    // step. Steady-state step times exclude the first-step warmup (kernels /
    // allocators not yet hot). Final checkpoint save happens after the loop and
    // is intentionally NOT part of step timing.
    size_t peak_used_mb = 0;
    std::vector<double> steady_step_times;
    const int warmup_steps_excluded = 1;   // first-step warmup excluded from median
    auto sample_peak_vram = [&]() {
        size_t fb = 0, tot = 0;
        if (hipMemGetInfo(&fb, &tot) == hipSuccess) {
            size_t u = tot - fb;
            if (u > peak_used_mb) peak_used_mb = u;
        }
    };

    // GLT: HIP events spanning a full step (forward/backward on main stream +
    // AdamW on adamw_stream) for completion-based, not submission-based, timing.
    hipEvent_t step_start_event, step_end_event;
    HIP_CHECK(hipEventCreate(&step_start_event));
    HIP_CHECK(hipEventCreate(&step_end_event));

    if (!load_base_model_safetensors(safetensors_path)) {
        return EXIT_FAILURE;
    }

    {
        // PATCH P3-19: rename `free` to `free_bytes` (shadowed C free()).
        size_t free_bytes = 0, total = 0;
        HIP_CHECK(hipMemGetInfo(&free_bytes, &total));
        size_t used = total - free_bytes;
        std::cout << "[VRAM] Weights loaded: " << (used / 1024 / 1024) << " MB used / "
                  << (total / 1024 / 1024) << " MB total\n";
    }

    VramArena arena{};
    allocate_arena(arena, batch_size, domain.seq_len);
    std::cout << "[Arena] Allocated " << (arena.total_bytes / 1024 / 1024) << " MB of VRAM.\n";

    // Phase 2: GraftWeights is the physical owner of persistent graft
    // parameters (delta weights + optional bf16 mirror). Allocated separately
    // from the arena, which retains only scratch/gradients/optimizer state.
    GraftWeights graft{};
    allocate_graft_weights(graft, domain_id, g_cfg.delta_bf16_mirror);
    sample_peak_vram();   // peak after base + arena + graft allocation

    // Ownership proof: graft pointers must NOT lie inside the arena range.
    {
        char* arena_base = static_cast<char*>(arena.raw_ptr);
        char* arena_end = arena_base + arena.total_bytes;
        assert(reinterpret_cast<char*>(graft.d_weight) < arena_base ||
               reinterpret_cast<char*>(graft.d_weight) >= arena_end);
        if (graft.d_weight_bf16 != nullptr) {
            assert(reinterpret_cast<char*>(graft.d_weight_bf16) < arena_base ||
                   reinterpret_cast<char*>(graft.d_weight_bf16) >= arena_end);
        }
        std::cerr << "[ownership] arena.raw_ptr=" << arena.raw_ptr
                  << " arena.total_bytes=" << arena.total_bytes
                  << " graft.d_weight=" << graft.d_weight
                  << " graft.d_weight_bf16=" << graft.d_weight_bf16
                  << std::endl;
    }

    // Training always operates with an active graft (the graft GEMM path runs
    // unconditionally; without a checkpoint the deltas are simply zero). The
    // graft.active gate in forward_layer therefore never suppresses graft math
    // during training — only eval-only base (no checkpoint) disables it.
    graft.active = true;

    if (!checkpoint_path.empty()) {
        std::cout << "[Checkpoint] Loading from " << checkpoint_path << "\n";
        if (!load_safetensors_checkpoint(checkpoint_path, domain_id, arena, graft)) return EXIT_FAILURE;
        graft.active = true;
    }

    // Parity mode: load deterministic initial delta weights (raw float32 flat array)
    if (!init_deltas_path.empty()) {
        std::cout << "[Parity] Loading initial deltas from " << init_deltas_path << "\n";
        size_t expected = static_cast<size_t>(g_cfg.total_projections) * g_cfg.slice_dim * g_cfg.h_slice_dim * sizeof(float);
        FILE* fp = fopen(init_deltas_path.c_str(), "rb");
        if (!fp) { std::cerr << "Error: cannot open " << init_deltas_path << "\n"; return EXIT_FAILURE; }
        fseek(fp, 0, SEEK_END); size_t file_bytes = ftell(fp); rewind(fp);
        if (file_bytes != expected) {
            std::cerr << "Error: " << init_deltas_path << " size " << file_bytes
                      << " != expected " << expected << "\n"; fclose(fp); return EXIT_FAILURE;
        }
        std::vector<float> host_buf(g_cfg.total_projections * g_cfg.slice_dim * g_cfg.h_slice_dim);
        size_t read_cnt = fread(host_buf.data(), sizeof(float), host_buf.size(), fp);
        fclose(fp);
        if (read_cnt != host_buf.size()) { std::cerr << "Error: short read\n"; return EXIT_FAILURE; }
        HIP_CHECK(hipMemcpy(graft.d_weight, host_buf.data(),
                            expected, hipMemcpyHostToDevice));
        // Also zero out adam states (re-initialized for clean run)
        size_t sz_w = static_cast<size_t>(g_cfg.total_projections) * g_cfg.slice_dim * g_cfg.h_slice_dim * sizeof(float);
        HIP_CHECK(hipMemset(arena.d_adam_m, 0, sz_w));
        HIP_CHECK(hipMemset(arena.d_adam_v, 0, sz_w));
        refresh_graft_weight_bf16(arena, graft);
        std::cout << "[Parity] Loaded " << read_cnt << " floats. Adam states zeroed.\n";

        if (!parity_dump_dir.empty()) {
            std::string pd = parity_dump_dir;
            if (!pd.empty() && pd.back() != '/' && pd.back() != '\\') pd += '/';
            std::filesystem::create_directories(pd);
            std::ofstream wf(pd + "delta_weights_loaded.raw", std::ios::binary);
            wf.write(reinterpret_cast<const char*>(host_buf.data()), expected);
            std::cout << "[Parity] Dumped " << host_buf.size() << " floats (delta_weights_loaded)\n";
        }
        graft.active = true;
    }


    if (g_cfg.contract_probe) {
        std::cout << "[ContractProbe] Running synthetic sparse contract-forward test.\n";
        run_contract_sparse_probe(arena, graft);
        return 0;
    }

    int half = batch_size / 2;
    int total_tokens = batch_size * domain.seq_len;
    std::vector<int32_t> host_input_ids(total_tokens);
    std::vector<int32_t> host_ood_mask(total_tokens);
    auto mix_seed = [](uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    };
    // ponytail: seed only offsets the corpus walk; no extra RNG state.
    uint64_t seed_u = static_cast<uint64_t>(seed);
    int domain_pos = static_cast<int>(mix_seed(seed_u) % static_cast<uint64_t>(domain.num_sequences));
    int ood_pos = static_cast<int>(mix_seed(seed_u ^ 0x9e3779b97f4a7c15ULL) % static_cast<uint64_t>(ood.num_sequences));

    // Structured logging to CSV
    std::string log_path = output_path.empty() ? "train_log.csv" :
        output_path.substr(0, output_path.find_last_of('.')) + "_log.csv";
    std::ofstream log_file(log_path);
    if (log_file) {
        log_file << "step,lm_loss,silence_loss,domain_graft_energy,ood_graft_energy,domain_ood_energy_ratio,total_loss,lr,grad_norm,step_time_sec\n";  // PATCH P2-13: max_inter_norm removed. PATCH P2-25: grad_norm added.
    }

    std::cout << "[Loop] Starting training for " << steps << " steps.\n"
              << "       Logging to " << log_path << "\n";
    std::cout << "[Loop] grad_norm logging: async one-step-delayed\n";

    // Pre-signal adamw_event so step 0's adamw wait passes immediately
    HIP_CHECK(hipEventRecord(arena.adamw_event, arena.adamw_stream));

    for (int step = 0; step < steps; ++step) {
        g_layer_activations.clear();
        g_layer_activations.reserve(g_cfg.num_layers);

        // Pipeline: wait for previous step's AdamW to finish on the current stream.
        HIP_CHECK(hipStreamWaitEvent(current_hip_stream(), arena.adamw_event, 0));
        // GLT: mark the start of THIS step's GPU work (after the prev-step wait).
        HIP_CHECK(hipEventRecord(step_start_event, current_hip_stream()));
        float host_scale = -1.0f;
        float grad_norm = -1.0f;

        for (int i = 0; i < half; ++i) {
            int ds = (domain_pos + i) % domain.num_sequences;
            std::memcpy(&host_input_ids[i * domain.seq_len],
                        &domain.tokens[ds * domain.seq_len],
                        domain.seq_len * sizeof(int32_t));
            std::memset(&host_ood_mask[i * domain.seq_len], 0,
                        domain.seq_len * sizeof(int32_t));
        }
        domain_pos = (domain_pos + half) % domain.num_sequences;

        for (int i = 0; i < half; ++i) {
            int os = (ood_pos + i) % ood.num_sequences;
            std::memcpy(&host_input_ids[(half + i) * domain.seq_len],
                        &ood.tokens[os * domain.seq_len],
                        domain.seq_len * sizeof(int32_t));
            std::fill(host_ood_mask.data() + (half + i) * domain.seq_len,
                      host_ood_mask.data() + (half + i + 1) * domain.seq_len, 1);
        }
        ood_pos = (ood_pos + half) % ood.num_sequences;

        HIP_CHECK(hipMemcpy(arena.d_input_ids, host_input_ids.data(),
                            total_tokens * sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(arena.d_ood_mask, host_ood_mask.data(),
                            total_tokens * sizeof(int32_t), hipMemcpyHostToDevice));

        // ============================================================
        // PHASE 1: FORWARD PASS - Store all layer activations
        // ============================================================
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
        auto input_ids = torch::from_blob(arena.d_input_ids, {batch_size, domain.seq_len}, options);

        // Shared full forward pass (embedding → layers → final norm → LM head).
        auto fwd = forward_model(input_ids, arena, graft, batch_size, domain.seq_len);
        at::Tensor logits = std::get<0>(fwd);
        at::Tensor x_before_final_norm = std::get<1>(fwd);
        at::Tensor final_norm_variance = std::get<2>(fwd);

        if (g_verbose) std::cerr << "[diag] lm_head_weight: " << g_model_weights.lm_head_weight.sizes() << std::endl;
        if (g_verbose) std::cerr << "[diag] x before lm_head: " << x_before_final_norm.sizes() << std::endl;
        if (!g_cfg.parity_dump_dir.empty()) {
            dump_f32_tensor_raw(x_before_final_norm, parity_dump_path("final_hidden.raw"));
            dump_f32_tensor_raw(logits[0], parity_dump_path("logits.raw"));
        }

        // Parity diagnostic: dump top-5 logits and first 8 logit values
        if (!g_cfg.parity_dump_dir.empty()) {
            // Compute per-token CE for first 10 domain tokens
            std::cout << "[Parity] CE per token:";
            for (int pos = 0; pos < 10; pos++) {
                auto lpos = logits[0][pos];
                auto label = input_ids[0][pos + 1];
                float log_p = at::log_softmax(lpos, 0)[label].item<float>();
                // Compute relative max logit difference
                float max_logit = std::get<0>(lpos.topk(1)).item<float>();
                std::cout << " ce=" << -log_p << "[max=" << max_logit << "]";
            }
            std::cout << "\n";
        }

        // ============================================================
        // LOSS COMPUTATION
        // ============================================================
        auto labels = input_ids.slice(1, 1, domain.seq_len);
        auto shift_logits = logits.slice(1, 0, domain.seq_len - 1);

        // Per-token cross-entropy via ATen ops (no torch::nn bloat)
        auto logits_flat = shift_logits.reshape({-1, g_cfg.vocab_size});
        auto labels_flat = labels.reshape({-1});
        auto ce_per_token = -at::log_softmax(logits_flat, -1)
            .gather(-1, labels_flat.unsqueeze(-1)).squeeze(-1);
        auto ood_full = torch::from_blob(
            arena.d_ood_mask, {batch_size, domain.seq_len},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)
        );
        auto domain_mask = (1 - ood_full.slice(1, 1, domain.seq_len).reshape({-1})).to(torch::kFloat32);
        float in_count = domain_mask.sum().item<float>();
        if (in_count < 1.0f) in_count = 1.0f;
        at::Tensor loss = (ce_per_token * domain_mask).sum() / in_count;

        auto target_logits = shift_logits.reshape({-1, g_cfg.vocab_size})
            .gather(-1, labels_flat.unsqueeze(-1)).squeeze(-1);
        auto target_probs = at::softmax(logits_flat, -1)
            .gather(-1, labels_flat.unsqueeze(-1)).squeeze(-1);
        float target_logit_mean = (target_logits * domain_mask).sum().item<float>() / in_count;
        float target_prob_mean = (target_probs * domain_mask).sum().item<float>() / in_count;

        float loss_val = loss.item<float>();
        if (!g_cfg.parity_dump_dir.empty()) {
            std::ofstream(parity_dump_path("lm_loss.txt")) << loss_val << "\n";
            std::ofstream(parity_dump_path("target_logits_mean.txt")) << target_logit_mean << "\n";
            std::ofstream(parity_dump_path("target_prob_mean.txt")) << target_prob_mean << "\n";
        }

        // TPHS-compatible silence loss: lambda * mean(||delta_out||/sqrt(V)) over OOD tokens and grafts
        float silence_val = 0.0f;
        float domain_graft_energy_val = 0.0f;
        float ood_graft_energy_val = 0.0f;
        float domain_ood_energy_ratio_val = 0.0f;
        float ood_count = 0.0f;
        int active_grafts = count_active_grafts();
        float lambda_effective = 0.0f;
        {
            auto ood_full_flat = ood_full.reshape({-1}).to(torch::kFloat32);
            auto domain_full_flat = (1 - ood_full_flat).to(torch::kFloat32);
            float domain_count = in_count;
            ood_count = ood_full_flat.sum().item<float>();
            if (ood_count > 0.0f && domain_count > 0.0f && active_grafts > 0) {
                at::Tensor domain_acc = at::zeros({1}, torch::kFloat32).to(torch::kCUDA);
                at::Tensor ood_acc = at::zeros({1}, torch::kFloat32).to(torch::kCUDA);
                for (int l = 0; l < g_cfg.num_layers; ++l) {
                    auto& acts = g_layer_activations[l];
                    if (l == 0 && !g_cfg.parity_dump_dir.empty()) {
                        float eg[4];
                        HIP_CHECK(hipMemcpy(eg, acts.energy_gate.data_ptr<float>(), 16, hipMemcpyDeviceToHost));
                        std::cerr << "[diag] stored energy[0..3]: " << eg[0] << " " << eg[1] << " " << eg[2] << " " << eg[3] << std::endl;
                    }
                    if (should_graft_proj(l, 0)) {
                        domain_acc += (acts.energy_gate * domain_full_flat).sum() / domain_count;
                        ood_acc += (acts.energy_gate * ood_full_flat).sum() / ood_count;
                        if (l == g_cfg.graft_layer_start && !g_cfg.parity_dump_dir.empty()) {
                            float s = ((acts.energy_gate * ood_full_flat).sum() / ood_count).item<float>();
                            std::cerr << "[diag] first gate sil_acc_term=" << s << std::endl;
                        }
                    }
                    if (should_graft_proj(l, 1)) {
                        domain_acc += (acts.energy_up * domain_full_flat).sum() / domain_count;
                        ood_acc += (acts.energy_up * ood_full_flat).sum() / ood_count;
                    }
                    if (should_graft_proj(l, 2)) {
                        domain_acc += (acts.energy_down * domain_full_flat).sum() / domain_count;
                        ood_acc += (acts.energy_down * ood_full_flat).sum() / ood_count;
                    }
                }
                domain_acc /= active_grafts;
                ood_acc /= active_grafts;
                domain_graft_energy_val = domain_acc.item<float>();
                ood_graft_energy_val = ood_acc.item<float>();
                domain_ood_energy_ratio_val = (ood_graft_energy_val > 0.0f) ? (domain_graft_energy_val / ood_graft_energy_val) : 0.0f;
                silence_val = g_cfg.lambda_silence * ood_graft_energy_val;
                lambda_effective = g_cfg.lambda_silence / (ood_count * active_grafts);
            }
        }
        if (!g_cfg.parity_dump_dir.empty()) {
            std::cerr << "[diag] active_grafts=" << active_grafts
                      << " ood_count=" << ood_count
                      << " lambda_silence=" << g_cfg.lambda_silence
                      << " lambda_effective=" << lambda_effective
                      << " silence_val=" << silence_val
                      << " domain_graft_energy=" << domain_graft_energy_val
                      << " ood_graft_energy=" << ood_graft_energy_val
                      << " domain_ood_energy_ratio=" << domain_ood_energy_ratio_val << std::endl;
        }
        float total_loss_val = g_cfg.lm_loss_scale * loss_val + silence_val;

        // ============================================================
        // PHASE 2: BACKWARD PASS - Reverse order
        // ============================================================
        at::Tensor grad_logits = torch::zeros_like(shift_logits);

        // (softmax - one_hot) * domain_mask / in_count
        at::Tensor probs = torch::softmax(logits_flat, -1);
        grad_logits = probs;
        grad_logits.index_put_({torch::arange(labels_flat.numel()), labels_flat},
                               grad_logits.index({torch::arange(labels_flat.numel()), labels_flat}) - 1.0f);
        grad_logits *= domain_mask.unsqueeze(-1);
        grad_logits /= in_count;
        grad_logits *= g_cfg.lm_loss_scale;
        grad_logits = grad_logits.reshape({batch_size, domain.seq_len - 1, g_cfg.vocab_size});

        at::Tensor grad_logits_full = torch::zeros({batch_size, domain.seq_len, g_cfg.vocab_size}, logits.options());
        grad_logits_full.slice(1, 0, domain.seq_len - 1).copy_(grad_logits);

        at::Tensor dx = at::matmul(grad_logits_full, g_model_weights.lm_head_weight);

        // Backprop through final RMSNorm to get gradient wrt layer 35 output
        dx = rms_norm_bwd(dx, x_before_final_norm, g_model_weights.final_norm_weight, final_norm_variance);

        // Scheduled LR: linear warmup then cosine decay
        float step_lr = get_lr(step, steps);

        for (int l = g_cfg.num_layers - 1; l >= 0; --l) {
            backward_layer(l, dx, arena, batch_size, domain.seq_len, step, step_lr, domain_id, graft, arena.d_ood_mask, lambda_effective);
        }

        // Gradient clipping: global L2 norm on all 108 deltas
            {
                float* d_grad = arena.d_delta_grad;
                int n = g_cfg.total_projections * g_cfg.slice_dim * g_cfg.h_slice_dim;
                int threads = 256;
                int nb = (n + threads * 4 - 1) / (threads * 4);
                int num_blocks = nb < 1024 ? nb : 1024;
                float* d_partial = arena.d_grad_clip_partial;
                float* d_scale = arena.d_grad_clip_scale;
        reduce_l2_kernel<<<num_blocks, threads, 0, current_hip_stream()>>>(d_grad, n, d_partial);
        HIP_CHECK(hipGetLastError());
                calc_scale_kernel<<<1, threads, 0, current_hip_stream()>>>(d_partial, num_blocks, g_cfg.gradient_clip_norm, d_scale);
                HIP_CHECK(hipGetLastError());
                apply_scale_kernel<<<num_blocks, threads, 0, current_hip_stream()>>>(d_grad, n, d_scale);
                HIP_CHECK(hipGetLastError());
            }
        // Pipeline: sequence backward -> AdamW across streams
        // backward_event fires after the current stream completes the backward pass
        HIP_CHECK(hipEventRecord(arena.backward_event, current_hip_stream()));
        // adamw_stream waits for backward, then processes OPTIM
        HIP_CHECK(hipStreamWaitEvent(arena.adamw_stream, arena.backward_event, 0));
        execute_adamw_all(arena, graft, step, step_lr, arena.adamw_stream);
        // adamw_event fires after AdamW completes ” next step waits on it
        HIP_CHECK(hipEventRecord(arena.adamw_event, arena.adamw_stream));
        // GLT: completion-based timing. step_end is recorded on the adamw stream
        // AFTER this step's AdamW, so the interval covers forward+backward+adamw.
        // We synchronize on step_end before reading, so the CPU timer cannot stop
        // while GPU kernels are still in flight (no fake "HIP is faster" result).
        HIP_CHECK(hipEventRecord(step_end_event, arena.adamw_stream));
        HIP_CHECK(hipEventSynchronize(step_end_event));
        float step_ms = 0;
        HIP_CHECK(hipEventElapsedTime(&step_ms, step_start_event, step_end_event));
        double step_time_sec = step_ms / 1000.0;
        // GLT instrumentation: sample peak VRAM each step; record steady-state
        // time (exclude first-step warmup) for the median.
        sample_peak_vram();
        if (step > 0) steady_step_times.push_back(step_time_sec);
        if (step > 0) {
            hipError_t copy_status = hipEventQuery(arena.grad_clip_copy_event);
            if (copy_status == hipSuccess) {
                float* prev_host_scale = ((step - 1) & 1) == 0 ? arena.h_grad_clip_scale_ping : arena.h_grad_clip_scale_pong;
                host_scale = *prev_host_scale;
                grad_norm = (host_scale > 0.0f) ? g_cfg.gradient_clip_norm / host_scale : 0.0f;
            } else if (copy_status != hipErrorNotReady) {
                HIP_CHECK(copy_status);
            }
        }

        if (log_file) {
            log_file << step << "," << loss_val << "," << silence_val << ","
                     << domain_graft_energy_val << "," << ood_graft_energy_val << ","
                     << domain_ood_energy_ratio_val << "," << total_loss_val << ","
                     << step_lr << "," << grad_norm << "," << step_time_sec << "\n";  // PATCH P2-13/P2-25
        }

        // Async host copy for next step's grad_norm logging.
        float* current_host_scale = (step & 1) == 0 ? arena.h_grad_clip_scale_ping : arena.h_grad_clip_scale_pong;
        HIP_CHECK(hipMemcpyAsync(current_host_scale, arena.d_grad_clip_scale, sizeof(float),
                                 hipMemcpyDeviceToHost, current_hip_stream()));
        HIP_CHECK(hipEventRecord(arena.grad_clip_copy_event, current_hip_stream()));

        // Parity dump: one-step forward/backward/optimizer tensors for TPHS comparison
        if (step == 0 && !parity_dump_dir.empty()) {
            std::string pd = parity_dump_dir;
            if (!pd.empty() && pd.back() != '/' && pd.back() != '\\') pd += '/';
            // Create directory (mkdir -p equivalent via system call)
            std::string mkdir_cmd = "mkdir -p " + pd;
            system(mkdir_cmd.c_str());
            std::cout << "[Parity] Dumping step 0 tensors to " << pd << "\n";

            // 1. Losses
            std::ofstream lf(pd + "losses.csv");
            lf << "lm_loss,silence_loss,domain_graft_energy,ood_graft_energy,domain_ood_energy_ratio,total_loss,grad_norm,step_time_sec\n";
            lf << loss_val << "," << silence_val << "," << domain_graft_energy_val << "," << ood_graft_energy_val << ","
               << domain_ood_energy_ratio_val << "," << total_loss_val << "," << grad_norm << "," << step_time_sec << "\n";
            std::ofstream(pd + "grad_clip_coef.txt") << host_scale << "\n";

            // 2. Delta gradients (post-backward, pre-optimizer)
            size_t sz_buf = static_cast<size_t>(g_cfg.total_projections) * g_cfg.slice_dim * g_cfg.h_slice_dim;
            std::vector<float> host_grads(sz_buf);
            HIP_CHECK(hipMemcpy(host_grads.data(), arena.d_delta_grad,
                                sz_buf * sizeof(float), hipMemcpyDeviceToHost));
            std::ofstream gf(pd + "delta_grads.raw", std::ios::binary);
            gf.write(reinterpret_cast<const char*>(host_grads.data()),
                     sz_buf * sizeof(float));
            std::cout << "[Parity] Dumped " << sz_buf << " floats (delta_grads)\n";

            // 3. Delta weights (post-optimizer)
            // Wait for AdamW stream to finish before reading back
            HIP_CHECK(hipStreamSynchronize(arena.adamw_stream));
            std::vector<float> host_weights(sz_buf);
            HIP_CHECK(hipMemcpy(host_weights.data(), graft.d_weight,
                                sz_buf * sizeof(float), hipMemcpyDeviceToHost));
            std::ofstream wf(pd + "delta_weights_post.raw", std::ios::binary);
            wf.write(reinterpret_cast<const char*>(host_weights.data()),
                     sz_buf * sizeof(float));
            std::cout << "[Parity] Dumped " << sz_buf << " floats (delta_weights_post)\n";
        }

        if (step % 50 == 0) {
            std::cout << "  Step " << step << "/" << steps
                      << " | lm=" << loss_val
                      << " | silence=" << silence_val
                      << " | total=" << total_loss_val
                      << " | lr=" << step_lr
                      << " | grad_norm=" << grad_norm << "\n";  // PATCH P2-25
        }
    }

    // Sync: ensure last step's AdamW completes before saving
    HIP_CHECK(hipStreamSynchronize(arena.adamw_stream));

    // ---- GLT: emit machine-readable steady-state timing / peak VRAM ----
    double sec_per_step_p50 = 0.0;
    if (!steady_step_times.empty()) {
        std::vector<double> tmp = steady_step_times;
        std::sort(tmp.begin(), tmp.end());
        size_t m = tmp.size() / 2;
        sec_per_step_p50 = (tmp.size() % 2 == 1) ? tmp[m] : (tmp[m - 1] + tmp[m]) / 2.0;
    }
    std::cout << "sec_per_step_p50=" << std::fixed << std::setprecision(6) << sec_per_step_p50 << "\n";
    std::cout << "peak_train_vram_mb=" << (peak_used_mb / 1024 / 1024) << "\n";
    std::cout << "timed_steps=" << steady_step_times.size() << "\n";
    std::cout << "warmup_steps_excluded=" << warmup_steps_excluded << "\n";
    HIP_CHECK(hipEventDestroy(step_start_event));
    HIP_CHECK(hipEventDestroy(step_end_event));

    std::cout << "[Save] Writing safetensors to " << output_path << "\n";
    save_safetensors(output_path, graft);
    write_gemm_profile_stats_json(output_path);

    // Free graft weights only after all GPU operations using them are done.
    free_graft_weights_noexcept(graft);

    // RAII: arena destructor will free resources automatically
    std::cout << "Experiment completed successfully.\n";
    }
    return EXIT_SUCCESS;
} catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return EXIT_FAILURE;
}