// test22_shadow_bake_sentinel.cpp
// Compile (host /workspace):
//   hipcc -std=c++17 -O3 -I/workspace \
//     -I/opt/venv/lib/python3.10/site-packages/torch/include \
//     -I/opt/venv/lib/python3.10/site-packages/torch/include/torch/csrc/api/include \
//     -L/opt/venv/lib/python3.10/site-packages/torch/lib -L/opt/rocm-7.2.1/lib \
//     -ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip -lrocblas -lhipblaslt \
//     -Wl,-rpath,/opt/venv/lib/python3.10/site-packages/torch/lib \
//     -Wl,-rpath,/opt/rocm-7.2.1/lib --hip-link \
//     -o /workspace/test22 /workspace/tests/test22_shadow_bake_sentinel.cpp
// Run: ./test22
//
// Sentinel for the Shadow Copy bake. Exercises the REAL apply_graft_to_shadow()
// with the production kernel in src/shadow_bake_impl.inc, including:
//   - h_slice_dim (513) > kernel_threads (256): kernel MUST loop
//     `for (int h = threadIdx.x; h < h_slice_dim; h += blockDim.x)`
//   - non-symmetric sentinel values so accidental transpose fails
//   - projection-mask logic (gate/up/down independently enabled/disabled)
//   - layer-range logic (should_graft_layer)
//   - double-bake guard (g_shadow_baked)

#include <torch/torch.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cassert>

#include "src/checks.hpp"            // HIP_CHECK, current_hip_stream
#include "src/config.hpp"            // g_cfg, should_graft_layer, PROJECTIONS_PER_LAYER
#include "src/types.hpp"             // ModelWeights, GraftWeights, at::Tensor
#include "src/kernels_elementwise.hpp"
#include "src/kernels_elementwise_impl.inc"  // bf16_to_f32 / f32_to_bf16 device fns

// delta_offset — local replica of src/graft_ops_impl.inc::delta_offset (single
// source of truth for the flat index; include graft_ops_impl.inc would drag in
// the full GEMM stack we don't need here). Defined before shadow_bake_impl.inc
// which references it.
int delta_offset(int layer_idx, int proj_idx) {
    return (layer_idx * PROJECTIONS_PER_LAYER + proj_idx) * g_cfg.slice_dim * g_cfg.h_slice_dim;
}

#include "src/shadow_bake.hpp"        // apply_graft_to_shadow, g_shadow_baked
#include "src/shadow_bake_impl.inc"   // REAL kernel + apply_graft_to_shadow

// bf16 helpers on host (for verification)
static inline float bf16_to_f32_host(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

int g_pass = 0, g_fail = 0;
void check(const char* name, bool ok) {
    if (ok) { printf("  PASS: %s\n", name); g_pass++; }
    else    { printf("  FAIL: %s\n", name); g_fail++; }
}

// Read a bf16 CUDA tensor to a host float vector.
std::vector<float> read_bf16(const at::Tensor& t) {
    auto f = t.to(torch::kFloat32).to(torch::kCPU).contiguous();
    return std::vector<float>(f.data_ptr<float>(), f.data_ptr<float>() + f.numel());
}

int main() {
    printf("=== Test 22: Shadow Bake Orientation Sentinel (REAL apply_graft_to_shadow) ===\n\n");

    // ---- Config: H=513 > kernel_threads=256 to force the h-loop ----
    const int S = 3;                       // slice_dim
    const int H = 513;                     // h_slice_dim  (H > 256)
    const int KT = 256;                    // kernel_threads
    const int NUM_LAYERS = 2;
    g_cfg.num_layers = NUM_LAYERS;
    g_cfg.slice_dim = S;
    g_cfg.h_slice_dim = H;
    g_cfg.kernel_threads = KT;
    g_cfg.hidden_dim = H;                  // hidden slice == h_slice_dim
    g_cfg.intermediate_dim = S;            // intermediate slice == slice_dim
    g_cfg.total_projections = NUM_LAYERS * PROJECTIONS_PER_LAYER;
    g_cfg.graft_layer_start = 0;
    g_cfg.graft_layer_end = NUM_LAYERS - 1;
    g_cfg.graft_proj_gate = 1;
    g_cfg.graft_proj_up = 1;
    g_cfg.graft_proj_down = 1;

    // ---- Host sentinel: graft[layer,proj,s,h] non-symmetric ----
    // value = (layer*100 + proj*10 + s) * 1000 + h + 1
    auto graft_val = [&](int layer, int proj, int s, int h) -> float {
        return static_cast<float>((layer * 100 + proj * 10 + s) * 1000 + h + 1);
    };
    auto check_close = [](float a, float b, float tol) -> bool {
        return std::fabs(a - b) <= tol + 0.005f * std::fabs(b);
    };

    const float TOL = 0.5f;  // ~0.5% relative + small absolute for bf16 rounding

    // ====================================================================
    // Build a REAL ModelWeights (2 layers) + GraftWeights
    // ====================================================================
    auto build_model = [&]() -> ModelWeights {
        ModelWeights m;
        for (int l = 0; l < NUM_LAYERS; ++l) {
            LayerWeights w;
            w.gate_proj_weight     = torch::zeros({S, H}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0));
            w.up_proj_weight       = torch::zeros({S, H}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0));
            w.down_proj_weight     = torch::zeros({H, S}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0));
            m.layers.push_back(w);
        }
        return m;
    };

    auto build_graft = [&](GraftWeights& g) {
        size_t elems = static_cast<size_t>(g_cfg.total_projections) * S * H;
        g.weight_elems = elems;
        HIP_CHECK(hipMalloc(&g.d_weight, elems * sizeof(float)));
        g.domain_id = 0;
        g.allocated = true;
        g.active = true;
        // Fill sentinel on host then upload
        std::vector<float> host(elems, 0.0f);
        for (int l = 0; l < NUM_LAYERS; ++l)
            for (int p = 0; p < PROJECTIONS_PER_LAYER; ++p)
                for (int s = 0; s < S; ++s)
                    for (int h = 0; h < H; ++h) {
                        int off = delta_offset(l, p);
                        host[off + s * H + h] = graft_val(l, p, s, h);
                    }
        HIP_CHECK(hipMemcpy(g.d_weight, host.data(), elems * sizeof(float), hipMemcpyHostToDevice));
    };

    // ====================================================================
    // Test 1: Full bake (gate+up+down, all layers) — verify orientation
    // ====================================================================
    printf("--- Test 1: full bake, all projections, all layers ---\n");
    {
        ModelWeights m = build_model();
        GraftWeights g; build_graft(g);
        g_shadow_baked = false;
        apply_graft_to_shadow(m, g);

        bool ok = true;
        for (int l = 0; l < NUM_LAYERS; ++l) {
            auto gate = read_bf16(m.layers[l].gate_proj_weight);     // [S,H]
            auto up   = read_bf16(m.layers[l].up_proj_weight);       // [S,H]
            auto down = read_bf16(m.layers[l].down_proj_weight);     // [H,S]
            for (int s = 0; s < S; ++s)
                for (int h = 0; h < H; ++h) {
                    float ge = graft_val(l, 0, s, h);
                    float ue = graft_val(l, 1, s, h);
                    float de = graft_val(l, 2, s, h);
                    if (!check_close(gate[s * H + h], ge, TOL)) { printf("  gate[%d,%d,%d,%d] %f vs %f\n", l, s, h, l, gate[s*H+h], ge); ok = false; }
                    if (!check_close(up[s * H + h],   ue, TOL)) { printf("  up[%d,%d,%d] %f vs %f\n", l, s, h, up[s*H+h], ue); ok = false; }
                    if (!check_close(down[h * S + s], de, TOL)) { printf("  down[%d,%d,%d,%d] %f vs %f\n", l, h, s, l, down[h*S+s], de); ok = false; }
                }
        }
        check("all projections baked with correct expand/contract orientation", ok);
        HIP_CHECK(hipFree(g.d_weight));
    }

    // ====================================================================
    // Test 2: H>blockDim loop actually ran (columns 256..512 baked)
    // ====================================================================
    printf("\n--- Test 2: h-loop coverage (H=513 > KT=256) ---\n");
    {
        ModelWeights m = build_model();
        GraftWeights g; build_graft(g);
        g_shadow_baked = false;
        apply_graft_to_shadow(m, g);
        auto gate = read_bf16(m.layers[0].gate_proj_weight);
        // column 300 can ONLY be filled if the loop `h += blockDim.x` executed
        bool loop_ran = true;
        for (int s = 0; s < S; ++s) {
            float v = gate[s * H + 300];
            if (!check_close(v, graft_val(0, 0, s, 300), TOL)) { loop_ran = false; printf("  gate col300 s=%d = %f (expected %f)\n", s, v, graft_val(0,0,s,300)); }
        }
        check("kernel loop baked h>=256 columns (h+=blockDim.x)", loop_ran);
        HIP_CHECK(hipFree(g.d_weight));
    }

    // ====================================================================
    // Test 3: projection-mask logic — disable gate, only up/down bake
    // ====================================================================
    printf("\n--- Test 3: projection masks (gate disabled) ---\n");
    {
        g_cfg.graft_proj_gate = 0;
        g_cfg.graft_proj_up = 1;
        g_cfg.graft_proj_down = 1;
        ModelWeights m = build_model();
        GraftWeights g; build_graft(g);
        g_shadow_baked = false;
        apply_graft_to_shadow(m, g);
        auto gate = read_bf16(m.layers[0].gate_proj_weight);
        auto up   = read_bf16(m.layers[0].up_proj_weight);
        auto down = read_bf16(m.layers[0].down_proj_weight);
        bool gate_untouched = true;
        for (size_t i = 0; i < gate.size(); ++i)
            if (gate[i] != 0.0f) { gate_untouched = false; break; }
        bool up_ok = true, down_ok = true;
        for (int s = 0; s < S; ++s)
            for (int h = 0; h < H; ++h) {
                if (!check_close(up[s * H + h],   graft_val(0, 1, s, h), TOL)) up_ok = false;
                if (!check_close(down[h * S + s], graft_val(0, 2, s, h), TOL)) down_ok = false;
            }
        check("gate projection mask disabled -> gate weights unchanged", gate_untouched);
        check("up still baked when gate disabled", up_ok);
        check("down still baked when gate disabled", down_ok);
        g_cfg.graft_proj_gate = 1;  // restore
        HIP_CHECK(hipFree(g.d_weight));
    }

    // ====================================================================
    // Test 4: layer-range logic — only layer 1 grafted
    // ====================================================================
    printf("\n--- Test 4: layer range (graft_layer_start=1) ---\n");
    {
        g_cfg.graft_layer_start = 1;
        g_cfg.graft_layer_end = 1;
        ModelWeights m = build_model();
        GraftWeights g; build_graft(g);
        g_shadow_baked = false;
        apply_graft_to_shadow(m, g);
        auto gate_l0 = read_bf16(m.layers[0].gate_proj_weight);
        auto gate_l1 = read_bf16(m.layers[1].gate_proj_weight);
        bool l0_untouched = true;
        for (size_t i = 0; i < gate_l0.size(); ++i)
            if (gate_l0[i] != 0.0f) { l0_untouched = false; break; }
        bool l1_ok = true;
        for (int s = 0; s < S; ++s)
            for (int h = 0; h < H; ++h)
                if (!check_close(gate_l1[s * H + h], graft_val(1, 0, s, h), TOL)) l1_ok = false;
        check("layer 0 outside range -> unchanged", l0_untouched);
        check("layer 1 inside range -> baked", l1_ok);
        g_cfg.graft_layer_start = 0;  // restore
        g_cfg.graft_layer_end = NUM_LAYERS - 1;
        HIP_CHECK(hipFree(g.d_weight));
    }

    // ====================================================================
    // Test 5: double-bake guard throws
    // ====================================================================
    printf("\n--- Test 5: double-bake guard ---\n");
    {
        ModelWeights m = build_model();
        GraftWeights g; build_graft(g);
        g_shadow_baked = false;
        apply_graft_to_shadow(m, g);
        bool threw = false;
        try {
            apply_graft_to_shadow(m, g);  // second call
        } catch (const std::exception& e) {
            threw = true;
            printf("  (caught: %s)\n", e.what());
        }
        check("second apply_graft_to_shadow throws (g_shadow_baked guard)", threw);
        HIP_CHECK(hipFree(g.d_weight));
    }

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
