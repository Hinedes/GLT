// ============================================================================
// Shadow Bake Subsystem
// Adds graft deltas directly into the loaded base model weights (bf16),
// creating a disposable Shadow Copy. Only legal in inference mode.
// ============================================================================

// Guard: prevents double application of graft to shadow
// g_shadow_baked is declared in shadow_bake.hpp

// ---------------------------------------------------------------------------
// Bake kernel — elementwise bf16 base += f32 graft delta
// mode=0: expand (gate/up): base[i_start+s, h_start+h] += graft[s, h]
// mode=1: contract (down):  base[h_start+h, i_start+s] += graft[s, h]
// ---------------------------------------------------------------------------
__global__ void shadow_bake_kernel(
    uint16_t* base_ptr,
    int base_stride_0,      // leading dimension of base tensor (row stride)
    const float* graft_ptr,
    int slice_dim,
    int h_slice_dim,
    int i_start,            // intermediate slice start (expand: rows, contract: cols)
    int h_start,            // hidden slice start (expand: cols, contract: rows)
    int mode                // 0=expand, 1=contract
) {
    int s = blockIdx.x;     // index into slice_dim

    if (s >= slice_dim) return;

    // PATCH: stride the h dimension across the block so all h_slice_dim
    // columns are baked. Prior version used `int h = threadIdx.x` with an
    // early return, which skipped h >= blockDim.x (e.g. only 256 of 512
    // hidden-slice columns for the default 4-domain config) — half the graft.
    for (int h = threadIdx.x; h < h_slice_dim; h += blockDim.x) {
        float delta = graft_ptr[s * h_slice_dim + h];

        int base_row, base_col;
        if (mode == 0) {
            // expand: base[i_start + s, h_start + h]
            base_row = i_start + s;
            base_col = h_start + h;
        } else {
            // contract: base[h_start + h, i_start + s]  (transpose)
            base_row = h_start + h;
            base_col = i_start + s;
        }

        int base_idx = base_row * base_stride_0 + base_col;
        float base_val = bf16_to_f32(base_ptr[base_idx]);
        base_ptr[base_idx] = f32_to_bf16(base_val + delta);
    }
}

// ---------------------------------------------------------------------------
// apply_graft_to_shadow — one-time startup assembly
// Reads the loaded bf16 model weights and adds graft delta weights in-place.
// After this call and a sync, the model weights ARE the shadow.
// ---------------------------------------------------------------------------
void apply_graft_to_shadow(ModelWeights& model, const GraftWeights& graft) {
    if (g_shadow_baked) {
        throw std::runtime_error("apply_graft_to_shadow called twice on the same model instance");
    }
    g_shadow_baked = true;

    if (!graft.allocated || graft.d_weight == nullptr) {
        std::cout << "[ShadowBake] Empty graft, no bake needed.\n";
        return;
    }

    std::cout << "[ShadowBake] Applying graft domain_id=" << graft.domain_id
              << " to shadow copy.\n";

    int num_layers = g_cfg.num_layers;
    int slice_dim = g_cfg.slice_dim;
    int h_slice_dim = g_cfg.h_slice_dim;
    int i_start = graft.domain_id * slice_dim;
    int h_start = graft.domain_id * h_slice_dim;

    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        if (!should_graft_layer(layer_idx)) continue;

        auto& w = model.layers[layer_idx];

        // Gate expand (proj_idx=0)
        if (g_cfg.graft_proj_gate) {
            auto& base = w.gate_proj_weight;
            int offset = delta_offset(layer_idx, 0);
            int base_stride = base.stride(0);
            shadow_bake_kernel<<<slice_dim, g_cfg.kernel_threads, 0, current_hip_stream()>>>(
                reinterpret_cast<uint16_t*>(base.data_ptr()),
                base_stride,
                graft.d_weight + offset,
                slice_dim, h_slice_dim, i_start, h_start, 0
            );
            HIP_CHECK(hipGetLastError());
        }

        // Up expand (proj_idx=1)
        if (g_cfg.graft_proj_up) {
            auto& base = w.up_proj_weight;
            int offset = delta_offset(layer_idx, 1);
            int base_stride = base.stride(0);
            shadow_bake_kernel<<<slice_dim, g_cfg.kernel_threads, 0, current_hip_stream()>>>(
                reinterpret_cast<uint16_t*>(base.data_ptr()),
                base_stride,
                graft.d_weight + offset,
                slice_dim, h_slice_dim, i_start, h_start, 0
            );
            HIP_CHECK(hipGetLastError());
        }

        // Down contract (proj_idx=2) — transpose layout
        if (g_cfg.graft_proj_down) {
            auto& base = w.down_proj_weight;
            int offset = delta_offset(layer_idx, 2);
            int base_stride = base.stride(0);
            shadow_bake_kernel<<<slice_dim, g_cfg.kernel_threads, 0, current_hip_stream()>>>(
                reinterpret_cast<uint16_t*>(base.data_ptr()),
                base_stride,
                graft.d_weight + offset,
                slice_dim, h_slice_dim, i_start, h_start, 1
            );
            HIP_CHECK(hipGetLastError());
        }
    }

    HIP_CHECK(hipStreamSynchronize(current_hip_stream()));
    std::cout << "[ShadowBake] Bake complete. Graft deltas added to shadow model.\n";
}
