// ============================================================================
// Fused AdamW Optimizer Kernel
// Operation: Single 2D grid launch processes all g_cfg.total_projections (108)
// delta weight matrices in parallel. grid.x = projection index, grid.y = 
// blocks within projection. Each thread: m = b1*m + (1-b1)*g, v = b2*v +
// (1-b2)*g^2, then weight -= lr * (m_hat/(sqrt(v_hat)+eps) + wd * weight).
// Deferred to adamw_stream to overlap optimizer compute with next step's
// backward pass (pipeline parallelism between the current stream and adamw_stream).
// CDNA3: 256 threads (4 warps) for full SIMD utilization.
// ============================================================================
__global__ __launch_bounds__(256, 4) void adamw_fused_kernel(float* d_weights, uint16_t* d_weights_bf16,
                                    float* d_grads, float* d_ms, float* d_vs,
                                    float lr, float beta1, float beta2, float beta1_corr, float beta2_corr,
                                    float eps, float weight_decay,
                                    int proj_size, int total_projs,
                                    int gate_enabled, int up_enabled, int down_enabled) {
    int proj = blockIdx.x;
    if (proj >= total_projs) return;

    // Skip masked projections to prevent weight-decay drift on zero channels
    int proj_in_layer = proj % 3;
    bool skip = (proj_in_layer == 0 && !gate_enabled) ||
                (proj_in_layer == 1 && !up_enabled) ||
                (proj_in_layer == 2 && !down_enabled);
    if (skip) return;

    int offset = proj * proj_size;

    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (idx >= proj_size) return;
    int gidx = offset + idx;

    float g = d_grads[gidx];
    float m = d_ms[gidx];
    float v = d_vs[gidx];

    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;

    d_ms[gidx] = m;
    d_vs[gidx] = v;

    float m_hat = m * beta1_corr;
    float v_hat = v * beta2_corr;

    float denom = sqrtf(v_hat) + eps;
    float new_w = d_weights[gidx] - lr * m_hat / denom - lr * weight_decay * d_weights[gidx];
    d_weights[gidx] = new_w;
    if (d_weights_bf16 != nullptr) {
        d_weights_bf16[gidx] = f32_to_bf16(new_w);
    }
}

// --------------------------------------------------------------------------
// execute_adamw_all — Launch fused AdamW across all 108 projections
// Operation: Single 2D grid launch on adamw_stream. Waits for backward_event
// (signaled by the current stream) before executing, then signals adamw_event. The
// next step's forward is blocked until AdamW completes — no stale D_1.
// --------------------------------------------------------------------------
void execute_adamw_all(VramArena& arena, GraftWeights& graft, int step, float lr, hipStream_t stream) {
    int proj_size = g_cfg.slice_dim * g_cfg.h_slice_dim;
    int blocks = (proj_size + g_cfg.kernel_threads - 1) / g_cfg.kernel_threads;

    float beta1_corr = 1.0f / (1.0f - powf(g_cfg.adam_beta1, step + 1));
    float beta2_corr = 1.0f / (1.0f - powf(g_cfg.adam_beta2, step + 1));
    dim3 grid(g_cfg.total_projections, blocks);
    adamw_fused_kernel<<<grid, g_cfg.kernel_threads, 0, stream>>>(
        graft.d_weight, graft.d_weight_bf16,
        arena.d_delta_grad,
        arena.d_adam_m, arena.d_adam_v,
        lr, g_cfg.adam_beta1, g_cfg.adam_beta2, beta1_corr, beta2_corr,
        g_cfg.adam_eps, g_cfg.adam_weight_decay,
        proj_size, g_cfg.total_projections,
        g_cfg.graft_proj_gate, g_cfg.graft_proj_up, g_cfg.graft_proj_down
    );
    HIP_CHECK(hipGetLastError());
}
