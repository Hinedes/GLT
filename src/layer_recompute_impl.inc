// Recompute FFN from Stored Activations (Checkpoint Recompute)
// Operation: Re-runs the FFN forward path using the saved x_ffn_input from
// Phase 1. Produces gate_pre, up, and intermediate as local tensors for use
// in backward computation. gate = silu(gate_pre) is no longer materialized
// (fused_silu_product_kernel computes it in registers). Avoids storing all
// intermediates from all 36 layers simultaneously, saving ~600 MB VRAM.
// ============================================================================
void recompute_ffn(int layer_idx, const at::Tensor& x_ffn_input,
                    VramArena& arena, int batch_size, int seq_len, int domain_id,
                    at::Tensor& gate_pre_out,
                    at::Tensor& up_out, at::Tensor& intermediate_out) {
    const auto& w = g_model_weights.layers[layer_idx];
    const auto& acts = g_layer_activations[layer_idx];
    int M = batch_size * seq_len;
    int h_start = domain_id * g_cfg.h_slice_dim;
    int i_start = domain_id * g_cfg.slice_dim;

    at::Tensor gate_pre;
    {
        torch::NoGradGuard guard;
        gate_pre = at::linear(x_ffn_input, w.gate_proj_weight);
    }
    if (should_graft_proj(layer_idx, 0)) {
        auto gate_pre_flat = gate_pre.view({M, g_cfg.intermediate_dim});
        auto gate_slice = gate_pre_flat.slice(1, i_start, i_start + g_cfg.slice_dim);
        auto delta_gate = acts.delta_gate_out.view({M, g_cfg.slice_dim});
        TORCH_CHECK(gate_slice.size(0) == delta_gate.size(0),
                    "gate recompute row mismatch: slice=", gate_slice.size(0),
                    " delta=", delta_gate.size(0));
        TORCH_CHECK(gate_slice.size(1) == g_cfg.slice_dim,
                    "gate recompute slice width mismatch: ", gate_slice.size(1),
                    " vs g_cfg.slice_dim=", g_cfg.slice_dim);
        TORCH_CHECK(delta_gate.size(1) == g_cfg.slice_dim,
                    "gate cached delta width mismatch: ", delta_gate.size(1),
                    " vs g_cfg.slice_dim=", g_cfg.slice_dim);
        if (delta_gate.dtype() != gate_slice.dtype()) {
            delta_gate = delta_gate.to(gate_slice.dtype());
        }
        gate_slice.add_(delta_gate);
    }

    at::Tensor up;
    {
        torch::NoGradGuard guard;
        up = at::linear(x_ffn_input, w.up_proj_weight);
    }
    if (should_graft_proj(layer_idx, 1)) {
        auto up_flat = up.view({M, g_cfg.intermediate_dim});
        auto up_slice = up_flat.slice(1, i_start, i_start + g_cfg.slice_dim);
        auto delta_up = acts.delta_up_out.view({M, g_cfg.slice_dim});
        TORCH_CHECK(up_slice.size(0) == delta_up.size(0),
                    "up recompute row mismatch: slice=", up_slice.size(0),
                    " delta=", delta_up.size(0));
        TORCH_CHECK(up_slice.size(1) == g_cfg.slice_dim,
                    "up recompute slice width mismatch: ", up_slice.size(1),
                    " vs g_cfg.slice_dim=", g_cfg.slice_dim);
        TORCH_CHECK(delta_up.size(1) == g_cfg.slice_dim,
                    "up cached delta width mismatch: ", delta_up.size(1),
                    " vs g_cfg.slice_dim=", g_cfg.slice_dim);
        if (delta_up.dtype() != up_slice.dtype()) {
            delta_up = delta_up.to(up_slice.dtype());
        }
        up_slice.add_(delta_up);
    }

    // Fused silu(gate_pre) * up -> intermediate.
    // Keep the output in the arena and expose it as a bf16 tensor view.
    {
        int n = M * g_cfg.intermediate_dim;
        int blocks = (n + g_cfg.kernel_threads * 8 - 1) / (g_cfg.kernel_threads * 8);
        fused_silu_product_strided_kernel<<<blocks, g_cfg.kernel_threads, 0, current_hip_stream()>>>(
            reinterpret_cast<const uint16_t*>(gate_pre.data_ptr()),
            reinterpret_cast<const uint16_t*>(up.data_ptr()),
            arena.d_intermediate,
            M,
            g_cfg.intermediate_dim,
            g_cfg.intermediate_dim,
            g_cfg.intermediate_dim,
            g_cfg.intermediate_dim
        );
        HIP_CHECK(hipGetLastError());
    }
    at::Tensor intermediate = torch::from_blob(
        arena.d_intermediate,
        {batch_size, seq_len, g_cfg.intermediate_dim},
        torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));

    gate_pre_out = std::move(gate_pre);
    up_out = std::move(up);
    intermediate_out = std::move(intermediate);
}
