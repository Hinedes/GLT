// ============================================================================
// RoPE: Rotary Position Embeddings (ATen-based)
// Operation: Computes cos/sin tables from position indices and theta,
// then applies rotary transformation to Q and K tensors in-place.
// Currently ATen-based; can be replaced with a fused HIP kernel if profiling
// identifies this as a bottleneck (unlikely — runs once per layer, frozen).
// ============================================================================
void apply_rope(at::Tensor& q, at::Tensor& k, int seq_len) {
    auto dtype = q.dtype();
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());

    auto i = torch::arange(0, g_cfg.head_dim, 2, opts_f32);
    auto inv_freq = 1.0f / torch::pow(g_cfg.rope_theta, i / g_cfg.head_dim);
    auto positions = torch::arange(seq_len, opts_f32);
    auto angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0);  // (S, 64)
    // SmolLM3 doubles angles via cat(freqs, freqs) for half-split pairing
    auto emb = torch::cat({angles, angles}, -1);                    // (S, 128)
    auto cos_vals = emb.cos().to(dtype);                            // (S, 128)
    auto sin_vals = emb.sin().to(dtype);                            // (S, 128)

    auto rotary = [&](at::Tensor& t, int num_heads) {
        auto shape = t.sizes();
        auto view = t.view({shape[0], shape[1], num_heads, g_cfg.head_dim});  // {B, S, nh, HD}
        int half = g_cfg.head_dim / 2;
        auto first_half = view.slice(-1, 0, half);
        auto second_half = view.slice(-1, half, g_cfg.head_dim);

        auto cs = cos_vals.unsqueeze(0).unsqueeze(2);  // {1, S, 1, 128}
        auto sn = sin_vals.unsqueeze(0).unsqueeze(2);

        auto r1 = first_half * cs.slice(-1, 0, half) - second_half * sn.slice(-1, 0, half);
        auto r2 = second_half * cs.slice(-1, half, g_cfg.head_dim) + first_half * sn.slice(-1, half, g_cfg.head_dim);
        auto stacked = torch::cat({r1, r2}, -1);
        t.copy_(stacked.view(shape));
    };

    rotary(q, g_cfg.num_heads);
    rotary(k, g_cfg.num_kv_heads);
}

// Autograd-friendly RoPE helper for backward recomputation.
static inline at::Tensor apply_rope_out_of_place(const at::Tensor& t, int num_heads, int seq_len) {
    auto dtype = t.dtype();
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(t.device());
    auto i = torch::arange(0, g_cfg.head_dim, 2, opts_f32);
    auto inv_freq = 1.0f / torch::pow(g_cfg.rope_theta, i / g_cfg.head_dim);
    auto positions = torch::arange(seq_len, opts_f32);
    auto angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0);
    auto emb = torch::cat({angles, angles}, -1);
    auto cos_vals = emb.cos().to(dtype);
    auto sin_vals = emb.sin().to(dtype);

    auto shape = t.sizes();
    auto view = t.view({shape[0], shape[1], num_heads, g_cfg.head_dim});
    int half = g_cfg.head_dim / 2;
    auto first_half = view.slice(-1, 0, half);
    auto second_half = view.slice(-1, half, g_cfg.head_dim);
    auto cs = cos_vals.unsqueeze(0).unsqueeze(2);
    auto sn = sin_vals.unsqueeze(0).unsqueeze(2);
    auto r1 = first_half * cs.slice(-1, 0, half) - second_half * sn.slice(-1, 0, half);
    auto r2 = second_half * cs.slice(-1, half, g_cfg.head_dim) + first_half * sn.slice(-1, half, g_cfg.head_dim);
    auto stacked = torch::cat({r1, r2}, -1);
    return stacked.view(shape);
}
