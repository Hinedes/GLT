at::Tensor attention_bwd_autograd(
    const LayerWeights& w,
    const at::Tensor& x_pre_attn,
    int batch_size,
    int seq_len,
    int layer_idx,
    const at::Tensor& grad_x_post_attn_total)
{
    torch::AutoGradMode grad_guard(true);
    auto x = x_pre_attn.detach().clone();
    x.requires_grad_(true);

    auto rms_result = rms_norm(x, w.input_layernorm_weight);
    auto x_norm = rms_result.first;
    auto q = at::linear(x_norm, w.q_proj_weight);
    auto k = at::linear(x_norm, w.k_proj_weight);
    auto v = at::linear(x_norm, w.v_proj_weight);

    if (g_cfg.no_rope_layers.empty() || g_cfg.no_rope_layers[layer_idx]) {
        q = apply_rope_out_of_place(q, g_cfg.num_heads, seq_len);
        k = apply_rope_out_of_place(k, g_cfg.num_kv_heads, seq_len);
    }

    int gqa_repeats = g_cfg.num_heads / g_cfg.num_kv_heads;
    auto split = [&](const at::Tensor& t, int nh) {
        return t.view({batch_size, seq_len, nh, g_cfg.head_dim}).transpose(1, 2);
    };
    auto q_mh = split(q, g_cfg.num_heads);
    auto k_mh = split(k, g_cfg.num_kv_heads).repeat_interleave(gqa_repeats, 1);
    auto v_mh = split(v, g_cfg.num_kv_heads).repeat_interleave(gqa_repeats, 1);
    auto attn_mh = at::scaled_dot_product_attention(q_mh, k_mh, v_mh, {}, 0.0, true);
    auto attn_out = attn_mh.transpose(1, 2).reshape({batch_size, seq_len, g_cfg.num_heads * g_cfg.head_dim});
    attn_out = at::linear(attn_out, w.o_proj_weight);
    auto x_post_attn = x + attn_out;

    auto grad_out = grad_x_post_attn_total.to(x_post_attn.dtype()).contiguous();
    x_post_attn.backward(grad_out);
    return x.grad().detach().to(torch::kFloat32).contiguous();
}
