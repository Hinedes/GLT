// ============================================================================
// Phase 1a: Forward Pass — Store all layer activations for backward
// Operation: For each transformer layer, runs base attention (frozen ATen),
// then the grafted FFN: gate_expand_delta + up_expand_delta are injected
// into gate_proj/up_proj outputs; down_contract_delta into down_proj.
// All graft operations use raw rocBLAS GEMM (bf16×bf16→f32) + custom HIP
// kernels with zero ATen overhead. Energy norms computed per-token for
// TPHS-compatible silence loss. Activations stored for Phase 2 backward.
// ============================================================================
// Diagnostic counter: number of times a graft GEMM was actually dispatched
// (i.e. forward_layer entered with graft.active == true). Used by eval-only
// to prove zero graft operations in base-only mode.
int g_graft_fwd_calls = 0;

void forward_layer(int layer_idx, at::Tensor& x, VramArena& arena, const GraftWeights& graft, int batch_size, int seq_len, bool use_explicit_graft) {
    const auto& w = g_model_weights.layers[layer_idx];
    
    LayerActivations acts;
    
    int M = batch_size * seq_len;
    int h_start = graft.domain_id * g_cfg.h_slice_dim;
    int i_start = graft.domain_id * g_cfg.slice_dim;
    const bool dump_chain = !g_cfg.parity_dump_dir.empty() && layer_idx == g_cfg.parity_dump_layer;

    // 1. Base Model: Attention (ATen — frozen, infrequent)
    at::Tensor x_ffn_bf16;
    {
        torch::NoGradGuard guard;
        at::Tensor x_norm, q, k, v, attn_out;
        acts.x_pre_attn = x.clone();
        auto rms_result = rms_norm(x, w.input_layernorm_weight);
        x_norm = rms_result.first;
        q = at::linear(x_norm, w.q_proj_weight);
        k = at::linear(x_norm, w.k_proj_weight);
        v = at::linear(x_norm, w.v_proj_weight);
        
        if (g_cfg.no_rope_layers.empty() || g_cfg.no_rope_layers[layer_idx]) {
            apply_rope(q, k, seq_len);
        }

        int gqa_repeats = g_cfg.num_heads / g_cfg.num_kv_heads;
        auto split = [&](const at::Tensor& t, int nh) {
            return t.view({batch_size, seq_len, nh, g_cfg.head_dim}).transpose(1, 2);
        };
        auto q_mh = split(q, g_cfg.num_heads);
        auto k_mh = split(k, g_cfg.num_kv_heads).repeat_interleave(gqa_repeats, 1);
        auto v_mh = split(v, g_cfg.num_kv_heads).repeat_interleave(gqa_repeats, 1);
        auto attn_mh = at::scaled_dot_product_attention(q_mh, k_mh, v_mh, {}, 0.0, true);
        attn_out = attn_mh.transpose(1, 2).reshape({batch_size, seq_len, g_cfg.num_heads * g_cfg.head_dim});
        attn_out = at::linear(attn_out, w.o_proj_weight);
        x = x + attn_out;
        if (g_verbose && layer_idx == 0) std::cerr << "[diag] L0 after attn: x=" << x.sizes() << " attn_out=" << attn_out.sizes() << std::endl;
        acts.x_post_attn = x.clone();
        
        auto rms_result2 = rms_norm(x, w.post_attention_layernorm_weight);
        x_ffn_bf16 = rms_result2.first;
        acts.rms_norm_variance = rms_result2.second.clone();
        acts.x_ffn_input = x_ffn_bf16.clone();
    }
    
    // 2. CAPTURE ffn_input to arena (single memcpy, bf16)
    auto ffn_input_bf16 = x_ffn_bf16.contiguous();
    HIP_CHECK(hipMemcpyAsync(arena.d_ffn_input, ffn_input_bf16.data_ptr(),
                             M * g_cfg.hidden_dim * sizeof(uint16_t), hipMemcpyDeviceToDevice,
                             current_hip_stream()));

    // 3. Base gate_proj + GATE EXPAND DELTA
    // gate_pre = x @ gate_weight^T + delta_gate(x_h_slice)
    at::Tensor gate_pre;
    {
        torch::NoGradGuard guard;
        gate_pre = at::linear(x_ffn_bf16, w.gate_proj_weight);
    }
    if (g_verbose && layer_idx == 0) std::cerr << "[diag] L0 gate_pre: " << gate_pre.sizes() << " " << gate_pre.dtype() << std::endl;
    if (graft.active && use_explicit_graft) ++g_graft_fwd_calls;
    if (graft.active && use_explicit_graft && should_graft_proj(layer_idx, 0)) {
        execute_expand_forward(arena, graft, layer_idx, 0, M, arena.d_ffn_input + h_start, arena.d_delta_out);
        {
            acts.delta_gate_out = torch::empty({M, g_cfg.slice_dim},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            acts.energy_gate = torch::empty({M},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            auto gp = gate_pre.slice(2, i_start, i_start + g_cfg.slice_dim);
            at::Tensor gp_before;
            if (layer_idx == 0) {
                gp_before = gp.clone();
            }
            int blocks = M;
            fused_inject_clone_energy_kernel<<<blocks, g_cfg.kernel_threads, 0, current_hip_stream()>>>(
                arena.d_delta_out,
                reinterpret_cast<uint16_t*>(gp.data_ptr()),
                acts.delta_gate_out.data_ptr<float>(),
                acts.energy_gate.data_ptr<float>(),
                M, g_cfg.slice_dim, gp.stride(1), 1e-8f
            );
            HIP_CHECK(hipGetLastError());
            if (layer_idx == 0) {
                float gp_diff = (gp - gp_before).abs().max().item<float>();
                std::cerr << "[diag] L0 gate_pre max|delta|=" << gp_diff << std::endl;
            }
        }
    } else {
        acts.delta_gate_out = torch::zeros({M, g_cfg.slice_dim},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        acts.energy_gate = torch::zeros({M},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }
    if (dump_chain) {
        dump_f32_tensor_raw(acts.delta_gate_out, parity_dump_path("delta_gate_out_l" + std::to_string(layer_idx) + ".raw"));
        auto gate_pre_slice = gate_pre.slice(2, i_start, i_start + g_cfg.slice_dim);
        auto silu_prime = gate_pre_slice.to(torch::kFloat32).sigmoid() *
            (1.0f + gate_pre_slice.to(torch::kFloat32) * (1.0f - gate_pre_slice.to(torch::kFloat32).sigmoid()));
        dump_f32_tensor_raw(gate_pre_slice, parity_dump_path("gate_pre_i_slice_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(silu_prime, parity_dump_path("silu_prime_gate_l" + std::to_string(layer_idx) + ".raw"));
    }

    // 4. Base up_proj + UP EXPAND DELTA  // PATCH P3-23: renumbered (was step 5, skipping 4)
    // up = x @ up_weight^T + delta_up(x_h_slice)
    at::Tensor up;
    {
        torch::NoGradGuard guard;
        up = at::linear(x_ffn_bf16, w.up_proj_weight);
    }
    if (graft.active && use_explicit_graft && should_graft_proj(layer_idx, 1)) {
        execute_expand_forward(arena, graft, layer_idx, 1, M, arena.d_ffn_input + h_start, arena.d_delta_out);
        {
            acts.delta_up_out = torch::empty({M, g_cfg.slice_dim},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            acts.energy_up = torch::empty({M},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            auto up_view = up.slice(2, i_start, i_start + g_cfg.slice_dim);
            int blocks = M;
            fused_inject_clone_energy_kernel<<<blocks, g_cfg.kernel_threads, 0, current_hip_stream()>>>(
                arena.d_delta_out,
                reinterpret_cast<uint16_t*>(up_view.data_ptr()),
                acts.delta_up_out.data_ptr<float>(),
                acts.energy_up.data_ptr<float>(),
                M, g_cfg.slice_dim, up_view.stride(1), 1e-8f
            );
            HIP_CHECK(hipGetLastError());
        }
    } else {
        acts.delta_up_out = torch::zeros({M, g_cfg.slice_dim},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        acts.energy_up = torch::zeros({M},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }
    if (dump_chain) {
        dump_f32_tensor_raw(acts.delta_up_out, parity_dump_path("delta_up_out_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(up.slice(2, i_start, i_start + g_cfg.slice_dim), parity_dump_path("up_i_slice_l" + std::to_string(layer_idx) + ".raw"));
    }

    // 5. Fused intermediate product: silu(gate_pre) * up.
    // Write bf16 output directly into the arena to avoid an ATen elementwise
    // launch and a separate intermediate copy.
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
    auto intermediate = torch::from_blob(
        arena.d_intermediate,
        {batch_size, seq_len, g_cfg.intermediate_dim},
        torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));

    // 6. Base down_proj  // PATCH P3-23: was 8
    at::Tensor ffn_out;
    {
        torch::NoGradGuard guard;
        ffn_out = at::linear(intermediate, w.down_proj_weight);
    }
    if (g_verbose && layer_idx == 0) std::cerr << "[diag] L0 ffn_out: " << ffn_out.sizes() << " intermediate: " << intermediate.sizes() << std::endl;

    // 7. CONTRACT DELTA: ffn_out += delta_contract(x_i_slice)  // PATCH P3-23: was 9
    if (graft.active && use_explicit_graft && should_graft_proj(layer_idx, 2)) {
        execute_contract_forward(arena, graft, layer_idx, M, arena.d_intermediate + i_start, arena.d_delta_out_contract);
        {
            acts.delta_down_out = torch::empty({M, g_cfg.h_slice_dim},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            acts.energy_down = torch::empty({M},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            auto fo = ffn_out.slice(2, h_start, h_start + g_cfg.h_slice_dim);
            at::Tensor fo_before;
            if (dump_chain) {
                fo_before = fo.clone();
            }
            if (dump_chain) {
                auto intermediate_i_slice = intermediate.slice(2, i_start, i_start + g_cfg.slice_dim).to(torch::kFloat32).contiguous();
                auto contract_input_slice = intermediate_i_slice.to(torch::kBFloat16).to(torch::kFloat32).contiguous();
                auto down_storage_view = at::from_blob(
                    graft.d_weight + delta_offset(layer_idx, 2),
                    {g_cfg.slice_dim, g_cfg.h_slice_dim},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                auto down_torch_view = down_storage_view.transpose(0, 1);
                HIP_CHECK(hipStreamSynchronize(current_hip_stream()));
                dump_f32_tensor_raw(contract_input_slice, parity_dump_path("contract_input_bf16_l" + std::to_string(layer_idx) + ".raw"));
                dump_f32_tensor_raw(intermediate_i_slice, parity_dump_path("intermediate_i_slice_l" + std::to_string(layer_idx) + ".raw"));
                dump_f32_tensor_raw(down_storage_view, parity_dump_path("down_storage_view_l" + std::to_string(layer_idx) + ".raw"));
                dump_f32_tensor_raw(down_torch_view, parity_dump_path("down_torch_view_l" + std::to_string(layer_idx) + ".raw"));
            }
            int blocks = M;
            fused_inject_clone_energy_kernel<<<blocks, g_cfg.kernel_threads, 0, current_hip_stream()>>>(
                arena.d_delta_out_contract,
                reinterpret_cast<uint16_t*>(fo.data_ptr()),
                acts.delta_down_out.data_ptr<float>(),
                acts.energy_down.data_ptr<float>(),
                M, g_cfg.h_slice_dim, fo.stride(1), 1e-8f
            );
            HIP_CHECK(hipGetLastError());
            if (dump_chain) {
                auto fo_after = fo.clone();
                float fo_diff = at::norm(fo_after - fo_before).item<float>();
                dump_f32_tensor_raw(acts.delta_down_out, parity_dump_path("delta_down_out_l" + std::to_string(layer_idx) + ".raw"));
                dump_f32_tensor_raw(at::from_blob(const_cast<int32_t*>(arena.d_ood_mask), {batch_size, seq_len},
                                    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)),
                                    parity_dump_path("ood_mask.raw"));
                dump_f32_buffer_raw(arena.d_delta_out_contract, static_cast<size_t>(M) * g_cfg.h_slice_dim,
                                    parity_dump_path("delta_out_contract_l35.raw"));
                dump_f32_tensor_raw(fo_before, parity_dump_path("ffn_out_hslice_before_l35.raw"));
                dump_f32_tensor_raw(fo_after, parity_dump_path("ffn_out_hslice_after_l35.raw"));
                std::ofstream(parity_dump_path("ffn_out_diff_norm_l35.txt")) << fo_diff << "\n";
                std::cerr << "[diag] L35 ffn_out diff_norm=" << fo_diff << std::endl;
                if (g_cfg.contract_probe) {
                    auto probe_weight = torch::zeros({g_cfg.slice_dim, g_cfg.h_slice_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                    probe_weight[0][0] = 1.0f;
                    auto probe_out = torch::empty({M, g_cfg.h_slice_dim},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                    graft_gemm_fwd(arena, GraftGemmKind::ContractForward, M, g_cfg.slice_dim, g_cfg.h_slice_dim,
                                   probe_weight.data_ptr<float>(),
                                   arena.d_intermediate + i_start,
                                   g_cfg.intermediate_dim,
                                   probe_out.data_ptr<float>());
                    dump_f32_tensor_raw(probe_out, parity_dump_path("contract_probe_out_l" + std::to_string(layer_idx) + ".raw"));
                }
            }
        }
    } else {
        acts.delta_down_out = torch::zeros({M, g_cfg.h_slice_dim},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        acts.energy_down = torch::zeros({M},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        if (dump_chain) {
            std::vector<float> zeros(static_cast<size_t>(M) * g_cfg.h_slice_dim, 0.0f);
            std::ofstream wf(parity_dump_path("delta_out_contract_l35.raw"), std::ios::binary);
            wf.write(reinterpret_cast<const char*>(zeros.data()),
                     static_cast<std::streamsize>(zeros.size() * sizeof(float)));
            std::vector<float> down_zeros(static_cast<size_t>(M) * g_cfg.h_slice_dim, 0.0f);
            std::ofstream ddf(parity_dump_path("delta_down_out_l" + std::to_string(layer_idx) + ".raw"), std::ios::binary);
            ddf.write(reinterpret_cast<const char*>(down_zeros.data()),
                      static_cast<std::streamsize>(down_zeros.size() * sizeof(float)));
            dump_f32_tensor_raw(at::from_blob(const_cast<int32_t*>(arena.d_ood_mask), {batch_size, seq_len},
                                torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)),
                                parity_dump_path("ood_mask.raw"));
            auto fo = ffn_out.slice(2, h_start, h_start + g_cfg.h_slice_dim);
            dump_f32_tensor_raw(fo, parity_dump_path("ffn_out_hslice_before_l35.raw"));
            dump_f32_tensor_raw(fo, parity_dump_path("ffn_out_hslice_after_l35.raw"));
            std::ofstream(parity_dump_path("ffn_out_diff_norm_l35.txt")) << 0.0f << "\n";
        }
    }
    // 9. Residual  // PATCH P3-23: was 10
    {
        torch::NoGradGuard guard;
        at::Tensor x_before_residual = x.clone();
        x = x + ffn_out;
        if (dump_chain) {
            auto x_after_residual = x.clone();
            float x_diff = at::norm(x_after_residual - x_before_residual).item<float>();
            dump_f32_tensor_raw(x_before_residual, parity_dump_path("x_before_residual_l35.raw"));
            dump_f32_tensor_raw(x_after_residual, parity_dump_path("x_after_residual_l35.raw"));
            std::ofstream(parity_dump_path("x_after_residual_diff_norm_l35.txt")) << x_diff << "\n";
            std::cerr << "[diag] L35 residual diff_norm=" << x_diff << std::endl;
        }
        if (g_verbose && layer_idx == 0) std::cerr << "[diag] L0 after residual: x=" << x.sizes() << std::endl;
    }
    
    if ((int)g_layer_activations.size() <= layer_idx) {
        g_layer_activations.resize(layer_idx + 1);
    }
    g_layer_activations[layer_idx] = std::move(acts);
}
