// ============================================================================
// Phase 2b: Backward Pass ” Per-Layer Gradient Computation
// Operation: Checkpoint-recomputes FFN from stored activations, then
// backpropagates through all paths in topological order:
//   1. base down_proj -> 2. contract delta -> 3. fused gate*up element-wise ->
//   4. expand deltas (gate+up with fused silence blend + energy) ->
//   5. base gate_proj/up_proj -> 6. post-attention RMSNorm.
// Gradient clipping and AdamW are deferred to pipelined post-processing.
// ============================================================================
void backward_layer(int layer_idx, at::Tensor& dx, VramArena& arena, int batch_size, int seq_len, int step, float lr, int domain_id, const GraftWeights& graft, const int32_t* ood_mask, float lambda_effective) {
    if (g_shadow_baked) {
        throw std::runtime_error("backward_layer called on shadow-baked model — forbidden");
    }
    const auto& w = g_model_weights.layers[layer_idx];
    const auto& acts = g_layer_activations[layer_idx];
    
    int M = batch_size * seq_len;
    int h_start = domain_id * g_cfg.h_slice_dim;
    int i_start = domain_id * g_cfg.slice_dim;
    const bool dump_gate_chain = !g_cfg.parity_dump_dir.empty() && layer_idx == g_cfg.parity_dump_layer;
    const bool dump_boundary_cur = !g_cfg.parity_dump_dir.empty() && layer_idx == g_cfg.parity_dump_layer;
    const bool dump_boundary_prev = !g_cfg.parity_dump_dir.empty() && (layer_idx + 1 == g_cfg.parity_dump_layer);

    if (dump_gate_chain) {
        dump_f32_tensor_raw(dx, parity_dump_path("grad_ffn_out_l" + std::to_string(layer_idx) + ".raw"));
    }
    if (dump_boundary_cur) {
        dump_f32_tensor_raw(dx, parity_dump_path("grad_x_out_l" + std::to_string(layer_idx) + ".raw"));
    }
    if (dump_boundary_prev) {
        dump_f32_tensor_raw(dx, parity_dump_path("grad_ffn_out_l" + std::to_string(layer_idx) + ".raw"));
    }

    // Checkpoint recompute: FFN intermediate tensors from stored x_ffn_input
    // gate (silu of gate_pre) is no longer materialized ” fused_gate_up_bwd
    // and fused_silu_product compute it in registers from gate_pre.
    at::Tensor gate_pre, up, intermediate;
    recompute_ffn(layer_idx, acts.x_ffn_input, arena, batch_size, seq_len, domain_id,
                  gate_pre, up, intermediate);

    // ========================================================================
    // Step 1: Backprop through base down_proj
    // grad_intermediate = dx @ down_proj_weight^T  -> [M, INTERMEDIATE]
    // ========================================================================
    at::Tensor grad_intermediate;
    {
        torch::NoGradGuard guard;
        grad_intermediate = at::matmul(dx.to(torch::kFloat32), w.down_proj_weight_f32);  // PATCH P2-14: cached f32
    }
    if (dump_gate_chain) {
        dump_f32_tensor_raw(grad_intermediate,
                            parity_dump_path("grad_intermediate_from_base_down_l" + std::to_string(layer_idx) + ".raw"));
    }

    // ========================================================================
    // Step 2: Contract delta backward (down_proj delta)
    // Forward math view:
    //   down_torch_view = [g_cfg.h_slice_dim, g_cfg.slice_dim]
    //   ffn_out[:, h_slice] += down_torch_view @ intermediate[:, i_slice]
    // Storage/canonical view:
    //   down_storage_view = [g_cfg.slice_dim, g_cfg.h_slice_dim]
    //   d_delta_grad and graft.d_weight must both stay in this flat orientation.
    // Gradient:
    //   grad_down_storage = intermediate_i_slice^T @ dx_h_slice -> [S, H]
    //   grad_intermediate_i_slice += dx_h_slice @ down_storage_view^T
    // ========================================================================
    at::Tensor dx_h_slice = dx.slice(2, h_start, h_start + g_cfg.h_slice_dim).to(torch::kFloat32).contiguous();
    at::Tensor intermediate_i_slice = intermediate.slice(2, i_start, i_start + g_cfg.slice_dim).to(torch::kFloat32).contiguous();

    if (dump_gate_chain) {
        dump_f32_tensor_raw(dx_h_slice, parity_dump_path("dx_h_slice_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(intermediate_i_slice, parity_dump_path("intermediate_i_slice_l" + std::to_string(layer_idx) + ".raw"));
    }

    if (should_graft_proj(layer_idx, 2)) {
        // Silence blend: add lambda * y_delta / (||y_delta|| * sqrt(V)) to dx_h_slice for OOD tokens
        {
            int blocks = M;
            silence_blend_kernel<<<blocks, g_cfg.kernel_threads, 0, current_hip_stream()>>>(
                reinterpret_cast<float*>(dx_h_slice.data_ptr()),
                acts.delta_down_out.data_ptr<float>(), ood_mask,
                batch_size, seq_len, g_cfg.h_slice_dim, lambda_effective
            );
            HIP_CHECK(hipGetLastError());
        }
        if (!g_cfg.parity_dump_dir.empty() && layer_idx == g_cfg.parity_dump_layer) {
            dump_f32_tensor_raw(dx_h_slice, parity_dump_path("dx_h_slice_post_silence_l" + std::to_string(layer_idx) + ".raw"));
        }

        // Contract delta weight gradient in canonical storage [S, H].
        // The natural math view is [H, S], but the flat buffer must match the
        // storage layout of graft.d_weight and the AdamW update path.
        execute_contract_backward_storage_canonical(arena, layer_idx, M,
            intermediate_i_slice.data_ptr<float>(),
            dx_h_slice.data_ptr<float>());
        if (dump_gate_chain) {
            dump_f32_buffer_raw(arena.d_delta_grad + delta_offset(layer_idx, 2),
                                static_cast<size_t>(g_cfg.slice_dim) * g_cfg.h_slice_dim,
                                parity_dump_path("down_weight_grad_l" + std::to_string(layer_idx) + ".raw"));
        }

        // Backprop through contract delta: grad_intermediate_i_slice += delta_down^T @ dx_h_slice
        if (g_cfg.use_aten_delta_input_grad) {
            torch::NoGradGuard guard;
            float* d_delta_w = graft.d_weight + delta_offset(layer_idx, 2);
            auto down_storage_view = torch::from_blob(d_delta_w, {g_cfg.slice_dim, g_cfg.h_slice_dim},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            auto grad_i_add = torch::matmul(dx_h_slice, down_storage_view.transpose(0, 1));
            auto grad_i_add_reshaped = grad_i_add.view({batch_size, seq_len, g_cfg.slice_dim});
            if (dump_gate_chain) {
                auto grad_i_add_full = torch::zeros_like(grad_intermediate);
                grad_i_add_full.slice(2, i_start, i_start + g_cfg.slice_dim).copy_(grad_i_add_reshaped);
                dump_f32_tensor_raw(grad_i_add_full,
                                    parity_dump_path("grad_intermediate_from_delta_down_l" + std::to_string(layer_idx) + ".raw"));
            }
            grad_intermediate.slice(2, i_start, i_start + g_cfg.slice_dim).add_(grad_i_add_reshaped);
        } else {
            auto dx_h_bf16 = dx_h_slice.to(torch::kBFloat16).contiguous();
            HIP_CHECK(hipMemcpyAsync(reinterpret_cast<uint16_t*>(arena.d_dy_slice_contract),
                                     dx_h_bf16.data_ptr(),
                                     static_cast<size_t>(M) * g_cfg.h_slice_dim * sizeof(uint16_t),
                                     hipMemcpyDeviceToDevice,
                                     current_hip_stream()));

            if (dump_gate_chain) {
                auto grad_i_add_full = torch::empty({batch_size, seq_len, g_cfg.slice_dim},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                float* d_delta_w = graft.d_weight + delta_offset(layer_idx, 2);
                const uint16_t* d_delta_w_bf16 = graft.d_weight_bf16 != nullptr
                    ? graft.d_weight_bf16 + delta_offset(layer_idx, 2)
                    : nullptr;
                graft_gemm_fwd(arena, GraftGemmKind::ContractInputGrad, M, g_cfg.h_slice_dim, g_cfg.slice_dim,
                               d_delta_w,
                               reinterpret_cast<const uint16_t*>(arena.d_dy_slice_contract),
                               g_cfg.h_slice_dim,
                               grad_i_add_full.data_ptr<float>(),
                               g_cfg.slice_dim,
                               1.0f, 0.0f,
                               d_delta_w_bf16);
                dump_f32_tensor_raw(grad_i_add_full,
                                    parity_dump_path("grad_intermediate_from_delta_down_l" + std::to_string(layer_idx) + ".raw"));
            }

            float* d_delta_w = graft.d_weight + delta_offset(layer_idx, 2);
            const uint16_t* d_delta_w_bf16 = graft.d_weight_bf16 != nullptr
                ? graft.d_weight_bf16 + delta_offset(layer_idx, 2)
                : nullptr;
            graft_gemm_fwd(arena, GraftGemmKind::ContractInputGrad, M, g_cfg.h_slice_dim, g_cfg.slice_dim,
                           d_delta_w,
                           reinterpret_cast<const uint16_t*>(arena.d_dy_slice_contract),
                           g_cfg.h_slice_dim,
                           grad_intermediate.data_ptr<float>() + i_start,
                           g_cfg.intermediate_dim,
                           1.0f, 1.0f,
                           d_delta_w_bf16);
        }
    }

    // ========================================================================
    // Step 3: Backprop through element-wise gate * up (fused HIP kernel)
    // intermediate = gate * up
    // grad_gate_pre = grad_intermediate * up * silu_deriv(gate_pre)
    // grad_up = grad_intermediate * gate
    // ========================================================================
    at::Tensor grad_gate_pre = torch::empty({batch_size, seq_len, g_cfg.intermediate_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    at::Tensor grad_up = torch::empty({batch_size, seq_len, g_cfg.intermediate_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    {
        int n = M * g_cfg.intermediate_dim;
        int blocks = (n + g_cfg.kernel_threads - 1) / g_cfg.kernel_threads;
        if (dump_gate_chain) {
            std::cout << "[gate_up_bwd_ptrs] layer=" << layer_idx
                      << " grad_intermediate_total=" << static_cast<const void*>(grad_intermediate.data_ptr<float>())
                      << " grad_gate_pre_out=" << static_cast<const void*>(grad_gate_pre.data_ptr<float>())
                      << " grad_up_out=" << static_cast<const void*>(grad_up.data_ptr<float>())
                      << std::endl;
        }
        fused_gate_up_bwd_kernel<<<blocks, g_cfg.kernel_threads, 0, current_hip_stream()>>>(
            grad_gate_pre.data_ptr<float>(),
            grad_up.data_ptr<float>(),
            grad_intermediate.data_ptr<float>(),
            reinterpret_cast<const uint16_t*>(up.data_ptr<at::BFloat16>()),
            reinterpret_cast<const uint16_t*>(gate_pre.data_ptr<at::BFloat16>()),
            n
        );
        HIP_CHECK(hipGetLastError());  // PATCH P1-07
    }
        if (dump_gate_chain) {
            auto gate_pre_i_slice = gate_pre.slice(2, i_start, i_start + g_cfg.slice_dim).to(torch::kFloat32).contiguous();
        auto up_i_slice = up.slice(2, i_start, i_start + g_cfg.slice_dim).to(torch::kFloat32).contiguous();
        auto silu_gate = gate_pre_i_slice.sigmoid() * gate_pre_i_slice;
        auto grad_gate_pre_expected = torch::zeros_like(grad_gate_pre);
        auto grad_up_expected = torch::zeros_like(grad_up);
        auto grad_gate_pre_i = grad_intermediate.slice(2, i_start, i_start + g_cfg.slice_dim) * up_i_slice *
            (gate_pre_i_slice.sigmoid() * (1.0f + gate_pre_i_slice * (1.0f - gate_pre_i_slice.sigmoid())));
        auto grad_up_i = grad_intermediate.slice(2, i_start, i_start + g_cfg.slice_dim) * gate_pre_i_slice.sigmoid() * gate_pre_i_slice;
        grad_gate_pre_expected.slice(2, i_start, i_start + g_cfg.slice_dim).copy_(grad_gate_pre_i);
        grad_up_expected.slice(2, i_start, i_start + g_cfg.slice_dim).copy_(grad_up_i);
        dump_f32_tensor_raw(gate_pre_i_slice, parity_dump_path("gate_pre_i_slice_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(up_i_slice, parity_dump_path("up_i_slice_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(silu_gate, parity_dump_path("silu_gate_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(gate_pre_i_slice.sigmoid() * (1.0f + gate_pre_i_slice * (1.0f - gate_pre_i_slice.sigmoid())),
                            parity_dump_path("silu_prime_gate_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(grad_up_expected, parity_dump_path("grad_up_expected_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(grad_gate_pre_expected, parity_dump_path("grad_gate_pre_expected_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(grad_gate_pre, parity_dump_path("grad_gate_pre_actual_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(grad_up, parity_dump_path("grad_up_actual_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(grad_intermediate,
                            parity_dump_path("grad_intermediate_total_l" + std::to_string(layer_idx) + ".raw"));
    }

    // ========================================================================
    // Steps 4+5: Expand delta backward (gate + up), fused silence blend
    // Forward:  gate_pre[:, i_slice] += delta_gate @ x_h_slice
    //           up[:, i_slice]       += delta_up @ x_h_slice
    // Gradient: grad_delta_gate = grad_gate_pre_i_slice^T @ x_h_slice
    //           grad_delta_up   = grad_up_i_slice^T @ x_h_slice
    //           grad_x_h       += grad_gate_pre_i_slice @ delta_gate
    //                          += grad_up_i_slice @ delta_up
    // ========================================================================
    at::Tensor grad_x_h = torch::zeros({batch_size, seq_len, g_cfg.h_slice_dim}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    {
        // Create dy tensors (no silence yet ” do both after dy_up exists)
        at::Tensor dy_gate = grad_gate_pre.slice(2, i_start, i_start + g_cfg.slice_dim).to(torch::kFloat32).contiguous();
        at::Tensor dy_up = grad_up.slice(2, i_start, i_start + g_cfg.slice_dim).to(torch::kFloat32).contiguous();

        // Fused silence blend + energy: gate + up deltas, single ood_mask read
        if (should_graft_proj(layer_idx, 0) || should_graft_proj(layer_idx, 1)) {
            int blocks = M;
            fused_silence_blend_energy_kernel<<<blocks, g_cfg.kernel_threads, 0, current_hip_stream()>>>(
                reinterpret_cast<float*>(dy_gate.data_ptr()),
                acts.delta_gate_out.data_ptr<float>(),
                reinterpret_cast<float*>(dy_up.data_ptr()),
                acts.delta_up_out.data_ptr<float>(),
                // PATCH P2-15: pass nullptr to skip energy recompute.
                // Forward pass already wrote these; loss was computed before
                // backward. Recomputing here is wasted work (~30% of kernel).
                nullptr,
                nullptr,
                ood_mask, batch_size, seq_len, g_cfg.slice_dim, lambda_effective
            );
            HIP_CHECK(hipGetLastError());
        }

        // PATCH PA-5 (Patch B): strided x_h pointer. Forward already wrote the
        // full [M, g_cfg.hidden_dim] bf16 input into arena.d_ffn_input (line ~2880) and
        // recompute_ffn operates on the ATen tensor, not the arena buffer, so the
        // full bf16 input is still live here. Read the H-slice in place with
        // stride g_cfg.hidden_dim — same pattern the forward expand path uses (lines
        // ~2893/2941). This removes one f32->bf16 cast kernel, one contiguous()
        // copy, and one D2D hipMemcpyAsync per layer.
        if (dump_gate_chain) {
            at::Tensor x_h_slice_view = acts.x_ffn_input.slice(2, h_start, h_start + g_cfg.h_slice_dim).to(torch::kFloat32);
            dump_f32_tensor_raw(x_h_slice_view, parity_dump_path("x_h_slice_l" + std::to_string(layer_idx) + ".raw"));
        }
        const uint16_t* d_x_h_strided = arena.d_ffn_input + h_start;

        auto run_expand_weight_grad = [&](int proj_idx, const at::Tensor& dy_src) {
            if (!should_graft_proj(layer_idx, proj_idx)) return;
            execute_expand_backward(arena, layer_idx, proj_idx, M,
                d_x_h_strided, dy_src.data_ptr<float>(), g_cfg.hidden_dim);
            if (dump_gate_chain) {
                dump_f32_buffer_raw(arena.d_delta_grad + delta_offset(layer_idx, proj_idx),
                                    static_cast<size_t>(g_cfg.slice_dim) * g_cfg.h_slice_dim,
                                    parity_dump_path(std::string(proj_idx == 0 ? "gate_grad_actual_l" : "up_grad_actual_l")
                                        + std::to_string(layer_idx) + ".raw"));
            }
        };

        if (g_cfg.use_aten_delta_input_grad) {
            // PATCH PA-4: gate the input-grad add by projection enablement.
            // Disabled projections must not contribute input gradients.
            // Gate expand backward GEMM + backprop (gate)
            run_expand_weight_grad(0, dy_gate);
            if (should_graft_proj(layer_idx, 0)) {
                torch::NoGradGuard guard;
                float* d_delta_w = graft.d_weight + delta_offset(layer_idx, 0);
                auto delta_gate = torch::from_blob(d_delta_w, {g_cfg.slice_dim, g_cfg.h_slice_dim},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                auto grad_x_add = torch::matmul(dy_gate, delta_gate);
                grad_x_h.add_(grad_x_add);
            }

            // Up expand backward GEMM + backprop
            run_expand_weight_grad(1, dy_up);
            if (should_graft_proj(layer_idx, 1)) {
                torch::NoGradGuard guard;
                float* d_delta_w = graft.d_weight + delta_offset(layer_idx, 1);
                auto delta_up = torch::from_blob(d_delta_w, {g_cfg.slice_dim, g_cfg.h_slice_dim},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                auto grad_x_add = torch::matmul(dy_up, delta_up);
                grad_x_h.add_(grad_x_add);
            }
        } else {
            // PATCH PA-2/PA-3: stage dy_gate (cast + memcpy) only when gate is enabled.
            if (should_graft_proj(layer_idx, 0)) {
                auto dy_gate_bf16 = dy_gate.to(torch::kBFloat16).contiguous();
                HIP_CHECK(hipMemcpyAsync(reinterpret_cast<uint16_t*>(arena.d_dy_slice),
                                         dy_gate_bf16.data_ptr(),
                                         static_cast<size_t>(M) * g_cfg.slice_dim * sizeof(uint16_t),
                                         hipMemcpyDeviceToDevice,
                                         current_hip_stream()));
                execute_expand_backward(arena, layer_idx, 0, M,
                    d_x_h_strided, dy_gate.data_ptr<float>(), g_cfg.hidden_dim);
                float* d_delta_w = graft.d_weight + delta_offset(layer_idx, 0);
                const uint16_t* d_delta_w_bf16 = graft.d_weight_bf16 != nullptr
                    ? graft.d_weight_bf16 + delta_offset(layer_idx, 0)
                    : nullptr;
                graft_gemm_fwd(arena, GraftGemmKind::ExpandInputGrad, M, g_cfg.slice_dim, g_cfg.h_slice_dim,
                               d_delta_w,
                               reinterpret_cast<const uint16_t*>(arena.d_dy_slice),
                               g_cfg.slice_dim,
                               grad_x_h.data_ptr<float>(),
                               g_cfg.h_slice_dim,
                               1.0f, 1.0f,
                               d_delta_w_bf16);
            }

            // PATCH PA-3: stage dy_up (cast + memcpy) only when up is enabled.
            if (should_graft_proj(layer_idx, 1)) {
                auto dy_up_bf16 = dy_up.to(torch::kBFloat16).contiguous();
                HIP_CHECK(hipMemcpyAsync(reinterpret_cast<uint16_t*>(arena.d_dy_slice),
                                         dy_up_bf16.data_ptr(),
                                         static_cast<size_t>(M) * g_cfg.slice_dim * sizeof(uint16_t),
                                         hipMemcpyDeviceToDevice,
                                         current_hip_stream()));
                execute_expand_backward(arena, layer_idx, 1, M,
                    d_x_h_strided, dy_up.data_ptr<float>(), g_cfg.hidden_dim);
                float* d_delta_w = graft.d_weight + delta_offset(layer_idx, 1);
                const uint16_t* d_delta_w_bf16 = graft.d_weight_bf16 != nullptr
                    ? graft.d_weight_bf16 + delta_offset(layer_idx, 1)
                    : nullptr;
                graft_gemm_fwd(arena, GraftGemmKind::ExpandInputGrad, M, g_cfg.slice_dim, g_cfg.h_slice_dim,
                               d_delta_w,
                               reinterpret_cast<const uint16_t*>(arena.d_dy_slice),
                               g_cfg.slice_dim,
                               grad_x_h.data_ptr<float>(),
                               g_cfg.h_slice_dim,
                               1.0f, 1.0f,
                               d_delta_w_bf16);
            }
        }
    }

    // ========================================================================
    // Step 6: Backprop through base gate_proj and up_proj
    // grad_x_ffn = grad_gate_pre @ gate_proj_weight + grad_up @ up_proj_weight
    // (use matmul, not linear, since we need grad @ weight, not grad @ weight.T)
    // grad_gate_pre/up are float32 from earlier .to(kFloat32); cast base weights to match
    // ========================================================================
    at::Tensor grad_x_ffn;
    {
        torch::NoGradGuard guard;
        grad_x_ffn = at::matmul(grad_gate_pre, w.gate_proj_weight_f32) +  // PATCH P2-14: cached f32
                     at::matmul(grad_up, w.up_proj_weight_f32);
    }

    // Add the H-slice gradient from expand deltas
    grad_x_ffn.slice(2, h_start, h_start + g_cfg.h_slice_dim).add_(grad_x_h);

    // ========================================================================
    // Step 7: Backprop through post-attention RMSNorm (fused HIP kernel)
    // Accumulate onto residual gradient from x_out = x_post_attn + ffn_out
    // ========================================================================
    at::Tensor grad_from_ffn_norm = rms_norm_bwd(grad_x_ffn, acts.x_post_attn, w.post_attention_layernorm_weight, acts.rms_norm_variance);
    at::Tensor grad_x_post_attn_total = dx + grad_from_ffn_norm;
    // attention_bwd_autograd re-runs the attention forward with x_pre_attn as the
    // only autograd leaf and backward(grad_x_post_attn_total); x.grad() is therefore
    // the full gradient wrt x_pre_attn (residual + attention path). That is exactly
    // the gradient that flows to the previous layer, so use it directly.
    at::Tensor grad_x_in_final = attention_bwd_autograd(
        w, acts.x_pre_attn, batch_size, seq_len, layer_idx, grad_x_post_attn_total);
    if (dump_boundary_cur) {
        dump_f32_tensor_raw(dx, parity_dump_path("grad_x_out_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(dx, parity_dump_path("grad_x_post_attn_initial_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(grad_from_ffn_norm, parity_dump_path("grad_from_ffn_norm_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(grad_x_post_attn_total, parity_dump_path("grad_x_post_attn_total_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(grad_x_post_attn_total, parity_dump_path("grad_attn_out_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(grad_x_in_final, parity_dump_path("grad_from_attention_l" + std::to_string(layer_idx) + ".raw"));
        dump_f32_tensor_raw(grad_x_in_final, parity_dump_path("grad_x_in_final_l" + std::to_string(layer_idx) + ".raw"));
    }
    dx = grad_x_in_final;
}
