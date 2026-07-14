// --------------------------------------------------------------------------
// delta_offset — Compute flat index into the unified delta weight array
// Layer l, projection p (0=gate, 1=up, 2=down) → offset in elements
// --------------------------------------------------------------------------
int delta_offset(int layer_idx, int proj_idx) {
    return (layer_idx * PROJECTIONS_PER_LAYER + proj_idx) * g_cfg.slice_dim * g_cfg.h_slice_dim;
}

// --------------------------------------------------------------------------
// execute_expand_forward — Expand delta forward (gate or up)
// Operation: C[M,g_cfg.slice_dim] = input[M,g_cfg.h_slice_dim] @ delta_weight^T[g_cfg.slice_dim,g_cfg.h_slice_dim]
// Stride is g_cfg.hidden_dim (d_input slices from full [M,g_cfg.hidden_dim] buffer)
// --------------------------------------------------------------------------
void execute_expand_forward(VramArena& arena, const GraftWeights& graft, int layer_idx, int proj_idx,
                            int M, const uint16_t* d_input, float* d_out) {
    int K = g_cfg.h_slice_dim;
    int N = g_cfg.slice_dim;
    int off = delta_offset(layer_idx, proj_idx);
    float* w = graft.d_weight + off;
    const uint16_t* w_bf16 = (graft.d_weight_bf16 != nullptr)
        ? (graft.d_weight_bf16 + off) : nullptr;
    graft_gemm_fwd(arena, GraftGemmKind::ExpandForward, M, K, N, w, d_input, g_cfg.hidden_dim, d_out,
                   /*out_stride=*/-1, /*alpha=*/1.0f, /*beta=*/0.0f, w_bf16);
}

// --------------------------------------------------------------------------
// execute_contract_forward — Contract delta forward (down)
// Operation: C[M,g_cfg.h_slice_dim] = input[M,g_cfg.slice_dim] @ delta_weight[g_cfg.slice_dim,g_cfg.h_slice_dim]
// Storage is canonical [S,H]. The row-major buffer is interpreted directly as
// a column-major [H,S] matrix for rocBLAS, so no explicit transpose is needed.
// Stride is g_cfg.intermediate_dim (d_input slices from full [M,g_cfg.intermediate_dim] buffer)
// --------------------------------------------------------------------------
void execute_contract_forward(VramArena& arena, const GraftWeights& graft, int layer_idx,
                              int M, const uint16_t* d_input, float* d_out) {
    int K = g_cfg.slice_dim;
    int N = g_cfg.h_slice_dim;
    float* w = graft.d_weight + delta_offset(layer_idx, 2);
    bool hipblaslt_fallback = false;
    if (g_cfg.graft_gemm_backend == GraftGemmBackend::HipblasltSelective) {
        const uint16_t* w_bf16 = nullptr;
        if (graft.d_weight_bf16 != nullptr) {
            w_bf16 = graft.d_weight_bf16 + delta_offset(layer_idx, 2);
        } else {
            int weight_elems = K * N;
            convert_f32_to_bf16(arena, w, arena.d_bf16_temp, weight_elems);
            w_bf16 = arena.d_bf16_temp;
        }
        RowMajorGemmSpec spec{
            GraftGemmKind::ContractForward,
            M, N, K,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            d_input, HIP_R_16BF,
            M, K, g_cfg.intermediate_dim,
            w_bf16, HIP_R_16BF,
            K, N, N,
            d_out, HIP_R_32F,
            M, N, N,
            1.0f, 0.0f
        };
        if (run_hipblaslt_row_major(arena, spec)) return;
        std::cerr << "[graft_gemm] hipBLASLt fallback to rocBLAS for ContractForward" << std::endl;
        hipblaslt_fallback = true;
    }
    int weight_elems = K * N;
    const uint16_t* w_bf16;
    if (graft.d_weight_bf16 != nullptr) {
        w_bf16 = graft.d_weight_bf16 + delta_offset(layer_idx, 2);
    } else {
        convert_f32_to_bf16(arena, w, arena.d_bf16_temp, weight_elems);
        w_bf16 = arena.d_bf16_temp;
    }
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    if (g_cfg.profile_graft_gemms) {
        HIP_CHECK(hipEventCreateWithFlags(&start, hipEventDefault));
        HIP_CHECK(hipEventCreateWithFlags(&stop, hipEventDefault));
        HIP_CHECK(hipEventRecord(start, current_hip_stream()));
    }
    gemm_core(arena,
        rocblas_operation_none, rocblas_operation_none,
        N, M, K,
        const_cast<uint16_t*>(w_bf16), rocblas_datatype_bf16_r, N,
        d_input,          rocblas_datatype_bf16_r, g_cfg.intermediate_dim,
        d_out,            rocblas_datatype_f32_r,   N,
        1.0f, 0.0f);
    if (g_cfg.profile_graft_gemms) {
        HIP_CHECK(hipEventRecord(stop, current_hip_stream()));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        RowMajorGemmSpec spec{
            GraftGemmKind::ContractForward,
            M, N, K,
            to_hipblas_operation(rocblas_operation_none),
            to_hipblas_operation(rocblas_operation_none),
            d_input, HIP_R_16BF,
            M, K, g_cfg.intermediate_dim,
            const_cast<uint16_t*>(w_bf16), HIP_R_16BF,
            K, N, N,
            d_out, HIP_R_32F,
            M, N, N,
            1.0f, 0.0f
        };
        record_gemm_profile(GraftGemmKind::ContractForward,
                            "rocblas",
                            static_cast<double>(ms),
                            spec,
                            0,
                            false,
                            hipblaslt_fallback ? "hipblaslt fallback" : "");
    }
}

// --------------------------------------------------------------------------
// execute_expand_grad_gemm — Expand delta gradient (gate or up)
// Operation: dW[N,K] = input[M,K]^T @ dy[M,N]  (GEMM only, AdamW deferred)
// --------------------------------------------------------------------------
void execute_expand_grad_gemm(VramArena& arena, int layer_idx, int proj_idx,
                              int M, const uint16_t* d_input, const float* dy,
                              int input_stride = g_cfg.h_slice_dim) {
    int K = g_cfg.h_slice_dim;
    int N = g_cfg.slice_dim;
    int off = delta_offset(layer_idx, proj_idx);
    float* d_grad = arena.d_delta_grad + off;
    grad_gemm(arena, GraftGemmKind::ExpandWeightGrad, M, K, N, d_input, input_stride, dy, d_grad);
}

// --------------------------------------------------------------------------
// execute_contract_grad_gemm — Contract delta gradient (down)
// Operation: dW[N,K] = input[M,K]^T @ dy[M,N]  (GEMM only, AdamW deferred)
// --------------------------------------------------------------------------
void execute_contract_grad_gemm(VramArena& arena, int layer_idx,
                                int M, const uint16_t* d_input, const float* dy) {
    int K = g_cfg.slice_dim;
    int N = g_cfg.h_slice_dim;
    int off = delta_offset(layer_idx, 2);
    float* d_grad = arena.d_delta_grad + off;
    grad_gemm_storage_canonical(arena, GraftGemmKind::ContractWeightGrad, M, K, N, d_input, g_cfg.slice_dim, dy, d_grad);
}

// --------------------------------------------------------------------------
// Legacy Backward Wrappers (gradient GEMM only, AdamW deferred to pipeline)
// Kept for forward-path compatibility; delegate to execute_*_grad_gemm.
// --------------------------------------------------------------------------
void execute_expand_backward(VramArena& arena, int layer_idx, int proj_idx,
                             int M, const uint16_t* d_input, const float* dy,
                             int input_stride = g_cfg.h_slice_dim) {
    execute_expand_grad_gemm(arena, layer_idx, proj_idx, M, d_input, dy, input_stride);
}

void execute_contract_backward(VramArena& arena, int layer_idx,
                               int M, const uint16_t* d_input, const float* dy) {
    execute_contract_grad_gemm(arena, layer_idx, M, d_input, dy);
}

void run_contract_sparse_probe(VramArena& arena, GraftWeights& graft) {
    const int M = 2;
    const int K = g_cfg.slice_dim;
    const int N = g_cfg.h_slice_dim;
    const int input_stride = g_cfg.intermediate_dim;
    const size_t in_elems = static_cast<size_t>(M) * input_stride;
    const size_t w_elems = static_cast<size_t>(K) * N;
    const size_t out_elems = static_cast<size_t>(M) * N;

    std::vector<uint16_t> h_in(in_elems, 0);
    std::vector<float> h_w(w_elems, 0.0f);

    h_in[0 * input_stride + 3] = f32_to_bf16_host(2.0f);
    h_in[1 * input_stride + 17] = f32_to_bf16_host(-5.0f);

    h_w[3 * N + 11] = 7.0f;
    h_w[17 * N + 23] = -13.0f;
    h_w[29 * N + 31] = 19.0f;

    HIP_CHECK(hipMemset(arena.d_intermediate, 0, in_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMemset(graft.d_weight, 0, static_cast<size_t>(g_cfg.total_projections) * g_cfg.slice_dim * g_cfg.h_slice_dim * sizeof(float)));
    HIP_CHECK(hipMemcpy(arena.d_intermediate, h_in.data(), in_elems * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(graft.d_weight + delta_offset(35, 2), h_w.data(), w_elems * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(arena.d_delta_out_contract, 0, out_elems * sizeof(float)));

    execute_contract_forward(arena, graft, 35, M, arena.d_intermediate, arena.d_delta_out_contract);
    HIP_CHECK(hipStreamSynchronize(current_hip_stream()));

    std::vector<float> h_out(out_elems);
    HIP_CHECK(hipMemcpy(h_out.data(), arena.d_delta_out_contract, out_elems * sizeof(float), hipMemcpyDeviceToHost));
    dump_f32_buffer_raw(arena.d_delta_out_contract, out_elems, parity_dump_path("contract_sparse_probe_out_l35.raw"));

    std::cout << "[ContractProbe] synthetic sparse mapping results:\n";
    float max_abs = 0.0f;
    int max_m = -1;
    int max_h = -1;
    for (int m = 0; m < M; ++m) {
        for (int h = 0; h < N; ++h) {
            float v = h_out[static_cast<size_t>(m) * N + h];
            float av = std::fabs(v);
            if (av > max_abs) {
                max_abs = av;
                max_m = m;
                max_h = h;
            }
            if (av > 1e-7f) {
                std::cout << "  out[" << m << "," << h << "]=" << v << "\n";
            }
        }
    }
    std::cout << "[ContractProbe] max_abs=" << max_abs
              << " at out[" << max_m << "," << max_h << "]=" << h_out[static_cast<size_t>(max_m) * N + max_h] << "\n";
    std::cout << "[ContractProbe] expected: out[0,11]=14, out[1,23]=65, others ~0\n";

}

void execute_contract_backward_storage_canonical(VramArena& arena, int layer_idx,
                                                 int M, const float* d_input, const float* dy) {
    int off = delta_offset(layer_idx, 2);
    float* d_grad = arena.d_delta_grad + off;
    // Canonical storage writeback:
    //   d_input is the intermediate_i_slice f32 buffer [M, S]
    //   dy is the grad_x_h_slice f32 buffer [M, H]
    // The helper converts the input slice to bf16 once in the arena
    // intermediate scratch buffer, then writes the result directly into
    // [S, H] storage to match graft.d_weight.
    convert_f32_to_bf16(arena, d_input, arena.d_intermediate, M * g_cfg.slice_dim);
    grad_gemm_storage_canonical(arena, GraftGemmKind::ContractWeightGrad, M, g_cfg.slice_dim, g_cfg.h_slice_dim,
                                arena.d_intermediate,
                                g_cfg.slice_dim, dy, d_grad);
}
