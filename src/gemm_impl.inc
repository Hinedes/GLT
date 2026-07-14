// GEMM cache instances — single definitions (extern in gemm.hpp)
std::unordered_map<GraftGemmKey, GraftGemmLtPlan, GraftGemmKeyHash> g_gemm_plan_cache;
std::unordered_map<GraftGemmKind, GraftGemmProfileStats> g_gemm_profile_stats;

// --------------------------------------------------------------------------
// gemm_core — General rocBLAS GEMM Wrapper
// Operation: Wraps rocblas_gemm_ex for forward (input @ weight^T) and
// backward (input^T @ dy) operations. Supports mixed precision via the
// compute_type parameter (always f32). All dimension/type parameters are
// explicit (no autodetect) to avoid rocBLAS ambiguity.
//   - Forward:  C[M,N] = op(A)[M,K] @ op(B)[K,N]
//   - Backward: C[K,N] = op(A)[K,M] @ op(B)[M,N]
// Parameters:
//   transa: transpose for A (none=forward, transpose=backward)
//   transb: transpose for B (none=forward, transpose=backward)
//   m, n, k: rocBLAS dimensions matching C[m,n] = A[m,k] @ B[k,n]
//   d_A, A_type, lda: first operand (weight/input)
//   d_B, B_type, ldb: second operand (input/dy)
//   d_C, C_type, ldc: output buffer (always f32)
// --------------------------------------------------------------------------
void gemm_core(
    VramArena& arena,
    rocblas_operation transa, rocblas_operation transb,
    int m, int n, int k,
    const void* d_A, rocblas_datatype A_type, int lda,
    const void* d_B, rocblas_datatype B_type, int ldb,
    void* d_C, rocblas_datatype C_type, int ldc,
    float alpha, float beta
) {
    // Bind rocBLAS lazily to the active HIP stream only when it changes.
    hipStream_t stream = current_hip_stream();
    if (arena.rocblas_stream != stream) {
        ROCBLAS_CHECK(rocblas_set_stream(arena.rocblas_handle, stream));
        arena.rocblas_stream = stream;
    }
    rocblas_status st = rocblas_gemm_ex(
        arena.rocblas_handle,
        transa, transb,
        m, n, k,
        &alpha,
        d_A, A_type, lda,
        d_B, B_type, ldb,
        &beta,
        d_C, C_type, ldc,
        d_C, C_type, ldc,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0
    );
    if (st != rocblas_status_success) {
        // Extra context (m, n, k, etc.) for debugging. Macro handles file:line + throw.
        std::cerr << "  gemm_ex context: transa=" << static_cast<int>(transa)
                  << " transb=" << static_cast<int>(transb)
                  << " m=" << m << " n=" << n << " k=" << k
                  << " lda=" << lda << " ldb=" << ldb << " ldc=" << ldc
                  << " status=" << static_cast<int>(st) << std::endl;
    }
    ROCBLAS_CHECK(st);
}

// ============================================================================
// Kernel: f32_to_bf16 — Float32 → BFloat16 Batch Conversion
// Operation: Converts an f32 buffer to bf16 for rocBLAS compatibility
// (rocBLAS requires A and B to share the same type). Uses round-to-nearest-
// even on the lower 16 bits of the mantissa before truncation.
// NOTE: __float2half converts to IEEE 754 fp16 (1-5-10), NOT bfloat16.
// Using __float2half and reinterpreting as bf16 silently corrupts values.
// The fallback kernel handles subnormals/infinity via explicit exponent math.
// ============================================================================
__global__ __launch_bounds__(256, 4) void f32_to_bf16_kernel(const float* __restrict__ in, uint16_t* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // float32: 1-8-23, bfloat16: 1-8-7
        // Conversion: take upper 16 bits of f32 mantissa (truncate lower 16 bits)
        unsigned int bits = *reinterpret_cast<const unsigned int*>(&in[i]);
        // Round to nearest even (standard IEEE rounding for bf16 conversion)
        unsigned int rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
        bits += rounding_bias;
        out[i] = static_cast<uint16_t>(bits >> 16);
    }
}

// --------------------------------------------------------------------------
// convert_f32_to_bf16 — Launch f32→bf16 conversion kernel (host wrapper)
// --------------------------------------------------------------------------
void convert_f32_to_bf16(VramArena& arena, const float* d_in, uint16_t* d_out, int n) {
    int blocks = (n + g_cfg.kernel_threads - 1) / g_cfg.kernel_threads;
    f32_to_bf16_kernel<<<blocks, g_cfg.kernel_threads, 0, current_hip_stream()>>>(d_in, d_out, n);
    HIP_CHECK(hipGetLastError());  // PATCH P1-07
}

void refresh_graft_weight_bf16(VramArena& scratch, GraftWeights& graft) {
    if (graft.d_weight_bf16 == nullptr) return;
    int n = static_cast<int>(graft.weight_elems);
    convert_f32_to_bf16(scratch, graft.d_weight, graft.d_weight_bf16, n);
}

static inline hipblasOperation_t to_hipblas_operation(rocblas_operation op) {
    return static_cast<hipblasOperation_t>(op);
}

static inline hipDataType to_hipblas_dtype(rocblas_datatype dt) {
    switch (dt) {
        case rocblas_datatype_f32_r: return HIP_R_32F;
        case rocblas_datatype_bf16_r: return HIP_R_16BF;
        default:
            throw std::runtime_error("Unsupported dtype for hipBLASLt graft GEMM");
    }
}

static inline hipblasLtMatrixLayout_t make_row_major_layout(hipDataType type, int rows, int cols, int ld) {
    hipblasLtMatrixLayout_t layout = nullptr;
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layout, type, rows, cols, ld));
    hipblasLtOrder_t order = HIPBLASLT_ORDER_ROW;
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        layout, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    return layout;
}

static inline void ensure_hipblaslt_workspace(VramArena& arena, size_t required_bytes) {
    if (required_bytes == 0) return;
    if (arena.d_hipblaslt_workspace != nullptr && arena.hipblaslt_workspace_bytes >= required_bytes) {
        return;
    }
    if (arena.d_hipblaslt_workspace != nullptr) {
        HIP_CHECK(hipFree(arena.d_hipblaslt_workspace));
        arena.d_hipblaslt_workspace = nullptr;
        arena.hipblaslt_workspace_bytes = 0;
    }
    HIP_CHECK(hipMalloc(&arena.d_hipblaslt_workspace, required_bytes));
    arena.hipblaslt_workspace_bytes = required_bytes;
}

static inline const char* gemm_backend_for_kind(GraftGemmKind kind) {
    switch (kind) {
        case GraftGemmKind::ContractForward:
        case GraftGemmKind::ExpandWeightGrad:
        case GraftGemmKind::ContractWeightGrad:
            return (g_cfg.graft_gemm_backend == GraftGemmBackend::HipblasltSelective) ? "hipblaslt" : "rocblas";
        case GraftGemmKind::ExpandForward:
        case GraftGemmKind::ExpandInputGrad:
        case GraftGemmKind::ContractInputGrad:
            return "rocblas";
    }
    return "rocblas";
}

static inline bool gemm_kind_is_profiled_backend(GraftGemmKind kind) {
    return kind == GraftGemmKind::ContractForward ||
           kind == GraftGemmKind::ExpandWeightGrad ||
           kind == GraftGemmKind::ContractWeightGrad;
}

static void record_gemm_profile(
    GraftGemmKind kind,
    const char* backend,
    double ms,
    const RowMajorGemmSpec& spec,
    size_t workspace_bytes,
    bool host_approx,
    const std::string& fallback_reason = std::string())
{
    // ponytail: hard gate. Even if a caller forgets to wrap the spec construction
    // in `if (g_cfg.profile_graft_gemms)`, this returns before any std::string work
    // or hash-map insertion. The default-constructed fallback_reason triggers a
    // heap allocation per call otherwise; we want zero-cost when profiling is off.
    if (!g_cfg.profile_graft_gemms) return;
    auto& stats = g_gemm_profile_stats[kind];
    stats.call_count += 1;
    if (std::string(backend) == "rocblas") {
        stats.rocblas_calls += 1;
    } else if (std::string(backend) == "hipblaslt") {
        stats.hipblaslt_calls += 1;
    } else if (std::string(backend) == "aten") {
        stats.aten_fallback_calls += 1;
    }
    if (!fallback_reason.empty()) {
        stats.fallback_calls += 1;
        stats.fallback_reason = fallback_reason;
    }
    stats.total_ms += ms;
    if (host_approx) stats.host_approx_ms += ms;
    stats.host_approx = host_approx;
    stats.workspace_bytes = workspace_bytes;
    stats.m = spec.m;
    stats.n = spec.n;
    stats.k = spec.k;
    stats.lda = spec.A_ld;
    stats.ldb = spec.B_ld;
    stats.ldc = spec.C_ld;
    stats.transa = static_cast<int>(spec.transa);
    stats.transb = static_cast<int>(spec.transb);
    stats.dtype_a = static_cast<int>(spec.A_type);
    stats.dtype_b = static_cast<int>(spec.B_type);
    stats.dtype_c = static_cast<int>(spec.C_type);
    stats.compute_type = static_cast<int>(HIPBLAS_COMPUTE_32F);
    stats.alpha = spec.alpha;
    stats.beta = spec.beta;
    stats.backend = backend;
}

static bool run_hipblaslt_row_major(VramArena& arena, const RowMajorGemmSpec& spec) {
    if (arena.hipblaslt_handle == nullptr) {
        return false;
    }

    GraftGemmKey key{
        spec.kind,
        spec.m,
        spec.n,
        spec.k,
        static_cast<int>(spec.transa),
        static_cast<int>(spec.transb),
        spec.A_ld,
        spec.B_ld,
        spec.C_ld,
        static_cast<int>(spec.A_type),
        static_cast<int>(spec.B_type),
        static_cast<int>(spec.C_type),
        static_cast<int>(HIPBLAS_COMPUTE_32F)
    };

    auto it = g_gemm_plan_cache.find(key);
    if (it != g_gemm_plan_cache.end() && !it->second.valid) {
        return false;
    }
    if (it == g_gemm_plan_cache.end()) {
        GraftGemmLtPlan plan{};
        hipblasLtMatmulDesc_t desc = nullptr;
        hipblasLtMatrixLayout_t a_layout = nullptr;
        hipblasLtMatrixLayout_t b_layout = nullptr;
        hipblasLtMatrixLayout_t c_layout = nullptr;
        hipblasLtMatrixLayout_t d_layout = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
        bool ok = true;
        std::string fail_reason;
        try {
            HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
            hipblasOperation_t ta = spec.transa;
            hipblasOperation_t tb = spec.transb;
            int32_t pointer_mode = HIPBLASLT_POINTER_MODE_HOST;
            HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
            HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
            HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));

            a_layout = make_row_major_layout(spec.A_type, spec.A_rows, spec.A_cols, spec.A_ld);
            b_layout = make_row_major_layout(spec.B_type, spec.B_rows, spec.B_cols, spec.B_ld);
            c_layout = make_row_major_layout(spec.C_type, spec.C_rows, spec.C_cols, spec.C_ld);
            d_layout = make_row_major_layout(spec.C_type, spec.C_rows, spec.C_cols, spec.C_ld);

            HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
            uint64_t max_workspace_bytes = 1024ull * 1024ull * 1024ull;
            HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &max_workspace_bytes, sizeof(max_workspace_bytes)));

            hipblasLtMatmulHeuristicResult_t heur[1]{};
            int returned = 0;
            hipblasStatus_t hstat = hipblasLtMatmulAlgoGetHeuristic(
                arena.hipblaslt_handle,
                desc,
                a_layout,
                b_layout,
                c_layout,
                d_layout,
                pref,
                1,
                heur,
                &returned);
            if (hstat != HIPBLAS_STATUS_SUCCESS || returned <= 0 || heur[0].state != HIPBLAS_STATUS_SUCCESS) {
                ok = false;
                fail_reason = "hipblasLt heuristic unavailable";
            } else {
                plan.desc = desc;
                plan.a_layout = a_layout;
                plan.b_layout = b_layout;
                plan.c_layout = c_layout;
                plan.d_layout = d_layout;
                plan.algo = heur[0];
                plan.workspace_bytes = heur[0].workspaceSize;
                plan.valid = true;
                if (plan.workspace_bytes > 0) {
                    ensure_hipblaslt_workspace(arena, plan.workspace_bytes);
                }
                desc = nullptr;
                a_layout = nullptr;
                b_layout = nullptr;
                c_layout = nullptr;
                d_layout = nullptr;
                pref = nullptr;
            }
        } catch (const std::exception& e) {
            ok = false;
            fail_reason = e.what();
        }
        if (pref) hipblasLtMatmulPreferenceDestroy(pref);
        if (d_layout) hipblasLtMatrixLayoutDestroy(d_layout);
        if (c_layout) hipblasLtMatrixLayoutDestroy(c_layout);
        if (b_layout) hipblasLtMatrixLayoutDestroy(b_layout);
        if (a_layout) hipblasLtMatrixLayoutDestroy(a_layout);
        if (desc) hipblasLtMatmulDescDestroy(desc);

        if (!ok) {
            GraftGemmLtPlan invalid{};
            invalid.valid = false;
            invalid.workspace_bytes = 0;
            g_gemm_plan_cache[key] = std::move(invalid);
            return false;
        }

        g_gemm_plan_cache[key] = std::move(plan);
        it = g_gemm_plan_cache.find(key);
    }

    auto& plan = it->second;
    if (!plan.valid) {
        return false;
    }

    const void* alpha_ptr = &spec.alpha;
    const void* beta_ptr = &spec.beta;

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    if (g_cfg.profile_graft_gemms) {
        HIP_CHECK(hipEventCreateWithFlags(&start, hipEventDefault));
        HIP_CHECK(hipEventCreateWithFlags(&stop, hipEventDefault));
        HIP_CHECK(hipEventRecord(start, current_hip_stream()));
    }

    hipblasStatus_t status = hipblasLtMatmul(
        arena.hipblaslt_handle,
        plan.desc,
        alpha_ptr,
        spec.A,
        plan.a_layout,
        spec.B,
        plan.b_layout,
        beta_ptr,
        spec.C,
        plan.c_layout,
        spec.C,
        plan.d_layout,
        &plan.algo.algo,
        arena.d_hipblaslt_workspace,
        arena.hipblaslt_workspace_bytes,
        current_hip_stream());

    if (g_cfg.profile_graft_gemms) {
        HIP_CHECK(hipEventRecord(stop, current_hip_stream()));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        if (status == HIPBLAS_STATUS_SUCCESS) {
            record_gemm_profile(spec.kind, "hipblaslt", static_cast<double>(ms), spec, plan.workspace_bytes, false);
        }
    }
    // ponytail: dropped the `else { record_gemm_profile(..., 0.0, ...) }` branch --
    // no spec bookkeeping when profiling is off. record_gemm_profile also has an
    // internal guard, so this is belt-and-suspenders.

    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipblasLtMatmul failed for kind=" << graft_gemm_kind_name(spec.kind)
                  << " status=" << static_cast<int>(status) << std::endl;
        return false;
    }
    return true;
}

// ============================================================================
// GEMM Wrappers: Expand/Contract Forward & Backward + Grad + AdamW
// Operation: Wraps rocBLAS gemm_ex for the four graft GEMM variants:
//   - expand_fwd:  input[Hslice] @ weight^T                  → f32 [Slice]
//   - contract_fwd: input[Slice] @ weight^T                  → f32 [Hslice]
//   - grad_expand:  input[Hslice]^T @ dy[Slice]              → f32 [Slice×Hslice]
//   - grad_contract: input[Slice]^T @ dy[Hslice]             → f32 [Hslice×Slice]
// rocBLAS requires same-type inputs, so f32→bf16 conversion is done via
// a lightweight HIP kernel before each GEMM. AdamW is a single fused kernel
// over all 108 projections deferred to adamw_stream for pipeline overlap.
// ============================================================================

// Forward GEMM: output[M, N] = input[M, K] @ weight[N, K]^T
// d_input: bf16 [M, K] row-major
// d_weight: f32 [N, K] row-major
// Output: f32 [M, N] row-major (out_stride defaults to N; can be wider for
// slice views such as the contract-input-gradient accumulation path).
void graft_gemm_fwd(VramArena& arena, GraftGemmKind kind, int M, int K, int N,
                    float* d_weight, const uint16_t* d_input, int input_stride,
                    float* d_out, int out_stride = -1,
                    float alpha = 1.0f, float beta = 0.0f,
                    const uint16_t* d_weight_bf16 = nullptr) {
    if (out_stride < 0) out_stride = N;
    // ponytail: when a bf16 mirror of d_weight is available (maintained by
    // --delta-bf16-mirror + refresh after AdamW), skip the f32->bf16 conversion
    // kernel and read directly from the mirror. Saves one kernel launch + one
    // full HBM round-trip per call. Per-step impact: ~144 fewer conversion
    // kernels (gate/up/down x 36 layers x fwd+bwd).
    const uint16_t* d_a_bf16;
    if (d_weight_bf16 != nullptr) {
        d_a_bf16 = d_weight_bf16;
    } else {
        int weight_elems = N * K;
        convert_f32_to_bf16(arena, d_weight, arena.d_bf16_temp, weight_elems);
        d_a_bf16 = arena.d_bf16_temp;
    }

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    if (g_cfg.profile_graft_gemms) {
        HIP_CHECK(hipEventCreateWithFlags(&start, hipEventDefault));
        HIP_CHECK(hipEventCreateWithFlags(&stop, hipEventDefault));
        HIP_CHECK(hipEventRecord(start, current_hip_stream()));
    }

    gemm_core(arena,
        rocblas_operation_transpose, rocblas_operation_none,
        N, M, K,
        d_a_bf16,          rocblas_datatype_bf16_r, K,
        d_input,           rocblas_datatype_bf16_r, input_stride,
        d_out,             rocblas_datatype_f32_r,   out_stride,
        alpha, beta);

    if (g_cfg.profile_graft_gemms) {
        HIP_CHECK(hipEventRecord(stop, current_hip_stream()));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        RowMajorGemmSpec spec{
            kind, M, N, K,
            to_hipblas_operation(rocblas_operation_transpose),
            to_hipblas_operation(rocblas_operation_none),
            const_cast<uint16_t*>(d_a_bf16), HIP_R_16BF,
            N, K, K,
            d_input, HIP_R_16BF,
            M, K, input_stride,
            d_out, HIP_R_32F,
            M, N, out_stride,
            alpha, beta
        };
        record_gemm_profile(kind, "rocblas", static_cast<double>(ms), spec, 0, false);
    }
}

// d_input: bf16 [M, K] row-major
// dy: f32 [M, N] row-major
// Output: grad [N, K] row-major
void grad_gemm(VramArena& arena, GraftGemmKind kind, int M, int K, int N,
               const uint16_t* d_input, int input_stride,
               const float* dy, float* d_grad,
               float alpha = 1.0f, float beta = 0.0f) {
    int dy_elems = M * N;
    convert_f32_to_bf16(arena, dy, arena.d_bf16_temp, dy_elems);
    bool hipblaslt_fallback = false;

    if (g_cfg.graft_gemm_backend == GraftGemmBackend::HipblasltSelective && gemm_kind_is_profiled_backend(kind)) {
        RowMajorGemmSpec spec{
            kind, K, N, M,
            HIPBLAS_OP_T, HIPBLAS_OP_N,
            d_input, HIP_R_16BF,
            M, K, input_stride,
            arena.d_bf16_temp, HIP_R_16BF,
            M, N, N,
            d_grad, HIP_R_32F,
            K, N, N,
            alpha, beta
        };
        if (run_hipblaslt_row_major(arena, spec)) return;
        std::cerr << "[graft_gemm] hipBLASLt fallback to rocBLAS for "
                  << graft_gemm_kind_name(kind) << std::endl;
        hipblaslt_fallback = true;
    }

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    if (g_cfg.profile_graft_gemms) {
        HIP_CHECK(hipEventCreateWithFlags(&start, hipEventDefault));
        HIP_CHECK(hipEventCreateWithFlags(&stop, hipEventDefault));
        HIP_CHECK(hipEventRecord(start, current_hip_stream()));
    }

    gemm_core(arena,
        rocblas_operation_none, rocblas_operation_transpose,
        K, N, M,
        d_input,          rocblas_datatype_bf16_r, input_stride,
        arena.d_bf16_temp, rocblas_datatype_bf16_r, N,
        d_grad,           rocblas_datatype_f32_r,   K,
        alpha, beta);

    if (g_cfg.profile_graft_gemms) {
        HIP_CHECK(hipEventRecord(stop, current_hip_stream()));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        RowMajorGemmSpec spec{
            kind, K, N, M,
            to_hipblas_operation(rocblas_operation_none),
            to_hipblas_operation(rocblas_operation_transpose),
            d_input, HIP_R_16BF,
            M, K, input_stride,
            arena.d_bf16_temp, HIP_R_16BF,
            M, N, N,
            d_grad, HIP_R_32F,
            K, N, N,
            alpha, beta
        };
        record_gemm_profile(kind, "rocblas", static_cast<double>(ms), spec, 0, false,
                            hipblaslt_fallback ? "hipblaslt fallback" : "");
    }
}

// grad_gemm_storage_canonical — Down-proj backward GEMM with canonical storage
// Output: grad [K, N] row-major, matching the flat d_delta_grad/graft.d_weight
// layout for the down projection. The math view is still the natural
// down_torch_view = [N, K] used by the forward path.
void grad_gemm_storage_canonical(VramArena& arena, GraftGemmKind kind, int M, int K, int N,
                                 const uint16_t* d_input, int input_stride,
                                 const float* dy, float* d_grad,
                                 float alpha = 1.0f, float beta = 0.0f) {
    int dy_elems = M * N;
    convert_f32_to_bf16(arena, dy, arena.d_bf16_temp, dy_elems);
    bool hipblaslt_fallback = false;

    if (g_cfg.graft_gemm_backend == GraftGemmBackend::HipblasltSelective && gemm_kind_is_profiled_backend(kind)) {
        RowMajorGemmSpec spec{
            kind, K, N, M,
            HIPBLAS_OP_T, HIPBLAS_OP_N,
            d_input, HIP_R_16BF,
            M, K, input_stride,
            arena.d_bf16_temp, HIP_R_16BF,
            M, N, N,
            d_grad, HIP_R_32F,
            K, N, N,
            alpha, beta
        };
        if (run_hipblaslt_row_major(arena, spec)) return;
        std::cerr << "[graft_gemm] hipBLASLt fallback to rocBLAS for "
                  << graft_gemm_kind_name(kind) << std::endl;
        hipblaslt_fallback = true;
    }

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    if (g_cfg.profile_graft_gemms) {
        HIP_CHECK(hipEventCreateWithFlags(&start, hipEventDefault));
        HIP_CHECK(hipEventCreateWithFlags(&stop, hipEventDefault));
        HIP_CHECK(hipEventRecord(start, current_hip_stream()));
    }

    gemm_core(arena,
        rocblas_operation_none, rocblas_operation_transpose,
        N, K, M,
        arena.d_bf16_temp, rocblas_datatype_bf16_r, N,
        d_input,          rocblas_datatype_bf16_r, input_stride,
        d_grad,           rocblas_datatype_f32_r,   N,
        alpha, beta);

    if (g_cfg.profile_graft_gemms) {
        HIP_CHECK(hipEventRecord(stop, current_hip_stream()));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        RowMajorGemmSpec spec{
            kind, K, N, M,
            to_hipblas_operation(rocblas_operation_none),
            to_hipblas_operation(rocblas_operation_transpose),
            arena.d_bf16_temp, HIP_R_16BF,
            M, N, N,
            d_input, HIP_R_16BF,
            M, K, input_stride,
            d_grad, HIP_R_32F,
            K, N, N,
            alpha, beta
        };
        record_gemm_profile(kind, "rocblas", static_cast<double>(ms), spec, 0, false,
                            hipblaslt_fallback ? "hipblaslt fallback" : "");
    }
}
