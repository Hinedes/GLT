// ---------------------------------------------------------------------------
// Helper: advance a byte offset within a malloced arena, return typed pointer
// ---------------------------------------------------------------------------
template <typename T>
[[nodiscard]] T* arena_offset(void* base, size_t& offset, size_t bytes) {
    offset = (offset + 15) & ~static_cast<size_t>(15);
    T* ptr = static_cast<T*>(static_cast<void*>(static_cast<char*>(base) + offset));
    offset += bytes;
    return ptr;
}

// ============================================================================

void allocate_arena(VramArena& arena, int batch_size, int seq_len) {
    if (arena.allocated)
        free_arena(arena);

    // Sizes (all typed via sizeof so the cast is implied)
    const size_t B  = static_cast<size_t>(batch_size) * seq_len;
    const size_t SZ_INPUT   = B * sizeof(int32_t);
    const size_t SZ_FFN_INPUT = B * g_cfg.hidden_dim * sizeof(uint16_t);
    const size_t SZ_INTER   = B * g_cfg.intermediate_dim * sizeof(uint16_t);
    const size_t SZ_DOUT    = B * g_cfg.slice_dim * sizeof(float);
    const size_t SZ_DOUT_C  = B * g_cfg.h_slice_dim * sizeof(float);
    const size_t SZ_DY      = B * g_cfg.slice_dim * sizeof(float);
    const size_t SZ_DY_C    = B * g_cfg.h_slice_dim * sizeof(float);
    const size_t SZ_OOD     = B * sizeof(int32_t);

    const size_t SZ_1PROJ = g_cfg.slice_dim * g_cfg.h_slice_dim * sizeof(float);
    const size_t SZ_W     = g_cfg.total_projections * SZ_1PROJ;

    // bf16 temp: large enough for fwd weight [D*Hd] or bwd dy [B*D]
    const size_t SZ_BF16 = std::max(B * g_cfg.slice_dim,
                                    static_cast<size_t>(g_cfg.slice_dim) * g_cfg.h_slice_dim)
                         * sizeof(uint16_t);

    const size_t SZ_GRAD_CLIP_PARTIAL = 1024 * sizeof(float);
    const size_t SZ_GRAD_CLIP_SCALE = sizeof(float);

    // PATCH PA-1: arena_offset aligns each block to 16 bytes (adds up to 15
    // bytes of padding per block). The prior "+256" slack could under-count:
    // 16 blocks x 15 bytes = 240 worst case, but future buffers could push it
    // over. Compute total_bytes through the SAME aligned-advance logic so the
    // allocation always covers the final offset, plus a 16-byte tail guard.
    auto aligned_total = [&](size_t init_off) -> size_t {
        size_t off = init_off;
        auto adv = [&](size_t bytes) { off = (off + 15) & ~static_cast<size_t>(15); off += bytes; };
        adv(SZ_INPUT);
        adv(SZ_FFN_INPUT);
        adv(SZ_INTER);
        adv(SZ_DOUT);
        adv(SZ_DOUT_C);
        adv(SZ_DY);
        adv(SZ_DY_C);
        adv(SZ_OOD);
        // Persistent graft parameters moved to GraftWeights (Phase 2): no arena
        // SZ_W weight block here. Arena retains grad + Adam m + Adam v only.
        adv(SZ_W); // grad
        adv(SZ_W); // Adam m
        adv(SZ_W); // Adam v
        adv(SZ_BF16);
        adv(SZ_GRAD_CLIP_PARTIAL);
        adv(SZ_GRAD_CLIP_SCALE);
        return (off + 15) & ~static_cast<size_t>(15);
    };
    arena.total_bytes = aligned_total(0);

    void* p = nullptr;
    HIP_CHECK(hipMalloc(&p, arena.total_bytes));
    arena.raw_ptr = p;

    size_t off = 0;
    arena.d_input_ids           = arena_offset<int32_t>(p, off, SZ_INPUT);
    arena.d_ffn_input           = arena_offset<uint16_t>(p, off, SZ_FFN_INPUT);
    arena.d_intermediate        = arena_offset<uint16_t>(p, off, SZ_INTER);
    arena.d_delta_out           = arena_offset<float>(p, off, SZ_DOUT);
    arena.d_delta_out_contract  = arena_offset<float>(p, off, SZ_DOUT_C);
    arena.d_dy_slice            = arena_offset<float>(p, off, SZ_DY);
    arena.d_dy_slice_contract   = arena_offset<float>(p, off, SZ_DY_C);
    arena.d_ood_mask            = arena_offset<int32_t>(p, off, SZ_OOD);
    arena.d_delta_grad          = arena_offset<float>(p, off, SZ_W);
    arena.d_adam_m              = arena_offset<float>(p, off, SZ_W);
    arena.d_adam_v              = arena_offset<float>(p, off, SZ_W);
    arena.d_bf16_temp           = arena_offset<uint16_t>(p, off, SZ_BF16);
    arena.d_grad_clip_partial   = arena_offset<float>(p, off, SZ_GRAD_CLIP_PARTIAL);
    arena.d_grad_clip_scale     = arena_offset<float>(p, off, SZ_GRAD_CLIP_SCALE);
    // PATCH PA-1: assert aligned advance never exceeds the pre-computed total.
    if (off > arena.total_bytes) {
        std::cerr << "[arena] FATAL: offset " << off << " > total_bytes " << arena.total_bytes << std::endl;
        throw std::runtime_error("arena alignment under-allocation");
    }
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&arena.h_grad_clip_scale_ping), sizeof(float)));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&arena.h_grad_clip_scale_pong), sizeof(float)));
    *arena.h_grad_clip_scale_ping = -1.0f;
    *arena.h_grad_clip_scale_pong = -1.0f;

    HIP_CHECK(hipMemset(arena.d_delta_grad, 0, SZ_W));
    HIP_CHECK(hipMemset(arena.d_adam_m, 0, SZ_W));
    HIP_CHECK(hipMemset(arena.d_adam_v, 0, SZ_W));

    rocblas_status status = rocblas_create_handle(&arena.rocblas_handle);
    if (status != rocblas_status_success) {
        std::cerr << "Failed to create rocblas handle: " << status << std::endl;
        throw std::runtime_error("rocblas_create_handle failed");
    }
    // Bind rocBLAS to the current HIP stream so GEMMs stay ordered with ATen.
    status = rocblas_set_stream(arena.rocblas_handle, current_hip_stream());
    if (status != rocblas_status_success) {
        std::cerr << "Failed to set rocblas stream: " << status << std::endl;
        throw std::runtime_error("rocblas_set_stream failed");
    }
    arena.rocblas_stream = current_hip_stream();

    hipblasStatus_t lt_status = hipblasLtCreate(&arena.hipblaslt_handle);
    if (lt_status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create hipBLASLt handle: " << static_cast<int>(lt_status) << std::endl;
        throw std::runtime_error("hipblasLtCreate failed");
    }

    HIP_CHECK(hipStreamCreate(&arena.adamw_stream));
    HIP_CHECK(hipEventCreateWithFlags(&arena.backward_event, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&arena.adamw_event, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&arena.grad_clip_copy_event, hipEventDisableTiming));
    
    arena.allocated = true;
}

// PATCH P0-04: free_arena must NOT throw. It is called from ~VramArena(),
// which fires during stack unwinding if a HIP error throws earlier. A
// throwing destructor + active exception = std::terminate (UB). All HIP
// errors are logged to std::cerr and swallowed instead.
void free_arena_noexcept(VramArena& arena) noexcept {
    if (arena.raw_ptr != nullptr) {
        hipError_t err = hipFree(arena.raw_ptr);
        if (err != hipSuccess) {
            std::cerr << "[free_arena] hipFree failed: " << hipGetErrorString(err) << std::endl;
        }
        arena.raw_ptr = nullptr;
    }
    if (arena.rocblas_handle != nullptr) {
        rocblas_status rs = rocblas_destroy_handle(arena.rocblas_handle);
        if (rs != rocblas_status_success) {
            std::cerr << "[free_arena] rocblas_destroy_handle failed: " << static_cast<int>(rs) << std::endl;
        }
        arena.rocblas_handle = nullptr;
    }
    if (arena.hipblaslt_handle != nullptr) {
        hipblasStatus_t hs = hipblasLtDestroy(arena.hipblaslt_handle);
        if (hs != HIPBLAS_STATUS_SUCCESS) {
            std::cerr << "[free_arena] hipblasLtDestroy failed: " << static_cast<int>(hs) << std::endl;
        }
        arena.hipblaslt_handle = nullptr;
    }
    if (arena.d_hipblaslt_workspace != nullptr) {
        hipError_t err = hipFree(arena.d_hipblaslt_workspace);
        if (err != hipSuccess) {
            std::cerr << "[free_arena] hipFree(workspace) failed: " << hipGetErrorString(err) << std::endl;
        }
        arena.d_hipblaslt_workspace = nullptr;
        arena.hipblaslt_workspace_bytes = 0;
    }
    if (arena.adamw_stream != nullptr) {
        hipError_t err = hipStreamDestroy(arena.adamw_stream);
        if (err != hipSuccess) {
            std::cerr << "[free_arena] hipStreamDestroy failed: " << hipGetErrorString(err) << std::endl;
        }
        arena.adamw_stream = nullptr;
    }
    if (arena.backward_event != nullptr) {
        hipError_t err = hipEventDestroy(arena.backward_event);
        if (err != hipSuccess) {
            std::cerr << "[free_arena] hipEventDestroy failed: " << hipGetErrorString(err) << std::endl;
        }
        arena.backward_event = nullptr;
    }
    if (arena.adamw_event != nullptr) {
        hipError_t err = hipEventDestroy(arena.adamw_event);
        if (err != hipSuccess) {
            std::cerr << "[free_arena] hipEventDestroy failed: " << hipGetErrorString(err) << std::endl;
        }
        arena.adamw_event = nullptr;
    }
    if (arena.grad_clip_copy_event != nullptr) {
        hipError_t err = hipEventDestroy(arena.grad_clip_copy_event);
        if (err != hipSuccess) {
            std::cerr << "[free_arena] hipEventDestroy(grad_clip_copy_event) failed: " << hipGetErrorString(err) << std::endl;
        }
        arena.grad_clip_copy_event = nullptr;
    }
    if (arena.h_grad_clip_scale_ping != nullptr) {
        hipError_t err = hipHostFree(arena.h_grad_clip_scale_ping);
        if (err != hipSuccess) {
            std::cerr << "[free_arena] hipHostFree(h_grad_clip_scale_ping) failed: " << hipGetErrorString(err) << std::endl;
        }
        arena.h_grad_clip_scale_ping = nullptr;
    }
    if (arena.h_grad_clip_scale_pong != nullptr) {
        hipError_t err = hipHostFree(arena.h_grad_clip_scale_pong);
        if (err != hipSuccess) {
            std::cerr << "[free_arena] hipHostFree(h_grad_clip_scale_pong) failed: " << hipGetErrorString(err) << std::endl;
        }
        arena.h_grad_clip_scale_pong = nullptr;
    }
    arena.allocated = false;
}

void free_arena(VramArena& arena) {
    if (!arena.allocated) {
        return;
    }
    free_arena_noexcept(arena);
}
