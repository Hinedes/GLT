// ---------------------------------------------------------------------------
// GraftWeights physical ownership — allocate/free persistent graft parameters
// and the optional bf16 mirror. After Phase 2, GraftWeights is the sole owner
// of these buffers; VramArena owns only scratch/gradients/optimizer state.
// ---------------------------------------------------------------------------
void allocate_graft_weights(GraftWeights& graft, int domain_id, bool with_bf16_mirror) {
    // Defensive: free any prior allocation before re-allocating.
    if (graft.allocated) free_graft_weights_noexcept(graft);

    graft.weight_elems =
        static_cast<size_t>(g_cfg.total_projections) *
        g_cfg.slice_dim *
        g_cfg.h_slice_dim;
    graft.domain_id = domain_id;
    graft.d_weight = nullptr;
    graft.d_weight_bf16 = nullptr;
    graft.active = false;

    hipError_t err = hipMalloc(&graft.d_weight, graft.weight_elems * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "[graft] hipMalloc(d_weight) failed: " << hipGetErrorString(err) << std::endl;
        free_graft_weights_noexcept(graft);
        throw std::runtime_error("allocate_graft_weights: d_weight alloc failed");
    }

    if (with_bf16_mirror) {
        err = hipMalloc(&graft.d_weight_bf16, graft.weight_elems * sizeof(uint16_t));
        if (err != hipSuccess) {
            std::cerr << "[graft] hipMalloc(d_weight_bf16) failed: " << hipGetErrorString(err) << std::endl;
            free_graft_weights_noexcept(graft);
            throw std::runtime_error("allocate_graft_weights: d_weight_bf16 alloc failed");
        }
    }

    HIP_CHECK(hipMemset(graft.d_weight, 0, graft.weight_elems * sizeof(float)));
    if (graft.d_weight_bf16 != nullptr) {
        HIP_CHECK(hipMemset(graft.d_weight_bf16, 0, graft.weight_elems * sizeof(uint16_t)));
    }

    graft.allocated = true;
}

void free_graft_weights_noexcept(GraftWeights& graft) noexcept {
    if (graft.d_weight != nullptr) {
        hipError_t err = hipFree(graft.d_weight);
        if (err != hipSuccess) {
            std::cerr << "[graft] hipFree(d_weight) failed: " << hipGetErrorString(err) << std::endl;
        }
        graft.d_weight = nullptr;
    }
    if (graft.d_weight_bf16 != nullptr) {
        hipError_t err = hipFree(graft.d_weight_bf16);
        if (err != hipSuccess) {
            std::cerr << "[graft] hipFree(d_weight_bf16) failed: " << hipGetErrorString(err) << std::endl;
        }
        graft.d_weight_bf16 = nullptr;
    }
    graft.weight_elems = 0;
    graft.domain_id = 0;
    graft.allocated = false;
    graft.active = false;
}
