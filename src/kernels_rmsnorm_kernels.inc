// ============================================================================
// Kernel: rms_norm_fwd ” Fused RMSNorm Forward
// Operation: One block per token. Phase 1: warp-reduce sum(x^2) across
// H via LDS -> rsqrt(var+eps) -> variance[token]. Phase 2: y = x * r * w.
// Single kernel replaces 4 ATen launches per layer (36L -> 144->36 launches).
// CDNA3: 256 threads (4 warps) to utilize all 4 SIMDs/CU.
// ============================================================================
__global__ __launch_bounds__(256, 4) void rms_norm_fwd_kernel(
    uint16_t* __restrict__ y,        // [M, H] bf16 ” output
    float* __restrict__ variance,     // [M] f32 ” per-token mean(x^2)
    const uint16_t* __restrict__ x,   // [M, H] bf16 ” input
    const uint16_t* __restrict__ w,   // [H] bf16 ” weight
    int H, float eps)
{
    int token_idx = blockIdx.x;
    int M = gridDim.x;
    if (token_idx >= M) return;

    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid & (warpSize - 1);
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    extern __shared__ float s_tmp[];

    // Phase 1: sum(x^2) across hidden dim
    float sum_sq = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float xv = bf16_to_f32(x[token_idx * H + i]);
        sum_sq += xv * xv;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, sum_sq, offset);

    if (lane_id == 0) s_tmp[warp_id] = sum_sq;
    __syncthreads();

    float total_sum_sq;
    if (warp_id == 0) {
        total_sum_sq = (lane_id < num_warps) ? s_tmp[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            total_sum_sq += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, total_sum_sq, offset);
        if (lane_id == 0) {
            float var = total_sum_sq / static_cast<float>(H);
            s_tmp[0] = rsqrtf(var + eps);
            variance[token_idx] = var;
        }
    }
    __syncthreads();

    float r = s_tmp[0];

    // Phase 2: y = x * r * w
    for (int i = tid; i < H; i += blockDim.x) {
        int idx = token_idx * H + i;
        float xv = bf16_to_f32(x[idx]);
        float wv = bf16_to_f32(w[i]);
        y[idx] = f32_to_bf16(xv * r * wv);
    }
}

// ============================================================================
// Kernel: rms_norm_bwd ” Fused RMSNorm Backward
// Operation: One block per token. Two sequential warp reductions via LDS.
// Pass 1: mean(dy*x*w) across H. Pass 2: dx[i] = dy[i]*r_var*w[i] -
// x[i]*mean_term*(r_var^3). Formula: r_var = rsqrt(var+eps).
// Single kernel replaces 3+ ATen operations.
// CDNA3: 256 threads (4 warps) to utilize all 4 SIMDs/CU.
// ============================================================================
__global__ __launch_bounds__(256, 4) void rms_norm_bwd_kernel(
    float* __restrict__ dx,             // [M, H] f32 ” gradient wrt x
    const float* __restrict__ dy,        // [M, H] f32 ” upstream gradient
    const uint16_t* __restrict__ x,      // [M, H] bf16 ” pre-norm input
    const uint16_t* __restrict__ w,      // [H] bf16 ” weight
    const float* __restrict__ variance,  // [M] f32 ” per-token variance (from forward)
    int H, float eps)
{
    int token_idx = blockIdx.x;
    int M = gridDim.x;
    if (token_idx >= M) return;

    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid & (warpSize - 1);
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    extern __shared__ float s_tmp[];

    float rsqrt_v = rsqrtf(variance[token_idx] + eps);
    float rsqrt_v3 = rsqrt_v * rsqrt_v * rsqrt_v;

    // Pass 1: mean(dy * x * w) across hidden dim
    float partial_mean = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        int idx = token_idx * H + i;
        float dyv = dy[idx];
        float xv = bf16_to_f32(x[idx]);
        float wv = bf16_to_f32(w[i]);
        partial_mean += dyv * xv * wv;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        partial_mean += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, partial_mean, offset);

    if (lane_id == 0) s_tmp[warp_id] = partial_mean;
    __syncthreads();

    float total_mean;
    if (warp_id == 0) {
        total_mean = (lane_id < num_warps) ? s_tmp[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            total_mean += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, total_mean, offset);
        if (lane_id == 0) s_tmp[0] = total_mean / static_cast<float>(H);
    }
    __syncthreads();

    float mean_term = s_tmp[0];

    // Pass 2: dx[i] = dy[i] * rsqrt_v * w[i] - x[i] * mean_term * rsqrt_v3
    for (int i = tid; i < H; i += blockDim.x) {
        int idx = token_idx * H + i;
        float dyv = dy[idx];
        float xv = bf16_to_f32(x[idx]);
        float wv = bf16_to_f32(w[i]);
        dx[idx] = dyv * rsqrt_v * wv - xv * mean_term * rsqrt_v3;
    }
}

// ============================================================================
// PATCH P2-11: Persistent RMSNorm kernels (rms_norm_fwd_persistent_kernel,
// rms_norm_bwd_persistent_kernel) deleted. Forward-declared at lines 36-37,
// defined here, never launched from any host code (~108 lines of dead code).
// If profiling shows RMSNorm needs optimization on MI300X, resurrect from git.
// ============================================================================
