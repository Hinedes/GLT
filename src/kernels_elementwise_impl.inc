// --------------------------------------------------------------------------
// Kernel: silence_blend — OOD Silence Regularization (contract delta variant)
// Operation: For each OOD token, computes ||delta|| then adds silence gradient
// lambda * delta / (||delta|| * sqrt(V)) to dy. Used in backward_layer for
// the contract delta path. Gate+up deltas use fused_silence_blend_energy instead.
// CDNA3: 128-bit vectorized loads/stores (float4) for memory-bound kernel.
// --------------------------------------------------------------------------
__global__ __launch_bounds__(256, 1) void silence_blend_kernel(float* dy_slice, const float* delta_out, const int32_t* ood_mask,
                                     int batch, int seq, int slice, float lambda_silence) {
    int token_idx = blockIdx.x;
    int total_tokens = batch * seq;
    if (token_idx >= total_tokens) return;
    if (ood_mask[token_idx] == 0) return;

    int tid = threadIdx.x;
    int lane_id = tid & (warpSize - 1);
    int warp_id = tid / warpSize;

    __shared__ float s_warp_sums[8];
    // PATCH P0-03: __shfl_sync broadcast is warp-local. Use shared memory
    // so all warps see the scale computed by warp 0.
    __shared__ float s_scale;

    float partial_norm = 0.0f;
    int off = token_idx * slice;

    // 128-bit vectorized loop (4 floats per iteration)
    int vec_slice = slice / 4;
    for (int i = tid; i < vec_slice; i += blockDim.x) {
        float4 v4 = *reinterpret_cast<const float4*>(delta_out + off + i * 4);
        partial_norm += v4.x * v4.x + v4.y * v4.y + v4.z * v4.z + v4.w * v4.w;
    }
    // PATCH P3-21: scalar tail removed (slice % 4 == 0 guaranteed; loop never ran).

    // Warp-level reduction via shuffle
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        partial_norm += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, partial_norm, offset);

    // Cross-warp reduction via shared memory
    if (lane_id == 0) s_warp_sums[warp_id] = partial_norm;
    __syncthreads();

    if (warp_id == 0) {
        float warp_sum = (lane_id < blockDim.x / warpSize) ? s_warp_sums[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            warp_sum += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, warp_sum, offset);
        if (lane_id == 0) {
            float norm = sqrtf(warp_sum + 1e-8f);
            s_scale = lambda_silence / (norm * sqrtf(static_cast<float>(slice)));
        }
    }
    __syncthreads();

    float scale = s_scale;

    // 128-bit vectorized write
    for (int i = tid; i < vec_slice; i += blockDim.x) {
        float4 v4 = *reinterpret_cast<const float4*>(delta_out + off + i * 4);
        float4 dy4 = *reinterpret_cast<float4*>(dy_slice + off + i * 4);
        dy4.x += scale * v4.x;
        dy4.y += scale * v4.y;
        dy4.z += scale * v4.z;
        dy4.w += scale * v4.w;
        *reinterpret_cast<float4*>(dy_slice + off + i * 4) = dy4;
    }
    // PATCH P3-21: scalar tail removed (slice % 4 == 0).
}

// --------------------------------------------------------------------------
// Kernel: fused_gate_up_bwd — Element-wise Gate*Up Backward
// Operation: Reads gate_pre(bf16) and up(bf16), plus the upstream gradient
// grad_intermediate(f32). Computes both grad_gate_pre and grad_up in
// registers. silu(gate_pre) is computed on-the-fly — no gate tensor needed.
// Fuses 7 ATen launches into 1 kernel.
// Formula: sigmoid(gate_pre) → silu_grad = sg * (1 + gp * (1 - sg))
//          grad_gate_pre = grad_intermediate * up * silu_grad
//          grad_up = grad_intermediate * silu(gate_pre)
// CDNA3: 128-bit vectorized loads (uint4 for 8x bf16, float4 for 4x f32).
// --------------------------------------------------------------------------
__global__ __launch_bounds__(256, 1) void fused_gate_up_bwd_kernel(
    float* __restrict__ grad_gate_pre,
    float* __restrict__ grad_up,
    const float* __restrict__ grad_intermediate,
    const uint16_t* __restrict__ up,      // bf16 from recompute_ffn
    const uint16_t* __restrict__ gate_pre, // bf16 from recompute_ffn
    int n)
{
    int vec_n = n / 8;  // 8 bf16 per uint4 (128-bit)
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx >= n) return;

    // Vectorized load: 8 bf16 elements per uint4
    uint4 up4 = *reinterpret_cast<const uint4*>(up + idx);
    uint4 gp4 = *reinterpret_cast<const uint4*>(gate_pre + idx);
    // grad_intermediate is f32, so load both 4-wide halves explicitly.
    float4 gi4a = *reinterpret_cast<const float4*>(grad_intermediate + idx);
    float4 gi4b = *reinterpret_cast<const float4*>(grad_intermediate + idx + 4);

    uint16_t* up16 = reinterpret_cast<uint16_t*>(&up4);
    uint16_t* gp16 = reinterpret_cast<uint16_t*>(&gp4);
    float* gi_a = reinterpret_cast<float*>(&gi4a);
    float* gi_b = reinterpret_cast<float*>(&gi4b);

    alignas(16) float results_gate[8];  // PATCH P2-10
    alignas(16) float results_up[8];    // PATCH P2-10

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float gp = bf16_to_f32(gp16[j]);
        float upv = bf16_to_f32(up16[j]);
        float sg = 1.0f / (1.0f + expf(-gp));
        float gv = gp * sg;
        float silu_grad = sg * (1.0f + gp * (1.0f - sg));
        float gi = (j < 4) ? gi_a[j] : gi_b[j - 4];

        results_gate[j] = gi * upv * silu_grad;
        results_up[j] = gi * gv;
    }

    *reinterpret_cast<float4*>(grad_gate_pre + idx) = *reinterpret_cast<float4*>(results_gate);
    *reinterpret_cast<float4*>(grad_gate_pre + idx + 4) = *reinterpret_cast<float4*>(results_gate + 4);
    *reinterpret_cast<float4*>(grad_up + idx) = *reinterpret_cast<float4*>(results_up);
    *reinterpret_cast<float4*>(grad_up + idx + 4) = *reinterpret_cast<float4*>(results_up + 4);
}

// --------------------------------------------------------------------------
// Kernels: Multi-block gradient clipping
// Three-kernel approach for utilizing all CUs on MI300X:
//   1) reduce_l2:    Multi-block partial L2 sums
//   2) calc_scale:   Single-block final reduction → scale factor
//   3) apply_scale:  Multi-block parallel scale application
// --------------------------------------------------------------------------
__global__ __launch_bounds__(256) void reduce_l2_kernel(
    const float* __restrict__ grad, int n, float* __restrict__ partial)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;

    int chunk_size = (n + num_blocks - 1) / num_blocks;
    int start = bid * chunk_size;
    int end = min(start + chunk_size, n);

    float sum_sq = 0.0f;
    for (int i = start + tid; i < end; i += blockDim.x) {
        float v = grad[i];
        sum_sq += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, sum_sq, offset);

    __shared__ float s_warp[32];
    int warp_id = tid / warpSize;
    int lane_id = tid & (warpSize - 1);
    if (lane_id == 0) s_warp[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < blockDim.x / warpSize) ? s_warp[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, val, offset);
        if (lane_id == 0) partial[bid] = val;
    }
}

__global__ void calc_scale_kernel(
    const float* __restrict__ partial, int num_partials,
    float clip_norm, float* __restrict__ d_scale)
{
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x)
        sum_sq += partial[i];

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, sum_sq, offset);

    __shared__ float s_warp[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x & (warpSize - 1);
    if (lane_id == 0) s_warp[warp_id] = sum_sq;
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        float val = (lane_id < blockDim.x / warpSize) ? s_warp[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, val, offset);
        total = val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float norm = sqrtf(total + 1e-8f);
        d_scale[0] = (norm > clip_norm) ? clip_norm / norm : 1.0f;
    }
}

__global__ void apply_scale_kernel(
    float* __restrict__ grad, int n, const float* __restrict__ d_scale)
{
    float scale = d_scale[0];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;

    int chunk_size = (n + num_blocks - 1) / num_blocks;
    int start = bid * chunk_size;
    int end = min(start + chunk_size, n);

    for (int i = start + tid; i < end; i += blockDim.x)
        grad[i] *= scale;
}

// --------------------------------------------------------------------------
// Kernel: compute_energy — Per-Token Delta Energy Norm
// Operation: For each token, computes ||delta_out|| / sqrt(slice) using
// warp-level tree reduction. Used for TPHS-compatible silence loss on
// gate/up/down delta outputs. Replaces PyTorch .norm(-1) reduce kernels.
// CDNA3: 128-bit vectorized loads (float4) for memory-bound reduction.
// --------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4) void compute_energy_kernel(
    const float* __restrict__ delta_out,
    float* __restrict__ energy_out,
    int total_tokens, int slice)
{
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    int tid = threadIdx.x;
    int lane_id = tid & (warpSize - 1);
    int warp_id = tid / warpSize;
    // num_warps is the actual warp count for g_cfg.kernel_threads=256:
    //   4 on AMD (warpSize=64), 8 on NVIDIA (warpSize=32). Array sized for max.
    // PATCH: fixed size — warpSize() is __device__, not constexpr.
    constexpr int max_warps = 8;  // safe for 256 threads with warpSize>=32
    __shared__ float s_warp_sum[max_warps];

    float pn = 0.0f;
    int off = token_idx * slice;

    // 128-bit vectorized loop (4 floats per iteration)
    int vec_slice = slice / 4;
    for (int i = tid; i < vec_slice; i += blockDim.x) {
        float4 v4 = *reinterpret_cast<const float4*>(delta_out + off + i * 4);
        pn += v4.x * v4.x + v4.y * v4.y + v4.z * v4.z + v4.w * v4.w;
    }
    // PATCH P3-21: scalar tail removed (slice % 4 == 0).

    // Warp-level reduction via shuffle
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        pn += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, pn, offset);
    }

    // PATCH P1-05: cross-warp reduction. Warp 0 lane 0 of each warp writes
    // its partial sum; all other lanes zero-init the entire shared array first
    // to prevent stale/garbage data in slots beyond the actual warp count
    // (e.g., AMD has 4 warps with 256 threads; we sized the array to 8).
    for (int i = tid; i < max_warps; i += blockDim.x) s_warp_sum[i] = 0.0f;
    __syncthreads();
    if (lane_id == 0) s_warp_sum[warp_id] = pn;
    __syncthreads();

    if (warp_id == 0) {
        // Clamp lane_id to array size to avoid OOB reads on shared memory.
        float total = (lane_id < max_warps) ? s_warp_sum[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, total, offset);
        }
        if (lane_id == 0) {
            float norm_factor = 1.0f / sqrtf(static_cast<float>(slice));
            energy_out[token_idx] = sqrtf(total + 1e-8f) * norm_factor;
        }
    }
}

// --------------------------------------------------------------------------
// Kernel: fused_inject_clone_energy — Fused Delta Injection + Clone + Energy
// Operation: Combines 3 memory-bound operations on the same delta_out buffer
// into 1 kernel: (1) inject f32 delta into bf16 base tensor at slice offset,
// (2) clone f32 delta to persistent activation storage for backward, and
// (3) compute per-token energy = ||delta||/sqrt(slice) for silence loss.
// All 3 share the same delta_out read — eliminates 2 redundant HBM passes.
// One block per token (M blocks total). Strided loop across slice dim.
// CDNA3: 128-bit vectorized loads/stores (float4 for f32, uint4 for 8x bf16).
// --------------------------------------------------------------------------
__global__ __launch_bounds__(256, 1) void fused_inject_clone_energy_kernel(
    const float* __restrict__ delta_out,   // [M, slice] f32 — source from GEMM
    uint16_t* __restrict__ base_slice,     // [M, base_stride] bf16 — injection target (strided)
    float* __restrict__ acts_delta,        // [M, slice] f32 — clone target
    float* __restrict__ energy_out,        // [M] f32 — per-token energy
    int M, int slice, int base_stride, float eps)
{
    int token_idx = blockIdx.x;
    if (token_idx >= M) return;
    int tid = threadIdx.x;
    int lane_id = tid & (warpSize - 1);
    int warp_id = tid / warpSize;
    // PATCH: fixed size — warpSize() is __device__, not constexpr.
    constexpr int max_warps = 8;  // safe for 256 threads with warpSize>=32
    __shared__ float s_warp_sum[max_warps];

    float sum_sq = 0.0f;
    int delta_off = token_idx * slice;
    int base_off = token_idx * base_stride;

    // 1:1 mapping: each f32 delta adds to exactly one bf16 base element.
    // Read 4 f32 (float4, 128-bit) + 4 bf16 (uint2, 64-bit) per iteration.
    // PATCH P0-01: was reading 8 bf16 and adding each delta to 2 base elements,
    // corrupting g_cfg.slice_dim*2 positions in the base tensor.
    int vec_slice = slice / 4;
    const bool slice_vec = (slice % 4 == 0);

    for (int i = tid; i < vec_slice; i += blockDim.x) {
        float4 dv4 = *reinterpret_cast<const float4*>(delta_out + delta_off + i * 4);
        uint2 base2 = *reinterpret_cast<const uint2*>(base_slice + base_off + i * 4);
        alignas(16) uint16_t results[4];
        uint16_t* base16 = reinterpret_cast<uint16_t*>(&base2);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float dv = (&dv4.x)[j];
            float bv = bf16_to_f32(base16[j]);
            results[j] = f32_to_bf16(bv + dv);
        }
        *reinterpret_cast<uint2*>(base_slice + base_off + i * 4) =
            *reinterpret_cast<const uint2*>(results);

        // Clone f32 delta to acts_delta (float4)
        *reinterpret_cast<float4*>(acts_delta + delta_off + i * 4) = dv4;

        sum_sq += dv4.x * dv4.x + dv4.y * dv4.y + dv4.z * dv4.z + dv4.w * dv4.w;
    }
    // Scalar tail (only reached if slice is not divisible by 4)
    if (!slice_vec) {
        for (int i = vec_slice * 4 + tid; i < slice; i += blockDim.x) {
            float dv = delta_out[delta_off + i];
            float bv = bf16_to_f32(base_slice[base_off + i]);
            base_slice[base_off + i] = f32_to_bf16(bv + dv);
            acts_delta[delta_off + i] = dv;
            sum_sq += dv * dv;
        }
    }

    // Warp-level reduction via shuffle
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, sum_sq, offset);

    // Cross-warp reduction via shared memory
    // PATCH P1-05 (also covers this kernel): previously only warp 0's lane 0
    // wrote energy, which was 1/num_warps of the true sum. Zero-init array
    // to prevent garbage in slots beyond the actual warp count.
    for (int i = tid; i < max_warps; i += blockDim.x) s_warp_sum[i] = 0.0f;
    __syncthreads();
    if (lane_id == 0) s_warp_sum[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        // Clamp lane_id to array size to avoid OOB reads on shared memory.
        float total = (lane_id < max_warps) ? s_warp_sum[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, total, offset);
        if (lane_id == 0) {
            float norm_factor = 1.0f / sqrtf((float)std::max(slice, 1));  // avoid div by 0
            energy_out[token_idx] = sqrtf(total + eps) * norm_factor;
        }
    }
}

// --------------------------------------------------------------------------
// Kernel: fused_silu_product — Fused SiLU + Gate*Up Product
// Operation: Combines gate = silu(gate_pre) and intermediate = gate * up
// into 1 kernel. Reads gate_pre and up (both bf16), writes intermediate (bf16).
// Eliminates the gate tensor entirely — no HBM write/read for intermediate
// gate activations (~43 GB/step saved). SiLU computed in registers.
// Used in both forward_layer (forward) and recompute_ffn (backward checkpoint).
// CDNA3: 128-bit vectorized loads/stores (uint4 for 8x bf16).
// --------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4) void fused_silu_product_kernel(
    const uint16_t* __restrict__ gate_pre,  // [M, g_cfg.intermediate_dim] bf16 — pre-activation (delta injected)
    const uint16_t* __restrict__ up,         // [M, g_cfg.intermediate_dim] bf16 — up projection (delta injected)
    uint16_t* __restrict__ intermediate,     // [M, g_cfg.intermediate_dim] bf16 — output: silu(gate_pre) * up
    int n)                                   // M * g_cfg.intermediate_dim elements
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx >= n) return;

    // Vectorized load: 8 bf16 elements per uint4 (128-bit)
    uint4 gp4 = *reinterpret_cast<const uint4*>(gate_pre + idx);
    uint4 up4 = *reinterpret_cast<const uint4*>(up + idx);

    uint16_t* gp16 = reinterpret_cast<uint16_t*>(&gp4);
    uint16_t* up16 = reinterpret_cast<uint16_t*>(&up4);
    alignas(16) uint16_t results[8];  // PATCH P2-10

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float gp = bf16_to_f32(gp16[j]);
        float upv = bf16_to_f32(up16[j]);
        float sg = 1.0f / (1.0f + expf(-gp));
        results[j] = f32_to_bf16(gp * sg * upv);
    }

    *reinterpret_cast<uint4*>(intermediate + idx) = *reinterpret_cast<uint4*>(results);
}

// --------------------------------------------------------------------------
// Kernel: fused_silu_product_strided
// Operation: row-major stride-aware fused SiLU * Up product.
// Used for contiguous tensors today, but explicit strides make the contract
// clear and keep the helper safe for future sliced views.
// --------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4) void fused_silu_product_strided_kernel(
    const uint16_t* __restrict__ gate_pre,
    const uint16_t* __restrict__ up,
    uint16_t* __restrict__ intermediate,
    int rows,
    int width,
    int gate_stride,
    int up_stride,
    int out_stride)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int total = rows * width;
    if (idx >= total) return;

    int row = idx / width;
    int col = idx % width;

    const uint16_t* gp_ptr = gate_pre + row * gate_stride + col;
    const uint16_t* up_ptr = up + row * up_stride + col;
    uint16_t* out_ptr = intermediate + row * out_stride + col;

    uint4 gp4 = *reinterpret_cast<const uint4*>(gp_ptr);
    uint4 up4 = *reinterpret_cast<const uint4*>(up_ptr);
    uint16_t* gp16 = reinterpret_cast<uint16_t*>(&gp4);
    uint16_t* up16 = reinterpret_cast<uint16_t*>(&up4);
    alignas(16) uint16_t results[8];

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float gp = bf16_to_f32(gp16[j]);
        float uv = bf16_to_f32(up16[j]);
        float sg = 1.0f / (1.0f + expf(-gp));
        results[j] = f32_to_bf16(gp * sg * uv);
    }

    *reinterpret_cast<uint4*>(out_ptr) = *reinterpret_cast<const uint4*>(results);
}

// --------------------------------------------------------------------------
// Kernel: fused_silence_blend_energy — Fused Silence Blend + Energy
// Operation: Combines silence blend and energy computation in one launch.
// For all tokens, computes energy = ||delta|| / sqrt(slice) for both gate
// and up deltas simultaneously. For OOD tokens only, adds silence gradient
// lambda * energy * delta to dy. Replaces 2 separate kernels per layer.
// CDNA3: 128-bit vectorized loads/stores (float4) for memory-bound kernel.
// --------------------------------------------------------------------------
__global__ __launch_bounds__(256, 1) void fused_silence_blend_energy_kernel(
    float* __restrict__ dy_gate,      const float* __restrict__ delta_gate,
    float* __restrict__ dy_up,        const float* __restrict__ delta_up,
    // PATCH P2-15: energy_gate/up outputs are optional (nullptr = skip write).
    // Forward pass already wrote these via fused_inject_clone_energy_kernel;
    // backward call site passes nullptr to avoid wasted recompute.
    float* __restrict__ energy_gate,  float* __restrict__ energy_up,
    const int32_t* __restrict__ ood_mask,
    int batch, int seq, int slice, float lambda_silence)
{
    int token_idx = blockIdx.x;
    int total_tokens = batch * seq;
    if (token_idx >= total_tokens) return;

    int tid = threadIdx.x;
    int lane_id = tid & (warpSize - 1);
    int warp_id = tid / warpSize;

    __shared__ float s_warp_g[8];
    __shared__ float s_warp_u[8];
    // PATCH P0-02: __shfl_sync broadcast is warp-local. Use shared memory
    // so all warps see the energy computed by warp 0.
    __shared__ float s_energy_g;
    __shared__ float s_energy_u;

    float pn_g = 0.0f, pn_u = 0.0f;
    int off = token_idx * slice;

    // 128-bit vectorized loop (4 floats per iteration)
    int vec_slice = slice / 4;
    for (int i = tid; i < vec_slice; i += blockDim.x) {
        float4 vg4 = *reinterpret_cast<const float4*>(delta_gate + off + i * 4);
        float4 vu4 = *reinterpret_cast<const float4*>(delta_up + off + i * 4);
        pn_g += vg4.x * vg4.x + vg4.y * vg4.y + vg4.z * vg4.z + vg4.w * vg4.w;
        pn_u += vu4.x * vu4.x + vu4.y * vu4.y + vu4.z * vu4.z + vu4.w * vu4.w;
    }
    // PATCH P3-21: scalar tail removed (slice % 4 == 0 guaranteed).

    // Warp-level reduction via shuffle
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        pn_g += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, pn_g, offset);
        pn_u += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, pn_u, offset);
    }

    // Cross-warp reduction via shared memory
    if (lane_id == 0) {
        s_warp_g[warp_id] = pn_g;
        s_warp_u[warp_id] = pn_u;
    }
    __syncthreads();

    if (warp_id == 0) {
        float wg = (lane_id < blockDim.x / warpSize) ? s_warp_g[lane_id] : 0.0f;
        float wu = (lane_id < blockDim.x / warpSize) ? s_warp_u[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            wg += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, wg, offset);
            wu += __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, wu, offset);
        }
        if (lane_id == 0) {
            float norm_factor = 1.0f / sqrtf(static_cast<float>(slice));
            float energy_g = sqrtf(wg + 1e-8f) * norm_factor;
            float energy_u = sqrtf(wu + 1e-8f) * norm_factor;
            // PATCH P2-15: skip energy output write if pointer is null.
            if (energy_gate != nullptr) energy_gate[token_idx] = energy_g;
            if (energy_up   != nullptr) energy_up[token_idx]   = energy_u;
            // Publish to shared memory so all warps can read the broadcast values.
            s_energy_g = energy_g;
            s_energy_u = energy_u;
        }
    }
    __syncthreads();

    float energy_g = s_energy_g;
    float energy_u = s_energy_u;

    float norm_g = energy_g * sqrtf(static_cast<float>(slice));
    float norm_u = energy_u * sqrtf(static_cast<float>(slice));
    float scale_g = lambda_silence / (norm_g * sqrtf(static_cast<float>(slice)) + 1e-8f);
    float scale_u = lambda_silence / (norm_u * sqrtf(static_cast<float>(slice)) + 1e-8f);

    // 128-bit vectorized write
    for (int i = tid; i < vec_slice; i += blockDim.x) {
        int idx = off + i * 4;
        if (ood_mask[token_idx] == 1) {
            float4 vg4 = *reinterpret_cast<const float4*>(delta_gate + idx);
            float4 dy_g4 = *reinterpret_cast<float4*>(dy_gate + idx);
            dy_g4.x += scale_g * vg4.x;
            dy_g4.y += scale_g * vg4.y;
            dy_g4.z += scale_g * vg4.z;
            dy_g4.w += scale_g * vg4.w;
            *reinterpret_cast<float4*>(dy_gate + idx) = dy_g4;

            float4 vu4 = *reinterpret_cast<const float4*>(delta_up + idx);
            float4 dy_u4 = *reinterpret_cast<float4*>(dy_up + idx);
            dy_u4.x += scale_u * vu4.x;
            dy_u4.y += scale_u * vu4.y;
            dy_u4.z += scale_u * vu4.z;
            dy_u4.w += scale_u * vu4.w;
            *reinterpret_cast<float4*>(dy_up + idx) = dy_u4;
        }
    }
    // PATCH P3-21: scalar tail removed (slice % 4 == 0).
}

// ============================================================================
// Device Helpers: BFloat16 ↔ Float32 Conversion
// Operation: bf16_to_f32 shifts uint16 left by 16 bits and reinterprets
// as float32. f32_to_bf16 applies round-to-nearest-even on the lower 16
// bits of the mantissa (standard IEEE for bf16 truncation), then takes
// the upper 16 bits. These are used everywhere for loading bf16 weights
// and activations from global memory into registers for computation.
// ============================================================================
__device__ float bf16_to_f32(uint16_t v) {
    unsigned int bits = static_cast<unsigned int>(v) << 16;
    return *reinterpret_cast<const float*>(&bits);
}

__device__ uint16_t f32_to_bf16(float f) {
    unsigned int bits = *reinterpret_cast<const unsigned int*>(&f);
    unsigned int rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}
