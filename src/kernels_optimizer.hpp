#pragma once
#include "types.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>

__global__ void adamw_fused_kernel(float* d_weights, uint16_t* d_weights_bf16,
    float* d_grads, float* d_ms, float* d_vs,
    float lr, float beta1, float beta2, float beta1_corr, float beta2_corr,
    float eps, float weight_decay,
    int proj_size, int total_projs,
    int gate_enabled, int up_enabled, int down_enabled);

void execute_adamw_all(VramArena& arena, GraftWeights& graft, int step, float lr, hipStream_t stream);
