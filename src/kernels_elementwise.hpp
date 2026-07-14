#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>

__device__ float bf16_to_f32(uint16_t v);
__device__ uint16_t f32_to_bf16(float f);

__global__ void silence_blend_kernel(float* dy_slice, const float* delta_out, const int32_t* ood_mask, int batch, int seq, int slice, float lambda_silence);
__global__ void fused_gate_up_bwd_kernel(float* __restrict__ grad_gate_pre, float* __restrict__ grad_up, const float* __restrict__ grad_intermediate, const uint16_t* __restrict__ up, const uint16_t* __restrict__ gate_pre, int n);
__global__ void reduce_l2_kernel(const float* __restrict__ grad, int n, float* __restrict__ partial);
__global__ void calc_scale_kernel(const float* __restrict__ partial, int num_partials, float clip_norm, float* __restrict__ d_scale);
__global__ void apply_scale_kernel(float* __restrict__ grad, int n, const float* __restrict__ d_scale);
__global__ void compute_energy_kernel(const float* __restrict__ delta_out, float* __restrict__ energy_out, int total_tokens, int slice);
__global__ void fused_inject_clone_energy_kernel(const float* __restrict__ delta_out, uint16_t* __restrict__ base_slice, float* __restrict__ acts_delta, float* __restrict__ energy_out, int M, int slice, int base_stride, float eps);
__global__ void fused_silu_product_kernel(const uint16_t* __restrict__ gate_pre, const uint16_t* __restrict__ up, uint16_t* __restrict__ intermediate, int n);
__global__ void fused_silu_product_strided_kernel(const uint16_t* __restrict__ gate_pre, const uint16_t* __restrict__ up, uint16_t* __restrict__ intermediate, int rows, int width, int gate_stride, int up_stride, int out_stride);
__global__ void fused_silence_blend_energy_kernel(float* __restrict__ dy_gate, const float* __restrict__ delta_gate, float* __restrict__ dy_up, const float* __restrict__ delta_up, float* __restrict__ energy_gate, float* __restrict__ energy_up, const int32_t* __restrict__ ood_mask, int batch, int seq, int slice, float lambda_silence);
