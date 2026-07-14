#pragma once
#include <torch/torch.h>
#include <cstdint>
#include <hip/hip_runtime.h>

__global__ void rms_norm_fwd_kernel(uint16_t* y, float* variance, const uint16_t* x, const uint16_t* w, int H, float eps);
__global__ void rms_norm_bwd_kernel(float* dx, const float* dy, const uint16_t* x, const uint16_t* w, const float* variance, int H, float eps);

std::pair<at::Tensor, at::Tensor> rms_norm(const at::Tensor& x, const at::Tensor& weight);
at::Tensor rms_norm_bwd(const at::Tensor& dy, const at::Tensor& x, const at::Tensor& weight, const at::Tensor& variance);
