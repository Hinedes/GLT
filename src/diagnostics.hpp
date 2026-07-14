#pragma once
#include <string>
#include <torch/torch.h>
#include <cstdint>

inline std::string parity_dump_path(const std::string& name);

void dump_f32_tensor_raw(const at::Tensor& t, const std::string& path);
void dump_f32_buffer_raw(const float* data, size_t numel, const std::string& path);
void dump_bf16_buffer_raw(const uint16_t* data, size_t numel, const std::string& path);

static inline float bf16_to_f32_host(uint16_t v);
static inline uint16_t f32_to_bf16_host(float f);
