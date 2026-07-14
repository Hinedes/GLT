#pragma once
#include <torch/torch.h>

void apply_rope(at::Tensor& q, at::Tensor& k, int seq_len);

static inline at::Tensor apply_rope_out_of_place(const at::Tensor& t, int num_heads, int seq_len);
