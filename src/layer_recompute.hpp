#pragma once
#include <torch/torch.h>
#include "src/types.hpp"

void recompute_ffn(int layer_idx, const at::Tensor& x_ffn_input,
                    VramArena& arena, int batch_size, int seq_len, int domain_id,
                    at::Tensor& gate_pre_out,
                    at::Tensor& up_out, at::Tensor& intermediate_out);
