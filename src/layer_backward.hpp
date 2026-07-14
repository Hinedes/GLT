#pragma once
#include <torch/torch.h>
#include "src/types.hpp"

void backward_layer(int layer_idx, at::Tensor& dx, VramArena& arena, int batch_size, int seq_len, int step, float lr, int domain_id, const GraftWeights& graft, const int32_t* ood_mask, float lambda_effective);
