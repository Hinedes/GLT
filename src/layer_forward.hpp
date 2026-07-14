#pragma once
#include <torch/torch.h>
#include "src/types.hpp"

void forward_layer(int layer_idx, at::Tensor& x, VramArena& arena, const GraftWeights& graft, int batch_size, int seq_len, bool use_explicit_graft = true);
