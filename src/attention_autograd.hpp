#pragma once
#include <torch/torch.h>
#include "src/types.hpp"

at::Tensor attention_bwd_autograd(
    const LayerWeights& w,
    const at::Tensor& x_pre_attn,
    int batch_size,
    int seq_len,
    int layer_idx,
    const at::Tensor& grad_x_post_attn_total);
