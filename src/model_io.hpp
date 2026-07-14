#pragma once
#include <string>
#include <vector>
#include <torch/torch.h>
#include "types.hpp"

// safetensors.h included earlier in grafting.hip (no include guard on that header).
// Types safetensors_TensorDescriptor and safetensors_File are in scope.

at::Tensor st_to_aten(safetensors_TensorDescriptor& td);
at::Tensor load_named_tensor(std::vector<std::pair<std::vector<uint8_t>, safetensors_File>>& files, const std::string& name);
bool load_base_model_safetensors(const std::string& safetensors_path);
