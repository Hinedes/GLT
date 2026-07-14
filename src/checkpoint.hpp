#pragma once
#include <string>
#include "types.hpp"

constexpr uint32_t GRFT_MAGIC = 0x54465247;
constexpr uint8_t GRFT_VERSION = 4;

void save_safetensors(const std::string& path, const GraftWeights& graft);
bool load_safetensors_checkpoint(const std::string& path, int expected_domain_id, VramArena& arena, GraftWeights& graft);
