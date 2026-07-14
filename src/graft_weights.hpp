#pragma once
#include "types.hpp"
#include <cstdint>

void allocate_graft_weights(GraftWeights& graft, int domain_id, bool with_bf16_mirror);
void free_graft_weights_noexcept(GraftWeights& graft) noexcept;
