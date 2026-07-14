#pragma once
#include "src/types.hpp"

void apply_graft_to_shadow(ModelWeights& model, const GraftWeights& graft);
inline bool g_shadow_baked = false;
