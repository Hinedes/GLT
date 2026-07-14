#pragma once
#include "types.hpp"
#include <cstdint>

int delta_offset(int layer_idx, int proj_idx);
void execute_expand_forward(VramArena& arena, const GraftWeights& graft, int layer_idx, int proj_idx, int M, const uint16_t* d_input, float* d_out);
void execute_contract_forward(VramArena& arena, const GraftWeights& graft, int layer_idx, int M, const uint16_t* d_input, float* d_out);
void execute_expand_grad_gemm(VramArena& arena, int layer_idx, int proj_idx, int M, const uint16_t* d_input, const float* dy, int input_stride);
void execute_contract_grad_gemm(VramArena& arena, int layer_idx, int M, const uint16_t* d_input, const float* dy);
void execute_expand_backward(VramArena& arena, int layer_idx, int proj_idx, int M, const uint16_t* d_input, const float* dy, int input_stride);
void execute_contract_backward(VramArena& arena, int layer_idx, int M, const uint16_t* d_input, const float* dy);
void run_contract_sparse_probe(VramArena& arena, GraftWeights& graft);
void execute_contract_backward_storage_canonical(VramArena& arena, int layer_idx, int M, const float* d_input, const float* dy);
