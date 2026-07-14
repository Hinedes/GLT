#pragma once
#include "checks.hpp"
#include "config.hpp"
#include "types.hpp"
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>
#include <unordered_map>

extern std::unordered_map<GraftGemmKey, GraftGemmLtPlan, GraftGemmKeyHash> g_gemm_plan_cache;
extern std::unordered_map<GraftGemmKind, GraftGemmProfileStats> g_gemm_profile_stats;

struct RowMajorGemmSpec {
    GraftGemmKind kind;
    int m; int n; int k;
    hipblasOperation_t transa;
    hipblasOperation_t transb;
    const void* A; hipDataType A_type; int A_rows; int A_cols; int A_ld;
    const void* B; hipDataType B_type; int B_rows; int B_cols; int B_ld;
    void* C; hipDataType C_type; int C_rows; int C_cols; int C_ld;
    float alpha; float beta;
};

void gemm_core(
    VramArena& arena,
    rocblas_operation transa, rocblas_operation transb,
    int m, int n, int k,
    const void* d_A, rocblas_datatype A_type, int lda,
    const void* d_B, rocblas_datatype B_type, int ldb,
    void* d_C, rocblas_datatype C_type, int ldc,
    float alpha, float beta);

__global__ void f32_to_bf16_kernel(const float* __restrict__ in, uint16_t* __restrict__ out, int n);

void convert_f32_to_bf16(VramArena& arena, const float* d_in, uint16_t* d_out, int n);

void refresh_graft_weight_bf16(VramArena& scratch, GraftWeights& graft);

static inline hipblasOperation_t to_hipblas_operation(rocblas_operation op);

static inline hipDataType to_hipblas_dtype(rocblas_datatype dt);

static inline hipblasLtMatrixLayout_t make_row_major_layout(hipDataType type, int rows, int cols, int ld);

static inline void ensure_hipblaslt_workspace(VramArena& arena, size_t required_bytes);

static inline const char* gemm_backend_for_kind(GraftGemmKind kind);

static inline bool gemm_kind_is_profiled_backend(GraftGemmKind kind);

static bool run_hipblaslt_row_major(VramArena& arena, const RowMajorGemmSpec& spec);

void graft_gemm_fwd(VramArena& arena, GraftGemmKind kind, int M, int K, int N,
                    float* d_weight, const uint16_t* d_input, int input_stride,
                    float* d_out, int out_stride,
                    float alpha, float beta,
                    const uint16_t* d_weight_bf16);

void grad_gemm(VramArena& arena, GraftGemmKind kind, int M, int K, int N,
               const uint16_t* d_input, int input_stride,
               const float* dy, float* d_grad,
               float alpha, float beta);

void grad_gemm_storage_canonical(VramArena& arena, GraftGemmKind kind, int M, int K, int N,
                                 const uint16_t* d_input, int input_stride,
                                 const float* dy, float* d_grad,
                                 float alpha, float beta);
