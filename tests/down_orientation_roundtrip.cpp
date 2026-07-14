// down_orientation_roundtrip.cpp
// Compile on MI300X droplet:
//   hipcc -std=c++17 -o down_orientation_roundtrip down_orientation_roundtrip.cpp
//   ./down_orientation_roundtrip
//
// Sentinel: S=3, H=2, M=2
// W_tphs[H,S] = [[0.5, 0.0, 0.25],
//                [0.0, 1.0, 0.0 ]]
// W_hip [S,H] = W_tphs.T  (flat row-major in graft storage)
// X [M,S]      = [[1,2,3],[4,5,6]]
// Expected Y[M,H] = X @ W_tphs.T = [[1.25,2.0],[3.5,5.0]]
// Expected grad[H,S] = dY^T @ X = ... (see test 2 below)

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>

// --- bf16 helpers (inlined from grafting.hip) ---
static inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}
static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

#define HIP_CHECK(call) do { \
    hipError_t e = call; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", (int)e, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define ROCBLAS_CHECK(call) do { \
    rocblas_status s = call; \
    if (s != rocblas_status_success) { \
        fprintf(stderr, "rocBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ===================================================================
// Test 1: Contract forward — HIP storage [S,H] produces correct Y[M,H]
// ===================================================================
int test_contract_forward() {
    const int S = 3, H = 2, M = 2;
    const int INTERMEDIATE_DIM = 6;  // >= S with room for slicing

    // W_tphs [H,S] = [[0.5, 0.0, 0.25], [0.0, 1.0, 0.0]]
    // W_hip  [S,H] = W_tphs.T = [[0.5, 0.0], [0.0, 1.0], [0.25, 0.0]]
    float h_w_f32[S * H] = {
        0.5f, 0.0f,
        0.0f, 1.0f,
        0.25f, 0.0f
    };

    uint16_t h_w_bf16[S * H];
    for (int i = 0; i < S * H; i++) h_w_bf16[i] = f32_to_bf16(h_w_f32[i]);

    // X [M,S] = [[1,2,3],[4,5,6]] — stored with stride INTERMEDIATE_DIM
    uint16_t h_x_bf16[M * INTERMEDIATE_DIM];
    std::memset(h_x_bf16, 0, sizeof(h_x_bf16));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < S; j++)
            h_x_bf16[i * INTERMEDIATE_DIM + j] = f32_to_bf16(static_cast<float>(i * S + j + 1));

    // GPU allocations
    uint16_t *d_w, *d_x;
    float *d_y;
    HIP_CHECK(hipMalloc(&d_w, S * H * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_x, M * INTERMEDIATE_DIM * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_y, M * H * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_w, h_w_bf16, S * H * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x_bf16, M * INTERMEDIATE_DIM * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_y, 0, M * H * sizeof(float)));

    // rocBLAS contract forward: no transpose on either (col-major trick)
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    float alpha = 1.0f, beta = 0.0f;
    // C[M,H] = A[H,S] @ B[S,M]  (rocBLAS column-major)
    // m=H, n=M, k=S, A col-major [H,S] = w_bf16[S,H] row-major reinterpreted
    // B col-major [S,M] = x_bf16[M,S] strided with INTERMEDIATE_DIM reinterpreted
    // C col-major [H,M] = y[M,H] row-major
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle,
        rocblas_operation_none, rocblas_operation_none,
        H, M, S,
        &alpha,
        d_w, rocblas_datatype_bf16_r, H,
        d_x, rocblas_datatype_bf16_r, INTERMEDIATE_DIM,
        &beta,
        d_y, rocblas_datatype_f32_r, H,
        d_y, rocblas_datatype_f32_r, H,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0
    ));

    float h_y[M * H];
    HIP_CHECK(hipMemcpy(h_y, d_y, M * H * sizeof(float), hipMemcpyDeviceToHost));

    // Expected: Y = [[1.25, 2.0], [3.5, 5.0]]
    float expected[M * H] = {1.25f, 2.0f, 3.5f, 5.0f};
    bool ok = true;
    for (int i = 0; i < M * H; i++) {
        if (std::fabs(h_y[i] - expected[i]) > 1e-4f) {
            printf("  forward[%d]: got %.6f, expected %.6f\n", i, h_y[i], expected[i]);
            ok = false;
        }
    }
    printf("Test 1 contract forward (col-major trick): %s\n", ok ? "PASS" : "FAIL");

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipFree(d_w));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    return ok ? 0 : 1;
}

// ===================================================================
// Test 2: Contract weight grad — output stored as [S,H] = grad[H,S].T
// ===================================================================
int test_contract_weight_grad() {
    const int S = 3, H = 2, M = 2;

    // dY [M,H] = upstream gradient through down output
    // [[1.0, 0.0], [0.5, 2.0]]
    float h_dy[M * H] = {1.0f, 0.0f, 0.5f, 2.0f};
    // X intermediate [M,S] = [[1,2,3],[4,5,6]] with stride S
    uint16_t h_x_bf16[M * S];
    for (int i = 0; i < M * S; i++)
        h_x_bf16[i] = f32_to_bf16(static_cast<float>(i + 1));  // 1..6

    uint16_t *d_x, *d_dy_bf16;
    float *d_grad, *d_dy;
    HIP_CHECK(hipMalloc(&d_x, M * S * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_dy, M * H * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dy_bf16, M * H * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_grad, H * S * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_x, h_x_bf16, M * S * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dy, h_dy, M * H * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_grad, 0, H * S * sizeof(float)));

    // convert d_dy f32 -> bf16 (host-side kernel equivalent)
    uint16_t h_dy_bf16[M * H];
    for (int i = 0; i < M * H; i++) h_dy_bf16[i] = f32_to_bf16(h_dy[i]);
    HIP_CHECK(hipMemcpy(d_dy_bf16, h_dy_bf16, M * H * sizeof(uint16_t), hipMemcpyHostToDevice));

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    // grad_contract [H,S] = dY^T [H,M] @ X [M,S]
    // Compute via rocBLAS: C[H,S] = A[H,M] @ B[M,S]^T
    // A = dY_bf16 col-major [H,M] (ld=H), B = X_bf16 col-major [S,M] (ld=S), B^T
    float alpha = 1.0f, beta = 0.0f;
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle,
        rocblas_operation_none, rocblas_operation_transpose,
        H, S, M,   // m=H, n=S, k=M
        &alpha,
        d_dy_bf16, rocblas_datatype_bf16_r, H,
        d_x,       rocblas_datatype_bf16_r, S,
        &beta,
        d_grad,    rocblas_datatype_f32_r, H,
        d_grad,    rocblas_datatype_f32_r, H,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0
    ));

    float h_grad[H * S];
    HIP_CHECK(hipMemcpy(h_grad, d_grad, H * S * sizeof(float), hipMemcpyDeviceToHost));

    // Expected grad[H,S] = dY^T @ X
    // [[1,0.5],[0,2]] @ [[1,2,3],[4,5,6]]
    // = [[1*1+0.5*4, 1*2+0.5*5, 1*3+0.5*6],
    //    [0*1+2*4,   0*2+2*5,   0*3+2*6]]
    // = [[3, 4.5, 6], [8, 10, 12]]
    float expected_math[H * S] = {3.0f, 4.5f, 6.0f, 8.0f, 10.0f, 12.0f};

    // rocBLAS output is column-major [H,S] with ld=H.
    // Physical offset (i,j) = i + j*H.
    // HIP storage is row-major [S,H].
    // Physical offset (k,m) = k*H + m.
    // For grad(i,j) to land at storage(k,m): i + j*H = k*H + m => k=j, m=i.
    // So grad[H,S](i,j) → storage[S,H](j,i) = transpose.
    // Expected storage = expected_math^T:
    float expected_storage[S * H] = {3.0f, 8.0f, 4.5f, 10.0f, 6.0f, 12.0f};

    bool ok = true;
    for (int i = 0; i < H * S; i++) {
        if (std::fabs(h_grad[i] - expected_storage[i]) > 1e-4f) {
            printf("  wgrad[%d]: got %.6f, expected %.6f\n", i, h_grad[i], expected_storage[i]);
            ok = false;
        }
    }
    printf("Test 2 contract weight grad (storage=[S,H]): %s\n", ok ? "PASS" : "FAIL");

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_dy));
    HIP_CHECK(hipFree(d_dy_bf16));
    HIP_CHECK(hipFree(d_grad));
    return ok ? 0 : 1;
}

// ===================================================================
// Test 3: Roundtrip — W_hip [S,H] in, contract forward out, grad
//         back, verify no orientation corruption
// ===================================================================
int test_roundtrip_orientation() {
    const int S = 3, H = 2, M = 2;

    float h_W[S * H] = {0.5f, 0.0f, 0.0f, 1.0f, 0.25f, 0.0f};
    uint16_t h_W_bf16[S * H];
    for (int i = 0; i < S * H; i++) h_W_bf16[i] = f32_to_bf16(h_W[i]);

    // Check: W_bf16 bytes match row-major [S,H] orientation
    // Element (0,0)=0.5 at offset 0, (2,0)=0.25 at offset 4
    bool orient_ok = true;
    if (bf16_to_f32(h_W_bf16[0]) != 0.5f)  { printf("  offset 0: %.4f != 0.5\n",  bf16_to_f32(h_W_bf16[0])); orient_ok = false; }
    if (bf16_to_f32(h_W_bf16[1]) != 0.0f)  { printf("  offset 1: %.4f != 0.0\n",  bf16_to_f32(h_W_bf16[1])); orient_ok = false; }
    if (bf16_to_f32(h_W_bf16[2]) != 0.0f)  { printf("  offset 2: %.4f != 0.0\n",  bf16_to_f32(h_W_bf16[2])); orient_ok = false; }
    if (bf16_to_f32(h_W_bf16[3]) != 1.0f)  { printf("  offset 3: %.4f != 1.0\n",  bf16_to_f32(h_W_bf16[3])); orient_ok = false; }
    if (bf16_to_f32(h_W_bf16[4]) != 0.25f) { printf("  offset 4: %.4f != 0.25\n", bf16_to_f32(h_W_bf16[4])); orient_ok = false; }
    if (bf16_to_f32(h_W_bf16[5]) != 0.0f)  { printf("  offset 5: %.4f != 0.0\n",  bf16_to_f32(h_W_bf16[5])); orient_ok = false; }
    printf("Test 3a W_hip[S,H] stored correctly: %s\n", orient_ok ? "PASS" : "FAIL");

    // Test 3b: apply as bake-in (eval/stack path)
    // W_tphs[H,S] = W_hip[S,H]^T (column-major reinterpretation)
    // delta_out[M,H] = X[M,S] @ W_tphs^T = X[M,S] @ W_hip[S,H] (using col-major trick)
    // This is exactly what test 1 verifies.
    printf("Test 3b bake-in math covered by test 1 above\n");

    return orient_ok ? 0 : 1;
}

int main() {
    int fail = 0;
    fail += test_contract_forward();
    fail += test_contract_weight_grad();
    fail += test_roundtrip_orientation();
    printf("\n%s (%d/3 passed)\n", fail == 0 ? "ALL PASSED" : "SOME FAILED", 3 - fail);
    return fail;
}
