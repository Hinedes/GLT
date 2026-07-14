// test4_down_checkpoint_roundtrip.cpp
// Compile on MI300X:
//   hipcc -std=c++17 -o test4_roundtrip test4_down_checkpoint_roundtrip.cpp -lrocblas
//   ./test4_roundtrip
//
// Verifies that the checkpoint f32→bf16→f32 conversion roundtrip
// preserves the down delta orientation byte-for-byte.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

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
        fprintf(stderr, "rocBLAS err %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static float bf16_to_f32(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out; std::memcpy(&out, &bits, sizeof(out)); return out;
}
static uint16_t f32_to_bf16(float f) {
    uint32_t bits; std::memcpy(&bits, &f, sizeof(bits));
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

int main() {
    const int S = 3, H = 2, M = 2;
    const int PROJ_SIZE = S * H;  // 6 elements per down delta

    // --- Sentinel: W_hip [S,H] row-major f32 ---
    float h_W_f32[PROJ_SIZE] = {
        0.5f, 0.0f,
        0.0f, 1.0f,
        0.25f, 0.0f
    };

    // === SAVE path: f32 → bf16 (matching save_safetensors host loop) ===
    uint16_t h_W_bf16_saved[PROJ_SIZE];
    for (int i = 0; i < PROJ_SIZE; i++) {
        // Match grafting.hip save_safetensors: line ~3667-3674
        uint32_t bits;
        std::memcpy(&bits, &h_W_f32[i], sizeof(bits));
        uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
        bits += rounding_bias;
        h_W_bf16_saved[i] = static_cast<uint16_t>(bits >> 16);
    }

    // === LOAD path: bf16 → f32 (matching load_safetensors_checkpoint) ===
    float h_W_f32_loaded[PROJ_SIZE];
    for (int i = 0; i < PROJ_SIZE; i++) {
        // Match grafting.hip load_safetensors_checkpoint: line ~3871-3873
        uint32_t bits = static_cast<uint32_t>(h_W_bf16_saved[i]) << 16;
        std::memcpy(&h_W_f32_loaded[i], &bits, sizeof(bits));
    }

    // --- Verify roundtrip ---
    bool ok = true;
    printf("=== Roundtrip element-wise check ===\n");
    for (int i = 0; i < PROJ_SIZE; i++) {
        float orig = h_W_f32[i];
        float reload = h_W_f32_loaded[i];
        printf("  [%d] orig=%.6f reload=%.6f diff=%.6f\n", i, (double)orig, (double)reload, (double)std::fabs(orig - reload));
        if (std::fabs(orig - reload) > 1e-4f) ok = false;
    }
    printf("Roundtrip values: %s\n\n", ok ? "PASS" : "FAIL");

    // --- Verify spatial mapping: which element landed where ---
    // W_hip[S,H]: element (s,h) should be at offset s*H + h
    // Row 0: W[0,0]=0.5 at offset 0, W[0,1]=0.0 at offset 1
    // Row 1: W[1,0]=0.0 at offset 2, W[1,1]=1.0 at offset 3
    // Row 2: W[2,0]=0.25 at offset 4, W[2,1]=0.0 at offset 5
    printf("=== Spatial orientation check ===\n");
    bool spatial_ok = true;
    auto check = [&](int s, int h, float expected) {
        int offset = s * H + h;
        float got = h_W_f32_loaded[offset];
        bool match = std::fabs(got - expected) < 1e-4f;
        printf("  W[%d,%d] offset=%d: %.2f (%.2f) %s\n", s, h, offset, (double)got, (double)expected, match ? "OK" : "FAIL");
        if (!match) spatial_ok = false;
    };
    check(0, 0, 0.5f);   check(0, 1, 0.0f);
    check(1, 0, 0.0f);   check(1, 1, 1.0f);
    check(2, 0, 0.25f);  check(2, 1, 0.0f);
    printf("Spatial orientation: %s\n\n", spatial_ok ? "PASS" : "FAIL");
    ok &= spatial_ok;

    // --- GPU: contract forward using loaded data (same as test 1) ---
    const int INTERMEDIATE_DIM = 6;

    uint16_t h_x_bf16[M * INTERMEDIATE_DIM];
    std::memset(h_x_bf16, 0, sizeof(h_x_bf16));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < S; j++)
            h_x_bf16[i * INTERMEDIATE_DIM + j] = f32_to_bf16(static_cast<float>(i * S + j + 1));

    uint16_t h_W_bf16[PROJ_SIZE];
    for (int i = 0; i < PROJ_SIZE; i++) h_W_bf16[i] = f32_to_bf16(h_W_f32_loaded[i]);

    uint16_t *d_W, *d_X;
    float *d_Y;
    HIP_CHECK(hipMalloc(&d_W, PROJ_SIZE * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_X, M * INTERMEDIATE_DIM * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_Y, M * H * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_W, h_W_bf16, PROJ_SIZE * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_X, h_x_bf16, M * INTERMEDIATE_DIM * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_Y, 0, M * H * sizeof(float)));

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    float alpha = 1.0f, beta = 0.0f;
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle,
        rocblas_operation_none, rocblas_operation_none,
        H, M, S,
        &alpha,
        d_W, rocblas_datatype_bf16_r, H,
        d_X, rocblas_datatype_bf16_r, INTERMEDIATE_DIM,
        &beta,
        d_Y, rocblas_datatype_f32_r, H,
        d_Y, rocblas_datatype_f32_r, H,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0
    ));

    float h_Y[M * H];
    HIP_CHECK(hipMemcpy(h_Y, d_Y, M * H * sizeof(float), hipMemcpyDeviceToHost));

    printf("=== Contract forward after checkpoint roundtrip ===\n");
    float expected[M * H] = {1.25f, 2.0f, 3.5f, 5.0f};
    bool fwd_ok = true;
    for (int i = 0; i < M * H; i++) {
        printf("  out[%d]=%.6f expected=%.6f %s\n", i, (double)h_Y[i], (double)expected[i],
               std::fabs(h_Y[i] - expected[i]) < 1e-4f ? "OK" : "FAIL");
        if (std::fabs(h_Y[i] - expected[i]) > 1e-4f) fwd_ok = false;
    }
    printf("Forward output: %s\n\n", fwd_ok ? "PASS" : "FAIL");
    ok &= fwd_ok;

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipFree(d_W)); HIP_CHECK(hipFree(d_X)); HIP_CHECK(hipFree(d_Y));

    printf("=== Test 4: down checkpoint roundtrip ===\n");
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
