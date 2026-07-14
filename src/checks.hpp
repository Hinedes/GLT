#pragma once

#include <iostream>
#include <stdexcept>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>
#include <c10/hip/HIPStream.h>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << hipGetErrorString(err) << "\"" << std::endl; \
            throw std::runtime_error("HIP error"); \
        } \
    } while (0)

#define HIPBLASLT_CHECK(call) \
    do { \
        hipblasStatus_t err = call; \
        if (err != HIPBLAS_STATUS_SUCCESS) { \
            std::cerr << "hipBLASLt error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << std::endl; \
            throw std::runtime_error("hipBLASLt error"); \
        } \
    } while (0)

static inline hipStream_t current_hip_stream() {
    return c10::hip::getCurrentHIPStream().stream();
}

// --------------------------------------------------------------------------
// rocBLAS Error Checker — Wrapper with file:line diagnostics
// --------------------------------------------------------------------------
inline void rocblas_check(rocblas_status status, const char* file, int line) {
    if (status != rocblas_status_success) {
        std::cerr << "rocBLAS error at " << file << ":" << line
                  << " status=" << static_cast<int>(status) << std::endl;
        throw std::runtime_error("rocBLAS error");
    }
}
#define ROCBLAS_CHECK(call) rocblas_check(call, __FILE__, __LINE__)
