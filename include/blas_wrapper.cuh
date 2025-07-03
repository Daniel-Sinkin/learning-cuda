#pragma once

#include <cublas_v2.h>

#include "common.cuh"

namespace DSCU {
namespace BLAS {
class CublasHandle {
public:
    CublasHandle() {
        CHECK_CUBLAS(cublasCreate(&handle_));
    }
    ~CublasHandle() {
        CHECK_CUBLAS(cublasDestroy(handle_));
    }
    [[nodiscard]] cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_;
};
} // namespace BLAS
} // namespace DSCU