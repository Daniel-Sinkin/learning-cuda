#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "blas_wrapper.cuh"
#include "common.cuh"
#include "cumatrix.cuh"
#include "curand.cuh"
#include "metrics.cuh"

namespace DSCU {
namespace BLAS {

void example_dgemm_matrix() {
    CublasHandle handle;
    constexpr int sizes[] = {32, 64, 96, 128, 256, 512};
    constexpr int n_iterations = 100;
    constexpr double alpha = 1.0, beta = 0.0;

    std::cout << "DGEMM Performance Test (CuMatrix)\n"
                 "=================================\n\n";

    for (int n : sizes) {
        std::cout << "Size " << n << " x " << n << "\n";
        CuMatrix<double> d_A(n, n), d_B(n, n), d_C(n, n);

        RANDOM::fill_with_random(d_A);
        RANDOM::fill_with_random(d_B);
        RANDOM::fill_with_random(d_C);

        for (int i = 0; i < 5; ++i)
            CHECK_CUBLAS(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_A.get(), d_A.ld(),
                d_B.get(), d_B.ld(),
                &beta,
                d_C.get(), d_C.ld()));
        CHECK_CUDA(cudaDeviceSynchronize());

        auto m_once = DgemmMetric("sync_once", n, n, n, n_iterations);
        m_once.start_timer();
        for (int i = 0; i < n_iterations; ++i)
            CHECK_CUBLAS(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_A.get(), d_A.ld(),
                d_B.get(), d_B.ld(),
                &beta,
                d_C.get(), d_C.ld()));
        CHECK_CUDA(cudaDeviceSynchronize());
        m_once.stop_timer();

        auto m_every = DgemmMetric("sync_every_iter", n, n, n, n_iterations);
        m_every.start_timer();
        for (int i = 0; i < n_iterations; ++i) {
            CHECK_CUBLAS(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_A.get(), d_A.ld(),
                d_B.get(), d_B.ld(),
                &beta,
                d_C.get(), d_C.ld()));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        m_every.stop_timer();

        double overhead = 100.0 * (m_every.elapsed_ns() - m_once.elapsed_ns()) / m_once.elapsed_ns();

        std::cout << "  " << m_once << "\n"
                  << "  " << m_every << "\n"
                  << "  sync overhead: " << std::fixed << std::setprecision(1)
                  << overhead << "%\n\n";
    }

    constexpr int big[] = {1024, 2048, 4096};
    constexpr int it_big = 10;

    std::cout << "Large matrices:\n--------------\n";
    for (int n : big) {
        std::cout << "Size " << n << " x " << n << "\n";
        CuMatrix<double> A(n, n), B(n, n), C(n, n);

        RANDOM::fill_with_random(A);
        RANDOM::fill_with_random(B);
        RANDOM::fill_with_random(C);

        for (int i = 0; i < 3; ++i)
            CHECK_CUBLAS(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, &alpha,
                A.get(), A.ld(),
                B.get(), B.ld(),
                &beta,
                C.get(), C.ld()));
        CHECK_CUDA(cudaDeviceSynchronize());

        auto m = DgemmMetric("large", n, n, n, it_big);
        m.start_timer();
        for (int i = 0; i < it_big; ++i)
            CHECK_CUBLAS(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, &alpha,
                A.get(), A.ld(),
                B.get(), B.ld(),
                &beta,
                C.get(), C.ld()));
        CHECK_CUDA(cudaDeviceSynchronize());
        m.stop_timer();

        std::cout << "  " << m << "\n\n";
    }
}

} // namespace BLAS
} // namespace DSCU
