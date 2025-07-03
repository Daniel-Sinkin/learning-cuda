#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "blas_wrapper.cuh"
#include "common.cuh"
#include "cuarray.cuh"
#include "curand.cuh"
#include "metrics.cuh"

namespace DSCU {
namespace BLAS {

void example_dgemm() {
    auto handle = CublasHandle();

    // Array of matrix sizes to test
    constexpr int sizes[] = {32, 64, 96, 128, 256, 512};
    constexpr int n_iterations = 100;
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    std::cout << "DGEMM Performance Test\n";
    std::cout << "======================\n\n";

    for (int size : sizes) {
        std::cout << "Testing matrix size: " << size << "x" << size << "x" << size << "\n";

        // Allocate matrices
        CuArray<double> d_A(size * size);
        CuArray<double> d_B(size * size);
        CuArray<double> d_C(size * size);

        // Initialize with random values
        RANDOM::fill_with_random(d_A);
        RANDOM::fill_with_random(d_B);
        RANDOM::fill_with_random(d_C);

        // Warmup run
        for (int i = 0; i < 5; ++i) {
            cublasDgemm(handle.get(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                size, size, size,
                &alpha,
                d_A.get(), size,
                d_B.get(), size,
                &beta,
                d_C.get(), size);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // Test 1: Sync once after all iterations
        auto metric_sync_once = DgemmMetric("sync_once", size, size, size, n_iterations);
        metric_sync_once.start_timer();
        for (int i = 0; i < n_iterations; ++i) {
            cublasDgemm(handle.get(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                size, size, size,
                &alpha,
                d_A.get(), size,
                d_B.get(), size,
                &beta,
                d_C.get(), size);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        metric_sync_once.stop_timer();

        // Test 2: Sync after every iteration
        auto metric_sync_every = DgemmMetric("sync_every_iter", size, size, size, n_iterations);
        metric_sync_every.start_timer();
        for (int i = 0; i < n_iterations; ++i) {
            cublasDgemm(handle.get(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                size, size, size,
                &alpha,
                d_A.get(), size,
                d_B.get(), size,
                &beta,
                d_C.get(), size);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        metric_sync_every.stop_timer();

        // Print results
        std::cout << "  " << metric_sync_once << "\n";
        std::cout << "  " << metric_sync_every << "\n";

        // Calculate and display the synchronization overhead
        double overhead_percent = 100.0 * (metric_sync_every.elapsed_ns() - metric_sync_once.elapsed_ns()) / metric_sync_once.elapsed_ns();
        std::cout << "  Synchronization overhead: " << std::fixed << std::setprecision(1)
                  << overhead_percent << "%\n";

        std::cout << "\n";
    }

    // Additional test with larger matrices to show peak performance
    std::cout << "Testing larger matrices for peak performance:\n";
    std::cout << "============================================\n\n";

    constexpr int large_sizes[] = {1024, 2048, 4096};
    constexpr int n_iterations_large = 10; // Fewer iterations for large matrices

    for (int size : large_sizes) {
        std::cout << "Testing matrix size: " << size << "x" << size << "x" << size << "\n";

        CuArray<double> d_A(size * size);
        CuArray<double> d_B(size * size);
        CuArray<double> d_C(size * size);

        RANDOM::fill_with_random(d_A);
        RANDOM::fill_with_random(d_B);
        RANDOM::fill_with_random(d_C);

        // Warmup
        for (int i = 0; i < 3; ++i) {
            cublasDgemm(handle.get(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                size, size, size,
                &alpha,
                d_A.get(), size,
                d_B.get(), size,
                &beta,
                d_C.get(), size);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        auto metric = DgemmMetric("large_matrix", size, size, size, n_iterations_large);
        metric.start_timer();
        for (int i = 0; i < n_iterations_large; ++i) {
            cublasDgemm(handle.get(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                size, size, size,
                &alpha,
                d_A.get(), size,
                d_B.get(), size,
                &beta,
                d_C.get(), size);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        metric.stop_timer();

        std::cout << "  " << metric << "\n\n";
    }
}
} // namespace BLAS
} // namespace DSCU