#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "blas_wrapper.cuh"
#include "common.cuh"
#include "cu_ptr_array.cuh"
#include "cuarray.cuh"
#include "curand.cuh"
#include "metrics.cuh"

namespace DSCU {
namespace BLAS {

void example_batched_dgemm() {
    auto handle = CublasHandle();

    std::cout << "Batched DGEMM Performance Test\n";
    std::cout << "==============================\n\n";

    // Test different batch sizes and matrix sizes
    constexpr int batch_sizes[] = {10, 50, 100, 500, 1000};
    constexpr int matrix_sizes[] = {32, 64, 128, 256};
    constexpr int n_iterations = 10;
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    for (int batch_size : batch_sizes) {
        std::cout << "Batch size: " << batch_size << "\n";
        std::cout << "-----------------\n";

        for (int size : matrix_sizes) {
            std::cout << "  Matrix size: " << size << "x" << size << "x" << size << "\n";

            // Calculate total memory required
            size_t total_elements = batch_size * size * size * 3; // A, B, C matrices
            size_t memory_mb = (total_elements * sizeof(double)) / (1024 * 1024);

            // Skip if it would use too much memory (>4GB)
            if (memory_mb > 4096) {
                std::cout << "    Skipping - would require " << memory_mb << " MB\n\n";
                continue;
            }

            // Allocate memory for all matrices in the batch
            CuArray<double> d_A(batch_size * size * size);
            CuArray<double> d_B(batch_size * size * size);
            CuArray<double> d_C(batch_size * size * size);

            // Initialize with random values
            RANDOM::fill_with_random(d_A);
            RANDOM::fill_with_random(d_B);
            RANDOM::fill_with_random(d_C);

            // Test 1: Baseline - Loop over individual DGEMMs
            auto metric_baseline = BatchedDgemmMetric("baseline_loop", batch_size, size, size, size, n_iterations);
            metric_baseline.start_timer();
            for (int iter = 0; iter < n_iterations; ++iter) {
                for (int i = 0; i < batch_size; ++i) {
                    cublasDgemm(handle.get(),
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        size, size, size,
                        &alpha,
                        d_A.get() + i * size * size, size,
                        d_B.get() + i * size * size, size,
                        &beta,
                        d_C.get() + i * size * size, size);
                }
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            metric_baseline.stop_timer();

            // Test 2: Batched DGEMM
            // Use CuPtrArray for device pointer arrays
            CuPtrArray<double> d_a_array(batch_size);
            CuPtrArray<double> d_b_array(batch_size);
            CuPtrArray<double> d_c_array(batch_size);

            setupBatchedPointers(d_a_array, d_b_array, d_c_array,
                d_A, d_B, d_C,
                batch_size, size, size, size);

            auto metric_batched = BatchedDgemmMetric("cublas_batched", batch_size, size, size, size, n_iterations);
            metric_batched.start_timer();
            for (int iter = 0; iter < n_iterations; ++iter) {
                cublasDgemmBatched(handle.get(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    size, size, size,
                    &alpha,
                    d_a_array.get(), size,
                    d_b_array.get(), size,
                    &beta,
                    d_c_array.get(), size,
                    batch_size);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            metric_batched.stop_timer();

            // Test 3: Strided Batched DGEMM
            auto metric_strided = BatchedDgemmMetric("strided_batched", batch_size, size, size, size, n_iterations);
            metric_strided.start_timer();
            for (int iter = 0; iter < n_iterations; ++iter) {
                cublasDgemmStridedBatched(handle.get(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    size, size, size,
                    &alpha,
                    d_A.get(), size, size * size,
                    d_B.get(), size, size * size,
                    &beta,
                    d_C.get(), size, size * size,
                    batch_size);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            metric_strided.stop_timer();

            // Print results
            std::cout << "    " << metric_baseline << "\n";
            std::cout << "    " << metric_batched << "\n";
            std::cout << "    " << metric_strided << "\n";

            // Calculate speedups
            double batched_speedup = static_cast<double>(metric_baseline.elapsed_ns()) /
                                     metric_batched.elapsed_ns();
            double strided_speedup = static_cast<double>(metric_baseline.elapsed_ns()) /
                                     metric_strided.elapsed_ns();

            std::cout << "    Speedup (batched vs baseline): " << std::fixed
                      << std::setprecision(2) << batched_speedup << "x\n";
            std::cout << "    Speedup (strided vs baseline): " << std::fixed
                      << std::setprecision(2) << strided_speedup << "x\n\n";
        }
        std::cout << "\n";
    }

    // Additional test with larger matrices and smaller batches
    std::cout << "Testing larger matrices with smaller batches:\n";
    std::cout << "============================================\n\n";

    constexpr int large_batch_sizes[] = {1, 5, 10, 20};
    constexpr int large_matrix_sizes[] = {512, 1024};

    for (int batch_size : large_batch_sizes) {
        for (int size : large_matrix_sizes) {
            std::cout << "Batch: " << batch_size << ", Matrix: " << size << "x" << size << "\n";

            CuArray<double> d_A(batch_size * size * size);
            CuArray<double> d_B(batch_size * size * size);
            CuArray<double> d_C(batch_size * size * size);

            RANDOM::fill_with_random(d_A);
            RANDOM::fill_with_random(d_B);
            RANDOM::fill_with_random(d_C);

            // Only test strided batched for large matrices (most efficient)
            auto metric = BatchedDgemmMetric("large_strided", batch_size, size, size, size, 5);
            metric.start_timer();
            for (int iter = 0; iter < 5; ++iter) {
                cublasDgemmStridedBatched(handle.get(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    size, size, size,
                    &alpha,
                    d_A.get(), size, size * size,
                    d_B.get(), size, size * size,
                    &beta,
                    d_C.get(), size, size * size,
                    batch_size);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            metric.stop_timer();

            std::cout << "  " << metric << "\n\n";
        }
    }
}

} // namespace BLAS
} // namespace DSCU