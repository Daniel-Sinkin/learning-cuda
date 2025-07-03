#pragma once

#include "common.cuh"
#include "cuarray.cuh"
#include "cumatrix.cuh"

#include <curand.h>
#include <curand_kernel.h>

namespace DSCU {
namespace RANDOM {
template <CudaCompatible T>
class CuRandGenerator {
private:
    curandGenerator_t gen_;

public:
    CuRandGenerator() {
        CHECK_CURAND(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL));
    }

    ~CuRandGenerator() {
        curandDestroyGenerator(gen_);
    }

    void generate_normal(T *data, size_t size, T mu, T sigma) {
        if constexpr (std::is_same_v<T, float>) {
            CHECK_CURAND(curandGenerateNormal(gen_, data, size, mu, sigma));
        } else if constexpr (std::is_same_v<T, double>) {
            CHECK_CURAND(curandGenerateNormalDouble(gen_, data, size, mu, sigma));
        }
    }
};

template <CudaCompatible T>
void generate_randn(CuArray<T> &array, T mu = 0.0, T sigma = 1.0) {
    static_assert(std::is_floating_point_v<T>, "generate_randn only works with floating point types");

    static thread_local CuRandGenerator<T> generator;
    generator.generate_normal(array.get(), array.get_size(), mu, sigma);
}
template <CudaCompatible T>
void generate_randn(CuMatrix<T> &matrix, T mu = T(0.0), T sigma = T(1.0)) {
    static_assert(std::is_floating_point_v<T>,
        "generate_randn only works with floating-point types");

    static thread_local CuRandGenerator<T> generator;
    generator.generate_normal(matrix.get(), matrix.size(), mu, sigma);
}

template <CudaCompatible T>
void fill_with_random(CuArray<T> &array) {
    generate_randn(array, T(0), T(1));
}

template <CudaCompatible T>
void fill_with_random(CuMatrix<T> &matrix) {
    generate_randn(matrix, T(0), T(1));
}
} // namespace RANDOM
} // namespace DSCU