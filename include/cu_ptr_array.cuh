#pragma once

#include <vector>

#include "common.cuh"

namespace DSCU {

// Specialized class for device arrays of pointers
template <typename T>
class CuPtrArray {
public:
    explicit CuPtrArray(size_t size) : size_(size), ptr_(nullptr) {
        CHECK_CUDA(cudaMalloc(&ptr_, size_ * sizeof(T *)));
    }

    ~CuPtrArray() {
        if (ptr_) cudaFree(ptr_);
    }

    // Set pointers from host arrays
    void set_from_base(const T *base_ptr, size_t stride, size_t count) {
        if (count > size_) PANIC("count exceeds array size");

        std::vector<T *> h_ptrs(count);
        for (size_t i = 0; i < count; ++i) {
            h_ptrs[i] = const_cast<T *>(base_ptr + i * stride);
        }

        CHECK_CUDA(cudaMemcpy(ptr_, h_ptrs.data(),
            count * sizeof(T *), cudaMemcpyHostToDevice));
    }

    // Set pointers from vector of host pointers
    void set_from_host(const std::vector<T *> &h_ptrs) {
        if (h_ptrs.size() > size_) PANIC("vector size exceeds array size");

        CHECK_CUDA(cudaMemcpy(ptr_, h_ptrs.data(),
            h_ptrs.size() * sizeof(T *), cudaMemcpyHostToDevice));
    }

    // Get raw pointer for cuBLAS calls
    T **get() { return ptr_; }
    const T *const *get() const { return ptr_; }

    size_t size() const { return size_; }

    // Delete copy operations
    CuPtrArray(const CuPtrArray &) = delete;
    CuPtrArray &operator=(const CuPtrArray &) = delete;

    // Move operations
    CuPtrArray(CuPtrArray &&other) noexcept
        : size_(other.size_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CuPtrArray &operator=(CuPtrArray &&other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

private:
    size_t size_;
    T **ptr_;
};

template <typename T>
void setupBatchedPointers(CuPtrArray<T> &a_array,
    CuPtrArray<T> &b_array,
    CuPtrArray<T> &c_array,
    const CuArray<T> &base_a,
    const CuArray<T> &base_b,
    CuArray<T> &base_c,
    int batch_size, int m, int k, int n) {
    a_array.set_from_base(base_a.get(), m * k, batch_size);
    b_array.set_from_base(base_b.get(), k * n, batch_size);
    c_array.set_from_base(base_c.get(), m * n, batch_size);
}

} // namespace DSCU