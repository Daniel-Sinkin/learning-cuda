#pragma once

#include <iostream>
#include <ostream>
#include <vector>

#include "common.cuh"
namespace DSCU {
template <CudaCompatible T>
class CuArray {
public:
    CuArray(size_t size) : size_(size), ptr_(nullptr) {
        CHECK_CUDA(cudaMalloc(&ptr_, size_ * sizeof(T)));
    }

    ~CuArray() {
        if (ptr_) cudaFree(ptr_);
    }

    T *get() const { return ptr_; }
    T *operator->() const { return ptr_; }
    T &operator*() const { return *ptr_; }

    __host__ __device__ size_t get_size() const { return size_; }
    __device__ T &operator[](size_t idx) const { return ptr_[idx]; }

    void copy_from_host(const T *h_data, size_t count) {
        if (count > size_) PANIC("count too big");
        CHECK_CUDA(cudaMemcpy(ptr_, h_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void copy_from_host(const std::vector<T> &h_data) {
        copy_from_host(h_data.data(), h_data.size());
    }
    void copy_to_host(T *h_data, size_t count) const {
        CHECK_CUDA(cudaMemcpy(h_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void copy_to_host(T *h_data) const {
        CHECK_CUDA(cudaMemcpy(h_data, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void copy_to_host(std::vector<T> &h_data) const {
        if (h_data.size() != size_) PANIC("host std::vector size mismatch");
        copy_to_host(h_data.data(), size_);
    }

    CuArray(const CuArray &) = delete;
    CuArray &operator=(const CuArray &) = delete;

    CuArray(CuArray &&other) noexcept
        : size_(other.size_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    CuArray &operator=(CuArray &&other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    friend std::ostream &operator<<(std::ostream &os, const CuArray &arr) {
        std::vector<T> host_data(arr.size_);
        arr.copy_to_host(host_data);
        os << "[";
        for (size_t i = 0; i < host_data.size(); ++i) {
            os << host_data[i];
            if (i + 1 != host_data.size()) os << ", ";
        }
        os << "]";
        return os;
    }

private:
    size_t size_;
    T *ptr_;
};
} // namespace DSCU
