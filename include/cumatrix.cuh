#pragma once

#include <vector>

#include "common.cuh"

namespace DSCU {

template <CudaCompatible T>
class CuMatrix {
public:
    CuMatrix() : rows_(0), cols_(0), ptr_(nullptr) {}

    CuMatrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), ptr_(nullptr) {
        if (rows_ == 0 || cols_ == 0) PANIC("rows and cols must be > 0");
        CHECK_CUDA(cudaMalloc(&ptr_, rows_ * cols_ * sizeof(T)));
    }

    ~CuMatrix() {
        if (ptr_) cudaFree(ptr_);
    }

    CuMatrix(const CuMatrix &) = delete;
    CuMatrix &operator=(const CuMatrix &) = delete;

    CuMatrix(CuMatrix &&other) noexcept { steal(std::move(other)); }
    CuMatrix &operator=(CuMatrix &&other) noexcept {
        if (this != &other) {
            release();
            steal(std::move(other));
        }
        return *this;
    }

    T *get() { return ptr_; }
    const T *get() const { return ptr_; }

    __host__ __device__ size_t rows() const { return rows_; }
    __host__ __device__ size_t cols() const { return cols_; }
    __host__ __device__ size_t size() const { return rows_ * cols_; }

    __host__ __device__ size_t ld() const { return cols_; }

    __device__ T &operator()(size_t r, size_t c) const {
        return ptr_[r * cols_ + c];
    }

    __device__ T *row_ptr(size_t r) const { return ptr_ + r * cols_; }

    void copy_from_host(const T *h_data) {
        CHECK_CUDA(cudaMemcpy(ptr_, h_data, size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    void copy_from_host(const std::vector<T> &h_vec) {
        if (h_vec.size() != size()) PANIC("host vector size mismatch");
        copy_from_host(h_vec.data());
    }

    void copy_to_host(T *h_data) const {
        CHECK_CUDA(cudaMemcpy(h_data, ptr_, size() * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void copy_to_host(std::vector<T> &h_vec) const {
        if (h_vec.size() != size()) PANIC("host vector size mismatch");
        copy_to_host(h_vec.data());
    }

    void zero() { CHECK_CUDA(cudaMemset(ptr_, 0, size() * sizeof(T))); }

    friend std::ostream &operator<<(std::ostream &os, const CuMatrix &m) {
        std::vector<T> host(m.size());
        m.copy_to_host(host);
        os << "[";
        for (size_t r = 0; r < m.rows_; ++r) {
            os << "[";
            for (size_t c = 0; c < m.cols_; ++c) {
                os << host[r * m.cols_ + c];
                if (c + 1 != m.cols_) os << ", ";
            }
            os << "]";
            if (r + 1 != m.rows_) os << ",\n ";
        }
        os << "]";
        return os;
    }

private:
    size_t rows_, cols_;
    T *ptr_;

    void steal(CuMatrix &&other) noexcept {
        rows_ = other.rows_;
        other.rows_ = 0;
        cols_ = other.cols_;
        other.cols_ = 0;
        ptr_ = other.ptr_;
        other.ptr_ = nullptr;
    }
    void release() noexcept {
        if (ptr_) cudaFree(ptr_);
    }
};

} // namespace DSCU
