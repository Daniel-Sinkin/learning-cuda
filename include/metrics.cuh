#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace DSCU {
namespace BLAS {

class MetricBase {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

public:
    MetricBase(const std::string &name, int num_iterations)
        : name_(name),
          num_iterations_(num_iterations),
          start_time_(std::nullopt),
          end_time_(std::nullopt) {}

    virtual ~MetricBase() = default;

    void start_timer() {
        start_time_ = Clock::now();
        end_time_ = std::nullopt;
    }

    void stop_timer() {
        if (!start_time_) {
            throw std::runtime_error("Cannot stop timer that was never started.");
        }
        end_time_ = Clock::now();
    }

    int64_t elapsed_ns() const {
        if (!start_time_ || !end_time_) {
            throw std::runtime_error("Cannot compute elapsed time; timer not fully set.");
        }
        return std::chrono::duration_cast<std::chrono::nanoseconds>(*end_time_ - *start_time_).count();
    }

    virtual double gflops() const = 0;
    virtual double memory_mb() const = 0;

    int iterations() const { return num_iterations_; }
    const std::string &name() const { return name_; }

protected:
    std::string name_;
    int num_iterations_;
    std::optional<TimePoint> start_time_;
    std::optional<TimePoint> end_time_;
};

class AxpyMetric : public MetricBase {
public:
    AxpyMetric(const std::string &name, int n, int num_iterations)
        : MetricBase(name, num_iterations),
          n_(n) {}

    AxpyMetric(int n, int num_iterations)
        : MetricBase("no_name", num_iterations),
          n_(n) {}

    double gflops() const override {
        int64_t elapsed = elapsed_ns();
        int64_t total_ops = 2LL * n_ * num_iterations_;

        return (static_cast<double>(total_ops) / 1e9) / (elapsed / 1e9);
    }

    double memory_mb() const override {
        return (2.0 * n_ * sizeof(double)) / (1024.0 * 1024.0);
    }

    int vector_size() const { return n_; }

    friend std::ostream &operator<<(std::ostream &os, const AxpyMetric &metric);

private:
    int n_;
};

class DgemmMetric : public MetricBase {
public:
    DgemmMetric(const std::string &name, int m, int n, int k, int num_iterations)
        : MetricBase(name, num_iterations),
          m_(m),
          n_(n),
          k_(k) {}

    DgemmMetric(int m, int n, int k, int num_iterations)
        : MetricBase("no_name", num_iterations),
          m_(m),
          n_(n),
          k_(k) {}

    double gflops() const override {
        int64_t elapsed = elapsed_ns();
        int64_t total_ops = 2LL * m_ * n_ * k_ * num_iterations_;

        return (static_cast<double>(total_ops) / 1e9) / (elapsed / 1e9);
    }

    double memory_mb() const override {
        size_t total_elements = static_cast<size_t>(m_) * k_ +
                                static_cast<size_t>(k_) * n_ +
                                static_cast<size_t>(m_) * n_;
        return (total_elements * sizeof(double)) / (1024.0 * 1024.0);
    }

    double theoretical_peak_gflops() const {
#ifdef CUDA_ARCH
#if CUDA_ARCH == 80           // A100 (sm_80)
        return 19.5 * 1000.0; // 19.5 TFLOPS
#elif CUDA_ARCH == 70         // V100 (sm_70)
        return 7.8 * 1000.0; // 7.8 TFLOPS
#elif CUDA_ARCH == 60         // P100 (sm_60)
        return 5.3 * 1000.0; // 5.3 TFLOPS
#elif CUDA_ARCH == 75         // T4 / Quadro RTX 8000 (sm_75)
        return 0.5 * 1000.0; // ~0.5 TFLOPS (FP64 limited on consumer Turing)
#else                         // Unknown arch
        return 0.0; // 0 prevents divide-by-zero
#endif
#else
        // CUDA_ARCH not defined, defaulting to A100
        return 19.5 * 1000.0;
#endif
    }

    double efficiency_percent() const {
        double peak = theoretical_peak_gflops();
        if (peak <= 0.0) return 0.0;
        return (gflops() / peak) * 100.0;
    }

    int m() const { return m_; }
    int n() const { return n_; }
    int k() const { return k_; }

    friend std::ostream &operator<<(std::ostream &os, const DgemmMetric &metric);

private:
    int m_, n_, k_;
};

inline std::ostream &operator<<(std::ostream &os, const AxpyMetric &metric) {
    try {
        os << std::fixed << std::setprecision(3);
        os << "[" << metric.name() << "]";
        os << "[Duration: " << metric.elapsed_ns() << " ns]";
        os << "[Performance: " << metric.gflops() << " GFLOPS]";
        os << "[Memory: " << metric.memory_mb() << " MB]";
        os << "[Vector Size: " << metric.vector_size() << "]";
        os << "[Iterations: " << metric.iterations() << "]";

        return os;
    } catch (const std::exception &e) {
        os << "[" << metric.name() << "][Error: " << e.what() << "]";
        return os;
    }
}

inline std::ostream &operator<<(std::ostream &os, const DgemmMetric &metric) {
    try {
        os << std::fixed << std::setprecision(3);
        os << "[" << metric.name() << "]";
        os << "[Duration: " << metric.elapsed_ns() << " ns]";
        os << "[Performance: " << metric.gflops() << " GFLOPS]";
        os << "[Efficiency: " << std::setprecision(1) << metric.efficiency_percent() << "%]";
        os << "[Memory: " << std::setprecision(3) << metric.memory_mb() << " MB]";
        os << "[DGEMM: (" << metric.m() << "x" << metric.k() << ") x ("
           << metric.k() << "x" << metric.n() << ") -> ("
           << metric.m() << "x" << metric.n() << ")]";
        os << "[Iterations: " << metric.iterations() << "]";

        return os;
    } catch (const std::exception &e) {
        os << "[" << metric.name() << "][Error: " << e.what() << "]";
        return os;
    }
}

inline std::ostream &operator<<(std::ostream &os, const MetricBase &metric) {
    try {
        os << std::fixed << std::setprecision(3);
        os << "[" << metric.name() << "]";
        os << "[Duration: " << metric.elapsed_ns() << " ns]";
        os << "[Performance: " << metric.gflops() << " GFLOPS]";
        os << "[Memory: " << metric.memory_mb() << " MB]";
        os << "[Iterations: " << metric.iterations() << "]";

        return os;
    } catch (const std::exception &e) {
        os << "[" << metric.name() << "][Error: " << e.what() << "]";
        return os;
    }
}

class BatchedDgemmMetric : public MetricBase {
public:
    BatchedDgemmMetric(const std::string &name, int batch_size, int m, int n, int k, int num_iterations)
        : MetricBase(name, num_iterations),
          batch_size_(batch_size),
          m_(m),
          n_(n),
          k_(k) {}

    BatchedDgemmMetric(int batch_size, int m, int n, int k, int num_iterations)
        : MetricBase("no_name", num_iterations),
          batch_size_(batch_size),
          m_(m),
          n_(n),
          k_(k) {}

    double gflops() const override {
        int64_t elapsed = elapsed_ns();
        // Each DGEMM in the batch performs 2*m*n*k operations
        int64_t total_ops = 2LL * m_ * n_ * k_ * batch_size_ * num_iterations_;

        return (static_cast<double>(total_ops) / 1e9) / (elapsed / 1e9);
    }

    double memory_mb() const override {
        // Memory for all matrices in the batch: batch_size * (A + B + C)
        size_t elements_per_batch = static_cast<size_t>(m_) * k_ +
                                    static_cast<size_t>(k_) * n_ +
                                    static_cast<size_t>(m_) * n_;
        size_t total_elements = batch_size_ * elements_per_batch;
        return (total_elements * sizeof(double)) / (1024.0 * 1024.0);
    }

    double theoretical_peak_gflops() const {
#ifdef CUDA_ARCH
#if CUDA_ARCH == 80           // A100 (sm_80)
        return 19.5 * 1000.0; // 19.5 TFLOPS
#elif CUDA_ARCH == 70         // V100 (sm_70)
        return 7.8 * 1000.0; // 7.8 TFLOPS
#elif CUDA_ARCH == 60         // P100 (sm_60)
        return 5.3 * 1000.0; // 5.3 TFLOPS
#elif CUDA_ARCH == 75         // T4 / Quadro RTX 8000 (sm_75)
        return 0.5 * 1000.0; // ~0.5 TFLOPS (FP64 limited on consumer Turing)
#else                         // Unknown arch
        return 0.0; // 0 prevents divide-by-zero
#endif
#else
        // CUDA_ARCH not defined, defaulting to A100
        return 19.5 * 1000.0;
#endif
    }

    double efficiency_percent() const {
        double peak = theoretical_peak_gflops();
        if (peak <= 0.0) return 0.0;
        return (gflops() / peak) * 100.0;
    }

    int batch_size() const { return batch_size_; }
    int m() const { return m_; }
    int n() const { return n_; }
    int k() const { return k_; }

    friend std::ostream &operator<<(std::ostream &os, const BatchedDgemmMetric &metric);

private:
    int batch_size_;
    int m_, n_, k_;
};

inline std::ostream &operator<<(std::ostream &os, const BatchedDgemmMetric &metric) {
    try {
        os << std::fixed << std::setprecision(3);
        os << "[" << metric.name() << "]";
        os << "[Duration: " << metric.elapsed_ns() << " ns]";
        os << "[Performance: " << metric.gflops() << " GFLOPS]";
        os << "[Efficiency: " << std::setprecision(1) << metric.efficiency_percent() << "%]";
        os << "[Memory: " << std::setprecision(3) << metric.memory_mb() << " MB]";
        os << "[Batch: " << metric.batch_size() << " × DGEMM: ("
           << metric.m() << "×" << metric.k() << ") × ("
           << metric.k() << "×" << metric.n() << ") → ("
           << metric.m() << "×" << metric.n() << ")]";
        os << "[Iterations: " << metric.iterations() << "]";

        return os;
    } catch (const std::exception &e) {
        os << "[" << metric.name() << "][Error: " << e.what() << "]";
        return os;
    }
}

} // namespace BLAS
} // namespace DSCU