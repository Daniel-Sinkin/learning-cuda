#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cnpy.h"

#include "blas_wrapper.cuh"
#include "common.cuh"
#include "cuarray.cuh"
#include "curand.cuh"
#include "metrics.cuh"

#include "src/example_dgemm.cu"
#include "src/example_dgemm_matrix.cu"

constexpr bool RUN_EXAMPLE_DGEMM = false;
constexpr bool RUN_EXAMPLE_DGEMM_MATRIX = false;

void setup(const char *npz_path) {
    try {
        cnpy::npz_t npz = cnpy::npz_load(npz_path);
        const char *key = "512x512_0";
        const cnpy::NpyArray &arr = npz[key];
        std::cout << "[PID " << getpid() << "] " << key << " → "
                  << arr.shape[0] << " x " << arr.shape[1] << '\n';
    } catch (const std::exception &e) {
        std::cerr << "[PID " << getpid() << "] Failed to load " << npz_path
                  << " : " << e.what() << '\n';
    }
}

void hello_world() {
    std::cout << "Hello from process ID: " << getpid() << "\n";
}

int main(int argc, char **argv) {
    if (RUN_EXAMPLE_DGEMM) DSCU::BLAS::example_dgemm();
    if (RUN_EXAMPLE_DGEMM_MATRIX) DSCU::BLAS::example_dgemm_matrix();

    const char *npz_path = (argc > 1) ? argv[1] : "matrices.npz";
    if (!std::filesystem::exists(npz_path)) {
        std::cerr << "Error: cannot find file \"" << npz_path << "\"\n";
        return 1;
    }

    using clock = std::chrono::high_resolution_clock;
    constexpr size_t NUM_PROCESSES = 4;
    std::array<pid_t, NUM_PROCESSES> pids;
    int pipefd[2];

    if (pipe(pipefd) == -1) {
        std::cerr << "Failed to create pipe\n";
        return 1;
    }

    for (size_t i = 0; i < NUM_PROCESSES; ++i) {
        pid_t pid = fork();
        if (pid < 0) {
            std::cerr << "Failed to fork\n";
            return 1;
        } else if (pid == 0) {
            close(pipefd[0]);
            setup(npz_path);
            hello_world();
            char done = 'x';
            write(pipefd[1], &done, 1);
            return 0;
        } else {
            pids[i] = pid;
        }
    }

    close(pipefd[1]);
    char buf;
    for (size_t i = 0; i < NUM_PROCESSES; ++i)
        read(pipefd[0], &buf, 1);
    std::cout << "Setup for all processes is done, now we time the execution\n";
    close(pipefd[0]);

    auto start = clock::now();
    for (pid_t pid : pids)
        waitpid(pid, nullptr, 0);
    auto end = clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    std::cout << "All child processes completed in " << duration << " s.\n";
}