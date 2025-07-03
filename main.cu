#include "src/example_batched_dgemm.cu"
#include "src/example_dgemm.cu"
#include "src/example_dgemm_matrix.cu"

constexpr bool RUN_EXAMPLE_DGEMM = false;
constexpr bool RUN_EXAMPLE_DGEMM_MATRIX = false;
constexpr bool RUN_EXAMPLE_BATCHED_DGEMM = false;

int main() {
    if (RUN_EXAMPLE_DGEMM) DSCU::BLAS::example_dgemm();
    if (RUN_EXAMPLE_DGEMM_MATRIX) DSCU::BLAS::example_dgemm_matrix();
    if (RUN_EXAMPLE_BATCHED_DGEMM) DSCU::BLAS::example_batched_dgemm();
    return 0;
}