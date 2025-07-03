#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import sys

matrix_sizes = [
    16,
    32,
    48,
    64,
    96,
    108,
    128,
    160,
    192,
    216,
    224,
    256,
    324,
    512,
    768,
    1024,
]


def describe_matrix(matrix: np.ndarray, name: str) -> None:
    print(f"{name}:")
    print(f"  Shape     : {matrix.shape}")
    print(f"  Dtype     : {matrix.dtype}")
    print(f"  Min       : {matrix.min():.4e}")
    print(f"  Max       : {matrix.max():.4e}")
    print(f"  Mean      : {matrix.mean():.4e}")
    print(f"  Std Dev   : {matrix.std():.4e}")
    print(f"  C-contig? : {matrix.flags['C_CONTIGUOUS']}")
    print(f"  F-contig? : {matrix.flags['F_CONTIGUOUS']}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate random square matrices and save them in a .npz archive."
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="where to write the .npz file (e.g. build/matrices.npz)",
    )
    args = p.parse_args()

    output_path: Path = args.output
    # ensure parent dirs exist
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {output_path.parent}: {e}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(seed=0)
    matrix_dict = {}
    for size in matrix_sizes:
        # Fortran-order to play nicely with BLAS
        m0 = rng.uniform(-1.0, 1.0, size=(size, size)).astype(np.float64, order="C")
        m1 = rng.uniform(-1.0, 1.0, size=(size, size)).astype(np.float64, order="C")

        key0 = f"{size}x{size}_0"
        key1 = f"{size}x{size}_1"

        describe_matrix(m0, key0)
        describe_matrix(m1, key1)

        matrix_dict[key0] = m0
        matrix_dict[key1] = m1

    np.savez(output_path, **matrix_dict)
    print(f"Saved all matrices to {output_path.resolve()}")


if __name__ == "__main__":
    main()
