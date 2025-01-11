from src.utils import (
    make_reconstructor,
    reconstruct_svd_broadcast,
    compare_matrices,
    random_svd,
    get_n_timings_from_kernels,
)
from src.kernels.global_mem import fp32 as kernel_globalmem_fp32
from src.kernels.global_mem import fp64 as kernel_globalmem_fp64
from src.kernels.global_mem import fp64_fma as kernel_globalmem_fp64_fma

INPUT_SIZE = (1080, 1920)  # small matrix size (bigger crashes my GPU)
BlOCK_SIZE = (8, 16)  # ideal block size

if __name__ == "__main__":
    # Dieses experiment soll fp64 mit fp32 vergleichen. Dazu werden die kernel mit nsight compute profiled.

    # create input
    input = random_svd(INPUT_SIZE)
    input = tuple(list(input) + [min(INPUT_SIZE)])

    # reference
    ref = reconstruct_svd_broadcast(*input)

    # create reco functions
    reco_fp64 = make_reconstructor(kernel_globalmem_fp64, BlOCK_SIZE)
    reco_fp32 = make_reconstructor(kernel_globalmem_fp32, BlOCK_SIZE)
    reco_fp64_fma = make_reconstructor(kernel_globalmem_fp64_fma, BlOCK_SIZE)

    # reconstruct
    print("reco fp64 (non fma) started")
    print("result precise: ", compare_matrices(ref, reco_fp64(*input)))

    print("reco fp32 (non fma) started")
    print("result precise: ", compare_matrices(ref, reco_fp32(*input)))

    print("reco fp64 (fma) started")
    print("result precise: ", compare_matrices(ref, reco_fp64_fma(*input)))
