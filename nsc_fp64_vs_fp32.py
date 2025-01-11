from src.utils import (
    make_reconstructor,
    reconstruct_svd_broadcast,
    compare_matrices,
    random_svd,
    get_n_timings_from_kernels,
)
from src.kernels.global_mem import fp32 as kernel_globalmem_fp32
from src.kernels.global_mem import fp64 as kernel_globalmem_fp64

INPUT_SIZE = (3072, 4096)  # medium matrix size
BlOCK_SIZE = (8, 16)  # ideal block size

if __name__ == "__main__":
    # Dieses experiment soll fp64 mit fp32 vergleichen. Dazu werden die kernel mit nsight compute profiled.

    # create input
    input = random_svd(INPUT_SIZE)
    input = tuple(list(input) + [min(INPUT_SIZE)])

    # create reco functions
    reco_fp64 = make_reconstructor(kernel_globalmem_fp64, BlOCK_SIZE)
    reco_fp32 = make_reconstructor(kernel_globalmem_fp32, BlOCK_SIZE)

    # reconstruct
    print("reco fp64 started")
    reco_fp64(*input)
    print("reco fp32 started")
    reco_fp32(*input)
