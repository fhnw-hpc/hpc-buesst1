from src.utils import (
    make_reconstructor,
    reconstruct_svd_broadcast,
    compare_matrices,
    random_svd,
    get_n_timings_from_kernels,
)
from src.kernels.shared_mem import fp32 as kernel_sharedmem_fp32

INPUT_SIZE = (1080, 1920)  # small matrix size (bigger crashes my GPU)
BlOCK_SIZE1 = (8, 16)  # ideal block size
BlOCK_SIZE2 = (4, 32)  # ideal block size (for coalescing)

if __name__ == "__main__":
    # Dieses experiment soll fp64 mit fp32 vergleichen. Dazu werden die kernel mit nsight compute profiled.

    # create input
    input = random_svd(INPUT_SIZE)
    input = tuple(list(input) + [min(INPUT_SIZE)])

    # reference
    ref = reconstruct_svd_broadcast(*input)

    # create reco functions
    reco_fp32 = make_reconstructor(kernel_sharedmem_fp32, BlOCK_SIZE1)
    reco_fp32_coalescing = make_reconstructor(kernel_sharedmem_fp32, BlOCK_SIZE2)

    # reconstruct
    print("reco (shared memory) fp32 started")
    print("result precise: ", compare_matrices(ref, reco_fp32(*input)))

    print("reco (shared memory) fp32 (more ideal coalescing) started")
    print("result precise: ", compare_matrices(ref, reco_fp32_coalescing(*input)))
