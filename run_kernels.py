from src.utils import (
    make_reconstructor,
    reconstruct_svd_broadcast,
    compare_matrices,
    reconstruct_svd_broadcast_timeit,
    compare_kernels,
    random_svd,
    get_timings,
    get_k_timings,
    get_k_timings_from_kernels,
)
from src.kernels.global_mem import fp32 as kernel_globalmem_fp32
from src.kernels.global_mem import fp64 as kernel_globalmem_fp64
from src.kernels.shared_mem import fp32 as kernel_sharedmem_fp32
from src.kernels.shared_mem import fp64 as kernel_sharedmem_fp64
from src.kernels.shared_mem import TILE_SIZE

RECO_SHAPE = (768, 1024)
BLOCK_SIZE = (8, 16)
PIN_MEMORY = True

# because of tiling -> num threads must be >= tile size because otherwise not all elements will be loaded
assert (
    BLOCK_SIZE[0] * BLOCK_SIZE[1] >= TILE_SIZE
), "number of threads must be bigger than tile size"

if __name__ == "__main__":
    input = random_svd(RECO_SHAPE)
    input = tuple(list(input) + [min(RECO_SHAPE)])

    ref = reconstruct_svd_broadcast(*input)

    reco_func = make_reconstructor(kernel_globalmem_fp64, BLOCK_SIZE)
    print("globalmem fp64 reco precise: ", compare_matrices(reco_func(*input), ref))

    reco_func = make_reconstructor(kernel_globalmem_fp32, BLOCK_SIZE)
    print("globalmem fp32 reco precise: ", compare_matrices(reco_func(*input), ref))

    reco_func = make_reconstructor(kernel_sharedmem_fp64, BLOCK_SIZE)
    print("sharedmem fp64 reco precise: ", compare_matrices(reco_func(*input), ref))

    reco_func = make_reconstructor(kernel_sharedmem_fp32, BLOCK_SIZE)
    print("sharedmem fp32 reco precise: ", compare_matrices(reco_func(*input), ref))
