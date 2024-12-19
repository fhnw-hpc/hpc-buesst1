from src.utils import (
    make_reconstructor,
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

RECO_SHAPE = (1000, 1000)
BLOCK_SIZE = (32, 32)
PIN_MEMORY = True

# because of tiling -> num threads must be >= tile size because otherwise not all elements will be loaded
assert (
    BLOCK_SIZE[0] * BLOCK_SIZE[1] >= TILE_SIZE
), "number of threads must be bigger than tile size"

if __name__ == "__main__":
    input = random_svd(RECO_SHAPE)
    input = tuple(list(input) + [min(RECO_SHAPE)])

    # for sanity check -> fp64 vs numpy
    isequal, _ = compare_kernels(
        input,
        make_reconstructor(kernel_globalmem_fp64, BLOCK_SIZE, PIN_MEMORY, timeit=True),
        reconstruct_svd_broadcast_timeit,
    )
    assert isequal

    isequal, _ = compare_kernels(
        input,
        make_reconstructor(kernel_sharedmem_fp64, BLOCK_SIZE, PIN_MEMORY, timeit=True),
        reconstruct_svd_broadcast_timeit,
    )
    assert isequal

    measurements = get_k_timings_from_kernels(
        input,
        [
            make_reconstructor(
                kernel_globalmem_fp64, BLOCK_SIZE, PIN_MEMORY, timeit=True
            ),
            make_reconstructor(
                kernel_globalmem_fp32, BLOCK_SIZE, PIN_MEMORY, timeit=True
            ),
            make_reconstructor(
                kernel_sharedmem_fp64, BLOCK_SIZE, PIN_MEMORY, timeit=True
            ),
            make_reconstructor(
                kernel_sharedmem_fp32, BLOCK_SIZE, PIN_MEMORY, timeit=True
            ),
        ],
        ["globalmem_fp64", "globalmem_fp32", "sharedmem_fp64", "sharedmem_fp32"],
    )

    print(measurements)
