from src.utils import (
    make_reconstructor,
    reconstruct_svd_broadcast,
    compare_matrices,
    random_svd,
)
from src.kernels.global_mem import fp32 as kernel_globalmem_fp32
from src.kernels.global_mem import fp64 as kernel_globalmem_fp64
from src.kernels.shared_mem import fp32 as kernel_sharedmem_fp32
from src.kernels.shared_mem import fp64 as kernel_sharedmem_fp64

RECO_SHAPE = (
    256,
    342,
)  #  (768, 1024) -> if kernel size is too big the kernels are run in series
BLOCK_SIZE = (8, 16)
PIN_MEMORY = True
NUM_STREAMS = 50

if __name__ == "__main__":
    input = random_svd(RECO_SHAPE)
    input = tuple(list(input) + [min(RECO_SHAPE)])

    ref = reconstruct_svd_broadcast(*input)

    reco_func = make_reconstructor(kernel_globalmem_fp64, BLOCK_SIZE, PIN_MEMORY)
    print("globalmem fp64 reco precise: ", compare_matrices(reco_func(*input), ref))

    reco_func = make_reconstructor(kernel_globalmem_fp32, BLOCK_SIZE, PIN_MEMORY)
    print("globalmem fp32 reco precise: ", compare_matrices(reco_func(*input), ref))

    reco_func = make_reconstructor(kernel_sharedmem_fp64, BLOCK_SIZE, PIN_MEMORY)
    print("sharedmem fp64 reco precise: ", compare_matrices(reco_func(*input), ref))

    reco_func = make_reconstructor(kernel_sharedmem_fp32, BLOCK_SIZE, PIN_MEMORY)
    print("sharedmem fp32 reco precise: ", compare_matrices(reco_func(*input), ref))

    reco_func = make_reconstructor(
        kernel_globalmem_fp64,
        BLOCK_SIZE,
        PIN_MEMORY,
        use_streams=True,
    )

    print(
        "globalmem fp64 reco streamed precise: ",
        [
            compare_matrices(result, ref)
            for result in reco_func(
                *(
                    [input[0]] * NUM_STREAMS,
                    [input[1]] * NUM_STREAMS,
                    [input[2]] * NUM_STREAMS,
                    [input[3]] * NUM_STREAMS,
                )
            )
        ],
    )

    reco_func = make_reconstructor(
        kernel_globalmem_fp32,
        BLOCK_SIZE,
        PIN_MEMORY,
        use_streams=True,
    )

    print(
        "globalmem fp32 reco streamed precise: ",
        [
            compare_matrices(result, ref)
            for result in reco_func(
                *(
                    [input[0]] * NUM_STREAMS,
                    [input[1]] * NUM_STREAMS,
                    [input[2]] * NUM_STREAMS,
                    [input[3]] * NUM_STREAMS,
                )
            )
        ],
    )
