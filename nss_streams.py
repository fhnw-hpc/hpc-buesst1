from src.utils import (
    make_reconstructor,
    random_svd,
)
from src.kernels.shared_mem import fp32 as kernel_sharedmem_fp32

import time

RECO_SHAPE_SMALL = (256, 170)  # same shape as mri images
RECO_SHAPE_BIG = (1080, 1920)  # big images
BLOCK_SIZE = (4, 32)
NUM_STREAMS = 50

if __name__ == "__main__":
    # construct small images
    input_small = random_svd(RECO_SHAPE_SMALL)
    input_small = tuple(list(input_small) + [min(RECO_SHAPE_SMALL)])
    input_small = (
        [input_small[0]] * NUM_STREAMS,
        [input_small[1]] * NUM_STREAMS,
        [input_small[2]] * NUM_STREAMS,
        [input_small[3]] * NUM_STREAMS,
    )

    # construct big images
    input_big = random_svd(RECO_SHAPE_BIG)
    input_big = tuple(list(input_big) + [min(RECO_SHAPE_BIG)])
    input_big = (
        [input_big[0]] * NUM_STREAMS,
        [input_big[1]] * NUM_STREAMS,
        [input_big[2]] * NUM_STREAMS,
        [input_big[3]] * NUM_STREAMS,
    )

    # serial reconstruction function
    reco_func_serial = make_reconstructor(
        kernel_sharedmem_fp32, BLOCK_SIZE, pin_memory=False, use_streams=False
    )

    # reconstruction function with pageable memory
    reco_func_pageable = make_reconstructor(
        kernel_sharedmem_fp32, BLOCK_SIZE, pin_memory=False, use_streams=True
    )

    # reconstruction function with pinned memory
    reco_func_pinned = make_reconstructor(
        kernel_sharedmem_fp32, BLOCK_SIZE, pin_memory=True, use_streams=True
    )

    print(f"reco of {NUM_STREAMS} small images started (serially)")
    [
        reco_func_serial(
            input_small[0][i], input_small[1][i], input_small[2][i], input_small[3][i]
        )
        for i in range(len(input_small[0]))
    ]

    time.sleep(1)

    print(f"reco of {NUM_STREAMS} big images started (serially)")
    [
        reco_func_serial(
            input_big[0][i], input_big[1][i], input_big[2][i], input_big[3][i]
        )
        for i in range(len(input_big[0]))
    ]

    time.sleep(1)

    print(f"reco of {NUM_STREAMS} small images started (pageable memory)")
    reco_func_pageable(*input_small)

    time.sleep(1)

    print(f"reco of {NUM_STREAMS} big images started (pageable memory)")
    reco_func_pageable(*input_big)

    time.sleep(1)

    print(f"reco of {NUM_STREAMS} small images started (pinned memory)")
    reco_func_pinned(*input_small)

    time.sleep(1)

    print(f"reco of {NUM_STREAMS} big images started (pinned memory)")
    reco_func_pinned(*input_big)
