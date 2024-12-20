from numba import cuda, float32, float64

TILE_SIZE = 32


@cuda.jit(
    "void(Array(float64, 2, 'C'), Array(float64, 1, 'C'), Array(float64, 2, 'C'), int32, Array(float64, 2, 'C'))",
    fastmath=True,
)
def fp64(u, s, vt, k, y):
    """SVD reconstruction for k components using cuda. FP64 operation (slower but more accurate)

    Inputs:
    u (m,n): array
    s (n): array (diagonal matrix)
    vt (n,n): array
    k int: number of reconstructed singular components
    y (m,n): output array
    """
    # col = x-dimension, row = y-dimension
    col, row = cuda.grid(2)

    # init shared array
    s_s = cuda.shared.array(shape=TILE_SIZE, dtype=float64)

    # calculate number of min tiles required
    num_tiles = (k + TILE_SIZE - 1) // TILE_SIZE

    # calculate thread id within block
    threadID = cuda.threadIdx.x + cuda.blockDim.x * cuda.threadIdx.y

    # element of final matrix will be stored here
    element = float64(0)

    # iterate over tiles
    for tile_nr in range(num_tiles):
        # only copy if threadID is smaller than TILE_SIZE
        if threadID < TILE_SIZE:
            idx = tile_nr * TILE_SIZE + threadID
            if idx < k:
                s_s[threadID] = s[idx]
            else:
                s_s[threadID] = float64(0)

        # wait for all threads to finish the copy process
        cuda.syncthreads()

        # only process if thread has a global index within final matrix
        if row < u.shape[0] and col < vt.shape[1]:
            # loop over tile
            for p in range(TILE_SIZE):
                idx = tile_nr * TILE_SIZE + p
                if idx < k:
                    element += u[row, idx] * s_s[p] * vt[idx, col]

        # wait for all threads to finish
        cuda.syncthreads()

    # only write to global memory if indices are in range
    if row < y.shape[0] and col < y.shape[1]:
        y[row, col] = element


@cuda.jit(
    "void(Array(float32, 2, 'C'), Array(float32, 1, 'C'), Array(float32, 2, 'C'), int32, Array(float32, 2, 'C'))",
    fastmath=True,
)
def fp32(u, s, vt, k, y):
    """SVD reconstruction for k components using cuda. FP32 operation (faster but lower precision)

    Inputs:
    u (m,n): array
    s (n): array (diagonal matrix)
    vt (n,n): array
    k int: number of reconstructed singular components
    y (m,n): output array
    """
    # col = x-dimension, row = y-dimension
    col, row = cuda.grid(2)

    # init shared array
    s_s = cuda.shared.array(shape=TILE_SIZE, dtype=float32)

    # calculate number of min tiles required
    num_tiles = (k + TILE_SIZE - 1) // TILE_SIZE

    # calculate thread id within block
    threadID = cuda.threadIdx.x + cuda.blockDim.x * cuda.threadIdx.y

    # element of final matrix will be stored here
    element = float32(0)

    # iterate over tiles
    for tile_nr in range(num_tiles):
        # only copy if threadID is smaller than TILE_SIZE
        if threadID < TILE_SIZE:
            idx = tile_nr * TILE_SIZE + threadID
            if idx < k:
                s_s[threadID] = s[idx]
            else:
                s_s[threadID] = float32(0)

        # wait for all threads to finish the copy process
        cuda.syncthreads()

        # only process if thread has a global index within final matrix
        if row < u.shape[0] and col < vt.shape[1]:
            # loop over tile
            for p in range(TILE_SIZE):
                idx = tile_nr * TILE_SIZE + p
                if idx < k:
                    element += u[row, idx] * s_s[p] * vt[idx, col]

        cuda.syncthreads()

    # only write to global memory if indices are in range
    if row < y.shape[0] and col < y.shape[1]:
        y[row, col] = element
