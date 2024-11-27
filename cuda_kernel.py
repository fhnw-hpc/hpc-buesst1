import numpy as np
from numba import cuda, float32, float64
import time

TILE_SIZE = 5

class time_region:
    def __init__(self, time_offset=0):
        self._time_offset = time_offset

    def __enter__(self):
        self._t_start = time.time()
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        self._t_end = time.time()

    def elapsed_time(self):
        return self._time_offset + (self._t_end - self._t_start)


class time_region_cuda:
    def __init__(self, time_offset=0, cuda_stream=0):
        self._t_start = cuda.event(timing=True)
        self._t_end = cuda.event(timing=True)
        self._time_offset = time_offset
        self._cuda_stream = cuda_stream

    def __enter__(self):
        self._t_start.record(self._cuda_stream)
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        self._t_end.record(self._cuda_stream)
        self._t_end.synchronize()

    def elapsed_time(self):
        return self._time_offset + 1e-3 * cuda.event_elapsed_time(
            self._t_start, self._t_end
        )


def random_svd(shape):
    """
    Generates random matrices U, S, and V^T for SVD decomposition.

    Parameters:
    - shape: tuple, the shape of the original matrix (m, n)

    Returns:
    - U: np.ndarray, matrix of shape (m, m)
    - S: np.ndarray, singular values (min(m, n),)
    - Vt: np.ndarray, matrix of shape (n, n)
    """
    m, n = shape
    k = min(m, n)  # Number of singular values

    # Generate random matrices U and V
    U = np.random.randn(m, m)  # QR decomposition ensures orthogonality
    Vt = np.random.randn(n, n)

    # Generate random singular values in descending order
    S = np.sort(np.random.rand(k))[::-1]

    return U, S, Vt


def reconstruct_svd_broadcast(u, s, vt, k):
    """SVD reconstruction for k components using broadcast

    Inputs:
    u: (m,n) numpy array
    s: (n) numpy array (diagonal matrix)
    vt: (n,n) numpy array
    k: number of reconstructed singular components

    Ouput:
    (m,n) numpy array U_mk * S_k * V^T_nk for k reconstructed components
    """

    return u[:, :k] @ (s[:k].reshape(-1, 1) * vt[:k, :])


@cuda.jit(
    "void(Array(float64, 2, 'C'), Array(float64, 1, 'C'), Array(float64, 2, 'C'), int32, Array(float64, 2, 'C'))",
    fastmath=True,
)
def svd_reco_kernel_fp64(u, s, vt, k, y):
    """SVD reconstruction for k components using cuda. FP64 operation (slower but more accurate than FP64)

    Inputs:
    u (m,n): array
    s (n): array (diagonal matrix)
    vt (n,n): array
    k int: number of reconstructed singular components
    y (m,n): output array
    """
    m, n = cuda.grid(2)

    if m >= u.shape[0] or n >= vt.shape[1]:
        return

    element = float64(0)
    for p in range(k):
        element += u[m, p] * s[p] * vt[p, n]

    y[m, n] = element


@cuda.jit(
    "void(Array(float32, 2, 'C'), Array(float32, 1, 'C'), Array(float32, 2, 'C'), int32, Array(float32, 2, 'C'))",
    fastmath=True,
)
def svd_reco_kernel_fp32(u, s, vt, k, y):
    """SVD reconstruction for k components using cuda. FP32 operation (faster but at the cost of lower precision)

    Inputs:
    u (m,n): array
    s (n): array (diagonal matrix)
    vt (n,n): array
    k int: number of reconstructed singular components
    y (m,n): output array
    """
    m, n = cuda.grid(2)

    if m >= u.shape[0] or n >= vt.shape[1]:
        return

    element = float32(0)
    for p in range(k):
        element += u[m, p] * s[p] * vt[p, n]

    y[m, n] = element

@cuda.jit(
    "void(Array(float64, 2, 'C'), Array(float64, 1, 'C'), Array(float64, 2, 'C'), int32, Array(float64, 2, 'C'))",
    fastmath=True,
)
def svd_reco_kernel_fp64_sharedmem(u, s, vt, k, y):
    """SVD reconstruction for k components using cuda. FP64 operation (slower but more accurate than FP64)

    Inputs:
    u (m,n): array
    s (n): array (diagonal matrix)
    vt (n,n): array
    k int: number of reconstructed singular components
    y (m,n): output array
    """
    m, n = cuda.grid(2)

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
            if (tile_nr*TILE_SIZE + threadID) < k:
                s_s[threadID] = s[tile_nr*TILE_SIZE + threadID]
            else:
                s_s[threadID] = float64(0.0)

        # wait for all threads to finish the copy process
        cuda.syncthreads()

        # only process if thread has a global index within final matrix
        if m < u.shape[0] and n < vt.shape[1]:
            # loop over tile
            for p in range(TILE_SIZE):
                # only add if element < k
                if tile_nr*TILE_SIZE+p < k:
                    element += u[m, tile_nr*TILE_SIZE+p] * s_s[p] * vt[tile_nr*TILE_SIZE+p, n]

        # wait for all threads to finish dot product
        cuda.syncthreads()

    y[m, n] = element


def get_transfers_and_fpo_per_thread(k):
    """Standard function to estimate number of data transfers and operations per thread"""

    num_transfers_per_thread = (4 * k) + 1
    num_fpo_per_thread = k * 3

    return num_transfers_per_thread, num_fpo_per_thread


def svd_reco_cuda(
    u: np.ndarray,
    s: np.ndarray,
    vt: np.ndarray,
    k: np.int32,
    block_size: tuple = (32, 32),
) -> np.ndarray:
    """Host function to perform SVD reconstruction using CUDA kernel. FP32 or FP64 operation is determined by dtype of input matrices

    Args:
        u (np.ndarray): Left singular vectors, shape (m, n).
        s (np.ndarray): Singular values, shape (n,).
        vt (np.ndarray): Right singular vectors, shape (n, n).
        k (int32): Number of singular components to use in reconstruction.
        block_size (tuple(int,int)): Number of threads per block

    Returns:
        np.ndarray: Reconstructed matrix, shape (m, n).
    """

    # assert correct datatype
    assert (
        u.dtype == s.dtype and u.dtype == vt.dtype
    ), "u, s and vt must be from the same dtype"
    assert (
        u.dtype == np.float32 or u.dtype == np.float64
    ), "u, s and vt must be float32 or float64"

    # fp64 operation enabled
    fp64 = u.dtype == np.float64

    print("reco in fp", 64 if fp64 else 32, " mode")

    u = np.ascontiguousarray(u)  # make row major
    s = np.ascontiguousarray(s)  # make row major
    vt = np.ascontiguousarray(vt)  # make row major

    # Send arrays to gpu
    u = cuda.to_device(u)
    s = cuda.to_device(s)
    vt = cuda.to_device(vt)

    # create array where results are stored. Also pin that array so no data gets moved out of the ram.
    m, n = u.shape[0], vt.shape[1]
    y = cuda.device_array((m, n), dtype=np.float64 if fp64 else np.float32, order="C")
    y_ret = cuda.pinned_array_like(y)

    # Define CUDA thread and block dimensions
    blocks_per_grid_x = (m + block_size[0] - 1) // block_size[0]
    blocks_per_grid_y = (n + block_size[1] - 1) // block_size[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # launch cuda kernel in fp64 mode
    if fp64:
        assert block_size[0]*block_size[1]== TILE_SIZE #at the moment number of threads per block must be equal to tile size
        svd_reco_kernel_fp64_sharedmem[blocks_per_grid, block_size](u, s, vt, np.int32(k), y)

    else:  # fp32 mode
        svd_reco_kernel_fp32[blocks_per_grid, block_size](u, s, vt, np.int32(k), y)

    y.copy_to_host(y_ret)

    return y_ret


def svd_reco_cuda_perfmeasure(
    u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: np.int32, block_size: tuple
):
    """Host function to perform SVD reconstruction using CUDA kernel and doing performance measurements. FP32 or FP64 operation is determined by dtype of input matrices

    Args:
        u (np.ndarray): Left singular vectors, shape (m, n).
        s (np.ndarray): Singular values, shape (n,).
        vt (np.ndarray): Right singular vectors, shape (n, n).
        k (int32): Number of singular components to use in reconstruction.
        block_size (tuple(int,int)): Number of threads per block

    Returns:
        dict containing times including memory bandwidth and TFLOPs
    """

    # assert correct datatype
    assert (
        u.dtype == s.dtype and u.dtype == vt.dtype
    ), "u, s and vt must be from the same dtype"
    assert (
        u.dtype == np.float32 or u.dtype == np.float64
    ), "u, s and vt must be float32 or float64"

    # fp64 operation enabled
    fp64 = u.dtype == np.float64

    print("reco in fp", 64 if fp64 else 32, " mode")

    u = np.ascontiguousarray(u)  # make row major
    s = np.ascontiguousarray(s)  # make row major
    vt = np.ascontiguousarray(vt)  # make row major

    # reconstruct using reference function
    with time_region() as t_cpu:
        y_ref = reconstruct_svd_broadcast(u, s, vt, k)

    with time_region_cuda() as t_xfer:
        # Send arrays to gpu
        u = cuda.to_device(u)
        s = cuda.to_device(s)
        vt = cuda.to_device(vt)

        # create array where results are stored. Also pin that array so no data gets moved out of the ram.
        m, n = u.shape[0], vt.shape[1]
        y = cuda.device_array(
            (m, n), dtype=np.float64 if fp64 else np.float32, order="C"
        )
        y_ret = cuda.pinned_array_like(y)

    # Define CUDA thread and block dimensions
    blocks_per_grid_x = (m + block_size[0] - 1) // block_size[0]
    blocks_per_grid_y = (n + block_size[1] - 1) // block_size[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    with time_region_cuda() as t_kernel:
        # launch cuda kernel in fp64 mode
        if fp64:
            svd_reco_kernel_fp64[blocks_per_grid, block_size](u, s, vt, np.int32(k), y)
        else:  # fp32 mode
            svd_reco_kernel_fp32[blocks_per_grid, block_size](u, s, vt, np.int32(k), y)

    with time_region_cuda(t_xfer.elapsed_time()) as t_xfer:
        # copy back to host
        y.copy_to_host(y_ret)

    # get number of transfers and number of fp64 operations per thread
    num_transfers_per_thread, num_fpo_per_thread = get_transfers_and_fpo_per_thread(k)

    # calculate number of transferred bytes in total
    number_of_GB_transferred_total = (
        num_transfers_per_thread * y.shape[0] * y.shape[1] * (8 if fp64 else 4)
    )

    # calculate number of floating point operations
    num_fpo_total = num_fpo_per_thread * y.shape[0] * y.shape[1]

    return {
        "cuda": {
            "time_transfer_ms": t_xfer.elapsed_time() * 1000,
            "time_kernel_ms": t_kernel.elapsed_time() * 1000,
            "time_reco_ms": (t_xfer.elapsed_time() + t_kernel.elapsed_time()) * 1000,
            "consumed_mem_bandwidth_GB/s": 1e-9
            * (number_of_GB_transferred_total / t_kernel.elapsed_time()),
            "consumed GFLOPs": 1e-9 * (num_fpo_total / t_kernel.elapsed_time()),
        },
        "cpu": {
            "time_reco_ms": t_cpu.elapsed_time() * 1000,
            "consumed_mem_bandwidth_GB/s": (
                1e-9 * (number_of_GB_transferred_total / t_cpu.elapsed_time())
                if t_cpu.elapsed_time() > 0
                else np.inf
            ),
            "consumed GFLOPs": (
                1e-9 * (num_fpo_total / t_cpu.elapsed_time())
                if t_cpu.elapsed_time() > 0
                else np.inf
            ),
        },
        "cpu_cuda_same": bool(np.isclose(y_ref, y_ret).all()),
    }


# if this script is called directly (eg profiling) -> perform random big reconstruction
if __name__ == "__main__":
    # create random matrices to reconstruct
    u, s, vt = random_svd((6, 6))

    print("u original:")
    print(u)

    print("s original:")
    print(s)

    print("vt original:")
    print(vt)

    # block size
    block_size = (1, 5)

    assert (
        block_size[0] * block_size[1] <= 1024
    ), "a block is now allowed ha have more threads than 1024"

    # do a full reconstruction in fp64 mode
    cuda64_start = time.time()
    result64 = svd_reco_cuda(u, s, vt, len(s), block_size)
    cuda64_end = time.time()

    # do a full reconstruction in fp32 mode
    cuda32_start = time.time()
    result32 = svd_reco_cuda(
        u.astype(np.float32),
        s.astype(np.float32),
        vt.astype(np.float32),
        len(s),
        block_size,
    )
    cuda32_end = time.time()

    # for reference -> calculate on cpu
    cpu_start = time.time()
    result_cpu = reconstruct_svd_broadcast(u, s, vt, len(s))
    cpu_end = time.time()

    print("cpu result:")
    print(result_cpu)

    print("gpu resuslt:")
    print(result64)

    print(
        "cuda in fp64 mode and cpu same: ",
        bool(np.isclose(result64, result_cpu).all()),
    )
    print(
        "cuda in fp32 mode and cpu same: ",
        bool(np.isclose(result32, result_cpu).all()),
    )
    print(f"total time cuda in fp64 mode: {(cuda64_end - cuda64_start)*1000} ms")
    print(f"total time cuda in fp32 mode: {(cuda32_end - cuda32_start)*1000} ms")
    print(f"total time cpu: {(cpu_end - cpu_start)*1000} ms")
