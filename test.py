import numpy as np
from numba import cuda
import time
import os
import imageio.v3 as imageio
import numpy as np
import glob


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


@cuda.jit("void(Array(float32, 2, 'C'), Array(float32, 1, 'C'), Array(float32, 2, 'F'), int32, Array(float32, 2, 'C'))")
def svd_reco_kernel(u, s, vt, k, y):
    """SVD reconstruction for k components using cuda
    
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

    element = 0.0
    for p in range(k):
        element += u[m, p] * s[p] * vt[p, n]

    y[m, n] = element

def calculate(
    u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: int, block_size: tuple = (32, 32)
) -> np.ndarray:
    """Host function to perform SVD reconstruction using CUDA kernel.

    Args:
        u (np.ndarray): Left singular vectors, shape (m, n).
        s (np.ndarray): Singular values, shape (n,).
        vt (np.ndarray): Right singular vectors, shape (n, n).
        k (int): Number of singular components to use in reconstruction.
        block_size (tuple(int,int)): Number of threads per block

    Returns:
        np.ndarray: Reconstructed matrix, shape (m, n).
    """

    # convert to correct dtype
    u = u.astype(np.float32)
    s = s.astype(np.float32)
    vt = np.asfortranarray(vt.astype(np.float32))

    with time_region_cuda() as t_xfer:
        # Ensure inputs are in the correct dtype and order
        u = cuda.to_device(u)
        s = cuda.to_device(s)
        vt = cuda.to_device(vt)

        # create array where results are stored. Also pin that array so no data gets moved out of the ram.
        m, n = u.shape[0], vt.shape[1]
        y = cuda.device_array((m, n), dtype=np.float32)
        y_ret = cuda.pinned_array_like(y)

    # Define CUDA thread and block dimensions
    blocks_per_grid_x = (m + block_size[0] - 1) // block_size[0]
    blocks_per_grid_y = (n + block_size[1] - 1) // block_size[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    with time_region_cuda() as t_kernel:
        # Launch the CUDA kernel
        svd_reco_kernel[blocks_per_grid, block_size](u, s, vt, k, y)

    with time_region_cuda(t_xfer.elapsed_time()) as t_xfer:
        # copy back to host
        y.copy_to_host(y_ret)

    # stop profiling
    cuda.profile_stop()

    # calculate number of transfers done in each thread
    num_transfers_per_thread = 1 + (4 * k) + 1
    number_of_GB_transferred = (
        1e-9 * 4 * num_transfers_per_thread * y.shape[0] * y.shape[1]
    )

    # calculate number of floating point operations
    num_fpo_per_thread = k*2
    num_fpo_total = 1e-12 * num_fpo_per_thread * y.shape[0] * y.shape[1]

    print(f"Cuda transfer overhead: {t_xfer.elapsed_time()*1000} ms")
    print(f"Cuda kernel time: {t_kernel.elapsed_time()*1000} ms")
    print(f"Consumed memory bandwidth: {number_of_GB_transferred / t_kernel.elapsed_time()} GB/s")
    print(f"Consumed TFLOPs: {num_fpo_total / t_kernel.elapsed_time()}")

    return y_ret

if __name__ == "__main__":
    subfolder = "001"
    folders = os.path.join("adni_png", subfolder)

    # Get all PNGs from 001 with 145 in the name
    files = sorted(glob.glob(f"{folders}/*145.png"))

    # Load all images using ImageIO and create a numpy array from them
    images = np.array([imageio.imread(f) for f in files])

    # Get all the names of the files
    names = [f[-17:-4] for f in files]

    im = np.random.normal(size=(2000, 20000))
    im = im - im.min() / im.max() - im.min()  # normalize image
    u, s, vt = np.linalg.svd(im, full_matrices=False)

    # convert to correct dtype
    u = u.astype(np.float32)
    s = s.astype(np.float32)
    vt = vt.astype(np.float32)

    calculate(u, s, vt, 100)
