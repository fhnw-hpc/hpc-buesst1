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
        return self._time_offset + 1e-3*cuda.event_elapsed_time(self._t_start, self._t_end)

def reconstruct_svd_broadcast(u,s,vt,k):
    """SVD reconstruction for k components using broadcast
    
    Inputs:
    u: (m,n) numpy array
    s: (n) numpy array (diagonal matrix)
    vt: (n,n) numpy array
    k: number of reconstructed singular components
    
    Ouput:
    (m,n) numpy array U_mk * S_k * V^T_nk for k reconstructed components
    """

    return u[:, :k]@(s[:k].reshape(-1, 1)*vt[:k, :])

@cuda.jit("void(Array(float64, 2, 'C'), Array(float64, 1, 'C'), Array(float64, 2, 'F'), int32, Array(float64, 2, 'C'))")
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

def get_transfers_fpo_per_thread(k):
    """ Standard function to estimate number of data transfers and operations per thread
    """
    
    num_transfers_per_thread = 1 + (4 * k) + 1
    num_fpo_per_thread = k*2

    return num_transfers_per_thread, num_fpo_per_thread

def svd_reco_cuda(
    u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: np.int32, block_size: tuple = (32, 32)
) -> np.ndarray:
    """Host function to perform SVD reconstruction using CUDA kernel.

    Args:
        u (np.ndarray): Left singular vectors, shape (m, n).
        s (np.ndarray): Singular values, shape (n,).
        vt (np.ndarray): Right singular vectors, shape (n, n).
        k (int32): Number of singular components to use in reconstruction.
        block_size (tuple(int,int)): Number of threads per block

    Returns:
        np.ndarray: Reconstructed matrix, shape (m, n).
    """

    # convert to correct dtype
    u = u.astype(np.float64)
    s = s.astype(np.float64)
    vt = np.asfortranarray(vt.astype(np.float64)) # convert and make column major

    # Send arrays to gpu
    u = cuda.to_device(u)
    s = cuda.to_device(s)
    vt = cuda.to_device(vt)

    # create array where results are stored. Also pin that array so no data gets moved out of the ram.
    m, n = u.shape[0], vt.shape[1]
    y = cuda.device_array((m, n), dtype=np.float64)
    y_ret = cuda.pinned_array_like(y)

    # Define CUDA thread and block dimensions
    blocks_per_grid_x = (m + block_size[0] - 1) // block_size[0]
    blocks_per_grid_y = (n + block_size[1] - 1) // block_size[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # launch cuda kernel
    svd_reco_kernel[blocks_per_grid, block_size](u, s, vt, np.int32(k), y)

    y.copy_to_host(y_ret)

    return y_ret

def svd_reco_cuda_perfmeasure(u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: np.int32, block_size: tuple):
    """Host function to perform SVD reconstruction using CUDA kernel and doing performance measurements.

    Args:
        u (np.ndarray): Left singular vectors, shape (m, n).
        s (np.ndarray): Singular values, shape (n,).
        vt (np.ndarray): Right singular vectors, shape (n, n).
        k (int32): Number of singular components to use in reconstruction.
        block_size (tuple(int,int)): Number of threads per block

    Returns:
        dict containing times including memory bandwidth and TFLOPs
    """

    # convert to correct dtype
    u = u.astype(np.float64)
    s = s.astype(np.float64)
    vt = vt.astype(np.float64)

    # reconstruct using reference function
    y_ref = reconstruct_svd_broadcast(u, s, vt, k)

    # make column major
    vt = np.asfortranarray(vt)

    # start profiling
    cuda.profile_start()

    with time_region_cuda() as t_xfer:
        # Send arrays to gpu
        u = cuda.to_device(u)
        s = cuda.to_device(s)
        vt = cuda.to_device(vt)

        # create array where results are stored. Also pin that array so no data gets moved out of the ram.
        m, n = u.shape[0], vt.shape[1]
        y = cuda.device_array((m, n), dtype=np.float64)
        y_ret = cuda.pinned_array_like(y)

    # Define CUDA thread and block dimensions
    blocks_per_grid_x = (m + block_size[0] - 1) // block_size[0]
    blocks_per_grid_y = (n + block_size[1] - 1) // block_size[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    with time_region_cuda() as t_kernel:
        # Launch the CUDA kernel
        svd_reco_kernel[blocks_per_grid, block_size](u, s, vt, np.int32(k), y)

    with time_region_cuda(t_xfer.elapsed_time()) as t_xfer:
        # copy back to host
        y.copy_to_host(y_ret)

    # stop profiling
    cuda.profile_stop()
    
    # get number of transfers and number of fp64 operations per thread
    num_transfers_per_thread, num_fpo_per_thread = get_transfers_fpo_per_thread(k)

    # calculate number of transfers in total
    number_of_GB_transferred_total = (
        1e-9 * 8 * num_transfers_per_thread * y.shape[0] * y.shape[1]
    )

    # calculate number of floating point operations
    num_fpo_total = 1e-12 * num_fpo_per_thread * y.shape[0] * y.shape[1]
    
    return {
        "time_transfer_ms": t_xfer.elapsed_time()*1000,
        "time_kernel_ms": t_kernel.elapsed_time()*1000,
        "consumed_mem_bandwidth_GB/s": number_of_GB_transferred_total / t_kernel.elapsed_time(),
        "consumed TFLOPs": num_fpo_total / t_kernel.elapsed_time(),
        "isclose": np.isclose(y_ref, y_ret)
    }

if __name__ == "__main__":
    subfolder = "001"
    folders = os.path.join("adni_png", subfolder)

    # Get all PNGs from 001 with 145 in the name
    files = sorted(glob.glob(f"{folders}/*145.png"))

    # Load all images using ImageIO and create a numpy array from them
    images = np.array([imageio.imread(f) for f in files])

    # Get all the names of the files
    names = [f[-17:-4] for f in files]

    # random BIG image
    im = np.random.normal(size=(2000, 20000))
    im = im - im.min() / im.max() - im.min()  # normalize image
    u, s, vt = np.linalg.svd(im, full_matrices=False)

    # do a full reconstruction
    svd_reco_cuda_perfmeasure(u, s, vt, len(s), (32, 32))
