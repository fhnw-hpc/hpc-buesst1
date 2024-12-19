import time
import numba
from numba import cuda
import numpy as np
import pandas as pd


class time_region:
    def __init__(self, time_offset=0):
        self._time_offset = time_offset

    def __enter__(self):
        self._t_start = time.time()
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        self._t_end = time.time()

    def elapsed_time_s(self):
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

    def elapsed_time_s(self):
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


def reconstruct_svd_broadcast_timeit(u, s, vt, k):
    """SVD reconstruction for k components using broadcast

    Inputs:
    u: (m,n) numpy array
    s: (n) numpy array (diagonal matrix)
    vt: (n,n) numpy array
    k: number of reconstructed singular components

    Ouput:
    (m,n) numpy array U_mk * S_k * V^T_nk for k reconstructed components
    """

    with time_region() as total:
        result = u[:, :k] @ (s[:k].reshape(-1, 1) * vt[:k, :])

    return result, {"total": total.elapsed_time_s()}


def compare_matrices(matrix1: np.ndarray, matrix2: np.ndarray):
    """
    Compares two matrices element-wise to determine if they are equal
    within a specified tolerance.

    Args:
        matrix1 (np.ndarray): The first matrix to compare.
        matrix2 (np.ndarray): The second matrix to compare.

    Returns:
        bool: True if all corresponding elements of the matrices are close
              within the tolerance defined by numpy's isclose function,
              otherwise False.
    """

    # Use numpy's isclose to check if all elements of the two matrices
    # are close to each other within the default tolerance.
    return bool(np.isclose(matrix1, matrix2).all())


def compare_kernels(input: tuple, reco_func1: callable, reco_func2: callable):
    """ """

    result_func1 = reco_func1(*input)
    result_func1, timings_func1 = (
        result_func1
        if isinstance(result_func1, tuple)
        else (result_func1, {})  # disassembe if also timings returned
    )

    result_func2 = reco_func2(*input)
    result_func2, timings_func2 = (
        result_func2
        if isinstance(result_func2, tuple)
        else (result_func2, {})  # disassembe if also timings returned
    )

    def transform(df):
        action = df.action.values[0]
        df = df.drop(columns=["action"])
        df.columns = pd.MultiIndex.from_product([[action], df.columns])

        return df

    # create timings table
    timings_ds = pd.concat(
        [
            transform(df).reset_index(drop=True)
            for _, df in pd.DataFrame.from_records(
                [timings_func1, timings_func2], index=["func1", "func2"]
            )
            .T.reset_index(names="action")
            .groupby("action")
        ],
        axis=1,
    )

    return compare_matrices(result_func1, result_func2), timings_ds


def make_reconstructor(
    kernel: callable, block_size: tuple, pin_memory: bool = False, timeit: bool = False
):
    """
    Creates a function to perform SVD reconstruction using a CUDA kernel.

    Args:
        kernel (callable): The CUDA kernel function to perform reconstruction.
        block_size (tuple): Tuple indicating the size of CUDA thread blocks (threads per block).
        pin_memory (bool): Whether to use pinned memory for the output array. Default is False.
        timeit (bool): Whether timings should also be calculated and returned

    Returns:
        callable: A function that performs SVD reconstruction on GPU.
    """

    def infer_types_and_orders(kernel: cuda.dispatcher.CUDADispatcher):
        """
        Infers the data types and memory orders of kernel arguments from the kernel signature.

        Args:
            kernel (cuda.dispatcher.CUDADispatcher): The CUDA kernel.

        Returns:
            list: A list of tuples, each containing dtype (NumPy type) and order (memory order, 'C' or 'F').
        """
        assert (
            len(kernel.signatures) == 1
        ), "Numba kernel is not allowed to have multiple signatures"

        signature = kernel.signatures[0]
        types_and_orders = []

        for arg in signature:
            if isinstance(arg, numba.types.npytypes.Array):
                dtype = arg.dtype.name
                order = "C" if arg.layout == "C" else "F"
                types_and_orders.append((dtype, order))

            elif isinstance(arg, numba.types.Integer):
                types_and_orders.append((arg.name, None))  # Scalars have no order

            else:
                raise ValueError(
                    f"Unsupported argument type in kernel signature: {arg}"
                )

        return types_and_orders

    def inner(u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: int):
        """
        Perform SVD reconstruction for k components using the provided CUDA kernel.

        Args:
            u (np.ndarray): Left singular vectors of shape (m, m).
            s (np.ndarray): Singular values of shape (min(m, n),).
            vt (np.ndarray): Right singular vectors of shape (n, n).
            k (int): Number of singular components to use for reconstruction.

        Returns:
            np.ndarray: Reconstructed matrix of shape (m, n).
        """

        # Infer types and orders for the inputs
        types_and_orders = infer_types_and_orders(kernel)
        u_dtype, u_order = types_and_orders[0]
        s_dtype, s_order = types_and_orders[1]
        vt_dtype, vt_order = types_and_orders[2]
        k_dtype, _ = types_and_orders[3]

        # Ensure inputs have correct dtype and memory order
        u = np.array(u, dtype=u_dtype, order=u_order)
        s = np.array(s, dtype=s_dtype, order=s_order)
        vt = np.array(vt, dtype=vt_dtype, order=vt_order)
        k = getattr(np, k_dtype)(k)

        # Send arrays to GPU
        u_gpu = cuda.to_device(u)
        s_gpu = cuda.to_device(s)
        vt_gpu = cuda.to_device(vt)

        # Determine output array dimensions
        m, n = u.shape[0], vt.shape[1]
        y_dtype, y_order = types_and_orders[
            4
        ]  # Assuming output type/order is defined in the kernel signature
        y_gpu = cuda.device_array((m, n), dtype=y_dtype, order=y_order)

        # Allocate pinned memory for output if required
        y_host = cuda.pinned_array_like(y_gpu) if pin_memory else np.empty_like(y_gpu)

        # Define CUDA grid dimensions
        blocks_per_grid_x = (m + block_size[0] - 1) // block_size[0]
        blocks_per_grid_y = (n + block_size[1] - 1) // block_size[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Launch CUDA kernel
        kernel[blocks_per_grid, block_size](u_gpu, s_gpu, vt_gpu, k, y_gpu)

        # Copy result back to host
        y_gpu.copy_to_host(y_host)

        return y_host

    def inner_timed(u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: int):
        """
        Perform SVD reconstruction for k components using the provided CUDA kernel.

        Args:
            u (np.ndarray): Left singular vectors of shape (m, m).
            s (np.ndarray): Singular values of shape (min(m, n),).
            vt (np.ndarray): Right singular vectors of shape (n, n).
            k (int): Number of singular components to use for reconstruction.

        Returns:
            np.ndarray: Reconstructed matrix of shape (m, n).
        """

        # Infer types and orders for the inputs
        types_and_orders = infer_types_and_orders(kernel)
        u_dtype, u_order = types_and_orders[0]
        s_dtype, s_order = types_and_orders[1]
        vt_dtype, vt_order = types_and_orders[2]
        k_dtype, _ = types_and_orders[3]

        # Ensure inputs have correct dtype and memory order
        u = np.array(u, dtype=u_dtype, order=u_order)
        s = np.array(s, dtype=s_dtype, order=s_order)
        vt = np.array(vt, dtype=vt_dtype, order=vt_order)
        k = getattr(np, k_dtype)(k)

        with time_region_cuda() as h2d:
            # Send arrays to GPU
            u_gpu = cuda.to_device(u)
            s_gpu = cuda.to_device(s)
            vt_gpu = cuda.to_device(vt)

        # Determine output array dimensions
        m, n = u.shape[0], vt.shape[1]
        y_dtype, y_order = types_and_orders[
            4
        ]  # Assuming output type/order is defined in the kernel signature

        with time_region_cuda() as d_maloc_y:
            y_gpu = cuda.device_array((m, n), dtype=y_dtype, order=y_order)

        with time_region_cuda() as h_maloc_y:
            # Allocate pinned memory for output if required
            y_host = (
                cuda.pinned_array_like(y_gpu) if pin_memory else np.empty_like(y_gpu)
            )

        # Define CUDA grid dimensions
        blocks_per_grid_x = (m + block_size[0] - 1) // block_size[0]
        blocks_per_grid_y = (n + block_size[1] - 1) // block_size[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Launch CUDA kernel
        with time_region_cuda() as kernel_t:
            kernel[blocks_per_grid, block_size](u_gpu, s_gpu, vt_gpu, k, y_gpu)

        # Copy result back to host
        with time_region_cuda() as d2h:
            y_gpu.copy_to_host(y_host)

        timings = {
            "h2d": h2d.elapsed_time_s(),
            "d_maloc_y": d_maloc_y.elapsed_time_s(),
            "h_maloc_y": h_maloc_y.elapsed_time_s(),
            "kernel": kernel_t.elapsed_time_s(),
            "d2h": d2h.elapsed_time_s(),
            "mem_operations_total": h2d.elapsed_time_s()
            + d_maloc_y.elapsed_time_s()
            + h_maloc_y.elapsed_time_s()
            + d2h.elapsed_time_s(),
            "total": h2d.elapsed_time_s()
            + d_maloc_y.elapsed_time_s()
            + h_maloc_y.elapsed_time_s()
            + d2h.elapsed_time_s()
            + kernel_t.elapsed_time_s(),
        }

        return y_host, timings

    return inner_timed if timeit else inner


# Example usage (to be implemented in a separate module):
# - Define a CUDA kernel for SVD reconstruction.
# - Pass the kernel, block size, and signature to `make_reconstructor`.
# - Use the resulting function for GPU-accelerated SVD reconstruction.
