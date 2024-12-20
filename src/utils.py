import time
import numba
from numba import cuda
import numpy as np
import pandas as pd
from typing import List


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
    """
    Compares the results and timings of two reconstruction functions using the same input data.

    Args:
        input (tuple): Input data for the reconstruction functions, typically (u, s, vt, k),
            where `u` is the left singular matrix, `s` is the singular values,
            `vt` is the right singular matrix, and `k` is the number of singular components to use.
        reco_func1 (callable): The first reconstruction function to evaluate.
        reco_func2 (callable): The second reconstruction function to evaluate.

    Returns:
        bool: True if the outputs of both functions are identical within a tolerance, otherwise False.
        pd.DataFrame: A DataFrame containing the timing comparisons of the two functions for different operations.
    """

    # Execute the first reconstruction function
    result_func1 = reco_func1(*input)
    result_func1, timings_func1 = (
        result_func1
        if isinstance(result_func1, tuple)
        else (result_func1, {})  # Handle case where timings are not returned
    )

    # Execute the second reconstruction function
    result_func2 = reco_func2(*input)
    result_func2, timings_func2 = (
        result_func2
        if isinstance(result_func2, tuple)
        else (result_func2, {})  # Handle case where timings are not returned
    )

    def transform(df):
        """
        Transforms a DataFrame by grouping actions and restructuring columns.

        Args:
            df (pd.DataFrame): Input DataFrame containing timing information.

        Returns:
            pd.DataFrame: Transformed DataFrame with hierarchical columns.
        """
        action = df.action.values[0]
        df = df.drop(columns=["action"])
        df.columns = pd.MultiIndex.from_product([[action], df.columns])

        return df

    # Create a DataFrame to compare timings between the two functions
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

    # Compare the matrices produced by both functions
    return compare_matrices(result_func1, result_func2), timings_ds


def get_timings(input: tuple, reco_func: callable):
    """
    Executes a reconstruction function and extracts the timing information.

    Args:
        input (tuple): Input data for the reconstruction function, typically (u, s, vt, k),
            where `u` is the left singular matrix, `s` is the singular values,
            `vt` is the right singular matrix, and `k` is the number of singular components to use.
        reco_func (callable): The reconstruction function to evaluate. This function should return
            a tuple containing the result and a dictionary of timings.

    Returns:
        pd.DataFrame: A DataFrame containing the timing information extracted from the reconstruction function.

    Raises:
        Exception: If the reconstruction function does not return a tuple with timings.
    """

    # Execute the reconstruction function
    result_func = reco_func(*input)

    # Ensure the function returns timings
    if not isinstance(result_func, tuple):
        raise Exception("No timings returned!")

    # Extract timings from the result
    _, timings_func = result_func

    # Convert timings to a DataFrame
    return pd.DataFrame.from_records([timings_func])


def get_k_timings(input: tuple, reco_func: callable, k=10):
    """
    Executes a reconstruction function multiple times and collects timing information.

    Args:
        input (tuple): Input data for the reconstruction function, typically (u, s, vt, k),
            where `u` is the left singular matrix, `s` is the singular values,
            `vt` is the right singular matrix, and `k` is the number of singular components to use.
        reco_func (callable): The reconstruction function to evaluate. This function should return
            a tuple containing the result and a dictionary of timings.
        k (int, optional): The number of repetitions to perform. Default is 10.

    Returns:
        pd.DataFrame: A DataFrame containing the timing information from all repetitions, with an
        additional column indicating the repetition index.
    """

    dfs = []
    for i in range(k):
        # Get timings for the current repetition
        df = get_timings(input, reco_func)
        df["repeat"] = i  # Add repetition index
        dfs.append(df)

    # Combine all repetitions into a single DataFrame
    return pd.concat(dfs)


def get_k_timings_from_kernels(
    input: tuple, reco_funcs=List[callable], names=List[str], k=10
):
    """
    Executes multiple reconstruction functions repeatedly and collects timing information.

    Args:
        input (tuple): Input data for the reconstruction functions, typically (u, s, vt, k),
            where `u` is the left singular matrix, `s` is the singular values,
            `vt` is the right singular matrix, and `k` is the number of singular components to use.
        reco_funcs (List[callable]): A list of reconstruction functions to evaluate. Each function
            should return a tuple containing the result and a dictionary of timings.
        names (List[str]): A list of names corresponding to the reconstruction functions.
        k (int, optional): The number of repetitions to perform for each function. Default is 10.

    Returns:
        pd.DataFrame: A DataFrame containing the timing information from all functions and repetitions,
        with additional columns indicating the function name and the repetition index.

    Raises:
        AssertionError: If the number of reconstruction functions does not match the number of names.
    """

    assert len(reco_funcs) == len(
        names
    ), "Each reconstruction function must have a corresponding name."

    dfs = []
    for i in range(len(reco_funcs)):
        reco_func = reco_funcs[i]
        name = names[i]

        # Get timings for the current reconstruction function
        df = get_k_timings(input, reco_func, k)
        df["name"] = name  # Add function name

        dfs.append(df)

    # Combine all results into a single DataFrame
    return pd.concat(dfs)


def make_reconstructor(
    kernel: callable, block_size: tuple, pin_memory: bool = False, timeit: bool = False
):
    """
    Creates a function to perform SVD reconstruction using a CUDA kernel.

    Args:
        kernel (callable): The CUDA kernel function to perform the reconstruction.
        block_size (tuple): The size of CUDA thread blocks as (num_rows, num_columns).
                            Internally, the kernel will still consider the first dimension as columns (x)
                            and the second dimension as rows (y).
        pin_memory (bool): Whether to use pinned memory for the output array. Defaults to False.
        timeit (bool): Whether to return timing information. Defaults to False.

    Returns:
        callable: A function that performs SVD reconstruction on the GPU.
    """

    def infer_types_and_orders(kernel: cuda.dispatcher.CUDADispatcher):
        """
        Infers the data types and memory orders of kernel arguments from the kernel signature.

        Args:
            kernel (cuda.dispatcher.CUDADispatcher): The CUDA kernel.

        Returns:
            list: A list of tuples, each containing the dtype (string) and order ('C' or 'F').
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
            u (np.ndarray): Left singular vectors of shape (m, n).
            s (np.ndarray): Singular values of shape (min(m, n),).
            vt (np.ndarray): Right singular vectors of shape (n, n).
            k (int): Number of singular components to use for reconstruction.

        Returns:
            np.ndarray: The reconstructed matrix of shape (m, n).
        """
        types_and_orders = infer_types_and_orders(kernel)
        u_dtype, u_order = types_and_orders[0]
        s_dtype, s_order = types_and_orders[1]
        vt_dtype, vt_order = types_and_orders[2]
        k_dtype, _ = types_and_orders[3]

        # Ensure inputs have the correct dtype and memory order
        u = np.array(u, dtype=u_dtype, order=u_order)
        s = np.array(s, dtype=s_dtype, order=s_order)
        vt = np.array(vt, dtype=vt_dtype, order=vt_order)
        k = getattr(np, k_dtype)(k)

        # Transfer arrays to GPU
        u_gpu = cuda.to_device(u)
        s_gpu = cuda.to_device(s)
        vt_gpu = cuda.to_device(vt)

        # Determine output array dimensions
        m, n = u.shape[0], vt.shape[1]
        y_dtype, y_order = types_and_orders[4]
        y_gpu = cuda.device_array((m, n), dtype=y_dtype, order=y_order)

        # Allocate host memory (pinned if requested)
        y_host = cuda.pinned_array_like(y_gpu) if pin_memory else np.empty_like(y_gpu)

        # block_size is given as (num_rows, num_columns)
        num_rows, num_columns = block_size

        # Since we want col = x-dim and row = y-dim, blockDim should be (num_columns, num_rows)
        # and grid dimensions should be calculated accordingly:
        blocks_per_grid_x = (n + num_columns - 1) // num_columns  # columns mapped to x
        blocks_per_grid_y = (m + num_rows - 1) // num_rows  # rows mapped to y
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Launch CUDA kernel with adjusted block dimensions
        kernel[blocks_per_grid, (num_columns, num_rows)](u_gpu, s_gpu, vt_gpu, k, y_gpu)

        # Copy result back to host
        y_gpu.copy_to_host(y_host)
        return y_host

    def inner_timed(u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: int):
        """
        Perform SVD reconstruction for k components using the provided CUDA kernel and return timing information.

        Args:
            u (np.ndarray): Left singular vectors of shape (m, n).
            s (np.ndarray): Singular values of shape (min(m, n),).
            vt (np.ndarray): Right singular vectors of shape (n, n).
            k (int): Number of singular components to use for reconstruction.

        Returns:
            (np.ndarray, dict): The reconstructed matrix and a dictionary with timing details.
        """
        types_and_orders = infer_types_and_orders(kernel)
        u_dtype, u_order = types_and_orders[0]
        s_dtype, s_order = types_and_orders[1]
        vt_dtype, vt_order = types_and_orders[2]
        k_dtype, _ = types_and_orders[3]

        u = np.array(u, dtype=u_dtype, order=u_order)
        s = np.array(s, dtype=s_dtype, order=s_order)
        vt = np.array(vt, dtype=vt_dtype, order=vt_order)
        k = getattr(np, k_dtype)(k)

        with time_region_cuda() as h2d:
            u_gpu = cuda.to_device(u)
            s_gpu = cuda.to_device(s)
            vt_gpu = cuda.to_device(vt)

        m, n = u.shape[0], vt.shape[1]
        y_dtype, y_order = types_and_orders[4]

        with time_region_cuda() as d_maloc_y:
            y_gpu = cuda.device_array((m, n), dtype=y_dtype, order=y_order)

        with time_region_cuda() as h_maloc_y:
            y_host = (
                cuda.pinned_array_like(y_gpu)
                if pin_memory
                else np.empty_like(
                    y_gpu, dtype=y_dtype, order=y_order
                )  # numpy doesn't derrive order...
            )

        # block_size is given as (num_rows, num_columns)
        num_rows, num_columns = block_size
        blocks_per_grid_x = (n + num_columns - 1) // num_columns
        blocks_per_grid_y = (m + num_rows - 1) // num_rows
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        with time_region_cuda() as kernel_t:
            kernel[blocks_per_grid, (num_columns, num_rows)](
                u_gpu, s_gpu, vt_gpu, k, y_gpu
            )

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
