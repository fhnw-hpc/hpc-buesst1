import time
import numba
from numba import cuda
import numpy as np
import pandas as pd
from typing import List, Union


class time_region:
    """Context Manager for CPU-based time measurement.

    Args:
        time_offset (float, optional): Optional time offset to be added. Defaults to 0.
    """

    def __init__(self, time_offset=0):
        self._time_offset = time_offset

    def __enter__(self):
        self._t_start = time.time()
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        self._t_end = time.time()

    def elapsed_time_s(self):
        """Get the elapsed time in seconds.

        Returns:
            float: The measured time including any time offset.
        """
        return self._time_offset + (self._t_end - self._t_start)


class time_region_cuda:
    """Context Manager for GPU-based time measurement with CUDA streams.

    Args:
        time_offset (float, optional): Optional time offset to be added. Defaults to 0.
        cuda_stream (int or cuda.cudadrv.stream.Stream, optional): CUDA stream to use for recording events.
            Defaults to 0 (the default stream).
    """

    def __init__(self, time_offset=0, cuda_stream=0):
        self._t_start = cuda.event(timing=True)
        self._t_end = cuda.event(timing=True)
        self._time_offset = time_offset
        self._cuda_stream = cuda_stream

    def __enter__(self):
        """Record the start event on the given stream."""
        self._t_start.record(self._cuda_stream)
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        """Record and synchronize the end event on the given stream."""
        self._t_end.record(self._cuda_stream)
        self._t_end.synchronize()

    def elapsed_time_s(self):
        """Get the elapsed time in seconds.

        Returns:
            float: The measured time including any time offset.
        """
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


def get_n_timings(input: tuple, reco_func: callable, n=10):
    """
    Executes a reconstruction function multiple times and collects timing information.

    Args:
        input (tuple): Input data for the reconstruction function, typically (u, s, vt, k),
            where `u` is the left singular matrix, `s` is the singular values,
            `vt` is the right singular matrix, and `k` is the number of singular components to use.
        reco_func (callable): The reconstruction function to evaluate. This function should return
            a tuple containing the result and a dictionary of timings.
        n (int, optional): The number of repetitions to perform. Default is 10.

    Returns:
        pd.DataFrame: A DataFrame containing the timing information from all repetitions, with an
        additional column indicating the repetition index.
    """

    dfs = []
    for i in range(n):
        # Get timings for the current repetition
        df = get_timings(input, reco_func)
        df["repeat"] = i  # Add repetition index
        dfs.append(df)

    # Combine all repetitions into a single DataFrame
    return pd.concat(dfs)


def get_n_timings_from_inputs(
    inputs: List[tuple], reco_func: callable, names=List[str], n=10
):
    """
    Executes multiple reconstructions from different inputs repeatedly and collects timing information.

    Args:
        inputs List(tuple): List of input data for the reconstruction function, typically (u, s, vt, k),
            where `u` is the left singular matrix, `s` is the singular values,
            `vt` is the right singular matrix, and `k` is the number of singular components to use.
        reco_func (callable): The reconstruction function to evaluate. This function should return
            a tuple containing the result and a dictionary of timings.
        names (List[str]): A list of names corresponding to the inputs.
        n (int, optional): The number of repetitions to perform for each input. Default is 10.

    Returns:
        pd.DataFrame: A DataFrame containing the timing information from all inputs and repetitions,
        with additional columns indicating the input name and the repetition index.

    Raises:
        AssertionError: If the number of inputs does not match the number of names.
    """

    assert len(inputs) == len(names), "Each inputs must have a corresponding name."

    dfs = []
    for i in range(len(inputs)):
        input = inputs[i]
        name = names[i]

        # Get timings for the current input
        df = get_n_timings(input, reco_func, n)
        df["input_name"] = name  # Add input name

        dfs.append(df)

    # Combine all results into a single DataFrame
    return pd.concat(dfs)


def get_n_timings_from_kernels(
    input: tuple, reco_funcs=List[callable], names=List[str], n=10
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
        n (int, optional): The number of repetitions to perform for each function. Default is 10.

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
        df = get_n_timings(input, reco_func, n)
        df["reco_name"] = name  # Add function name

        dfs.append(df)

    # Combine all results into a single DataFrame
    return pd.concat(dfs)


def get_n_timings_from_inputs_and_kernels(
    inputs: List[tuple],
    reco_funcs=List[callable],
    input_names=List[str],
    func_names=List[str],
    n=10,
):
    """
    Args:
        inputs List(tuple): List of input data for the reconstruction function, typically (u, s, vt, k),
            where `u` is the left singular matrix, `s` is the singular values,
            `vt` is the right singular matrix, and `k` is the number of singular components to use.
        reco_funcs (List[callable]): A list of reconstruction functions to evaluate. Each function
            should return a tuple containing the result and a dictionary of timings.
        input_names (List[str]): A list of names corresponding to the inputs.
        func_names (List[str]): A list of names corresponding to the reconstruction functions.
        n (int, optional): The number of repetitions to perform for each input and function. Default is 10.
    """

    assert len(inputs) == len(
        input_names
    ), "Each inputs must have a corresponding name."

    assert len(reco_funcs) == len(
        func_names
    ), "Each reconstruction function must have a corresponding name."

    dfs = []
    for i, reco_func in enumerate(reco_funcs):
        df = get_n_timings_from_inputs(inputs, reco_func, input_names, n)
        df["reco_name"] = func_names[i]  # Add function name

        dfs.append(df)

    # Combine all results into a single DataFrame
    return pd.concat(dfs)


def make_reconstructor(
    kernel: callable,
    block_size: tuple,
    pin_memory: bool = False,
    timeit: bool = False,
    use_streams: bool = False,
):
    """Creates one or more reconstruction functions using a provided CUDA kernel.

    Args:
        kernel (callable): The CUDA kernel function to be used for reconstruction.
        block_size (tuple): The size of the CUDA thread block, as (num_rows, num_columns).
            Note: internally columns map to x-dim and rows map to y-dim.
        pin_memory (bool, optional): If True, the output array will use pinned host memory. Defaults to False.
        timeit (bool, optional): If True, a timed version of the reconstruction function is returned. Defaults to False.
        use_streams (bool, optional): If True, a version accepting lists of (u, s, vt) to process in parallel streams
            is returned. Defaults to False.

    Returns:
        callable: Depending on the flags, one of the following functions:
            - If `use_streams=False` and `timeit=False`:
                `inner(u, s, vt, k) -> np.ndarray`
            - If `use_streams=False` and `timeit=True`:
                `inner_timed(u, s, vt, k) -> (np.ndarray, dict)`
            - If `use_streams=True` and `timeit=False`:
                `inner_streams_no_timing(u_list, s_list, vt_list, k) -> List[np.ndarray]`
            - If `use_streams=True` and `timeit=True`:
                `inner_streams_timed(u_list, s_list, vt_list, k) -> (List[np.ndarray], List[dict])`
    """

    def infer_types_and_orders(kernel: cuda.dispatcher.CUDADispatcher):
        """Infers data types and memory order from the kernel's signature.

        Args:
            kernel (cuda.dispatcher.CUDADispatcher): The Numba CUDA kernel with exactly one signature.

        Returns:
            list: A list of tuples (dtype_string, order), or (dtype_string, None) for scalars.

        Raises:
            ValueError: If an unsupported argument type is found in the kernel signature.
        """
        assert (
            len(kernel.signatures) == 1
        ), "Numba kernel is not allowed to have multiple signatures."
        signature = kernel.signatures[0]
        types_and_orders = []
        for arg in signature:
            if isinstance(arg, numba.types.npytypes.Array):
                dtype = arg.dtype.name  # e.g., 'float64'
                order = "C" if arg.layout == "C" else "F"
                types_and_orders.append((dtype, order))
            elif isinstance(arg, numba.types.Integer):
                # For int32, int64, etc.
                types_and_orders.append((arg.name, None))
            else:
                raise ValueError(
                    f"Unsupported argument type in kernel signature: {arg}"
                )
        return types_and_orders

    def inner(u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: int):
        """Reconstruct a single matrix using the provided CUDA kernel.

        Args:
            u (np.ndarray): Left singular matrix of shape (m, n).
            s (np.ndarray): Singular values of shape (min(m, n),).
            vt (np.ndarray): Right singular matrix of shape (n, n).
            k (int): Number of singular components to use.

        Returns:
            np.ndarray: The reconstructed matrix of shape (m, n).
        """
        types_and_orders = infer_types_and_orders(kernel)
        u_dtype, u_order = types_and_orders[0]
        s_dtype, s_order = types_and_orders[1]
        vt_dtype, vt_order = types_and_orders[2]
        k_dtype, _ = types_and_orders[3]
        y_dtype, y_order = types_and_orders[4]

        # Match dtypes/orders
        u = np.array(u, dtype=u_dtype, order=u_order)
        s = np.array(s, dtype=s_dtype, order=s_order)
        vt = np.array(vt, dtype=vt_dtype, order=vt_order)
        k = getattr(np, k_dtype)(k)

        # pinning of u,s,vt isn't done here since DMA (Direct Memory Access) brings not really an advantage here

        # Transfer to GPU
        u_gpu = cuda.to_device(u)
        s_gpu = cuda.to_device(s)
        vt_gpu = cuda.to_device(vt)

        # Determine output array dimensions
        m, n = u.shape[0], vt.shape[1]
        y_gpu = cuda.device_array((m, n), dtype=y_dtype, order=y_order)

        # block_size is given as (num_rows, num_columns)
        num_rows, num_columns = block_size

        # Since we want col = x-dim and row = y-dim, blockDim should be (num_columns, num_rows)
        # and grid dimensions should be calculated accordingly:
        blocks_per_grid_x = (n + num_columns - 1) // num_columns  # columns mapped to x
        blocks_per_grid_y = (m + num_rows - 1) // num_rows  # rows mapped to y
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Launch CUDA kernel with adjusted block dimensions
        kernel[blocks_per_grid, (num_columns, num_rows)](u_gpu, s_gpu, vt_gpu, k, y_gpu)

        # allocate memory on host and pin if requrested
        if pin_memory:
            y_host = cuda.pinned_array_like(y_gpu)
        else:
            # Ensure correct shape/order
            y_host = np.empty((m, n), dtype=y_dtype, order=y_order)

        # Copy result back to host
        y_gpu.copy_to_host(y_host)

        return y_host

    def inner_timed(u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: int):
        """Reconstruct a single matrix using the provided CUDA kernel and measure timings.

        Args:
            u (np.ndarray): Left singular matrix of shape (m, n).
            s (np.ndarray): Singular values of shape (min(m, n),).
            vt (np.ndarray): Right singular matrix of shape (n, n).
            k (int): Number of singular components to use.

        Returns:
            Tuple[np.ndarray, dict]:
                A tuple of (reconstructed matrix, dictionary of timing information).
        """
        types_and_orders = infer_types_and_orders(kernel)
        u_dtype, u_order = types_and_orders[0]
        s_dtype, s_order = types_and_orders[1]
        vt_dtype, vt_order = types_and_orders[2]
        k_dtype, _ = types_and_orders[3]
        y_dtype, y_order = types_and_orders[4]

        # Data conversions
        u = np.array(u, dtype=u_dtype, order=u_order)
        s = np.array(s, dtype=s_dtype, order=s_order)
        vt = np.array(vt, dtype=vt_dtype, order=vt_order)
        k = getattr(np, k_dtype)(k)

        # pinning of u,s,vt isn't done here since DMA (Direct Memory Access) brings not really an advantage here

        with time_region_cuda() as h2d:
            u_gpu = cuda.to_device(u)
            s_gpu = cuda.to_device(s)
            vt_gpu = cuda.to_device(vt)

        m, n = u.shape[0], vt.shape[1]

        with time_region_cuda() as d_maloc_y:
            y_gpu = cuda.device_array((m, n), dtype=y_dtype, order=y_order)

        # block_size is given as (num_rows, num_columns)
        num_rows, num_columns = block_size
        blocks_per_grid_x = (n + num_columns - 1) // num_columns
        blocks_per_grid_y = (m + num_rows - 1) // num_rows
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        with time_region_cuda() as kernel_t:
            # start kernel execution
            kernel[blocks_per_grid, (num_columns, num_rows)](
                u_gpu, s_gpu, vt_gpu, k, y_gpu
            )

        with time_region_cuda() as h_maloc_y:
            # allocate memory on host and pin if requrested
            if pin_memory:
                y_host = cuda.pinned_array_like(y_gpu)
            else:
                # Ensure correct shape/order
                y_host = np.empty((m, n), dtype=y_dtype, order=y_order)

        with time_region_cuda() as d2h:
            # copy data back to host
            y_gpu.copy_to_host(y_host)

        # Collect timings
        timings = {
            "h2d": h2d.elapsed_time_s(),
            "d_maloc_y": d_maloc_y.elapsed_time_s(),
            "h_maloc_y": h_maloc_y.elapsed_time_s(),
            "kernel": kernel_t.elapsed_time_s(),
            "d2h": d2h.elapsed_time_s(),
        }
        mem_operations_total = (
            timings["h2d"]
            + timings["d_maloc_y"]
            + timings["h_maloc_y"]
            + timings["d2h"]
        )
        timings["mem_operations_total"] = mem_operations_total
        timings["total"] = mem_operations_total + timings["kernel"]

        return y_host, timings

    def inner_streams(
        u_list: List[np.ndarray],
        s_list: List[np.ndarray],
        vt_list: List[np.ndarray],
        k: Union[int, List[int]],
    ):
        """Reconstruct multiple matrices in parallel streams.

        Args:
            u_list (List[np.ndarray]): List of 'u' matrices (each shape (m, n)).
            s_list (List[np.ndarray]): List of 's' vectors (each shape (min(m, n),)).
            vt_list (List[np.ndarray]): List of 'vt' matrices (each shape (n, n)).
            k (Union[int, List[int]]): Number of singular components to use. Can be a single int
                applied to all jobs or a list of length len(u_list).

        Returns:
            List[np.ndarray]: A list of reconstructed matrices, one per input triplet.
        """

        assert (
            len(u_list) == len(s_list) == len(vt_list)
        ), "Input lists must have the same length."

        n_jobs = len(u_list)

        # Normalize k to a list
        if isinstance(k, int):
            k_list = [k] * n_jobs
        else:
            k_list = k
            assert (
                len(k_list) == n_jobs
            ), "If k is a list, it must match the number of jobs."

        # Kernel signature
        types_and_orders = infer_types_and_orders(kernel)
        u_dtype, u_order = types_and_orders[0]
        s_dtype, s_order = types_and_orders[1]
        vt_dtype, vt_order = types_and_orders[2]
        k_dtype, _ = types_and_orders[3]
        y_dtype, y_order = types_and_orders[4]

        # Create streams
        streams = [cuda.stream() for _ in range(n_jobs)]
        results = [None] * n_jobs

        # stop memory cleanup during operation
        with cuda.defer_cleanup():

            # Transfer, launch, copy back in each stream
            for i in range(n_jobs):
                # here in steams also the input arrays must be pinned to use the DMA engines (Direct Memory Access)
                if pin_memory:
                    # create pinned array
                    u_i = cuda.pinned_array(
                        u_list[i].shape, dtype=u_dtype, order=u_order
                    )
                    s_i = cuda.pinned_array(
                        s_list[i].shape, dtype=s_dtype, order=s_order
                    )
                    vt_i = cuda.pinned_array(
                        vt_list[i].shape, dtype=vt_dtype, order=vt_order
                    )

                    # copy data to pinned array
                    np.copyto(u_i, u_list[i])
                    np.copyto(s_i, s_list[i])
                    np.copyto(vt_i, vt_list[i])

                else:

                    u_i = np.array(u_list[i], dtype=u_dtype, order=u_order)
                    s_i = np.array(s_list[i], dtype=s_dtype, order=s_order)
                    vt_i = np.array(vt_list[i], dtype=vt_dtype, order=vt_order)

                k_i = getattr(np, k_dtype)(k_list[i])

                u_gpu = cuda.to_device(u_i, stream=streams[i])
                s_gpu = cuda.to_device(s_i, stream=streams[i])
                vt_gpu = cuda.to_device(vt_i, stream=streams[i])

                m, n = u_i.shape[0], vt_i.shape[1]
                y_gpu = cuda.device_array(
                    (m, n), dtype=y_dtype, order=y_order, stream=streams[i]
                )

                num_rows, num_columns = block_size
                blocks_per_grid_x = (n + num_columns - 1) // num_columns
                blocks_per_grid_y = (m + num_rows - 1) // num_rows
                blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

                kernel[
                    blocks_per_grid,
                    (num_columns, num_rows),
                    streams[i],
                ](u_gpu, s_gpu, vt_gpu, k_i, y_gpu)

                # allocate memory on host and pin if requrested
                if pin_memory:
                    y_host = cuda.pinned_array_like(y_gpu)
                else:
                    # Ensure correct shape/order
                    y_host = np.empty((m, n), dtype=y_dtype, order=y_order)

                # Copy result back to host
                y_gpu.copy_to_host(y_host, stream=streams[i])
                results[i] = y_host

            # Synchronize
            for i in range(n_jobs):
                streams[i].synchronize()

        return results

    # Decide which function to return
    if use_streams:
        if timeit:
            raise NotImplementedError(
                "timeit is not implemented for streamed operation"
            )

        return inner_streams
    else:
        if timeit:
            return inner_timed
        else:
            return inner


# Example usage (to be implemented in a separate module):
# - Define a CUDA kernel for SVD reconstruction.
# - Pass the kernel, block size, and signature to `make_reconstructor`.
# - Use the resulting function for GPU-accelerated SVD reconstruction.
