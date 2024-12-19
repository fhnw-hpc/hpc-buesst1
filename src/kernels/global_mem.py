from numba import cuda, float32, float64


@cuda.jit(
    "void(Array(float64, 2, 'C'), Array(float64, 1, 'C'), Array(float64, 2, 'C'), int32, Array(float64, 2, 'C'))",
    fastmath=True,
)
def fp64(u, s, vt, k, y):
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
def fp32(u, s, vt, k, y):
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
