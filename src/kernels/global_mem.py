from numba import cuda, float32, float64


@cuda.jit(
    "void(Array(float64, 2, 'C'), Array(float64, 1, 'C'), Array(float64, 2, 'C'), int32, Array(float64, 2, 'C'))",
    fastmath=True,
)
def fp64(u, s, vt, k, y):
    # row = y-dimension, col = x-dimension
    col, row = cuda.grid(2)

    if row >= u.shape[0] or col >= vt.shape[1]:
        return

    element = float64(0)
    for p in range(k):
        element += u[row, p] * s[p] * vt[p, col]

    y[row, col] = element


@cuda.jit(
    "void(Array(float32, 2, 'C'), Array(float32, 1, 'C'), Array(float32, 2, 'C'), int32, Array(float32, 2, 'C'))",
    fastmath=True,
)
def fp32(u, s, vt, k, y):
    # row = y-dimension, col = x-dimension
    col, row = cuda.grid(2)

    if row >= u.shape[0] or col >= vt.shape[1]:
        return

    element = float32(0)
    for p in range(k):
        element += u[row, p] * s[p] * vt[p, col]

    y[row, col] = element
