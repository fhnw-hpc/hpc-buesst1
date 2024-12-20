from numba import cuda, float32, float64


@cuda.jit(
    "void(float64[:,::1], float64[::1], float64[:,::1], int32, float64[:,::1])",
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
    "void(float32[:,::1], float32[::1], float32[:,::1], int32, float32[:,::1])",
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
