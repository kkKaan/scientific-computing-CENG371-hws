import numpy as np


def crouts(A):
    """
    Recursive LU Decomposition based on Crout's algorithm.
    :param A: The input square matrix.
    :return: L (Lower triangular matrix), U (Upper triangular matrix)
    """
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=float)
    n = len(A)

    # Initialize matrices
    L = np.zeros((n, n))
    U = np.eye(n)

    # Recursive implementation
    _crouts_recursive(A, L, U, 0, n)

    return L, U


def _crouts_recursive(A, L, U, j, n):
    # Base case
    if j >= n:
        return

    # Compute L[:, j]
    for i in range(j, n):
        sum_l = sum(L[i, k] * U[k, j] for k in range(j))
        L[i, j] = A[i, j] - sum_l

    # Compute U[j, :]
    for i in range(j + 1, n):
        sum_u = sum(L[j, k] * U[k, i] for k in range(j))
        U[j, i] = (A[j, i] - sum_u) / L[j, j]

    # Recursive call for next column
    _crouts_recursive(A, L, U, j + 1, n)
