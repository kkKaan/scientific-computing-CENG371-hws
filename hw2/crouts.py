import numpy as np


def crouts(A):
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=float)
    n = len(A)

    # Base case
    if n == 1:
        return np.array([[A[0, 0]]]), np.array([[1]])

    # Initialize L and U
    L = np.zeros((n, n))
    U = np.eye(n)

    # First column of L is directly from A
    L[:, 0] = A[:, 0]

    # First row of U (except U[0,0]=1)
    U[0, 1:] = A[0, 1:] / L[0, 0]

    A_reduced = A[1:, 1:] - np.outer(L[1:, 0], U[0, 1:])
    L_sub, U_sub = crouts(A_reduced)

    # Fill in submatrices
    L[1:, 1:] = L_sub
    U[1:, 1:] = U_sub

    return L, U
