import numpy as np


def picketts(A):
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=float)
    n = len(A)

    # Base case
    if n == 1:
        return np.array([[1]]), np.array([[A[0, 0]]])

    L = np.eye(n)  # Initialize L as identity matrix
    U = np.zeros((n, n))

    # First row of U directly from A
    U[0, :] = A[0, :]

    # First column of L
    L[1:, 0] = A[1:, 0] / U[0, 0]

    A_reduced = A[1:, 1:] - np.outer(L[1:, 0], U[0, 1:])
    L_sub, U_sub = picketts(A_reduced)

    # Fill in recursive results
    L[1:, 1:] = L_sub
    U[1:, 1:] = U_sub

    return L, U
