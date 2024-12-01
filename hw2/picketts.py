import numpy as np


def picketts(A):
    """
    Recursive LU Decomposition based on Pickett's Charge algorithm.
    :param A: The input square matrix.
    :return: L (Lower triangular matrix), U (Upper triangular matrix)
    """
    # Base case for recursion: 1x1 matrix
    if A.shape[0] == 1:
        return np.array([[1]]), A  # L=1, U=A for 1x1 matrix

    # A += 1e-12 * np.eye(A.shape[0])  # Add small perturbation to avoid singular matrix

    # Partition A into blocks
    k = 1  # Current column of focus
    n = A.shape[0]

    A11 = A[:k, :k]  # Top-left submatrix
    a1k = A[:k, k:]  # Top-right column vector (2D)
    ak1 = A[k:, :k]  # Bottom-left row vector (2D)
    Akk = A[k:, k:]  # Bottom-right submatrix

    # Recursive step to compute LU for A11
    L11, U11 = picketts(A11)

    # Compute intermediate terms
    u1k = np.linalg.solve(L11, a1k)  # Solve L11 * u1k = a1k
    l_k1 = np.linalg.solve(U11.T, ak1.T).T  # Solve U11^T * l_k1^T = ak1^T

    # Compute Schur complement (reduced Akk block)
    vkk = Akk - l_k1 @ u1k

    # Recursively compute LU decomposition for the Schur complement
    Lkk, Ukk = picketts(vkk)

    # Assemble the full L and U matrices
    L = np.block([[L11, np.zeros((k, n - k))], [l_k1, Lkk]])
    U = np.block([[U11, u1k], [np.zeros((n - k, k)), Ukk]])

    return L, U
