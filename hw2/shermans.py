import numpy as np


def shermans(A):
    """
    Recursive LU Decomposition based on Sherman's algorithm.
    :param A: The input square matrix.
    :return: L (Lower triangular matrix), U (Upper triangular matrix)
    """
    # Base case for recursion: 1x1 matrix
    if A.shape[0] == 1:
        return np.array([[1]]), A  # L=1, U=A for 1x1 matrix

    A += 1e-12 * np.eye(A.shape[0])  # Add small perturbation to avoid singular matrix

    # Partition A into blocks
    n = A.shape[0]
    A11 = A[:-1, :-1]  # Top-left submatrix
    a1k = A[:-1, -1]  # Top-right column vector
    ak1 = A[-1, :-1]  # Bottom-left row vector
    akk = A[-1, -1]  # Bottom-right scalar

    # Recursively decompose A11
    L11, U11 = shermans(A11)

    # Compute other components
    u1k = np.linalg.solve(L11, a1k)  # u1k = L11⁻¹ * a1k
    l1k = np.linalg.solve(U11.T, ak1.T).T  # l1k = (U11⁻¹)ᵀ * ak1
    ukk = akk - np.dot(l1k, u1k)  # ukk = akk - l1k * u1k

    # Assemble the full L and U matrices
    L = np.block([[L11, np.zeros((n - 1, 1))], [l1k, np.array([[1]])]])
    U = np.block([[U11, u1k.reshape(-1, 1)], [np.zeros((1, n - 1)), np.array([[ukk]])]])

    return L, U
