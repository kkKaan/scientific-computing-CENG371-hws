import numpy as np


def power_method(A, V=None):
    """
    Power method to compute the dominant eigenvalue and corresponding eigenvector of matrix A.

    Args:
        A (numpy.ndarray): The input matrix
        V (numpy.ndarray, optional): Initial guess vector. If None, uses ones vector

    Returns:
        tuple: (eigenvalue, eigenvector)
    """
    if V is None:
        n = A.shape[0]
        V = np.ones(n)

    V = V / np.linalg.norm(V)
    max_iter = 1000
    tol = 1e-10

    for _ in range(max_iter):
        V_old = V.copy()
        V = A @ V
        V = V / np.linalg.norm(V)

        if np.allclose(abs(V), abs(V_old), rtol=tol):
            break

    eigenvalue = np.dot(A @ V, V) / np.dot(V, V)
    return eigenvalue, V
