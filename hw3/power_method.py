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
        # Give random vector of size n
        V = np.random.rand(n)

    V = V / np.linalg.norm(V)
    max_iter = 1000
    tol = 1e-8

    for _ in range(max_iter):
        V_old = V.copy()
        V = A @ V
        V = V / np.linalg.norm(V)

        if np.allclose(abs(V), abs(V_old), rtol=tol):
            break

    eigenvalue = np.dot(A @ V, V) / np.dot(V, V)
    return eigenvalue, V


if __name__ == "__main__":
    # Q1.3
    A = np.array([[2, -1, 0, 0, 0], [-1, 2, -1, 0, 0], [0, -1, 2, -1, 0], [0, 0, -1, 2, -1], [0, 0, 0, -1,
                                                                                              2]])
    eVal, eVec = power_method(A)
    # print("Eigenvalue:", eVal)
    # print("Eigenvector:", eVec)

    # Q1.4
    B = np.array([[0.2, 0.3, -0.5], [0.6, -0.8, 0.2], [-1, 0.1, 0.9]])
    eVal, eVec = power_method(B)
    print("Eigenvalue:", eVal)
    print("Eigenvector:", eVec)
