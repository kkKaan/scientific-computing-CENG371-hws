import numpy as np
import scipy.io
import time


def subspace_iteration(A, k):
    """
    Find k largest eigenvalues and eigenvectors using subspace iteration
    
    Args:
        A (ndarray): Input matrix
        k (int): Number of eigenvalues to find
        
    Returns:
        tuple: (eigenvalues array, eigenvectors matrix)
    """
    n = A.shape[0]
    max_iter = 1000
    tol = 1e-8

    # Initialize random matrix
    Q = np.random.rand(n, k)
    Q, _ = np.linalg.qr(Q)

    # Power iteration with QR
    for _ in range(max_iter):
        Q_old = Q.copy()
        Z = A @ Q
        Q, _ = np.linalg.qr(Z)

        # Check convergence
        if np.allclose(abs(Q), abs(Q_old), rtol=tol):
            break

    # Compute Rayleigh quotient
    R = Q.T @ A @ Q
    eVals, eVecs = np.linalg.eig(R)

    # Sort by magnitude of eigenvalues
    idx = np.argsort(abs(eVals))[::-1]
    eVals = eVals[idx]
    eVecs = Q @ eVecs[:, idx]

    return eVals[:k], eVecs[:, :k]


if __name__ == "__main__":
    # # Q1.3
    # A = np.array([[2, -1, 0, 0, 0], [-1, 2, -1, 0, 0], [0, -1, 2, -1, 0], [0, 0, -1, 2, -1], [0, 0, 0, -1,
    #                                                                                           2]])
    # eVals, eVecs = subspace_iteration(A, 5)
    # print("Eigenvalues:", eVals)
    # print("Eigenvectors:", eVecs)

    # # Q1.4
    # B = np.array([[0.2, 0.3, -0.5], [0.6, -0.8, 0.2], [-1, 0.1, 0.9]])
    # eVals, eVecs = subspace_iteration(B, 3)
    # print("Eigenvalues:", eVals)
    # print("Eigenvectors:", eVecs)

    # Q2.4
    # Read from mtx file and keep the time of that operation
    A = scipy.io.mmread('hw3/can_229.mtx').toarray()

    start = time.time()
    eVals, eVecs = subspace_iteration(A, 229)
    end = time.time()
    # print("Eigenvalues:", eVals)
    # print("Eigenvectors:", eVecs)
    print("Time (s): ", end - start)
