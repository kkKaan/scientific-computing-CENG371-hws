import numpy as np
import scipy.io
import time

from power_method import power_method


def power_k(A, k):
    """
    Find k largest eigenvalues and eigenvectors using deflation
    
    Args:
        A (ndarray): Input matrix
        k (int): Number of eigenvalues to find
        
    Returns:
        tuple: (eigenvalues array, eigenvectors matrix)
    """
    n = A.shape[0]
    eVals = np.zeros(k)
    eVecs = np.zeros((n, k))
    A_deflated = A.copy()

    for i in range(k):
        # Find current largest eigenvalue/vector
        eval_i, evec_i = power_method(A_deflated)
        eVals[i] = eval_i
        eVecs[:, i] = evec_i

        # Proper deflation using projection matrix
        P = np.outer(evec_i, evec_i) / np.inner(evec_i, evec_i)
        A_deflated = A_deflated - eval_i * P

    return eVals, eVecs


if __name__ == "__main__":
    # # Q1.3
    # A = np.array([[2, -1, 0, 0, 0], [-1, 2, -1, 0, 0], [0, -1, 2, -1, 0], [0, 0, -1, 2, -1], [0, 0, 0, -1,
    #                                                                                           2]])
    # eVals, eVecs = power_k(A, 5)
    # print("Eigenvalues:", eVals)
    # print("Eigenvectors:", eVecs)

    # # Q1.4
    # B = np.array([[0.2, 0.3, -0.5], [0.6, -0.8, 0.2], [-1, 0.1, 0.9]])
    # eVals, eVecs = power_k(B, 3)
    # print("Eigenvalues:", eVals)
    # print("Eigenvectors:", eVecs)

    # Q2.4
    # Read from mtx file and keep the time of that operation
    A = scipy.io.mmread('hw3/can_229.mtx').toarray()

    start = time.time()
    eVals, eVecs = power_k(A, 229)
    end = time.time()
    # print("Eigenvalues:", eVals)
    # print("Eigenvectors:", eVecs)
    print("Time (s): ", end - start)
