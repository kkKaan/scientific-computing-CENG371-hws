import numpy as np
from power_method import power_method


def inverse_power(A, alpha):
    """
    Implements inverse power method using power_method function
    
    Args:
        A (ndarray): Input square matrix
        alpha (float): Shift value
    
    Returns:
        tuple: (eigenvalue, eigenvector)
    """
    n = len(A)
    I = np.eye(n)
    A_shifted = A - alpha * I

    # Use power method on inverse of shifted matrix
    eVal, eVec = power_method(np.linalg.inv(A_shifted))

    # Convert back to original eigenvalue
    eigenvalue = alpha + 1 / eVal

    return eigenvalue, eVec
