import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.linalg import lu, hilbert
from shermans import shermans
from picketts import picketts
from crouts import crouts

# Test size
n = 4
A = hilbert(n)  # Generate Hilbert matrix

print("Matrix A:")
print(A)
print()

# # Sherman's March
# start = time()
# L1, U1 = shermans(A)
# time_sherman = time() - start
# rel_error_sherman = np.linalg.norm(A - L1 @ U1, 2) / np.linalg.norm(A, 2)

# print("Sherman L:")
# print(L1)
# print("Sherman U:")
# print(U1)
# print(f"Sherman error: {rel_error_sherman}\n")
# print(f"Multiplied L1 and U1:")
# print(L1 @ U1)

# # Pickett's Charge
# start = time()
# L2, U2 = picketts(A)
# time_pickett = time() - start
# rel_error_pickett = np.linalg.norm(A - L2 @ U2, 2) / np.linalg.norm(A, 2)

# print("Pickett L:")
# print(L2)
# print("Pickett U:")
# print(U2)
# print(f"Pickett error: {rel_error_pickett}\n")
# print(f"Multiplied L2 and U2:")
# print(L2 @ U2)

# Crout's Method
start = time()
L3, U3 = crouts(A)
time_crout = time() - start
rel_error_crout = np.linalg.norm(A - L3 @ U3, 2) / np.linalg.norm(A, 2)

print("Crout L:")
print(L3)
print("Crout U:")
print(U3)
print(f"Crout error: {rel_error_crout}\n")
print(f"Multiplied L3 and U3:")
print(L3 @ U3)

# Built-in LU
start = time()
P, L4, U4 = lu(A)
time_builtin = time() - start
rel_error_builtin = np.linalg.norm(A - P @ L4 @ U4, 2) / np.linalg.norm(A, 2)

print("Built-in L:")
print(L4)
print("Built-in U:")
print(U4)
print(f"Multiplied L4 and U4:")
print(P @ L4 @ U4)
print(f"Built-in error: {rel_error_builtin}\n")
