import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.linalg import lu, hilbert
from shermans import shermans
from picketts import picketts
from crouts import crouts

max_n = 300
n_values = range(2, max_n + 1, 2)

time_sherman = []
rel_error_sherman = []
offset_sherman = 1e-18

time_pickett = []
rel_error_pickett = []
offset_pickett = 2e-18

time_crout = []
rel_error_crout = []
offset_crout = 3e-18

time_builtin = []
rel_error_builtin = []

for n in n_values:
    A = hilbert(n)
    A_norm = np.linalg.norm(A, 2)

    # Sherman's March
    start = time()
    L1, U1 = shermans(A)
    time_sherman.append(time() - start)
    rel_error_sherman.append(np.linalg.norm(A - L1 @ U1, 2) / A_norm + offset_sherman)

    # Pickett's Charge
    start = time()
    L2, U2 = picketts(A)
    time_pickett.append(time() - start)
    rel_error_pickett.append(np.linalg.norm(A - L2 @ U2, 2) / A_norm + offset_pickett)

    # Crout's Method
    start = time()
    L3, U3 = crouts(A)
    time_crout.append(time() - start)
    rel_error_crout.append(np.linalg.norm(A - L3 @ U3, 2) / A_norm + offset_crout)

    # Built-in LU
    start = time()
    P, L4, U4 = lu(A)
    time_builtin.append(time() - start)
    rel_error_builtin.append(np.linalg.norm(A - P @ L4 @ U4, 2) / A_norm)

    print(f'Matrix size {n} completed')

# Runtime comparison plot
plt.figure(figsize=(10, 6))
plt.plot(n_values, time_sherman, 'r', label='Sherman')
plt.plot(n_values, time_pickett, 'g', label='Pickett')
plt.plot(n_values, time_crout, 'b', label='Crout')
plt.plot(n_values, time_builtin, 'k', label='Built-in')
plt.title('Runtime Comparison')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)
plt.savefig('runtime_comparison.png')
plt.close()

# Relative Error Comparison plot
plt.figure(figsize=(10, 6))
plt.semilogy(n_values, rel_error_sherman, 'r', label='Sherman')
plt.semilogy(n_values, rel_error_pickett, 'g', label='Pickett')
plt.semilogy(n_values, rel_error_crout, 'b', label='Crout')
plt.semilogy(n_values, rel_error_builtin, 'k', label='Built-in')
plt.title('Relative Error Comparison')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Relative Error (offset added around 1e-18)')
plt.legend()
plt.grid(True)
plt.savefig('error_comparison.png')
plt.close()

# print(f'Sherman relative error for n=23: {rel_error_sherman[22]:e}')
# print(f'Pickett relative error for n=23: {rel_error_pickett[22]:e}')
# print(f'Crout relative error for n=23: {rel_error_crout[22]:e}')
# print(f'Built-in relative error for n=23: {rel_error_builtin[22]:e}')
