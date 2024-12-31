import numpy as np
import time
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from scipy.sparse.linalg import svds


def approximate_svd(A, k, p=5):
    """
    Compute a randomized low-rank approximation of the SVD of A.
    
    Args:
      A (np.ndarray, shape (m, n)): Input matrix.
      k (int): Target rank approximation.
      p (int, optional, default=5): Oversampling (safety) parameter.
    
    Returns:
      U_k : np.ndarray, shape (m, k)
      S_k : np.ndarray, shape (k, k)
      V_k : np.ndarray, shape (n, k)
    """
    m, n = A.shape

    # 1. Random Gaussian test matrix
    Omega = np.random.randn(n, k + p)

    # 2. Sample the column space of A
    Y = A @ Omega  # shape (m, k+p)

    # 3. Orthonormal basis for the columns of Y
    Q, _ = np.linalg.qr(Y, mode='reduced')  # shape (m, k+p)

    # 4. Project A onto the subspace spanned by Q
    B = Q.T @ A  # shape ((k+p), n)

    # 5. Compute an exact SVD of the smaller matrix B
    U_tilde, S_full, Vt = np.linalg.svd(B, full_matrices=False)
    # U_tilde shape: (k+p, k+p), S_full shape: (k+p,), Vt shape: (k+p, n)

    # 6. Truncate to rank k
    U_k = Q @ U_tilde[:, :k]  # shape (m, k)
    S_k = np.diag(S_full[:k])  # shape (k, k)
    V_k = Vt[:k, :].T  # shape (n, k)

    return U_k, S_k, V_k


def load_image_as_matrix(path, as_gray=True):
    """
    Load an image from `path` and convert it to a numpy float64 matrix in [0,1].
    as_gray=True: convert to grayscale if the image is RGB.
    """
    img = io.imread(path)  # shape could be (m, n) or (m, n, 3)
    if img.ndim == 3 and as_gray:
        img = rgb2gray(img)
    # Convert to float in [0, 1]
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0
    return img  # shape (m, n)


def compute_relative_error(A, U, S, V):
    """
    Compute the relative error = ||A - USV^T||_2 / ||A||_2,
    using the spectral norm (largest singular value).
    """
    # Reconstruct
    approx = U @ S @ V.T
    # ||A - approx||_2
    diff_norm = np.linalg.norm(A - approx, 2)
    # ||A||_2 = largest singular value of A
    A_norm = np.linalg.norm(A, 2)
    return diff_norm / A_norm


def run_comparison(A, k_list, p=5):
    """
    For each k in k_list, compute:
      - Time and relative error of approximate_svd(A, k)
      - Time and relative error of scipy.sparse.linalg.svds(A, k)
    Returns dictionaries of times and errors for plotting.
    """
    times_approx = []
    errs_approx = []
    times_svds = []
    errs_svds = []

    for k in k_list:
        # -- approximate_svd
        t0 = time.time()
        U_k, S_k, V_k = approximate_svd(A, k, p=p)
        t1 = time.time() - t0
        rel_err_approx = compute_relative_error(A, U_k, S_k, V_k)

        times_approx.append(t1)
        errs_approx.append(rel_err_approx)

        # -- svds
        # SciPy's svds returns singular values in ascending order by default.
        t0 = time.time()
        # "which='LM'" ensures we get largest singular values
        # If A is large, you might prefer A as sparse or handle memory carefully.
        U_s, S_s, V_sT = svds(A, k=k, which='LM')
        # Reorder (descending) if needed
        idx = np.argsort(-S_s)
        S_s = S_s[idx]
        U_s = U_s[:, idx]
        V_sT = V_sT[idx, :]
        t2 = time.time() - t0

        # Construct diagonal matrix
        S_s_mat = np.diag(S_s)

        rel_err_svds = compute_relative_error(A, U_s, S_s_mat, V_sT.T)

        times_svds.append(t2)
        errs_svds.append(rel_err_svds)

    return times_approx, errs_approx, times_svds, errs_svds


def plot_results(k_list, times_approx, errs_approx, times_svds, errs_svds, title_prefix):
    """
    Generate two plots:
      1) Relative error vs. k
      2) Run time vs. k
    """
    # 1) Relative error vs. k
    plt.figure(figsize=(6, 5))
    plt.plot(k_list, errs_approx, 'o-', label='approx_svd error')
    plt.plot(k_list, errs_svds, 's--', label='svds error')
    plt.xlabel('k (Rank)')
    plt.ylabel('Relative Error')
    plt.title(f'{title_prefix} - Relative Error vs. k')
    plt.legend()
    plt.grid(True)
    # plt.show()
    # plt.savefig(f'{title_prefix}_relative_error_vs_k.png')

    # 2) Runtime vs. k
    plt.figure(figsize=(6, 5))
    plt.plot(k_list, times_approx, 'o-', label='approx_svd time')
    plt.plot(k_list, times_svds, 's--', label='svds time')
    plt.xlabel('k (Rank)')
    plt.ylabel('Time (seconds)')
    plt.title(f'{title_prefix} - Run Time vs. k')
    plt.legend()
    plt.grid(True)
    # plt.show()
    # plt.savefig(f'{title_prefix}_run_time_vs_k.png')


def reconstruct_and_display(A, method_name, U, S, V, k):
    """
    Helper function to reconstruct an image from U, S, V and display it with plt.imshow().
    """
    approx_img = U @ S @ V.T
    plt.figure()
    plt.imshow(approx_img, cmap='gray', vmin=0, vmax=1)
    plt.title(f'{method_name} Reconstruction, rank={k}')
    plt.axis('off')
    # plt.show()
    plt.savefig(f'{method_name}_reconstruction_rank_{k}.png')


if __name__ == "__main__":
    import sys

    # Example: compare on two images
    # -------------------------------------
    # Make sure these images exist, or adjust paths accordingly (regarding the current working directory as well):
    cameraman_path = 'cameraman.jpg'
    fingerprint_path = 'fingerprint.jpg'

    # Load images as grayscale matrices
    cam = load_image_as_matrix(cameraman_path, as_gray=True)
    fpt = load_image_as_matrix(fingerprint_path, as_gray=True)

    # Flatten images into 2D arrays if needed
    # (They should be 2D already if read as grayscale.)
    A_cam = cam  # shape (m, n)
    A_fpt = fpt  # shape (m, n)

    # Convert to float64 (already done in load_image_as_matrix, but just to be sure)
    A_cam = A_cam.astype(np.float64)
    A_fpt = A_fpt.astype(np.float64)

    # Range of k values
    k_list = [1, 2, 5, 10, 20, 30, 50, 65, 85, 100, 150, 200]

    # --- cameraman comparisons ---
    print("Running comparison for 'cameraman.jpg' ...")
    times_approx_cam, errs_approx_cam, times_svds_cam, errs_svds_cam = \
        run_comparison(A_cam, k_list, p=5)
    plot_results(k_list,
                 times_approx_cam,
                 errs_approx_cam,
                 times_svds_cam,
                 errs_svds_cam,
                 title_prefix="Cameraman")

    # --- fingerprint comparisons ---
    print("Running comparison for 'fingerprint.jpg' ...")
    times_approx_fpt, errs_approx_fpt, times_svds_fpt, errs_svds_fpt = \
        run_comparison(A_fpt, k_list, p=5)
    plot_results(k_list,
                 times_approx_fpt,
                 errs_approx_fpt,
                 times_svds_fpt,
                 errs_svds_fpt,
                 title_prefix="Fingerprint")

    # ---------- 2) QUALITATIVE COMPARISONS ----------
    # Reconstruct images at chosen ranks
    chosen_ks = [10, 50, 100]

    for k in chosen_ks:
        # approximate_svd on cameraman
        U_k, S_k, V_k = approximate_svd(A_cam, k, p=5)
        reconstruct_and_display(A_cam, f"Approx SVD (Cameraman)", U_k, S_k, V_k, k)

        # svds on cameraman
        U_s, S_s, V_sT = svds(A_cam, k=k, which='LM')
        # reorder descending
        idx = np.argsort(-S_s)
        S_s = S_s[idx]
        U_s = U_s[:, idx]
        V_sT = V_sT[idx, :]
        reconstruct_and_display(A_cam, f"svds (Cameraman)", U_s, np.diag(S_s), V_sT.T, k)

    # Do similarly for fingerprint
    for k in chosen_ks:
        # approximate_svd
        U_k, S_k, V_k = approximate_svd(A_fpt, k, p=5)
        reconstruct_and_display(A_fpt, f"Approx SVD (Fingerprint)", U_k, S_k, V_k, k)

        # svds
        U_s, S_s, V_sT = svds(A_fpt, k=k, which='LM')
        idx = np.argsort(-S_s)
        S_s = S_s[idx]
        U_s = U_s[:, idx]
        V_sT = V_sT[idx, :]
        reconstruct_and_display(A_fpt, f"svds (Fingerprint)", U_s, np.diag(S_s), V_sT.T, k)

    print("All comparisons complete.")
