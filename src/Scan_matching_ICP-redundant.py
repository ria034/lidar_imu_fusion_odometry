import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from scipy.linalg import expm
import os.path as osp
from pathlib import Path
from math import sqrt


def load_velodyne_scan(bin_path):
    """
    Load a single KITTI velodyne scan from a .bin file.

    Parameters
    ----------
    bin_path : str — path to .bin file

    Returns
    -------
    points : ndarray (N, 3) — XYZ points (intensity dropped)
    """
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3]  # drop intensity, keep XYZ only


def load_velodyne_sequence(sequence_path):
    """
    Load all .bin files from a KITTI velodyne sequence folder.

    Parameters
    ----------
    sequence_path : str — path to velodyne folder
                          e.g. 'sequences/00/velodyne/'

    Returns
    -------
    generator yielding (index, ndarray (N, 3)) one scan at a time
    """
    bin_files = sorted(Path(sequence_path).glob('*.bin'))

    if len(bin_files) == 0:
        raise FileNotFoundError(f"No .bin files found in {sequence_path}")

    print(f"Found {len(bin_files)} scans in {sequence_path}")

    for i, f in enumerate(bin_files):
        yield i, load_velodyne_scan(f)


def EstimateCorrespondence(X, Y, t, R, dmax):
    """
    Estimate point correspondences (Algorithm 1, lines 5-10).

    Parameters
    ----------
    X    : ndarray (nX, d) — source pointcloud
    Y    : ndarray (nY, d) — target pointcloud
    t    : ndarray (d, 1)  — current translation estimate
    R    : ndarray (d, d)  — current rotation estimate
    dmax : float           — max distance for correspondence

    Returns
    -------
    C : ndarray (K, 2) — correspondence pairs [(i, j), ...]
    """
    C = []
    for i in range(len(X)):
        x_transformed = (R @ X[i].reshape(3, 1) + t).ravel()
        dists = np.linalg.norm(Y - x_transformed, axis=1)
        j = np.argmin(dists)
        if dists[j] < dmax:
            C.append([i, j])
    return np.array(C)


def ComputeOptimalRigidRegistration(X, Y, C):
    """
    Compute optimal (R, t) aligning corresponding points (Algorithm 2).

    Parameters
    ----------
    X : ndarray (nX, d) — source pointcloud
    Y : ndarray (nY, d) — target pointcloud
    C : ndarray (K, 2)  — correspondence pairs

    Returns
    -------
    R : ndarray (d, d) — optimal rotation
    t : ndarray (d, 1) — optimal translation
    """
    src_pts = X[C[:, 0]]  # (K, 3)
    tgt_pts = Y[C[:, 1]]  # (K, 3)

    centroid_X = src_pts.mean(axis=0, keepdims=True).T  # (3, 1)
    centroid_Y = tgt_pts.mean(axis=0, keepdims=True).T  # (3, 1)

    A = src_pts - centroid_X.T  # (K, 3)
    B = tgt_pts - centroid_Y.T  # (K, 3)

    W = A.T @ B  # (3, 3)

    U, S, Vt = np.linalg.svd(W)

    D = np.diag([1, 1, np.linalg.det(Vt.T @ U.T)])
    R = Vt.T @ D @ U.T

    t = centroid_Y - R @ centroid_X

    return R, t


def SE3_transform(X, R, t):
    """Apply rigid transformation."""
    return (R @ X.T).T + t.T


def RMSE(X, Y, C):
    """Root-mean-squared error for corresponding points."""
    return np.sqrt(np.linalg.norm(X[C[:, 0]] - Y[C[:, 1]], axis=1).mean())


def ICP(X, Y, t0, R0, dmax, num_ICP_iters):
    """
    Iterative Closest Point (Algorithm 1).

    Parameters
    ----------
    X              : ndarray (nX, d) — source pointcloud
    Y              : ndarray (nY, d) — target pointcloud
    t0             : ndarray (d, 1)  — initial translation
    R0             : ndarray (d, d)  — initial rotation
    dmax           : float           — max correspondence distance
    num_ICP_iters  : int             — number of iterations

    Returns
    -------
    R : ndarray (d, d) — estimated rotation
    t : ndarray (d, 1) — estimated translation
    C : ndarray (K, 2) — final correspondences
    """
    t = t0.copy()
    R = R0.copy()
    for i in range(num_ICP_iters):
        C = EstimateCorrespondence(X, Y, t, R, dmax)
        R, t = ComputeOptimalRigidRegistration(X, Y, C)
    return R, t, C


def voxel_downsample(points, voxel_size=0.3):
    """
    Downsample pointcloud using a voxel grid.
    Needed for KITTI scans (~100k points) to make ICP fast enough.

    Parameters
    ----------
    points     : ndarray (N, 3)
    voxel_size : float — voxel grid resolution in meters

    Returns
    -------
    downsampled : ndarray (M, 3) where M << N
    """
    voxel_indices = np.floor(points / voxel_size).astype(int)
    unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    downsampled = np.array([
        points[inverse == i].mean(axis=0) for i in range(len(unique_voxels))
    ])
    return downsampled


if __name__ == "__main__":

    # --- Config ---
    VELODYNE_PATH = r"C:\Users\lenovo\Downloads\KITTI\sequences\00\velodyne"
    VOXEL_SIZE    = 0.3      # downsample resolution (meters)
    DMAX          = 1.0      # larger than before — KITTI scale
    NUM_ICP_ITERS = 30

    t0 = np.zeros((3, 1))
    R0 = np.eye(3)

    prev_scan = None

    for i, scan in load_velodyne_sequence(VELODYNE_PATH):

        # Downsample — KITTI scans are ~100k points, too slow for naive ICP
        scan_down = voxel_downsample(scan, voxel_size=VOXEL_SIZE)

        if prev_scan is None:
            prev_scan = scan_down
            continue

        print(f"\nRegistering scan {i-1} → {i}  "
              f"(src: {prev_scan.shape[0]} pts, tgt: {scan_down.shape[0]} pts)")

        # Run ICP between consecutive scans
        R_est, t_est, C_est = ICP(prev_scan, scan_down, t0, R0, DMAX, NUM_ICP_ITERS)

        # Validate rotation matrix
        assert np.allclose(R_est @ R_est.T, np.eye(3), atol=1e-4), "R is not orthogonal!"
        assert np.isclose(np.linalg.det(R_est), 1.0, atol=1e-4),   "det(R) != 1!"

        # RMSE
        src_transformed = SE3_transform(prev_scan, R_est, t_est)
        rmse = RMSE(src_transformed, scan_down, C_est)
        print(f"  RMSE      : {rmse:.4f}")
        print(f"  Translation: {t_est.ravel()}")

        # Visualize every 10th pair
        if i % 10 == 1:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(scan_down[:, 0],      scan_down[:, 1],      scan_down[:, 2],
                       c='red',   label='Target',             alpha=0.4, s=1)
            ax.scatter(src_transformed[:, 0], src_transformed[:, 1], src_transformed[:, 2],
                       c='green', label='Source (Transformed)', alpha=0.6, s=1)
            ax.set_title(f'ICP Alignment — scans {i-1} → {i}')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.legend()
            plt.tight_layout()
            plt.show()

        prev_scan = scan_down  # slide window forward