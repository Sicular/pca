import numpy as np
import h5py
import h5py
from scipy import sparse
from time import time
from pathlib import Path

def orthogonal_col_mat(col_mat):
    for i in range(col_mat.shape[1]):
        for j in range(i):
            proj = np.dot(np.dot(col_mat[:, i], col_mat[:, j]), col_mat[:, j])
            col_mat[:, i] -= proj
        col_mat[:, i] /= np.linalg.norm(col_mat[:, i])


def power_method(mat: sparse.csr_matrix, k):
    np.random.seed(42)
    first_vectors = np.random.rand(mat.shape[1], k)

    # orthogonal_col_mat(first_vectors)
    first_vectors, _ = np.linalg.qr(first_vectors)
    eig_mat = mat.transpose() @ (mat @ first_vectors)

    # orthogonal_col_mat(eig_mat)
    eig_mat, _ = np.linalg.qr(eig_mat)

    return mat @ eig_mat


def center_power_method(mat: sparse.csr_matrix, k):
    np.random.seed(42)
    
    first_vectors = np.random.rand(mat.shape[1], k)
    first_vectors, _ = np.linalg.qr(first_vectors)

    mean_col = np.mean(mat, axis=0)

    mul_part_1 = mat @ first_vectors - \
        np.ones((mat.shape[0], 1)) @ (mean_col @ first_vectors)

    eig_mat = mat.transpose() @ mul_part_1 - \
        mean_col.T @ (np.ones((1, mat.shape[0])) @ mul_part_1)

    eig_mat, _ = np.linalg.qr(eig_mat)

    return mat @ eig_mat