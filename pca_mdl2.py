from scipy.sparse import spmatrix
from typing import Union
import numpy as np
from matplotlib import pyplot as plt
from _oja import _oja_batch
import cupy as cp
import cupyx as cpx

def _oja_grad(
    x: Union[np.ndarray, spmatrix], mean: np.ndarray, w: np.ndarray, lib = np
) -> np.ndarray:
    colsums = lib.array(x.sum(axis=0)).reshape(1, -1)
    return -(
        x.T @ (x @ w)
        - mean @ (colsums @ w)
        + (mean * x.shape[0] - colsums.T) @ (mean.T @ w)
    )

def _adam_coeffs(G, t, m, v, beta1, beta2, lib = np):
    m = beta1 * m + (1 - beta1) * G
    v = beta2 * v + (1 - beta2) * (lib.linalg.norm(G, axis=0) ** 2)
    mh = m / (1 - beta1**t)
    vh = v / (1 - beta2**t)

    return (m, v, mh, vh)

def oja_batch(
    matrix: Union[np.ndarray, spmatrix],
    k: int,
    batch_sz=2048,
    step_sz=1e-3,
    epsilon=1e-8,
    beta1=0.9,
    beta2=0.999,
) -> np.ndarray:
    """AdaOja principal component analysis implementation.

    Parameters
    ----------
        matrix : array or sparse matrix
                Input (objects x features) matrix for PCA.
        k : int
                Number of principal components.

    Returns
    -------
        pca : array
                (objects x k) PCA matrix.

    Examples
    --------
        >>> pca_mtx = history_pca(matrix, 50)
        array([[ 1.33843906e+02,  5.88548225e+02, -4.29797162e+02, ...,
        -5.39586482e+00,  9.15233611e+00,  1.46966907e-01],
        ...,
        [ 2.85430135e+02,  8.86089300e+02,  1.25807185e+02, ...,
        -4.81416391e+00,  3.71223338e+00,  3.77609306e+00]])
    """
    indices = np.arange(matrix.shape[0])
    np.random.shuffle(indices)
    matrix = matrix[indices]

    n_rows, n_cols = matrix.shape
    w = np.random.normal(0, 1, (n_cols, k)).astype("float32")
    w, _ = np.linalg.qr(w)
    batch_sz = min(batch_sz, n_rows // 10)
    mean = np.array(matrix.mean(axis=0)).reshape(-1, 1)
    m, v, t = 0, 0, 0

    t += 1
    sX = matrix[0:batch_sz]
    g = _oja_grad(sX, mean, w)

    m, v, mh, vh = _adam_coeffs(g, t, m, v, beta1, beta2)
    step_sz = max(step_sz, 0.5 / (np.mean(np.abs(mh) / np.sqrt(vh + epsilon))))
    s = w - step_sz * (mh / (np.sqrt(vh) + epsilon))
    w, _ = np.linalg.qr(s)

    print("step size: ", step_sz)
    for start in range(0, n_rows, batch_sz):
        t += 1
        end = min(start + batch_sz, n_rows)
        sX = matrix[start:end]
        g = _oja_grad(sX, mean, w)

        m, v, mh, vh = _adam_coeffs(g, t, m, v, beta1, beta2)
        s = w - step_sz * (mh / (np.sqrt(vh) + epsilon))
        w, _ = np.linalg.qr(s)
        
    return matrix @ w - mean.T @ w

def oja_batch_cupy(
    matrix: Union[np.ndarray, spmatrix],
    k: int = 6,
    batch_sz=2048,
    step_sz=1e-3,
    epsilon=1e-8,
    beta1=0.9,
    beta2=0.999,
) -> np.ndarray:
    """AdaOja principal component analysis implementation.

    Parameters
    ----------
        matrix : array or sparse matrix
                Input (objects x features) matrix for PCA.
        k : int
                Number of principal components.

    Returns
    -------
        pca : array
                (objects x k) PCA matrix.

    Examples
    --------
        >>> pca_mtx = history_pca(matrix, 50)
        array([[ 1.33843906e+02,  5.88548225e+02, -4.29797162e+02, ...,
        -5.39586482e+00,  9.15233611e+00,  1.46966907e-01],
        ...,
        [ 2.85430135e+02,  8.86089300e+02,  1.25807185e+02, ...,
        -4.81416391e+00,  3.71223338e+00,  3.77609306e+00]])
    """
    from time import time

    t1 = time()
    indices = cp.arange(matrix.shape[0])
    cp.random.shuffle(indices)
    print(time() - t1) #0.4s
    
    matrix = matrix[indices.get()]
    print(time() - t1) #19s

    n_rows, n_cols = matrix.shape
    w = cp.random.normal(0, 1, (n_cols, k)).astype("float32")
    w, _ = cp.linalg.qr(w)
    batch_sz = min(batch_sz, n_rows // 10)
    print(time() - t1) #1.2s
    
    sum = cp.zeros((1, n_cols))
    for start in range(0, n_rows, batch_sz):
        end = min(start + batch_sz, n_rows)
        sX = cpx.scipy.sparse.csr_matrix(
            matrix[start:end],
            shape=(end - start, matrix.shape[1]),
            dtype=cp.float32,
        )
        sum += cp.ones((1,sX.shape[0])) @ sX 

    mean = sum.reshape(-1, 1) / n_rows
    m, v, t = 0, 0, 0
    print(time() - t1) #40s

    t += 1
    sX = cpx.scipy.sparse.csr_matrix(
        matrix[indices[0:batch_sz].get()],
        shape=(batch_sz, matrix.shape[1]),
        dtype=cp.float32,
    )
    g = _oja_grad(sX, mean, w, cp)

    m, v, mh, vh = _adam_coeffs(g, t, m, v, beta1, beta2, cp)
    step_sz = max(step_sz, 0.5 / (cp.mean(cp.abs(mh) / cp.sqrt(vh + epsilon))))
    s = w - step_sz * (mh / (cp.sqrt(vh) + epsilon))
    w, _ = cp.linalg.qr(s)
    print(time() - t1)

    for start in range(0, n_rows, batch_sz):
        t += 1
        end = min(start + batch_sz, n_rows)
        sX = cpx.scipy.sparse.csr_matrix(
            matrix[start:end],
            shape=(end - start, matrix.shape[1]),
            dtype=cp.float32,
        )
        g = _oja_grad(sX, mean, w, cp)
        m, v, mh, vh = _adam_coeffs(g, t, m, v, beta1, beta2, cp)
        s = w - step_sz * (mh / (cp.sqrt(vh) + epsilon))
        w, _ = cp.linalg.qr(s)
    print(time() - t1)
    
    output = cp.zeros((n_rows,k))
    mean_w = mean.T @ w
    for start in range(0, n_rows, batch_sz):
        end = min(start + batch_sz, n_rows)
        sX = cpx.scipy.sparse.csr_matrix(
            matrix[start:end],
            shape=(end - start, matrix.shape[1]),
            dtype=cp.float32,
        )
        output[start:end] = sX @ w
    output = (output - mean_w).get()
    # print("step size: ", step_sz)

    return output


def oja_batch_cpp(
    matrix: Union[np.ndarray, spmatrix],
    k: int,
    batch_sz=2048,
    step_sz=1e-3,
    epsilon=1e-8,
    beta1=0.9,
    beta2=0.999,
) -> np.ndarray:
    indices = np.arange(matrix.shape[0])
    np.random.shuffle(indices)
    matrix = matrix[indices]
    mean = np.array(matrix.mean(axis=0)).reshape(-1, 1)
    
    return _oja_batch(
        matrix,
        mean,
        k,
        batch_sz,
        step_sz,
        epsilon,
        beta1,
        beta2,
    )

def oja_pca(
    matrix: Union[np.ndarray, spmatrix],
    k: int = 6,
    batch_sz=2048,
    step_sz=1e-3,
    epsilon=1e-8,
    beta1=0.9,
    beta2=0.999,
) -> np.ndarray:
    if matrix.shape[0]*matrix.shape[1] < 7e7:
        print("oja_batch")
        return oja_batch(**locals())
    else:
        print("oja_cupy")
        return oja_batch_cupy(**locals())
    