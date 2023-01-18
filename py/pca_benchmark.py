import numpy as np
from typing import Union
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator, svds
import cupy as cp
import cupyx as cpx
from matplotlib import pyplot as plt

def _svd_flip(u: np.ndarray):
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    return u


def _power_iter(
    x: Union[np.ndarray, spmatrix], origin: np.ndarray, w: np.ndarray, lib = np
) -> np.ndarray:
    colsums = lib.array(x.sum(axis=0)).reshape(1, -1)
    return (
        x.T @ (x @ w)
        - origin @ (colsums @ w)
        + (origin * x.shape[0] - colsums.T) @ (origin.T @ w)
    )


def history_pca_cupy(matrix: Union[np.ndarray, spmatrix], k: int) -> np.ndarray:
    """History principal component analysis implementation.
    Reference: https://arxiv.org/pdf/1802.05447.pdf
    
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
    # indices = np.arange(matrix.shape[0])
    # np.random.shuffle(indices)
    # matrix = matrix[indices]

    n_rows, n_cols = matrix.shape
    batch_sz = min(2048, n_rows // 10)
    cp.random.seed(42)
    prev_w = cp.random.normal(0, 1, (n_cols, k), dtype=cp.float32)
    prev_w = cp.eye(n_cols, k)
    prev_w, _ = cp.linalg.qr(prev_w)

    origin = cp.array(matrix.mean(axis=0)).reshape(-1, 1)
    his = [[],[]]
    for i in range(3):
        x = cpx.scipy.sparse.csr_matrix(
            matrix[:batch_sz],
            shape=(batch_sz, matrix.shape[1]),
            dtype=cp.float32,
        )
        s = _power_iter(x, origin, prev_w, cp) / batch_sz
        prev_w, _ = cp.linalg.qr(s)
        # his[0].append(cp.sum(cp.var(cp.array(matrix.todense()) @ prev_w, axis = 0)).get())

    eigen_vals = cp.diag(cp.linalg.norm(s, axis=0))

    for start in range(0, n_rows, batch_sz):
        end = min(start + batch_sz, n_rows)
        w = prev_w.copy()
        x = cpx.scipy.sparse.csr_matrix(
            matrix[start:end],
            shape=(end - start, matrix.shape[1]),
            dtype=cp.float32,
        )

        alpha = cp.square(start / end)
        y = prev_w @ eigen_vals
        s = alpha * (y @ (prev_w.T @ w)) + (
            (1.0 - alpha) / (end - start)
        ) * _power_iter(x, origin, w, cp)

        w, _ = cp.linalg.qr(s)
        eigen_vals = cp.diag(cp.linalg.norm(s, axis=0))
        prev_w = w
        # his[0].append(cp.sum(cp.var(cp.array(matrix.todense()) @ w, axis = 0)).get())
        
    # fig, ax = plt.subplots(len(his))
    # for i in range(len(his)):
    #     ax[i].plot(his[i])
    # fig.savefig("./b.png")
    # return matrix @ w.get()
    # result = _svd_flip((matrix @ w.get() - (origin.T @ w).get()))
    result = (matrix @ w.get() - (origin.T @ w).get())
    return result

def history_pca(matrix: Union[np.ndarray, spmatrix], k: int) -> np.ndarray:
    """History principal component analysis implementation.
    Reference: https://arxiv.org/pdf/1802.05447.pdf
    
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
    # indices = np.arange(matrix.shape[0])
    # np.random.shuffle(indices)
    # matrix = matrix[indices]

    n_rows, n_cols = matrix.shape
    batch_sz = min(2048, n_rows // 10)
    # np.random.seed(42)
    # prev_w = np.random.normal(0, 1, (n_cols, k), dtype=np.float32)
    prev_w = np.eye(n_cols, k)
    prev_w, _ = np.linalg.qr(prev_w)

    origin = np.array(matrix.mean(axis=0)).reshape(-1, 1)
    his = [[],[]]
    for i in range(3):
        x = matrix[:batch_sz]
        s = _power_iter(x, origin, prev_w) / batch_sz
        prev_w, _ = np.linalg.qr(s)
        # his[0].append(np.sum(np.var(np.array(matrix.todense()) @ prev_w, axis = 0)).get())

    eigen_vals = np.diag(np.linalg.norm(s, axis=0))

    for start in range(0, n_rows, batch_sz):
        end = min(start + batch_sz, n_rows)
        w = prev_w.copy()
        x = matrix[start:end]

        alpha = np.square(start / end)
        y = prev_w @ eigen_vals
        s = alpha * (y @ (prev_w.T @ w)) + (
            (1.0 - alpha) / (end - start)
        ) * _power_iter(x, origin, w)

        w, _ = np.linalg.qr(s)
        eigen_vals = np.diag(np.linalg.norm(s, axis=0))
        prev_w = w
        # his[0].append(np.sum(np.var(np.array(matrix.todense()) @ w, axis = 0)).get())
        
    # fig, ax = plt.subplots(len(his))
    # for i in range(len(his)):
    #     ax[i].plot(his[i])
    # fig.savefig("./b.png")
    # return matrix @ w.get()
    # result = _svd_flip((matrix @ w.get() - (origin.T @ w).get()))
    result = (matrix @ w - (origin.T @ w))
    return result


def exact_pca(matrix: Union[np.ndarray, spmatrix], k: int) -> np.ndarray:
    """Exact principal component analysis.
    
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
    s = matrix.shape
    mu = (matrix.mean(axis=0)).astype(matrix.dtype)

    mdot = mu.dot
    mhdot = mu.T.dot
    x_dot = matrix.dot

    xh = matrix.T.tocsr()
    xh_dot = xh.dot

    def matvec(x):
        x = np.array(x, dtype="float32")
        ans = x_dot(x) if len(x.shape) == 1 else np.array([x_dot(x.T[0])]).T
        ans = ans - mdot(x)
        return ans

    def rmatvec(x):
        x = np.array(x, dtype="float32")
        ans = xh_dot(x) if len(x.shape) == 1 else np.array([xh_dot(x.T[0])]).T
        ans = ans - mhdot(x.sum(keepdims=True))
        return ans

    xl = LinearOperator(
        matvec=matvec,
        dtype="float32",
        shape=s,
        rmatvec=rmatvec,
    )

    u, s, _ = svds(xl, solver="arpack", k=k)
    u = _svd_flip(u)

    idx = np.argsort(-s)
    result = np.ascontiguousarray((u * s)[:, idx])

    return result


def pca(matrix: Union[np.ndarray, spmatrix], k: int) -> np.ndarray:
    """Principal component analysis.
    
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

    if matrix.shape[0] < 10000:
        return exact_pca(matrix, k)
    else:
        indices = np.arange(matrix.shape[0])
        np.random.shuffle(indices)
        pca_mtx = history_pca(matrix[indices], k)
        pca_mtx = pca_mtx[np.argsort(indices)]
        return pca_mtx
