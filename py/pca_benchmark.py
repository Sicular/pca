import numpy as np
from typing import Union
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator, svds


def _svd_flip(u: np.ndarray):
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    return u


def _power_iter(
    x: Union[np.ndarray, spmatrix], origin: np.ndarray, w: np.ndarray
) -> np.ndarray:
    colsums = np.array(x.sum(axis=0)).reshape(1, -1)
    return (
        x.T @ (x @ w)
        - origin @ (colsums @ w)
        + (origin * x.shape[0] - colsums.T) @ (origin.T @ w)
    )


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
    indices = np.arange(matrix.shape[0])
    np.random.shuffle(indices)
    matrix = matrix[indices]

    n_rows, n_cols = matrix.shape
    batch_sz = min(2048, n_rows // 10)
    prev_w = np.random.randn(n_cols, k).astype("float32")
    prev_w, _ = np.linalg.qr(prev_w)

    origin = np.array(matrix.mean(axis=0)).reshape(-1, 1)

    for i in range(3):
        s = _power_iter(matrix[:batch_sz], origin, prev_w) / batch_sz
        prev_w, _ = np.linalg.qr(s)

    eigen_vals = np.diag(np.linalg.norm(s, axis=0))

    for start in range(batch_sz, n_rows, batch_sz):
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

    result = _svd_flip(np.array(matrix @ w - origin.T @ w))
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
