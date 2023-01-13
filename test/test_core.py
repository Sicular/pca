import pytest
from oja import oja_batch
import utils.csr_matrix as csr
import numpy as np
from scipy.sparse.linalg import svds


@pytest.mark.parametrize("acc", [1])
def test_accuracy(acc):
    k = 5
    input = csr.get_multirow(0, 100000)[:, :100]
    pca = oja_batch(input, k)
    output = np.sort(np.diag(np.cov(pca.T)))
    _, S, _ = svds(input.astype("float"), k=k)
    true_ground = np.sort(S[:k] ** 2 / (input.shape[0]))
    rate = output / true_ground
    rate[rate > 1] = 1 / rate[rate > 1]
    print(true_ground)
    print(output)
    assert (
        np.sum(rate < acc) == 0
    ), f"(min; mean; max) accuracy = ({np.min(rate)}; {np.mean(rate)}; {np.max(rate)})"
