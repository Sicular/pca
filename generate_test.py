import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


def get_correlated_dataset(n, d=3):
    # np.random.seed(42)
    k = 5
    dependency = np.hstack((np.random.rand(d, k) + 1, np.ones((d, d - k))))
    scale = np.hstack((np.random.rand(k) + 5, np.ones((d - k))))
    mu = np.random.rand(d) * 1000 + 1000
    latent = np.abs(np.random.randn(n, d) * 10 + 10)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    return scaled_with_offset


input = get_correlated_dataset(10000, 100)
input = sparse.csr_matrix(input)

from pca_mdl2 import oja_pca
from pca import history_pca
from pca_pm import center_power_method, power_method

k = 5

oja_pca_mtx = oja_pca(input, k)
history_pca_mtx = history_pca(input, k)
power_pca_mtx = center_power_method(input, k)

oja_vars = np.sum(np.var(oja_pca_mtx, axis=0))
history_vars = np.sum(np.var(history_pca_mtx, axis=0))
power_vars = np.sum(np.var(power_pca_mtx, axis=0))
input = np.array(input.todense())
input_vars = np.sum(np.var(input, axis=0))
print("data mean:", np.mean(input, axis=0))
print("oja vars: ", oja_vars, np.sort(np.var(oja_pca_mtx, axis=0)))
print("his vars: ", history_vars, np.sort(np.var(history_pca_mtx, axis=0)))
print("pow vars: ", power_vars, np.sort(np.var(power_pca_mtx, axis=0)))
print("inp vars: ", input_vars)
print("oja acc : ", oja_vars / input_vars)
print("his acc: ", history_vars / input_vars)
print("pow acc: ", power_vars / input_vars)
print("oja/his: ", oja_vars / history_vars)
print("oja/pow: ", oja_vars / power_vars)
print(power_pca_mtx[0,0])

plt.scatter(input.T[0], input.T[1])
plt.xlim((0,3000))
plt.ylim((0,3000))
plt.savefig("./a.png")
