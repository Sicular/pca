import numpy as np


input = np.load('/data1/intern/hanl/find_polygon/input/basic/00.npy')


def pca_exact(X, k):
    sum = np.mean(X, axis=1)
    std_input = (X.T - sum).T
    S = std_input @ std_input.T / X.shape[1]
    eigen_value, eigen_vector = np.linalg.eig(S)
    Uk = eigen_vector[:,: k]
    return np.dot(Uk.T,std_input)






