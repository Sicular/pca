import numpy as np
import cupy as cp
from time import time

t1 = time()
a_np = np.random.rand(1000,10000)
a_cp = cp.zeros((a_np.shape[1],a_np.shape[1]))
s = cp.cuda.Stream(non_blocking=True)
# a_cp = a_np.T @ a_np
a_cp.set(a_np.T @ a_np, stream=s)
s.synchronize()
print(time() - t1)