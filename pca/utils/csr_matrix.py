import numpy as np

data_path = "/data1/intern/csr_matrix/data.bin"
indices_path = "/data1/intern/csr_matrix/indices.bin"
indptr_path = "/data1/intern/csr_matrix/indptr.bin"

n_cols = 63683
# n_rows = 83441495
n_rows = 204800


def read_file(filename, start, end, dtype=np.int32):
    with open(filename, "rb") as fin:
        dtype_size = np.dtype(dtype).itemsize
        fin.seek(start * dtype_size)
        data = fin.read((end - start) * dtype_size)

        return np.frombuffer(data, dtype=dtype)


def get_multirow(i_start, i_end):
    output = np.zeros(((i_end - i_start), n_cols), dtype=np.int32)
    indptr = read_file(indptr_path, i_start, i_end + 1, np.int64)
    indices = read_file(indices_path, indptr[0], indptr[-1])
    data = read_file(data_path, indptr[0], indptr[-1])
    indptr = indptr - indptr[0]
    for i in range(i_end - i_start):
        output[i, indices[indptr[i] : indptr[i + 1]]] = data[indptr[i] : indptr[i + 1]]

    return output


def get_row(i):
    return get_multirow(i, i + 1)


def get_col(j):
    result = np.zeros((1, n_rows))
    for i in range(n_rows):
        indptr = read_file(indptr_path, i, i + 1 + 1, np.int64)
        indices = read_file(indices_path, indptr[0], indptr[-1])
        data = read_file(data_path, indptr[0], indptr[-1])
        index = np.where(indices == j)[0]
        if index.size > 0:
            result[:, i] = data[index[0]]
    return result


def get_value(i, j):
    indptr = read_file(indptr_path, i, (i + 1) + 1, np.int64)
    indices = read_file(indices_path, indptr[0], indptr[-1] + 1)
    data = read_file(data_path, indptr[0], indptr[-1] + 1)
    return data[indices == j][0] if data[indices == j].size > 0 else 0


# import time
# t1 = time.time()
# output = get_multirow(2, 3)
# print(np.unique(output).shape)
# # for i in range(10000):
# #     output = get_row(i)
# #     # np.save("../data/output/demo_large_query.npy", output)
# #     # print(output)
# t2 = time.time()
# print(t2 - t1)
