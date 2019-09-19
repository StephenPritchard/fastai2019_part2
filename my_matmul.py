from my_testing import test_near, times_100, times_1, times_10
import torch
from torch import tensor


# slow vanilla python
def three_loop_matmul(m1, m2):
    a_rows, a_cols = m1.shape
    b_rows, b_cols = m2.shape
    assert a_cols == b_rows, "These matrices cannot be multiplied, a_rows != b_cols"

    result = torch.zeros(a_rows, b_cols)
    for i in range(a_rows):
        for j in range(b_cols):
            for k in range(a_cols):  # or b_rows
                result[i, j] += m1[i, k] * m2[k, j]

    return result


# elementwise multiplication done by python
def two_loop_matmul(m1, m2):
    m1_rows, m1_cols = m1.shape
    m2_rows, m2_cols = m2.shape

    assert m1_cols == m2_rows, "These matrices cannot be multiplied, a_rows != b_cols"
    result = torch.zeros(m1_rows, m2_cols)
    for i in range(m1_rows):
        for j in range(m2_cols):
            result[i, j] = (m1[i, :] * m2[:, j]).sum()

    return result


def broadcast_matmul(m1, m2):
    m1_rows, m1_cols = m1.shape
    m2_rows, m2_cols = m2.shape

    assert m1_cols == m2_rows, "These matrices cannot be multiplied, a_rows != b_cols"
    result = torch.zeros(m1_rows, m2_cols)
    for i in range(m1_rows):
        result[i] = (m1[i].unsqueeze(-1) * m2).sum(dim=0)

    return result


def einstein_sum_matmul(m1, m2):
    return torch.einsum('ik,kj->ij', m1, m2)


def torch_matmul(m1, m2):
    return m1.matmul(m2)


if __name__ == "__main__":
    a = tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    b = tensor([[4.], [5.], [6.]])

    test_near(three_loop_matmul(a, b), tensor([[32.], [77.], [122.]]))
    test_near(two_loop_matmul(a, b), tensor([[32.], [77.], [122.]]))
    test_near(broadcast_matmul(a, b), tensor([[32.], [77.], [122.]]))
    test_near(einstein_sum_matmul(a, b), tensor([[32.], [77.], [122.]]))
    test_near(torch_matmul(a, b), tensor([[32.], [77.], [122.]]))

    t1 = torch.randn(10000, 10)
    t2 = torch.randn(10, 1)

    print("******************************************************************")
    print("Timing the execution of the various matrix multiplication methods.")
    print("******************************************************************")
    print(f'Three loop matmul time, 1 execution: {times_1(three_loop_matmul, t1, t2) * 1000} ms')
    print(f'Two loop matmul time, 1 execution: {times_1(two_loop_matmul, t1, t2) * 1000} ms')
    print(f'Broadcast matmul time, 1 execution: {times_1(broadcast_matmul, t1, t2) * 1000} ms')
    print(f'Einsum matmul time, 1 execution: {times_1(einstein_sum_matmul, t1, t2) * 1000} ms')
    print(f'Torch matmul time, 1 execution: {times_1(torch_matmul, t1, t2) * 1000} ms')
    print(f'Two loop matmul time, mean of 10: {times_10(two_loop_matmul, t1, t2) * 1000} ms')
    print(f'Broadcast matmul time, mean of 10: {times_10(broadcast_matmul, t1, t2) * 1000} ms')
    print(f'Einsum matmul time, mean of 100: {times_100(einstein_sum_matmul, t1, t2) * 1000} ms')
    print(f'Torch matmul time, mean of 100: {times_100(torch_matmul, t1, t2) * 1000} ms')
    print('\nThere might be some kind of caching at play in the looped runs, not sure')
