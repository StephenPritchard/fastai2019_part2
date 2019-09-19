import torch
import operator
from timeit import default_timer as timer


def test(a, b, cmp, cname=None):
    if cname is None:
        cname = cmp.__name__
    assert cmp(a, b), f"{cname}:\n{a}\n{b}"


def test_equality(a, b):
    test(a, b, operator.eq, "==")


def near(a, b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)


def test_near(a, b): test(a, b, near)


def test_near_zero(a, tol=1e-3): assert a.abs() < tol, f"Near zero: {a}"


def times_100(func, m1, m2):
    start = timer()
    for i in range(100):
        func(m1, m2)
    end = timer()
    return (end - start) / 100.


def times_1(func, m1, m2):
    start = timer()
    func(m1, m2)
    end = timer()
    return end - start


def times_10(func, m1, m2):
    start = timer()
    for i in range(10):
        func(m1, m2)
    end = timer()
    return (end - start) / 10.
