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


def times_100(func, *args):
    start = timer()
    for i in range(100):
        func(args[0], args[1])
    end = timer()
    return (end - start) / 100.


def times_1(func, *args):
    start = timer()
    func(args[0], args[1])
    end = timer()
    return end - start


def times_10(func, *args):
    start = timer()
    for i in range(10):
        func(args[0], args[1])
    end = timer()
    return (end - start) / 10.
