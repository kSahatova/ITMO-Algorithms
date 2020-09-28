import random
import numpy as np
from decimal import *
getcontext().prec = 10


def random_vector_creation(n):
    v = [list(np.random.random(i)) for i in range(1, n+1)]
    return v


def constant_function(v):
    return 1


def sum_function(v):
    return sum(v)


def product_function(v):
    return np.prod(v)


def direct_polynomial(v, x=1.5):
    p = [Decimal(v_k) * Decimal(x) ** Decimal(k) for k, v_k in enumerate(v)]
    return np.sum(p)


def Horner_method(v):
    n_ = len(v)
    b1 = Decimal(v[len(v) - 1])
    b2 = 0
    if n_ == 1:
        return b1
    else:
        for i in range(2, n_ + 1):
            b2 = Decimal(b1) * Decimal(1.5) + Decimal(v[len(v) - i])
            b1 = b2
    return b2


def bubble_sort(v):
    for i in range(len(v)-1):
        for j in range(len(v)-i-1):
            if v[j] > v[j+1]:
                v[j], v[j+1] = v[j+1], v[j]
    return v


def quick_sort(nums):
    if len(nums) <= 1:
        return nums
    else:
        q = random.choice(nums)
        s_nums = []
        m_nums = []
        e_nums = []
        for n in nums:
            if n < q:
                s_nums.append(n)
            elif n > q:
                m_nums.append(n)
            else:
                e_nums.append(n)
        return quick_sort(s_nums) + e_nums + quick_sort(m_nums)


def nlogn(x, a, b, c):
    return a * x * np.log(b * x) + c


def timsort(v):
    return sorted(v)


def matrix_multiplication(matrix1, matrix2):
    return matrix1.dot(matrix2)


