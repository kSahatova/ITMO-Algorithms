import timeit
import functools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from functions import constant_function, sum_function, \
    product_function, direct_polynomial, Horner_method, \
    bubble_sort, quick_sort, timsort, random_vector_creation, matrix_multiplication
from algorithm_execution_time import execution_time_measuring,\
    linear_approximation, polynomial_approximation, opt_curve


# task 1.1

FUNCTIONS = [constant_function, sum_function, product_function, direct_polynomial,
             Horner_method, bubble_sort, quick_sort, timsort]

lin_approx_functions = [constant_function, sum_function, product_function, direct_polynomial, Horner_method]
polyn_approx_functions = [bubble_sort]
optcurve_approx_functions = [quick_sort, timsort]

iterations = 5
n = 2000

v = random_vector_creation(n)

for function in FUNCTIONS:
    average_execution_time = execution_time_measuring(function, v, n, iterations)

    if function in lin_approx_functions:
        linear_approximation(function, average_execution_time, n)

    elif function in polyn_approx_functions:
        polynomial_approximation(function, average_execution_time, n)

    elif function in optcurve_approx_functions:
        opt_curve(function, average_execution_time, n)


# task 1.2
execution_times = np.zeros(n)
for n in tqdm(range(1, n+1), desc=matrix_multiplication.__name__):
    m1 = np.random.rand(n, n)
    m2 = np.random.rand(n, n)
    time = timeit.timeit(functools.partial(matrix_multiplication, m1, m2), number=5)
    execution_times[n-1] = time

coefficients = np.polyfit(range(1, 2001), execution_times, 3)
for_matrix_prod = []
for i in range(2000):
    for_matrix_prod.append(coefficients[0] * i ** 3+coefficients[1] * i ** 2 +
                           coefficients[2] * i + coefficients[3])

plt.plot(range(1, n+1), execution_times, label=matrix_multiplication.__name__)
plt.plot(range(1, 2001), for_matrix_prod, linewidth=3)
plt.legend(loc="upper left")
plt.show()





