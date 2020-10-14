import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import optimize


# Data generation with noise
delta = np.array([random.uniform(0, 1) for i in range(0, 1001)])
x = np.array([3*k/1000 for k in range(0, 1001)])
f_x = np.array([1/(x_k**2 - 3*x_k + 2) for x_k in x])
y = []
for f, d_ in zip(f_x, delta):
    if f < -100:
        y.append(-100+d_)
    elif -100 <= f <= 100:
        y.append(f + d_)
    elif f > 100:
        y.append(100+d_)
y = np.asarray(y)

epsilon = 0.001


def approx_function(data_x, a, b, c, d):
    res = (a*data_x+b)/(data_x**2 + c*data_x + d)
    return res


def least_squares(args):
    a, b, c, d = args
    res = 0
    for k in range(0, 1001):
        res += (approx_function(x[k], a, b, c, d) - y[k])**2
    return res


result_nm = optimize.minimize(least_squares, np.array([1, 1, 1, 1]), method='Nelder-Mead',
                            tol=epsilon)
print(f'Optimization by Nelder-Mead algorithm terminated successfully.\n\
         Current function value: {result_nm.fun}\n\
         Function evaluations: {result_nm.nfev}\n\
         Number of iterations: {result_nm.nit}\n\
         Parameters: {result_nm.x}')


def fun_linear(args, x, y):
    return approx_function(x, args[0], args[1], args[2], args[3]) - y


result_lsq = optimize.least_squares(fun_linear, np.array([1, 1, 1, 1]), args=(x, y), method='lm', gtol=0.001,)
print(f'Optimization by Levenberg-Marquardt algorithm terminated successfully.\n\
         Current function value: {result_lsq.cost}\n\
         Function evaluations: {result_lsq.nfev}\n\
         Parameters: {result_lsq.x}')

bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10)]
result_de = optimize.differential_evolution(least_squares, bounds=bounds)
print(f'Optimization by Differential evolution algorithm terminated successfully.\n\
         Function evaluations: {result_de.nfev}\n\
         Number of iterations: {result_de.nit}\n\
         Parameters: {result_de.x}')


result_sa = optimize.dual_annealing(least_squares, bounds=bounds)
print(f'Optimization by Simulated annealing algorithm terminated successfully.\n\
         Function evaluations: {result_sa.nfev}\n\
         Number of iterations: {result_sa.nit}\n\
         Parameters: {result_sa.x}')

results = [result_nm, result_lsq, result_de, result_sa]
methods = ['Nelder-Mead','Levenberg-Marquardt','Differential evolution', 'Simulated annealing']
for result, method in zip(results, methods):
    plt.plot(x, y, marker='*', label='Experimental data')
    plt.plot(x, approx_function(x, result.x[0], result.x[1], result.x[2], result.x[3]), label=method)
    plt.legend(loc='upper right')
    plt.show()
