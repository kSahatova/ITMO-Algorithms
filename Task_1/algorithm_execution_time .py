import time
import numpy as np
from tqdm import trange
from functions import nlogn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def execution_time_measuring(function, vector, n, iterations):
    time_measure = np.zeros([iterations, n])

    for i in trange(iterations):
        for j in range(len(vector)):
            start_time = time.perf_counter()
            function(vector[j])
            end_time = time.perf_counter()
            time_measure[i][j] = end_time - start_time

    average_execution_time = np.array([])
    for i in range(time_measure.shape[1]):
        d = 0
        for j in range(time_measure.shape[0]):
            d = d + time_measure[j][i]
        average_execution_time = np.append(average_execution_time, d / iterations)
    return average_execution_time


def linear_approximation(function, average_execution_time, n):
    x = range(1, n+1)
    A = np.vstack([range(1, 2001), np.ones(n)]).T
    a, b = np.linalg.lstsq(A, average_execution_time, rcond=None)[0]
    y = a * x + b

    plt.plot(average_execution_time)
    plt.plot(x, y, linewidth=2)
    plt.legend((function.__name__, 'linear approx'), loc='upper left')
    plt.savefig(function.__name__ + '.png', format='png')
    plt.show()


def polynomial_approximation(function, average_execution_time, n):
    x = range(1, n + 1)
    coefficients = np.polyfit(x, average_execution_time, 2, rcond=None)
    y = [coefficients[0]*i ** 2 + coefficients[1] * i + coefficients[2] for i in range(n)]

    plt.plot(average_execution_time)
    plt.plot(x, y, linewidth=2)
    plt.legend((function.__name__, 'polynomial approx'), loc='upper left')
    plt.savefig(function.__name__ + '.png', format='png')
    plt.show()


def opt_curve(function, average_execution_time, n):
    x = range(1, n + 1)
    popt, pcov = curve_fit(nlogn, x, average_execution_time)
    y = nlogn(x, *popt)

    plt.plot(average_execution_time)
    plt.plot(x, y, linewidth=2)
    plt.legend((function.__name__, 'optimal curve'), loc='upper left')
    plt.savefig(function.__name__ + '.png', format='png')
    plt.show()


