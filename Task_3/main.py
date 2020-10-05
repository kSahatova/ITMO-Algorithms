import numpy as np
import random
from sympy import symbols, diff
import matplotlib.pyplot as plt
from scipy import optimize
from autograd import jacobian


alpha = random.uniform(0, 1)
betta = random.uniform(0, 1)

# Data generation with noise
delta = [random.uniform(0, 1) for i in range(0, 101)]
data_x = np.array([k/100 for k in range(0, 101)])
data_y = np.array([alpha * x_k + betta + delta[k] for k, x_k in enumerate(data_x)])

epsilon = 0.001


def linear_function(x, a, b):
    f = a * x + b
    return f


def rational_function(x, a, b):
    f = a /(1 + x * b)
    return f


def lin_cost_function(x):
    a, b = x
    res = 0
    for k in range(101):
        res += (linear_function(data_x[k], a, b) - data_y[k]) ** 2
    return res


def rat_cost_function(x):
    a, b = x
    res = 0
    for k in range(101):
        res += (rational_function(data_x[k], a, b) - data_y[k]) ** 2
    return res


def gradient_descent(a_init, b_init, learning_rate, cost_func):
    a_curr = a_init
    b_curr = b_init
    curr_coeffs = (a_curr, b_curr)
    cost_start = cost_func(curr_coeffs)
    iterations = 0
    fCalc = 1
    running = True

    while running:
        iterations += 1
        fCalc += 1
        curr_coeffs = symbols('a b', real=True)
        a, b = curr_coeffs
        d_a = diff(cost_func(curr_coeffs), a)
        d_a = float(d_a.subs({a: a_curr, b: b_curr}))
        d_b = diff(cost_func(curr_coeffs), b)
        d_b = float(d_b.subs({a: a_curr, b: b_curr}))

        a_curr = a_curr - learning_rate*d_a
        b_curr = b_curr - learning_rate*d_b
        curr_coeffs = (a_curr, b_curr)

        cost_curr = cost_func(curr_coeffs)

        if abs(cost_curr - cost_start) < epsilon:
            running = False
        else:
            cost_start = cost_curr

    return [a_curr, b_curr, cost_func(curr_coeffs), iterations, fCalc]


a, b, cost_curr, iterations, fCalc = gradient_descent(a_init=0, b_init=0, learning_rate=0.001, cost_func=lin_cost_function)
print(f'Optimization terminated successfully.\n\
         Current function value: {cost_curr}\n\
         Iterations: {iterations}\n\
         Function evaluations: {fCalc}\n\
Gradient Descent Linear: [{a}, {b}]')

plt.plot(data_x, data_y,  label='Experimental data')
plt.plot(data_x, linear_function(data_x, a, b), label='Linear approx with GD')
plt.legend(loc='upper left')
plt.show()

res_CG_linear = optimize.minimize(lin_cost_function, np.array([0, 0]), method='CG', tol=epsilon, options={'disp': True})
print('CG Linear: {}'.format(res_CG_linear.x))

plt.plot(data_x, data_y,  label='Experimental data')
plt.plot(data_x, linear_function(data_x, res_CG_linear.x[0], res_CG_linear.x[1]), label='Linear approx with CG')
plt.legend(loc='upper left')
plt.show()


res_NewtonCG_linear = optimize.minimize(lin_cost_function, np.array([0, 0]),method='Newton-CG',
                                        jac=jacobian(lin_cost_function), tol=epsilon, options={'disp': True})
print('NewtonCG Linear: {}'.format(res_NewtonCG_linear.x))

plt.plot(data_x, data_y,  label='Experimental data')
plt.plot(data_x, linear_function(data_x, res_NewtonCG_linear.x[0], res_NewtonCG_linear.x[1]), label='Linear approx with Newton-CG')
plt.legend(loc='upper left')
plt.show()


def fun_linear(args, x, y):
    return args[0]*x + args[1] - y


res_lsq_lin = optimize.least_squares(fun_linear, np.array([0, 0]), args=(data_x, data_y), method='lm', gtol=0.001,)
print(f'Optimization terminated successfully.\n\
         Current function value: {res_lsq_lin.cost}\n\
         Function evaluations: {res_lsq_lin.nfev}\n\
Levenberg-Marquart Linear: {res_lsq_lin.x}')

plt.plot(data_x, data_y,  label='Experimental data')
plt.plot(data_x, linear_function(data_x, res_lsq_lin.x[0], res_lsq_lin.x[1]), label='Linear approx with LMA')
plt.legend(loc='upper left')
plt.show()

##########################################

a, b, cost_curr, iterations, fCalc = gradient_descent(a_init=0, b_init=0, learning_rate=0.001, cost_func=rat_cost_function)
print(f'Optimization terminated successfully.\n\
         Current function value: {cost_curr}\n\
         Iterations: {iterations}\n\
         Function evaluations: {fCalc}\n\
Gradient Descent Rational: [{a}, {b}]')

res_CG_rat = optimize.minimize(rat_cost_function, np.array([0, 0]), method='CG', tol=epsilon, options={'disp': True})
print('CG Rational: {}'.format(res_CG_linear.x))


res_NewtonCG_rat = optimize.minimize(rat_cost_function, np.array([0, 0]),method='Newton-CG',
                                        jac=jacobian(rat_cost_function), tol=epsilon, options={'disp': True})
print('NewtonCG Rational: {}'.format(res_NewtonCG_linear.x))


def fun_rational(args, x, y):
    return args[0]/(1 + args[1]*x) - y


res_lsq_rat = optimize.least_squares(fun_rational, np.array([0, 0]),
                                 args=(data_x, data_y), method='lm', gtol=0.001,)
print(f'Optimization terminated successfully.\n\
         Current function value: {res_lsq_rat.cost}\n\
         Function evaluations: {res_lsq_rat.nfev}\n\
Levenberg-Marquart Rational: {res_lsq_rat.x}')


plt.plot(data_x, data_y,  label='Experimental data')
plt.plot(data_x, rational_function(data_x, a, b), label='Rational approx with GD')
plt.legend(loc='upper left')
plt.show()

plt.plot(data_x, data_y,  label='Experimental data')
plt.plot(data_x, rational_function(data_x, res_CG_rat.x[0], res_CG_rat.x[1]), label='Rational approx with CG')
plt.legend(loc='upper left')
plt.show()

plt.plot(data_x, data_y,  label='Experimental data')
plt.plot(data_x, rational_function(data_x, res_NewtonCG_rat.x[0], res_NewtonCG_rat.x[1]), label='Rational approx with Newton-CG')
plt.legend(loc='upper left')
plt.show()

plt.plot(data_x, data_y,  label='Experimental data')
plt.plot(data_x, rational_function(data_x, res_lsq_rat.x[0], res_lsq_rat.x[1]), label='Rational approx with LMA')
plt.legend(loc='upper left')
plt.show()
