import timeit, functools


def max(x, y):
    if x > y:
        return x
    return y


def top_down_rod_cutting(c, n):
    global r
    if r[n] >= 0:
        return r[n]

    maximum_revenue = -1*INF

    for i in range(1, n+1):
        maximum_revenue = max(maximum_revenue, c[i] + top_down_rod_cutting(c, n-i))

    r[n] = maximum_revenue
    return r[n]


def bottom_up_rod_cutting(c, n):
    r = [0]*(n+1)
    r[0] = 0

    for j in range(1, n+1):
        maximum_revenue = -1*INF
        for i in range(1, j+1):
            maximum_revenue = max(maximum_revenue, c[i] + r[j-i])
        r[j] = maximum_revenue
    return r[n]


if __name__ == '__main__':
    c = [0, 10, 18, 27, 30, 38, 45, 50, 67]
    parts = len(c)-1
    INF = 100000
    r = [0] + [-1 * INF] * parts
    print(f'Maximum revenue from the rod divided into {parts} parts: { top_down_rod_cutting(c, parts)}')
    time_measure1 = timeit.Timer(functools.partial(top_down_rod_cutting, c, parts))
    print('Executed time with top - down approach:', time_measure1.timeit(1))

    print(f'Maximum revenue from the rod divided into {parts} parts: {bottom_up_rod_cutting(c, parts)}')
    time_measure2 = timeit.Timer(functools.partial(bottom_up_rod_cutting, c, parts))
    print('Executed time with bottom-up approach :', time_measure2.timeit(1))

