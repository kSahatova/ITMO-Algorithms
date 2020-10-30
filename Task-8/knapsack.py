import random
import timeit, functools


# A Dynamic Programming approach for 0-1 Knapsack problem
def knapSackDP(W, wt, val, n):
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]

    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    return K[n][W]


def knapSackNR(W, wt, val, n):
    # Base Case
    if n == 0 or W == 0:
        return 0

    # If weight of the nth item is more than Knapsack of capacity
    # W, then this item cannot be included in the optimal solution
    if wt[n - 1] > W:
        return knapSackNR(W, wt, val, n - 1)

        # return the maximum of two cases:
    # (1) nth item included
    # (2) not included
    else:
        return max(val[n - 1] + knapSackNR(W - wt[n - 1], wt, val, n - 1),
                   knapSackNR(W, wt, val, n - 1))


W = 80
wt = [random.randrange(1, 100) for i in range(150)]
val = [random.randrange(1, 100) for j in range(150)]
n = len(val)
print(knapSackDP(W, wt, val, n))
print(knapSackNR(W, wt, val, n))

time_measure1 = timeit.Timer(functools.partial(knapSackDP, W, wt, val, n))
time_measure2 = timeit.Timer(functools.partial(knapSackNR, W, wt, val, n))
print('Executed time with DP approach for 0-1 KnapSack problem:', time_measure1.timeit(1))
print('Executed time with NR approach for 0-1 KnapSack problem:', time_measure2.timeit(1))
