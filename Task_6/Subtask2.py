import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

# Grid generation
k = 0
obstacles = []
grid = np.zeros((10, 10))
i, j = random.randrange(0, 10), random.randrange(0, 10)
while k < 30:
    if grid[i][j] == 0:
        grid[i][j] = 1
        obstacles.append((i, j))
        k += 1
    else:
        i, j = random.randrange(0, 10), random.randrange(0, 10)


graph = nx.grid_graph(dim=(10, 10))
print(list(graph.nodes))
print('Nodes of obstacles:\n',obstacles)
for obstacle in obstacles:
    graph.remove_node(obstacle)

print(list(graph.nodes))
print(list(graph.edges))

start_point = random.choice(list(graph.nodes))
end_point = random.choice(list(graph.nodes))
print(start_point, end_point)


def euclidean(start_point, end_point):
    x1, y1 = start_point
    x2, y2 = end_point
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

shortest_path = []
if start_point == end_point:
    print('The shortest path is equal to 0, start point and end point are the same')
else:
    try:
        shortest_path = nx.astar_path(graph, source=start_point, target=end_point, heuristic=euclidean)
        print(shortest_path)
    except nx.exception.NetworkXNoPath:
        print(f"Node {end_point} not reachable from {start_point}")

x_ = []
y_ = []
for x, y in shortest_path:
    y_.append(y)
    x_.append(x)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(grid)

if start_point not in obstacles and end_point not in obstacles:
    ax.scatter(start_point[1], start_point[0], s=18**2, c='green', marker="o")
    ax.scatter(end_point[1], end_point[0], s=18**2, c='red', marker="o")
else:
    print('Error!')
ax.scatter(y_, x_, marker="o")
plt.show()

