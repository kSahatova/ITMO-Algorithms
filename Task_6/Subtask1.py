import random
import networkx as nx
from pprint import pprint
import time
import matplotlib.pyplot as plt

G = nx.Graph()
while len(list(G.nodes)) < 100:
    G.add_node(random.randrange(1, 400))

while len(list(G.edges)) < 500:
    G.add_edge(random.choice(list(G.nodes)), random.choice(list(G.nodes)),
               weight=random.uniform(0, 1))

print('Number of graph nodes:', G.number_of_nodes())
print('Number of graph edges:', G.number_of_edges())

ad_matrix = nx.adjacency_matrix(G)
print('Adjacency matrix: \n', ad_matrix.todense())


def adjacency_list(adj_matrix):
    adj_list = []
    for i in range(0,100):
        lst = []
        for j in range(0,100):
            if adj_matrix[i, j] != 0:
                lst.append(j)
        adj_list.append(lst)
    vertices = []
    for i in range(100):
        vertices.append(i)

    return dict(zip(vertices, adj_list))


ad_list = adjacency_list(ad_matrix)
print('Adjacency list:')
pprint(ad_list)


start_point = random.choice(list(G.nodes))
end_point = random.choice(list(G.nodes))
start_time = time.perf_counter()
average_time = 0
methods = ['dijkstra', 'bellman-ford']
for method in methods:
    for i in range(10):
        start_time = time.perf_counter()
        shortest_path_dijkstra = nx.shortest_path(G, source=start_point, method=method)
        delta_time = time.perf_counter()-start_time
        average_time += delta_time
    print(f"Average execution time for {method} algorithm : {average_time/10}")

