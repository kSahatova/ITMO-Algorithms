import random
import networkx as nx
from pprint import pprint
import matplotlib.pyplot as plt


G = nx.Graph()
while len(list(G.nodes)) < 100:
    G.add_node(random.randrange(1, 400))

while len(list(G.edges)) < 200:
    G.add_edge(random.choice(list(G.nodes)), random.choice(list(G.nodes)))

print('Number of graph nodes:', G.number_of_nodes())
print('Number of graph edges:', G.number_of_edges())

ad_matrix = nx.adjacency_matrix(G)
print('Adjacency matrix: ', ad_matrix.todense()[0],'...')


def ad_matrix_to_ad_list(adjacency_matrix):
    adjacency_list = []
    for i in range(0,100):
        lst = []
        for j in range(0,100):
            if adjacency_matrix[i, j] == 1:
                lst.append(j)
        adjacency_list.append(lst)
    return adjacency_list


ad_list = ad_matrix_to_ad_list(ad_matrix)
print('Adjacency list:')
pprint(ad_list[0:5])

connected_components = list(nx.dfs_edges(G, source=random.choice(list(G.nodes))))
print('Connected components by using DFS:\n', connected_components)

start_point = random.choice(list(G.nodes))
end_point = random.choice(list(G.nodes))
print(f'The shortest path between {start_point} and {end_point} by Dijkstra algorithm:')
print(nx.shortest_path(G, source=start_point, target=end_point))


plt.figure()
nx.draw_shell(G, with_labels=True, node_size=100, width=0.6, font_size=6)
plt.show()
