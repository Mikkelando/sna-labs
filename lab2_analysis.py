"""
Задание 2. Центральность по собственному вектору.

Граф собран вручную в веб-редакторе (lab2_ui.html) под характеристику от
преподавателя: 50 узлов, последовательность центральностей образует «яму»
и «горбик» — т.е. сначала падает, затем возрастает до пика, потом снова
опускается.

Этот скрипт загружает сохранённый JSON, считает центральность через NetworkX
(nx.eigenvector_centrality_numpy)
и строит итоговый график для отчёта.
"""

import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


GRAPH_FILE = 'lab2_graph.json'  # сохранённый из веб-редактора


# =========================================================================
# Загрузка графа из JSON и сборка nx.Graph
# =========================================================================
with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

n = data['n']
edges = data['edges']
positions = data.get('positions')  # координаты узлов из редактора (если сохранены)

G = nx.Graph()
G.add_nodes_from(range(n))
G.add_edges_from(edges)

print(f'Вершин: {G.number_of_nodes()}, рёбер: {G.number_of_edges()}')
print(f'Связных компонент: {nx.number_connected_components(G)}')

# nx.eigenvector_centrality_numpy кидает AmbiguousSolution на разорванном графе.
# Если что-то пошло не так — проверь редактор и пересохрани.
assert nx.is_connected(G), (
    'Граф разорван — eigenvector_centrality_numpy не работает на несвязных '
    'графах. Соедини компоненты в редакторе и пересохрани JSON.'
)


# =========================================================================
# Центральность по собственному вектору
# =========================================================================
centrality = nx.eigenvector_centrality_numpy(G)
c = np.array([centrality[i] for i in range(n)])

print('\nЗначения c(i):')
for i in range(n):
    print(f'  c({i:2d}) = {c[i]:.6f}')


# =========================================================================
# Главный график: c(i) по индексу вершины.
# По нему должно быть видно требуемую форму «спуск → яма → подъём → пик → спуск».
# =========================================================================
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(range(n), c, 'o-', color='#2c7fb8', linewidth=2, markersize=6)
ax.set_xlabel('Индекс вершины i')
ax.set_ylabel('Центральность по собств. вектору  c(i)')
ax.set_title('Центральность вершин (упорядочены по индексу)')
ax.grid(alpha=0.3)
ax.set_xticks(range(0, n, 2))
plt.tight_layout()
plt.show()


# =========================================================================
# Визуализация самого графа.
# Используем координаты, в которых граф собирался в редакторе.
# =========================================================================
if positions is not None:
    # инверсия Y, чтобы совпадало с canvas (там Y растёт вниз)
    pos = {i: (positions[i][0], -positions[i][1]) for i in range(n)}
else:
    pos = nx.spring_layout(G, seed=42)

fig, ax = plt.subplots(figsize=(11, 8))
node_sizes = 200 + 1500 * (c / c.max())
nx.draw_networkx_edges(G, pos, edge_color='#cccccc', width=1, ax=ax)
nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=c,
                                cmap='viridis', ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, font_color='white', ax=ax)
plt.colorbar(nodes, ax=ax, label='c(v)')
ax.set_title('Граф: размер и цвет узла = центральность по собств. вектору')
ax.set_axis_off()
plt.tight_layout()
plt.show()
