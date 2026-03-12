from utils import benchmark
from tempo import tempo
import numpy as np

def neighbors_test(g: tempo.TGraph, node: int):
    return [n for n, t, e in g.neighbors(node) if e == 0]

def neighbors_array_test(g: tempo.TGraph, node: int):
    n, t, e = g.neighbors_array(node)
    return n[e == 0]

if __name__ == '__main__':
    stream = tempo.CSVEventStream('../data/soc-sign-bitcoinotc.csv')
    g = tempo.TGraph()
    g.ingest(stream)

    nodes = sorted([u for u in range(g.num_nodes())], key=lambda u: g.degree(u), reverse=True)

    for u in range(min(10, len(nodes))):
        print(f'Neighbors tests for node: {nodes[u]}')
        print('Non-batched:')
        print(benchmark(neighbors_test, g, nodes[u]))
        print('Batched:')
        print(benchmark(neighbors_array_test, g, nodes[u]))
        print('-' * 10)