import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def slice_graph(graph: nx.Graph, t: int) -> nx.Graph:
    '''
    Take a snapshot of a temporal graph.
    
    :param graph: The graph to snapshot.
    :type graph: nx.Graph
    :param t: What time stamp to snapshot at.
    :type t: int
    :return: A static snapshot of the graph at time stamp t.
    :rtype: Graph
    '''
    H = nx.Graph()
    for node, data in graph.nodes(data=True):
        if t in data['t']:
            H.add_node(node)
    for u, v, data in graph.edges(data=True):
        if t in data['t']:
            H.add_edge(u, v)
    return H

def vis_graph(G: nx.Graph, time_stamps: int, destination_folder: str) -> None:
    '''
    Visualize a temporal graph through a series of snapshots.
    
    :param G: Temporal graph to visualize.
    :type G: nx.Graph
    :param time_stamps: How many discrete time stamps to slice to.
    :type time_stamps: int
    :param destination_folder: Save the visualizations in this folder.
    :type destination_folder: str
    '''
    for t in range(time_stamps):
        H = slice_graph(G, t)
        plt.figure()
        nx.draw(H, with_labels=True)
        plt.title(f't = {t}')
        plt.savefig(f'{destination_folder}/frame_{t}.png')
        plt.close()

def temporal_conn(graph: nx.Graph, R: int, epochs: int) -> dict[any, float]:
    '''
    Calculate the temporal connectivity of each node in a temporal graph.
    
    :param graph: Inputted temporal graph.
    :type graph: nx.Graph
    :param epochs: How many times to run the algorithm before stopping.
    :type epochs: int
    :param R: Length of each random walk.
    :type R: int
    :return: Map from node to connectivity value.
    :rtype: dict[Any, float]
    '''
    tc = { node: 1.0 for node in graph.nodes }
    for _ in range(epochs):
        for node in graph.nodes:
            # Start the random walk
            current_node = node
            t_current = 0
            for r in range(R):
                # Sample edges that exist in the future from the current time stamp
                temporal_edges = [
                    (edge, t)
                    for edge in graph.edges(current_node) if t_current <= graph.edges[edge]['t'][-1]
                    for t in graph.edges[edge]['t'] if t_current <= t
                ]

                # End the walk early if a dead end occurs
                if len(temporal_edges) == 0:
                    break

                '''Randomly choose an edge where Probability(edge) decreases as Delta(time) increases.'''
                times = np.array([e[1] for e in temporal_edges])

                # Exponential decay
                delta_t = times - t_current
                lmbda = 1.0

                # Calculate probabilities
                weights = np.exp(-lmbda * delta_t)
                probs = weights / weights.sum() # Normalize

                # Make the selection based on probabilities
                idx = np.random.choice(len(temporal_edges), p=probs)

                # Update connectivity values
                (_, next_node), t_current = temporal_edges[idx]
                tc[next_node] += 1.0
                current_node = next_node
    
    # Monte Carlo sampling
    total_samples = epochs * graph.number_of_nodes() * R
    for node in tc:
        tc[node] /= total_samples

    return tc

# Create temporal graph
G = nx.Graph()

# Add nodes
G.add_node('A', t=(0, 1, 2, 3, 4))
G.add_node('B', t=(0, 1, 2, 3, 4))
G.add_node('C', t=(0, 1, 2, 3, 4))
G.add_node('D', t=(0, 1, 2, 3, 4))

# Add temporal edges
G.add_edge('A', 'B', t=(0, 1, 2, 3, 4))
G.add_edge('A', 'D', t=(0, 1, 2, 3, 4))
G.add_edge('B', 'D', t=(0, 1, 2, 3, 4))
G.add_edge('B', 'C', t=(0, 3, 4))
G.add_edge('C', 'D', t=(0, 1, 4))

with open('test.txt', 'w') as file:
    for i in range(1, 21):
        for _ in range(5):
            temporal_connectivity = temporal_conn(G, 5, i)
            file.write(f'Epochs: {i}:\n')
            file.write(str(temporal_connectivity))
            file.write('\n')
        file.write('\n')