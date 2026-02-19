import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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

def connect(graph: nx.Graph, R: int, epochs: int) -> dict[any, float]:
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
    tc = { node: 0.0 for node in graph.nodes }
    for _ in range(epochs):
        for node in graph.nodes:
            # Start the random walk
            current_node = node
            t_current = 0
            for _ in range(R):
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

def events(graph: nx.Graph) -> list[tuple]:
    '''
    Generate a stream of events from a temporal graph.

    :param graph: Graph to stream.
    :type graph: nx.Graph
    :return: List of all graph events that occur in the graph.
    :rtype: list[tuple]
    '''
    e = []
    t_end = max([t for node in graph.nodes for t in graph.nodes[node]['t']])
    
    '''Calculate all node events.'''
    for node in graph.nodes:
        if not graph.nodes[node]['t'][0] == 0:
            e.append(('insert', 'node', node, graph.nodes[node]['t'][0]))
        for i in range(1, len(graph.nodes[node]['t'])):
            t = graph.nodes[node]['t'][i]
            last_seen = graph.nodes[node]['t'][i-1]
            if t - last_seen > 1:
                e.append(('delete', 'node', node, last_seen+1))
                e.append(('insert', 'node', node, t))
        if t < t_end:
            e.append(('delete', 'node', node, t+1))
    
    '''Calculate all edge events.'''
    for edge in graph.edges:
        if not graph.edges[edge]['t'][0] == 0:
            e.append(('insert', 'edge', edge, graph.edges[edge]['t'][0]))
        for i in range(1, len(graph.edges[edge]['t'])):
            t = graph.edges[edge]['t'][i]
            last_seen = graph.edges[edge]['t'][i-1]
            if t - last_seen > 1:
                e.append(('delete', 'edge', edge, last_seen+1))
                e.append(('insert', 'edge', edge, t))
        if t < t_end:
            e.append(('delete', 'edge', edge, t+1))

    return sorted(e, key=lambda event: event[3])

def entropy(graph: nx.Graph, clusters: dict[any, int]=None) -> dict[any, float]:
    '''
    For each node, calculate how much predictability information would be lost
    if that node was deleted.
    
    :param graph: Temporal graph to calculate eventual_energy for.
    :type graph: nx.Graph
    :param clusters: A partitioning of the temporal graph.
    :type: dict[any, int]
    :return: Map from node to eventual_energy.
    :rtype: dict[Any, float]
    '''
    te = { v: 0.0 for v in graph.nodes }

    e = events(graph)

    if clusters is None:
        N = len(e)

        # P(e = E) is the probability of an event e being type E
        p_e = {
            ('insert', 'node'): 0.0,
            ('delete', 'node'): 0.0,
            ('insert', 'edge'): 0.0,
            ('delete', 'edge'): 0.0,
        }
        
        delta_v = defaultdict(list)
        last_t = {}

        for event in e:
            # Calculate the probability of an event being a certain type
            e_type, e_target, e_data, t = event
            p_e[(e_type, e_target)] += 1 / N

            # Need to calculate the mean event time per node to determine
            # if an event is a long-term or short term event
            if e_target == 'node':
                delta_t = t - last_t.get(e_data, 0)
                delta_v[e_data].append(delta_t)
                last_t[e_data] = t
            elif e_target == 'edge':
                u, v = e_data

                delta_t = t - last_t.get(u, 0)
                delta_v[u].append(delta_t)
                last_t[u] = t

                delta_t = t - last_t.get(v, 0)
                delta_v[v].append(delta_t)
                last_t[v] = t
            
        delta_m = { v: np.mean(delta_v[v]) for v in delta_v }

        # P(e = E, t = T) is the probability of event e being type E and delta_t = {short-term, long-te}
        p_et = {
            e: {
                'short': 0,
                'long': 0,
            }
            for e in p_e
        }

        last_t = {}
        for event in e:
            e_type, e_target, e_data, t = event
            if e_target == 'node':
                # Node events affect one bucket so add 1 / N to the bucket
                delta_t = t - last_t.get(e_data, 0)
                if delta_t <= delta_m[e_data]:
                    p_et[(e_type, e_target)]['short'] += 1 / N
                else:
                    p_et[(e_type, e_target)]['long'] += 1 / N
                last_t[e_data] = t
            elif e_target == 'edge':
                # Edge events affect potentially two buckets 
                # (bucket for u, bucket for v) so add 0.5 / N to the bucket
                u, v = e_data

                delta_t = t - last_t.get(u, 0)
                if delta_t <= delta_m[u]:
                    p_et[(e_type, e_target)]['short'] += 0.5 / N
                else:
                    p_et[(e_type, e_target)]['long'] += 0.5 / N
                last_t[u] = t

                delta_t = t - last_t.get(v, 0)
                if delta_t <= delta_m[v]:
                    p_et[(e_type, e_target)]['short'] += 0.5 / N
                else:
                    p_et[(e_type, e_target)]['long'] += 0.5 / N
                last_t[v] = t

        '''Calculate conditional entropy of H(T|E). This is the predictabilty of the time of an event given its type.'''
        '''Temporal GNNs try to predict events in time so we are given event type (e.g., predict when an edge will be inserted here).'''
        last_t = {}
        for event in e:
            e_type, e_target, e_data, t = event
            if e_target == 'node':
                # If target is a node, update the node with the event entropy
                delta_t = t - last_t.get(e_data, 0)
                if delta_t <= delta_m[e_data]:
                    te[e_data] -= float(
                        p_et[(e_type, e_target)]['short'] *
                        np.log2(
                            p_et[(e_type, e_target)]['short'] / 
                            p_e[(e_type, e_target)]
                        )
                    )
                else:
                    te[e_data] -= float(
                        p_et[(e_type, e_target)]['long'] *
                        np.log2(
                            p_et[(e_type, e_target)]['long'] / 
                            p_e[(e_type, e_target)]
                        )
                    )
                last_t[e_data] = t
            elif e_target == 'edge':
                # If target is an edge, update both nodes affected with the event entropy
                u, v = e_data

                delta_t = t - last_t.get(u, 0)
                if delta_t <= delta_m[u]:
                    te[u] -= float(
                        p_et[(e_type, e_target)]['short'] *
                        np.log2(
                            p_et[(e_type, e_target)]['short'] / 
                            p_e[(e_type, e_target)]
                        )
                    )
                else:
                    te[u] -= float(
                        p_et[(e_type, e_target)]['long'] *
                        np.log2(
                            p_et[(e_type, e_target)]['long'] / 
                            p_e[(e_type, e_target)]
                        )
                    )
                last_t[u] = t

                delta_t = t - last_t.get(v, 0)
                if delta_t <= delta_m[v]:
                    te[v] -= float(
                        p_et[(e_type, e_target)]['short'] *
                        np.log2(
                            p_et[(e_type, e_target)]['short'] / 
                            p_e[(e_type, e_target)]
                        )
                    )
                else:
                    te[v] -= float(
                        p_et[(e_type, e_target)]['long'] *
                        np.log2(
                            p_et[(e_type, e_target)]['long'] / 
                            p_e[(e_type, e_target)]
                        )
                    )
                last_t[v] = t
    else:
        # Calculate the number of nodes in each cluster
        N_clusters = { n: 0 for n in clusters.values() }
        for node in graph.nodes:
            N_clusters[clusters[node]] += 1

        # Calculate the number of events that occur in each cluster
        E_clusters = { n: 0 for n in clusters.values() }
        for _, e_target, e_data, _ in e:
            if e_target == 'node':
                E_clusters[clusters[e_data]] += 1
            elif e_target == 'edge':
                # For an edge event, each cluster receives half of the event (totalling a single event)
                u, v = e_data
                E_clusters[clusters[u]] += 0.5
                E_clusters[clusters[v]] += 0.5

        # P(x) is the probability of node n being a member of cluster C
        p_x = { n: N_clusters[n] / graph.number_of_nodes() for n in clusters.values() }

        # P(x, y) is the probability of node n being apart of event e where n and e are both members
        # of cluster C
        p_xy = {
            n: {
                ('insert', 'node'): 0.0,
                ('delete', 'node'): 0.0,
                ('insert', 'edge'): 0.0,
                ('delete', 'edge'): 0.0,
            }
            for n in clusters.values()
        }

        for e_type, e_target, e_data, _ in e:
            if e_target == 'node':
                p_xy[clusters[e_data]][(e_type, e_target)] += 1 / E_clusters[clusters[e_data]] * p_x[clusters[e_data]]
            elif e_target == 'edge':
                u, v = e_data
                p_xy[clusters[u]][(e_type, e_target)] += 0.5 / E_clusters[clusters[u]] * p_x[clusters[u]]
                p_xy[clusters[v]][(e_type, e_target)] += 0.5 / E_clusters[clusters[v]] * p_x[clusters[v]]

        # Calculate the entropy of each node given its event type within a cluster
        for e_type, e_target, e_data, _ in e:
            if e_target == 'node':
                # If the target is a node, update the node with the event entropy
                te[e_data] -= float(
                    p_xy[clusters[e_data]][(e_type, e_target)] * 
                    np.log2(
                        p_xy[clusters[e_data]][(e_type, e_target)] /
                        p_x[clusters[e_data]]
                    )
                )
            elif e_target == 'edge':
                # If the target is an edge, update both nodes with the event entropy
                u, v = e_data
                te[u] -= float(
                    p_xy[clusters[u]][(e_type, e_target)] * 
                    np.log2(
                        p_xy[clusters[u]][(e_type, e_target)] /
                        p_x[clusters[u]]
                    )
                )
                te[v] -= float(
                    p_xy[clusters[v]][(e_type, e_target)] * 
                    np.log2(
                        p_xy[clusters[v]][(e_type, e_target)] /
                        p_x[clusters[v]]
                    )
                )
                
    return te

# Create temporal graph
G = nx.Graph()

# Add nodes
G.add_node('a', t=(0, 1, 2, 3))
G.add_node('b', t=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_node('c', t=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_node('d', t=(1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_node('e', t=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_node('f', t=(3, 4, 5, 7, 8, 9))
G.add_node('g', t=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_node('h', t=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_node('i', t=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_node('j', t=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_node('k', t=(6, 7, 8, 9))
G.add_node('l', t=(5, 6, 7, 8, 9))
G.add_node('m', t=(6, 7, 8, 9))

# Add edges
G.add_edge('a', 'b', t=(0, 1, 2, 3))
G.add_edge('a', 'd', t=(1, 2, 3))
G.add_edge('a', 'f', t=(0, 1, 2))
G.add_edge('b', 'c', t=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_edge('b', 'f', t=(3, 4, 5, 8, 9))
G.add_edge('c', 'd', t=(1, 2, 3, 4, 6, 7, 8, 9))
G.add_edge('c', 'e', t=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_edge('d', 'e', t=(1, 3, 5, 7, 9))
G.add_edge('d', 'm', t=(6, 7, 8, 9))
G.add_edge('e', 'f', t=(3, 4, 5, 6, 7, 8, 9))
G.add_edge('e', 'g', t=(4, 5, 6, 7, 8, 9))
G.add_edge('e', 'k', t=(6, 7, 8, 9))
G.add_edge('g', 'h', t=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_edge('g', 'i', t=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_edge('g', 'j', t=(0, 1, 2, 3, 4, 7, 8, 9))
G.add_edge('h', 'i', t=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
G.add_edge('i', 'j', t=(0, 1, 2, 3, 4, 5, 9))
G.add_edge('k', 'l', t=(6, 7, 8, 9))
G.add_edge('k', 'm', t=(8, 9))
G.add_edge('l', 'm', t=(6, 7, 9))

clusters = {
    'a': 1,
    'b': 1,
    'c': 1,
    'd': 1,
    'e': 1,
    'f': 1,
    'g': 2,
    'h': 2,
    'i': 2,
    'j': 2,
    'k': 3,
    'l': 3,
    'm': 3,
}

e = entropy(G)

print(e)