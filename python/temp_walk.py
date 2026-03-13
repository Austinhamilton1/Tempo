from tempo import tempo
import numpy as np

def node2vec_step(g: tempo.TGraph, prev_node: int, current_node: int, current_time: int, p: float, q: float):
    '''
    Get the next node in node2vec temporal walk.

    :param g: The graph to step through.
    :type g: tempo.TGraph
    :param prev_node: Last node to be visited.
    :type prev_node: int
    :param current_node: The current node (get next node after this node).
    :type current_node: int
    :param current_time: The current timestamp.
    :type current_time: int
    :param p: How likely is the walk to return to prev_node.
    :type p: float
    :param q: How likely is the walk to leave prev_node's temporal neighborhood.
    :type q: float
    '''
    # Get the temporal neighborhood of the nodes
    nbr, ts, _ = g.neighbors_array(current_node, current_time)

    # Break early if no more edges
    if len(nbr) == 0:
        return None, None
    
    if prev_node is None:
        # First step in node2vec (equal probabilities)
        weights = np.ones_like(nbr, dtype=float)
        weights /= weights.sum()
    else:
        # 1/p for returning to previous node
        weights = np.ones_like(nbr, dtype=float)
        weights[nbr == prev_node] = 1/p
        
        # 1/q for leaving previous node neighborhood
        t_neighbors, _, _ = g.neighbors_array(prev_node, current_time)
        connected = np.isin(nbr, t_neighbors)
        weights[~connected] = 1/q
        
        # Normalize
        weights /= weights.sum()

    # Select next node
    idx = np.random.choice(len(nbr), p=weights)
    return nbr[idx], ts[idx]

def node2vec_walk(g: tempo.TGraph, start_node: int, num_walks: int, walk_length: int, p: float, q: float):
    '''
    Generate a temporal random walk for a node in a temporal graph.

    :param g: The graph to generate walks for.
    :type g: tempo.TGraph
    :param start_node: Generate walks for this node.
    :type start_node: int
    :param num_walks: How many walks should be generated for start_node.
    :type num_walks: int
    :param walk_length: How long should each walk be.
    :type walk_length: int
    :param p: How likely is the walk to return to prev_node.
    :type p: float
    :param q: How likely is the walk to leave prev_node's temporal neighborhood.
    :type q: float
    '''
    # Initialize walks to -1
    walks = np.full((num_walks, walk_length), -1)
    
    # Generate walks
    for i in range(num_walks):
        t = None
        v = start_node
        ts = 0
        walks[i,0] = start_node
        for j in range(1, walk_length):
            # Get next neighbor in random walk
            nbr, ts = node2vec_step(g, t, v, ts, p, q)
            
            # No new neigbors, break early
            if nbr is None:
                break

            # Set the walk value
            walks[i,j] = nbr
            t = v
            v = nbr
    
    return walks

def node2vec_walks(g: tempo.TGraph, num_walks: int, walk_length: int, p: float=1.0, q: float=1.0):
    '''
    Generate all temporal random walks for a temporal graph.

    :param g: The graph to generate walks for.
    :type g: tempo.TGraph
    :param num_walks: How many walks should be generated for start_node.
    :type num_walks: int
    :param walk_length: How long should each walk be.
    :type walk_length: int
    :param p: How likely is the walk to return to prev_node.
    :type p: float
    :param q: How likely is the walk to leave prev_node's temporal neighborhood.
    :type q: float
    '''
    walks = []
    for u in range(g.num_nodes()):
        walks.append(node2vec_walk(g, u, num_walks, walk_length, p, q))
    return walks

if __name__ == '__main__':
    stream = tempo.CSVEventStream('../data/soc-sign-bitcoinotc.csv')
    g = tempo.TGraph()
    g.ingest(stream)

    walks = node2vec_walks(g, 10, 10, 1.0, 0.5)

    print(walks)