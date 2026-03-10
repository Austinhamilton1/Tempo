#include <algorithm>
#include <assert.h>

#include "tgraph.hpp"

/*
 * Reorder edges based on indexed permutation.
 * Arguments:
 *     uint32_t start - Reorder starting from here.
 *     std::vector<uint32_t>& perm - Reorder based on this permutation.
 */
void TGraph::reorder_range(uint32_t start, std::vector<uint32_t>& perm) {
    std::vector<uint32_t> n_tmp(perm.size());
    std::vector<uint32_t> t_tmp(perm.size());
    std::vector<uint16_t> e_tmp(perm.size());

    for(size_t i = 0; i < perm.size(); i++) {
        n_tmp[i] = neighbor[perm[i]];
        t_tmp[i] = timestamp[perm[i]];
        e_tmp[i] = event_type[perm[i]];
    }

    for(size_t i = 0; i < perm.size(); i++) {
        neighbor[start+i] = n_tmp[i];
        timestamp[start+i] = t_tmp[i];
        event_type[start+i] = e_tmp[i];
    }
}

/*
 * Initialize a temporal graph from an edge stream.
 * Arguments:
 *     EventStream& stream - Initialize the graph from these events.
 */
void TGraph::ingest(EventStream& stream) {
    size_t m = 0;   // Event count
    std::vector<uint32_t> degree;   // Degrees of nodes

    // PASS 1: Compute degrees and event count
    Event e;
    while(stream.next(e)) {
        if(e.src >= degree.size()) {
            degree.resize(e.src + 1, 0);
        }

        degree[e.src]++;
        m++;
    }

    uint32_t n = degree.size(); // Node count

    // Build node_index
    node_index.resize(n + 1);
    node_index[0] = 0;
    for(uint32_t i = 0; i < n; i++) {
        node_index[i+1] = node_index[i] + degree[i];
    }

    // Allocate edge arrays
    neighbor.resize(m);
    timestamp.resize(m);
    event_type.resize(m);

    // Cursor positions
    std::vector<uint32_t> cursor = node_index;

    // Pass 2: Build event arrays
    stream.reset();
    while(stream.next(e)) {
        uint32_t pos = cursor[e.src]++;

        neighbor[pos] = e.dest;
        timestamp[pos] = e.t;
        event_type[pos] = e.type;
    }

    /* Sort edges by timestamp */
    for(uint32_t u = 0; u < node_index.size() - 1; u++) {
        uint32_t start = node_index[u];
        uint32_t end = node_index[u+1];
        uint32_t len = end - start;

        if(len <= 1) continue;

        std::vector<uint32_t> perm(len);

        for(uint32_t i = 0; i < len; i++) {
            perm[i] = start + i;
        }

        std::sort(perm.begin(), perm.end(),
            [&](uint32_t a, uint32_t b) {
                return timestamp[a] < timestamp[b];
            });

        reorder_range(start, perm);
    }

    // Mark all nodes active
    node_active.assign(n, 1);
}


/*
 * Return the number of nodes in the graph.
 * Returns:
 *     size_t - Number of nodes in the graph.
 */
size_t TGraph::num_nodes() const {
    return node_index.size() - 1;
}

/*
 * Returns the number of temporal edges in the graph.
 * Returns:
 *     size_t - Number of temporal edges in the graph.
 */
size_t TGraph::num_edges() const {
    return neighbor.size();
}

/*
 * Get the neighbor at a specific edge ID.
 * Arguments:
 *     uint32_t edge_id - Index into the neighbor array.
 * Returns:
 *     uint32_t - Value of neighbor[edge_id].
 */
uint32_t TGraph::get_neighbor(uint32_t edge_id) const {
    return neighbor[edge_id];
}

/*
 * Get the timestamp at a specific edge ID.
 * Arguments: 
 *     uint32_t edge_id - Index into the timestamp array.
 * Returns:
 *     uint32_t - Value of timestamp[edge_id].
 */
uint32_t TGraph::get_timestamp(uint32_t edge_id) const {
    return timestamp[edge_id];
}

/*
 * Get the event type at a specific edge ID.
 * Arguments:
 *     uint32_t edge_id - Index into the event_type array.
 * Returns:
 *     uint16_t - Value of event_type[edge_id].
 */
uint16_t TGraph::get_event_type(uint32_t edge_id) const {
    return event_type[edge_id];
}

/*
 * Get all temporal neighbors of a node.
 * Arguments:
 *     uint32_t u - Query this node.
 * Returns:
 *     EdgeRange - The temporal neighborhood of the node.
 */
EdgeRange TGraph::neighbors_range(uint32_t u) const {
    assert(u + 1 < node_index.size());

    return { node_index[u], node_index[u+1] };
}

/*
 * Get the temporal neighbors of a node after a start time.
 * Arguments:
 *     uint32_t u - Query this node.
 *     uint32_t start_time - Get neighbors starting at this time.
 * Returns:
 *     EdgeRange - The temporal neighborhood of the node.
 */
EdgeRange TGraph::neighbors_range(uint32_t u, uint32_t start_time) const {
    assert(u + 1 < node_index.size());

    uint32_t start = node_index[u];
    uint32_t end = node_index[u+1];

    auto t_begin = timestamp.begin() + start;
    auto t_end = timestamp.begin() + end;

    // Binary search for lower bound
    auto it = std::lower_bound(
        t_begin,
        t_end,
        start_time
    );

    return { (uint32_t)(it - timestamp.begin()), end };
}

/*
 * Get the temporal neighbors of a node between a start and end time.
 * Arguments:
 *     uint32_t u - Query this node.
 *     uint32_t start_time - Get neighbors starting at this time.
 *     uint32_t end_time - Get neighbors up until this time.
 * Returns:
 *     EdgeRange - The temporal neighborhood of the node.
 */
EdgeRange TGraph::neighbors_range(uint32_t u, uint32_t start_time, uint32_t end_time) const {
    assert(u + 1 < node_index.size());

    uint32_t start = node_index[u];
    uint32_t end = node_index[u+1];

    auto t_begin = timestamp.begin() + start;
    auto t_end = timestamp.begin() + end;

    // Binary search for lower bound
    auto lit = std::lower_bound(
        t_begin,
        t_end,
        start_time
    );

    // Binary search for upper bound
    auto hit = std::upper_bound(
        lit,
        t_end,
        end_time
    );

    return { (uint32_t)(lit - timestamp.begin()), (uint32_t)(hit - timestamp.begin()) };
}

/*
 * Get all temporal neighbors of a node.
 * Arguments:
 *     uint32_t u - Query this node.
 * Returns:
 *     NeighborView - The temporal neighborhood of the node.
 */
NeighborView TGraph::neighbors(uint32_t u) const {
    assert(u + 1 < node_index.size());

    auto r = neighbors_range(u);
    return NeighborView{ this, r.start, r.end };
}

/*
 * Get the temporal neighbors of a node after a start time.
 * Arguments:
 *     uint32_t u - Query this node.
 *     uint32_t start_time - Get neighbors starting at this time.
 * Returns:
 *     NeighborView - The temporal neighborhood of the node.
 */
NeighborView TGraph::neighbors(uint32_t u, uint32_t start_time) const {
    assert(u + 1 < node_index.size());

    auto r = neighbors_range(u, start_time);
    return NeighborView{ this, r.start, r.end };
}

/*
 * Get the temporal neighbors of a node between a start and end time.
 * Arguments:
 *     uint32_t u - Query this node.
 *     uint32_t start_time - Get neighbors starting at this time.
 *     uint32_t end_time - Get neighbors up until this time.
 * Returns:
 *     NeighborView - The temporal neighborhood of the node.
 */
NeighborView TGraph::neighbors(uint32_t u, uint32_t start_time, uint32_t end_time) const {
    assert(u + 1 < node_index.size());

    auto r = neighbors_range(u, end_time);
    return NeighborView{ this, r.start, r.end };
}

/*
 * Get the temporal degree of a node.
 * Arguments:
 *     uint32_t u - Query this node.
 * Returns:
 *     size_t - The temporal degree of node u.
 */
size_t TGraph::degree(uint32_t u) const {
    assert(u + 1 < node_index.size());

    return node_index[u+1] - node_index[u];
}

/*
 * Check if an edge exists between two nodes between two times.
 * Arguments:
 *     uint32_t u - Source node.
 *     uint32_t v - Destination node.
 *     uint32_t start_time - Check for edges on or after this time.
 *     uint32_t end_time - Check for edges on or before this time.
 * Returns:
 *     bool - true if an edge exists, false otherwise.
 */
bool TGraph::has_edge(uint32_t u, uint32_t v, uint32_t start_time, uint32_t end_time) const {
    // Get all edges within the time window
    EdgeRange range = neighbors_range(u, start_time, end_time);

    // Check if any of the edges in the time window connect to v
    if(range.size() == 0) return false;
    for(size_t w = range.start; w < range.end; w++) {
        if(neighbor[w] == v) return true;
    }

    return false;
}