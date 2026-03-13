#pragma once

#include <cstdint>
#include <vector>
#include <tuple>

#include "event.hpp"

struct NeighborView;

/*
 * Represents a range of edges. Using start and end,
 * the user can slice into the graph data.
 */
struct EdgeRange {
    uint32_t start;
    uint32_t end;

    /*
     * Return the size of the iterator.
     * Returns:
     *     size_t - How many edges are in the range.
     */
    size_t size() const {
        return end - start;
    }
};

/*
 * This data structure represents a temporal graph. Temporal graphs need to 
 * be stored memory efficient and have the ability to be easily ported to device.
 * As such we went with a structure of arrays (SoA) temporal compressed sparse row
 * (T-CSR) storage format.
 */
class TGraph {
private:
    // Edge arrays
    std::vector<uint32_t>   neighbor;       // Destination endpoints
    std::vector<uint64_t>   timestamp;      // Time stamp of edge
    std::vector<uint16_t>   event_type;     // Type of event an edge represents

    // Node index
    std::vector<uint32_t>   node_index;     // T-CSR pointer
    std::vector<uint8_t>    node_active;    // Mask for node deletion

    // Node embeddings
    std::vector<float>      X;              // Feature embeddings
    std::vector<float>      S;              // Structural embeddings
    std::vector<float>      T;              // Temporal embeddings

    /*
     * Reorder edges based on indexed permutation.
     * Arguments:
     *     uint32_t start - Reorder starting from here.
     *     std::vector<uint32_t>& perm - Reorder based on this permutation.
     */
    void reorder_range(uint32_t start, std::vector<uint32_t>& perm);

public:
    // Create/destroy a temporal graph
    TGraph() {}
    ~TGraph() {};

    /*
     * Initialize a temporal graph from an edge stream.
     * Arguments:
     *     EventStream& stream - Initialize the graph from these events.
     */
    void ingest(EventStream& stream);

    /*
     * Return the number of nodes in the graph.
     * Returns:
     *     size_t - Number of nodes in the graph.
     */
    size_t num_nodes() const;

    /*
     * Returns the number of temporal edges in the graph.
     * Returns:
     *     size_t - Number of temporal edges in the graph.
     */
    size_t num_edges() const;

    /*
     * Return the temporal degree of a node.
     * Arguments:
     *     uint32_t - Query this node.
     * Returns:
     *     size_t - Temporal degree of node u.
     */
    size_t degree(uint32_t u) const;

    /*
     * Check if an edge ever exists between two nodes.
     * Arguments:
     *     uint32_t u - Source node.
     *     uint32_t v - Destination node.
     * Returns:
     *     bool - true if an edge exists, false otherwise.
     */
    bool has_edge(uint32_t u, uint32_t v) const;

    /*
     * Check if an edge exists between two nodes after a certain time.
     * Arguments:
     *     uint32_t u - Source node.
     *     uint32_t v - Destination node.
     *     uint64_t start_time - Check for edges after this time.
     * Returns:
     *     bool - true if an edge exists, false otherwise.
     */
    bool has_edge(uint32_t u, uint32_t v, uint64_t start_time) const;

    /*
     * Check if an edge exists between two nodes between two times.
     * Arguments:
     *     uint32_t u - Source node.
     *     uint32_t v - Destination node.
     *     uint64_t start_time - Check for edges on or after this time.
     *     uint64_t end_time - Check for edges on or before this time.
     * Returns:
     *     bool - true if an edge exists, false otherwise.
     */
    bool has_edge(uint32_t u, uint32_t v, uint64_t start_time, uint64_t end_time) const;

    /*
     * Get the neighbor at a specific edge ID.
     * Arguments:
     *     uint32_t edge_id - Index into the neighbor array.
     * Returns:
     *     uint32_t - Value of neighbor[edge_id].
     */
    uint32_t get_neighbor(uint32_t edge_id) const;

    /*
     * Get a constant pointer to the neighbor array.
     * Returns:
     *     const uint32_t * - Constant pointer.
     */
    const uint32_t *get_neighbor_ptr() const;

    /*
     * Get the timestamp at a specific edge ID.
     * Arguments: 
     *     uint32_t edge_id - Index into the timestamp array.
     * Returns:
     *     uint64_t - Value of timestamp[edge_id].
     */
    uint64_t get_timestamp(uint32_t edge_id) const;

    /*
     * Get a constant pointer to the timestamp array.
     * Returns:
     *     const uint64_t * - Constant pointer.
     */
    const uint64_t *get_timestamp_ptr() const;
 
    /*
     * Get the event type at a specific edge ID.
     * Arguments:
     *     uint32_t edge_id - Index into the event_type array.
     * Returns:
     *     uint16_t - Value of event_type[edge_id].
     */
    uint16_t get_event_type(uint32_t edge_id) const;

    /*
     * Get a constant pointer to the event_type array.
     * Returns:
     *     const uint16_t * - Constant pointer.
     */
    const uint16_t *get_event_type_ptr() const;

    /*
     * Get the temporal neighbors of a node.
     * Arguments:
     *     uint32_t u - Query this node.
     * Returns:
     *     EdgeRange - The temporal neighborhood of the node.
     */
    EdgeRange neighbors_range(uint32_t u) const;

    /*
     * Get the temporal neighbors of a node after a start time.
     * Arguments:
     *     uint32_t u - Query this node.
     *     uint64_t start_time - Get neighbors starting at this time.
     * Returns:
     *     EdgeRange - The temporal neighborhood of the node.
     */
    EdgeRange neighbors_range(uint32_t u, uint64_t start_time) const;

    /*
     * Get the temporal neighbors of a node between a start and end time.
     * Arguments:
     *     uint32_t u - Query this node.
     *     uint64_t start_time - Get neighbors starting at this time.
     *     uint64_t end_time - Get neighbors up until this time.
     * Returns:
     *     EdgeRange - The temporal neighborhood of the node.
     */
    EdgeRange neighbors_range(uint32_t u, uint64_t start_time, uint64_t end_time) const;

    /*
     * Get the temporal neighbors of a node as an iterator.
     * Arguments:
     *     uint32_t u - Query this node.
     * Returns:
     *     NeighborView - The temporal neighborhood of the node.
     */
    NeighborView neighbors(uint32_t u) const;

    /*
     * Get the temporal neighbors of a node after a start time as an iterator.
     * Arguments:
     *     uint32_t u - Query this node.
     *     uint64_t start_time - Get neighbors starting at this time.
     * Returns:
     *     NeighborView - The temporal neighborhood of the node.
     */
    NeighborView neighbors(uint32_t u, uint64_t start_time) const;

    /*
     * Get the temporal neighbors of a node between a start and end time as an iterator.
     * Arguments:
     *     uint32_t u - Query this node.
     *     uint64_t start_time - Get neighbors starting at this time.
     *     uint64_t end_time - Get neighbors up until this time.
     * Returns:
     *     NeighborView - The temporal neighborhood of the node.
     */
    NeighborView neighbors(uint32_t u, uint64_t start_time, uint64_t end_time) const;
};

/*
 * This data structure is used to iterate through
 * neighbors of a node.
 */
struct NeighborIterator {
    const TGraph *g;
    uint32_t pos;

    NeighborIterator(const TGraph *g, uint32_t pos)
        : g(g), pos(pos) {};

    NeighborIterator& operator++() {
        pos++;
        return *this;
    }

    bool operator==(const NeighborIterator& other) const {
        return pos == other.pos;
    }
    
    bool operator!=(const NeighborIterator& other) const {
        return pos != other.pos;
    }

    std::tuple<uint32_t, uint32_t, uint16_t> operator*() const {
        return {
            g->get_neighbor(pos),
            g->get_timestamp(pos),
            g->get_event_type(pos)
        };
    }
};

/*
 * Give data from a TGraph's neighbor function.
 */
struct NeighborView {
    const TGraph *g;
    uint32_t start;
    uint32_t stop;

    NeighborIterator begin() const {
        return NeighborIterator(g, start);
    }

    NeighborIterator end() const {
        return NeighborIterator(g, stop);
    }
};