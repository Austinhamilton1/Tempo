#pragma once

#include <cstdint>
#include <vector>
#include <span>

#include "event.hpp"

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
    std::vector<uint32_t>   neighbor;               // Destination endpoints
    std::vector<uint64_t>   timestamp;              // Time stamp of edge
    std::vector<uint16_t>   event_type;             // Type of event an edge represents

    // Node index
    std::vector<uint32_t>   node_index;             // T-CSR pointer
    std::vector<uint8_t>    node_active;            // Mask for node deletion

    // Node embeddings
    std::vector<std::vector<float>> X;              // Feature embeddings
    std::vector<std::vector<float>> S;              // Structural embeddings
    std::vector<std::vector<std::vector<float>>> T; // Temporal embeddings

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
     * Get the temporal neighbors of a node between a start and end time.
     * Arguments:
     *     uint32_t u - Query this node.
     * Returns:
     *     EdgeRange - The temporal neighborhood of the node.
     */
    EdgeRange neighbors_range(uint32_t u);

    /*
     * Get the temporal neighbors of a node between a start and end time.
     * Arguments:
     *     uint32_t u - Query this node.
     *     uint32_t start_time - Get neighbors starting at this time.
     * Returns:
     *     EdgeRange - The temporal neighborhood of the node.
     */
    EdgeRange neighbors_range(uint32_t u, uint32_t start_time);

    /*
     * Get the temporal neighbors of a node between a start and end time.
     * Arguments:
     *     uint32_t u - Query this node.
     *     uint32_t start_time - Get neighbors starting at this time.
     *     uint32_t end_time - Get neighbors up until this time.
     * Returns:
     *     EdgeRange - The temporal neighborhood of the node.
     */
    EdgeRange neighbors_range(uint32_t u, uint32_t start_time, uint32_t end_time);
};