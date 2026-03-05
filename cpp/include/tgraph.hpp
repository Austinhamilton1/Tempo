#pragma once

#include <cstdint>
#include <vector>

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
    std::vector<uint32_t>   timestamp;              // Time stamp of edge
    std::vector<uint8_t>    event_type;             // Type of event an edge represents

    // Node index
    std::vector<uint32_t>   node_index;             // T-CSR pointer
    std::vector<uint8_t>    node_active;            // Mask for node deletion

    // Node embeddings
    std::vector<std::vector<float>> X;              // Feature embeddings
    std::vector<std::vector<float>> S;              // Structural embeddings
    std::vector<std::vector<std::vector<float>>> T; // Temporal embeddings

public:
};