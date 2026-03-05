#include <vector>

#include "event.hpp"

/*
 * Get the next event from the CSV file.
 * Arguments:
 *     Event& e - Store the next event here.
 * Returns:
 *     bool - true if there is another event, false otherwise.
 */
bool CSVEventStream::next(Event& e) {
    std::string line;

    // No more data
    if(!std::getline(file, line)) return false;
    
    // Consume header
    if(header) {
        if(!std::getline(file, line)) return false;
        header = false;
    }

    /* Split the line on commas */
    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> cols;

    while(std::getline(ss, token, ',')) {
        cols.push_back(token);
    }

    // Assign each node a unique node ID
    if(node_lookup.find(cols[src]) == node_lookup.end()) {
        node_lookup[cols[src]] = current_node++;
    }
    if(node_lookup.find(cols[dest]) == node_lookup.end()) {
        node_lookup[cols[dest]] = current_node++;
    }

    // Assign each event type a unique type ID
    if(type_lookup.find(cols[type]) == type_lookup.end()) {
        type_lookup[cols[type]] = current_type++;
    }

    // Save the data into the event
    e.src = node_lookup[cols[src]];
    e.dest = node_lookup[cols[dest]];
    if(type > 0) e.type = type_lookup[cols[type]];
    e.t = std::stof(cols[time]);

    return true;
}

/*
 * Get the node ID of a node based on what it was inputted as.
 * Arguments:
 *     const std::string& node - The input name of the node.
 * Returns:
 *     std::optional<uint32_t> - The unique node ID if it exists, otherwise null.
 */
std::optional<uint32_t> CSVEventStream::lookup_node(const std::string& node) {
    if(node_lookup.find(node) == node_lookup.end()) return std::nullopt;
    return node_lookup[node];
}

/*
 * Get the type ID of an event type based on what is was inputted as.
 * Arguments:
 *     const std::string& type - The input name of the type.
 * Returns:
 *     std::optional<uint16_t> - The unique type ID if it exists, otherwise null.
 */
std::optional<uint16_t> CSVEventStream::lookup_type(const std::string& type) {
    if(type_lookup.find(type) == type_lookup.end()) return std::nullopt;
    return type_lookup[type];
}

/*
 * Get the node ID of a node based on what it was inputted as.
 * Arguments:
 *     const std::string& node - The input name of the node.
 * Returns:
 *     std::optional<uint32_t> - The unique node ID if it exists, otherwise null.
 */
std::optional<uint32_t> TSVEventStream::lookup_node(const std::string& node) {
    if(node_lookup.find(node) == node_lookup.end()) return std::nullopt;
    return node_lookup[node];
}

/*
 * Get the type ID of an event type based on what is was inputted as.
 * Arguments:
 *     const std::string& type - The input name of the type.
 * Returns:
 *     std::optional<uint16_t> - The unique type ID if it exists, otherwise null.
 */
std::optional<uint16_t> TSVEventStream::lookup_type(const std::string& type) {
    if(type_lookup.find(type) == type_lookup.end()) return std::nullopt;
    return type_lookup[type];
}

/*
 * Get the next event from the TSV file.
 * Arguments:
 *     Event& e - Store the next event here.
 * Returns:
 *     bool - true if there is another event, false otherwise.
 */
bool TSVEventStream::next(Event& e) {
    std::string line;

    // No more data
    if(!std::getline(file, line)) return false;
    
    // Consume header
    if(header) {
        if(!std::getline(file, line)) return false;
        header = false;
    }

    /* Split the line on commas */
    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> cols;

    while(std::getline(ss, token, '\t')) {
        cols.push_back(token);
    }

    // Assign each node a unique node ID
    if(node_lookup.find(cols[src]) == node_lookup.end()) {
        node_lookup[cols[src]] = current_node++;
    }
    if(node_lookup.find(cols[dest]) == node_lookup.end()) {
        node_lookup[cols[dest]] = current_node++;
    }

    // Assign each event type a unique type ID
    if(type_lookup.find(cols[type]) == type_lookup.end()) {
        type_lookup[cols[type]] = current_type++;
    }

    // Save the data into the event
    e.src = node_lookup[cols[src]];
    e.dest = node_lookup[cols[dest]];
    if(type > 0) e.type = type_lookup[cols[type]];
    e.t = std::stof(cols[time]);

    return true;
}

/*
 * Get the next event from the base stream with the filter applied.
 * Arguments:
 *     Event& e - Store the next event here.
 * Returns:
 *     bool - true if there is another event, false otherwise.
 */
bool FilterEventStream::next(Event& e) {
    while(input.next(e)) {
        if(filter(e)) return true;
    }
    return false;
}