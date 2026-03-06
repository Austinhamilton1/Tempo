#pragma once

#include <cstdint>
#include <string>
#include <fstream>
#include <unordered_map>
#include <sstream>
#include <optional>
#include <functional>
#include <queue>

/*
 * This data structure represents an event (edge) in the graph.
 * It will only be used for processing on the host, these will not be stored.
 */
struct Event {
    uint32_t    src;    // Source node
    uint32_t    dest;   // Destination node
    uint16_t    type;   // Type of event
    float       t;      // Timestamp

    Event() : src(0), dest(0), type(0), t(0) {}
    Event(uint32_t src, uint32_t dest, uint16_t type, float t) 
        : src(src), dest(dest), type(type), t(t) {}
};

/*
 * This interface allows iteration of events from some source
 * without the need to store all events in memory.
 */
class EventStream {
public:
    virtual ~EventStream() {};

    // Get the next event in the stream
    virtual bool next(Event &) = 0;
};

/*
 * This stream pulls in graph events from a CSV file.
 */
class CSVEventStream : public EventStream {
private:
    std::ifstream file; // File to pull events from
    bool header;        // Does the file have a header?
    
    /* Column indexes of the data */
    int src;
    int dest;
    int time;
    int type;

    /* Node lookup table */
    uint32_t current_node;
    std::unordered_map<std::string, uint32_t> node_lookup;
    
    /* Event type lookup table */
    uint16_t current_type;
    std::unordered_map<std::string, uint16_t> type_lookup;

public:
    /*
     * Create an instance of a CSVEventStream.
     * Arguments:
     *     const std::string& path - File name to ingest.
     *     bool header - Does the file have a header?
     *     int src - Column index of the source node.
     *     int dest - Column index of the destination node.
     *     int type - Column index of the event type.
     *     int time - Column index of the timestamp.
     */
    CSVEventStream(const std::string& path, bool header=false, int src=0, int dest=1, int type=2, int time=3)
        : file(path), header(header), src(src), dest(dest), time(time), type(type), current_node(0), current_type(0) {}

    /*
     * Get the next event from the CSV file.
     * Arguments:
     *     Event& e - Store the next event here.
     * Returns:
     *     bool - true if there is another event, false otherwise.
     */
    bool next(Event &e) override;

    /*
     * Get the node ID of a node based on what it was inputted as.
     * Arguments:
     *     const std::string& node - The input name of the node.
     * Returns:
     *     std::optional<uint32_t> - The unique node ID if it exists, otherwise null.
     */
    std::optional<uint32_t> lookup_node(const std::string& node);

    /*
     * Get the type ID of an event type based on what is was inputted as.
     * Arguments:
     *     const std::string& type - The input name of the type.
     * Returns:
     *     std::optional<uint16_t> - The unique type ID if it exists, otherwise null.
     */
    std::optional<uint16_t> lookup_type(const std::string& type);
};

/*
 * This stream pulls in graph events from a TSV file.
 */
class TSVEventStream : public EventStream {
private:
    std::ifstream file; // File to pull events from
    bool header;        // Does the file have a header?
    
    /* Column indexes of the data */
    int src;
    int dest;
    int time;
    int type;

    /* Node lookup table */
    uint32_t current_node;
    std::unordered_map<std::string, uint32_t> node_lookup;
    
    /* Event type lookup table */
    uint16_t current_type;
    std::unordered_map<std::string, uint16_t> type_lookup;

public:
    /*
     * Create an instance of a TSVEventStream.
     * Arguments:
     *     const std::string& path - File name to ingest.
     *     bool header - Does the file have a header?
     *     int src - Column index of the source node.
     *     int dest - Column index of the destination node.
     *     int type - Column index of the event type.
     *     int time - Column index of the timestamp.
     */
    TSVEventStream(const std::string& path, int header=false, int src=0, int dest=1, int type=2, int time=3)
        : file(path), header(header), src(src), dest(dest), time(time), type(type), current_node(0), current_type(0) {}

    /*
     * Get the next event from the CSV file.
     * Arguments:
     *     Event& e - Store the next event here.
     * Returns:
     *     bool - true if there is another event, false otherwise.
     */
    bool next(Event &e) override;

    /*
     * Get the node ID of a node based on what it was inputted as.
     * Arguments:
     *     const std::string& node - The input name of the node.
     * Returns:
     *     std::optional<uint32_t> - The unique node ID if it exists, otherwise null.
     */
    std::optional<uint32_t> lookup_node(const std::string& node);

    /*
     * Get the type ID of an event type based on what is was inputted as.
     * Arguments:
     *     const std::string& type - The input name of the type.
     * Returns:
     *     std::optional<uint16_t> - The unique type ID if it exists, otherwise null.
     */
    std::optional<uint16_t> lookup_type(const std::string& type);
};

/*
 * A filter stream sits on top of another stream and filters based on the event.
 */
class FilterEventStream : public EventStream {
private:
    EventStream& input;
    std::function<bool(const Event& e)> filter;

public:
    /*
     * Create a FilterEventStream.
     * Arguments:
     *     EventStream& stream - Base stream to filter on.
     *     std::function<bool(const Event& e, const EventStream& stream)> filter - Filter based on this function (return true to keep, false to discard).
     */
    FilterEventStream(EventStream& stream, std::function<bool(const Event& e)> filter)
        : input(stream), filter(filter) {};

    /*
     * Get the next event from the base stream with the filter applied.
     * Arguments:
     *     Event& e - Store the next event here.
     * Returns:
     *     bool - true if there is another event, false otherwise.
     */
    bool next(Event& e) override;
};