#include <iostream>
#include <functional>

#include "event.hpp"

int main(int argc, char **argv) {
    CSVEventStream stream("../data/soc-sign-bitcoinotc.csv");
    FilterEventStream filter(stream, [&stream](const Event &e) {
        return e.type == stream.lookup_type("8");
    });

    Event e;

    for(int i = 0; i < 5; i++) {
        if(!filter.next(e)) break;
        std::cout << "Event " << i+1 << ": (" << e.src << ", " << e.dest << ", " << e.type << ", " << e.t << ")" << std::endl; 
    }

    return 0;
}