#include <iostream>
#include <functional>

#include "event.hpp"
#include "tgraph.hpp"

int main(int argc, char **argv) {
    CSVEventStream stream("~/code/Tempo/data/soc-sign-bitcoinotc.csv");
    TGraph graph;

    graph.ingest(stream);

    return 0;
}