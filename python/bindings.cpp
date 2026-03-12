#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "event.hpp"
#include "tgraph.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tempo, m) {
    m.doc() = "Tempo Temporal Graph Engine";

    /*
     * Event 
     */
    py::class_<Event>(m, "Event")
        .def(py::init<>())
        .def(py::init<uint32_t, uint32_t, uint16_t, uint32_t>(),
            py::arg("src"),
            py::arg("dest"),
            py::arg("type"),
            py::arg("t"))
        .def_readwrite("src", &Event::src)
        .def_readwrite("dest", &Event::dest)
        .def_readwrite("type", &Event::type)
        .def_readwrite("t", &Event::t);

    /*
     * EdgeRange
     */
    py::class_<EdgeRange>(m, "EdgeRange")
        .def_readonly("start", &EdgeRange::start)
        .def_readonly("end", &EdgeRange::end)
        .def("size", &EdgeRange::size);

    /*
     * Abstract EventStream
     */
    py::class_<EventStream>(m, "EventStream");

    /*
     * CSVEventStream
     */
    py::class_<CSVEventStream, EventStream>(m, "CSVEventStream")
        .def(py::init<
            const std::string&,
            bool,
            int,
            int,
            int,
            int>(),
            py::arg("path"),
            py::arg("header") = false,
            py::arg("src") = 0,
            py::arg("dest") = 1,
            py::arg("type") = 2,
            py::arg("time") = 3)
        .def("reset", &CSVEventStream::reset)
        .def("lookup_node", &CSVEventStream::lookup_node)
        .def("lookup_type", &CSVEventStream::lookup_type);

    /*
     * TSVEventStream
     */
    py::class_<TSVEventStream, EventStream>(m, "TSVEventStream")
        .def(py::init<
            const std::string&,
            bool,
            int,
            int,
            int,
            int>(),
            py::arg("path"),
            py::arg("header") = false,
            py::arg("src") = 0,
            py::arg("dest") = 1,
            py::arg("type") = 2,
            py::arg("time") = 3)
        .def("reset", &TSVEventStream::reset)
        .def("lookup_node", &TSVEventStream::lookup_node)
        .def("lookup_type", &TSVEventStream::lookup_type);

    py::class_<NeighborView>(m, "NeighborView")
        .def("__iter__", [](const NeighborView &v) {
            return py::make_iterator(v.begin(), v.end());
        }, py::keep_alive<0, 1>());

    /*
     * TGraph
     */
    py::class_<TGraph>(m, "TGraph")
        .def(py::init<>())

        .def("ingest", &TGraph::ingest)

        .def("num_nodes", &TGraph::num_nodes)
        .def("num_edges", &TGraph::num_edges)

        .def("degree", &TGraph::degree)

        .def("get_neighbor", &TGraph::get_neighbor)
        .def("get_timestamp", &TGraph::get_timestamp)
        .def("get_event_type", &TGraph::get_event_type)

        .def("has_edge", &TGraph::has_edge,
            py::arg("u"),
            py::arg("v"),
            py::arg("start_time"),
            py::arg("end_time"))

        /*
         * neighbors_range overloads
         */
        .def("neighbors_range",
            py::overload_cast<uint32_t>(&TGraph::neighbors_range, py::const_))

        .def("neighbors_range",
            py::overload_cast<uint32_t, uint64_t>(
                &TGraph::neighbors_range, py::const_))

        .def("neighbors_range",
            py::overload_cast<uint32_t, uint64_t, uint64_t>(
                &TGraph::neighbors_range, py::const_))

        /*
         * neighbors overloads
         */
        .def("neighbors",
            py::overload_cast<uint32_t>(&TGraph::neighbors, py::const_))

        .def("neighbors",
            py::overload_cast<uint32_t, uint64_t>(
                &TGraph::neighbors, py::const_))

        .def("neighbors",
            py::overload_cast<uint32_t, uint64_t, uint64_t>(
                &TGraph::neighbors, py::const_))

        .def("neighbors_array",
            [](TGraph &g, uint32_t u) {
                EdgeRange r = g.neighbors_range(u);
                size_t n = r.size();

                auto nbr = py::array_t<uint32_t>(
                    {n},
                    {sizeof(uint32_t)},
                    g.get_neighbor_ptr() + r.start,
                    py::cast(&g)
                );

                auto ts = py::array_t<uint64_t>(
                    {n},
                    {sizeof(uint64_t)},
                    g.get_timestamp_ptr() + r.start,
                    py::cast(&g)
                );

                auto et = py::array_t<uint16_t>(
                    {n},
                    {sizeof(uint16_t)},
                    g.get_event_type_ptr() + r.start,
                    py::cast(&g)
                );

                return py::make_tuple(nbr, ts, et);
            })

        .def("neighbors_array",
            [](TGraph &g, uint32_t u, uint64_t start_time) {
                EdgeRange r = g.neighbors_range(u, start_time);
                size_t n = r.size();

                auto nbr = py::array_t<uint32_t>(
                    {n},
                    {sizeof(uint32_t)},
                    g.get_neighbor_ptr() + r.start,
                    py::cast(&g)
                );

                auto ts = py::array_t<uint64_t>(
                    {n},
                    {sizeof(uint64_t)},
                    g.get_timestamp_ptr() + r.start,
                    py::cast(&g)
                );

                auto et = py::array_t<uint16_t>(
                    {n},
                    {sizeof(uint16_t)},
                    g.get_event_type_ptr() + r.start,
                    py::cast(&g)
                );

                return py::make_tuple(nbr, ts, et);
            })

        .def("neighbors_array",
            [](TGraph &g, uint32_t u, uint64_t start_time, uint64_t end_time) {
                EdgeRange r = g.neighbors_range(u, start_time, end_time);
                size_t n = r.size();

                auto nbr = py::array_t<uint32_t>(
                    {n},
                    {sizeof(uint32_t)},
                    g.get_neighbor_ptr() + r.start,
                    py::cast(&g)
                );

                auto ts = py::array_t<uint64_t>(
                    {n},
                    {sizeof(uint64_t)},
                    g.get_timestamp_ptr() + r.start,
                    py::cast(&g)
                );

                auto et = py::array_t<uint16_t>(
                    {n},
                    {sizeof(uint16_t)},
                    g.get_event_type_ptr() + r.start,
                    py::cast(&g)
                );

                return py::make_tuple(nbr, ts, et);
            });
}