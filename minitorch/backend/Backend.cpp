#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "Buffer.h"

namespace minitorch
{
    PYBIND11_MODULE(Backend, m)
    {
        m.doc() = "C++ backend for the minitorch library.";
    
        pybind11::class_<MiniBuffer>(m, "MiniBuffer")
                // Ctor
                .def(pybind11::init<const std::vector<float>&, const std::vector<int>&>())
                // Getters
                .def("get_data", &MiniBuffer::get_data)
                .def("get_shape", &MiniBuffer::get_shape)
                .def("get_strides", &MiniBuffer::get_strides)
                .def("get_rank", &MiniBuffer::get_rank)
                // Generation methods
                .def("arange", &MiniBuffer::arange)
                .def("fill", &MiniBuffer::fill)
                .def("replace", &MiniBuffer::replace)
                .def("full_like", &MiniBuffer::full_like)
                .def("masked_fill", &MiniBuffer::masked_fill)
                .def("tril", &MiniBuffer::tril)
                // Unary
                .def(-pybind11::self)
                .def("log", &MiniBuffer::log)
                .def("log2", &MiniBuffer::log2)
                // Binary
                .def(pybind11::self + pybind11::self)
                .def(pybind11::self - pybind11::self)
                .def(pybind11::self * pybind11::self)
                .def(pybind11::self / pybind11::self)
                .def("pow", &MiniBuffer::pow)
                .def("max", &MiniBuffer::max)
                .def(pybind11::self == pybind11::self)
                .def(pybind11::self == float())
                .def(pybind11::self <  float())
                .def(pybind11::self >  float())
                // Reduction
                .def("sum", static_cast<MiniBuffer (MiniBuffer::*)()>(&MiniBuffer::sum))
                .def("sum", static_cast<MiniBuffer (MiniBuffer::*)(int)>(&MiniBuffer::sum))
                // Mutation
                .def("reshape", &MiniBuffer::reshape)
                .def("flatten", &MiniBuffer::flatten)
                .def("permute", &MiniBuffer::permute)
                .def("expand", &MiniBuffer::expand)
                .def("pad", &MiniBuffer::pad)
                .def("shrink", &MiniBuffer::shrink)
                // Utility
                .def("__len__", &MiniBuffer::len)
                .def("is_scalar", &MiniBuffer::is_scalar)
                .def("is_square", &MiniBuffer::is_square)
                .def("__repr__", &MiniBuffer::to_string);
    }
}
