#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "Buffer.h"

namespace minitorch
{
    PYBIND11_MODULE(Backend, m)
    {
        m.doc() = "C++ backend for the minitorch library.";

        // Float MiniBuffer
        pybind11::class_<MiniBuffer<float>>(m, "MiniBufferF32")
                // Ctor
                .def(pybind11::init<const std::vector<float>&, const std::vector<int>&>())
                .def(pybind11::init<const std::vector<float>&, const std::vector<int>&, const std::vector<int>&>())
                // Getters
                .def("get_data", &MiniBuffer<float>::get_data)
                .def("get_shape", &MiniBuffer<float>::get_shape)
                .def("get_strides", &MiniBuffer<float>::get_strides)
                .def("get_rank", &MiniBuffer<float>::get_rank)
                // Generation methods
                .def("arange", &MiniBuffer<float>::arange)
                .def("fill", &MiniBuffer<float>::fill)
                .def("replace", &MiniBuffer<float>::replace)
                .def("full_like", &MiniBuffer<float>::full_like)
                .def("masked_fill", &MiniBuffer<float>::masked_fill)
                .def("tril", &MiniBuffer<float>::tril)
                // Unary
                .def(-pybind11::self)
                .def("log", &MiniBuffer<float>::log)
                .def("log2", &MiniBuffer<float>::log2)
                // Binary
                .def(pybind11::self + pybind11::self)
                .def(pybind11::self - pybind11::self)
                .def(pybind11::self * pybind11::self)
                .def(pybind11::self / pybind11::self)
                .def("pow", &MiniBuffer<float>::pow)
                .def("max", &MiniBuffer<float>::max)
                .def(pybind11::self == pybind11::self)
                .def(pybind11::self == float())
                .def(pybind11::self <  float())
                .def(pybind11::self >  float())
                // Reduction
                .def("sum", static_cast<MiniBuffer<float> (MiniBuffer<float>::*)()>(&MiniBuffer<float>::sum))
                .def("sum", static_cast<MiniBuffer<float> (MiniBuffer<float>::*)(int)>(&MiniBuffer<float>::sum))
                // Mutation
                .def("reshape", &MiniBuffer<float>::reshape)
                .def("flatten", &MiniBuffer<float>::flatten)
                .def("permute", &MiniBuffer<float>::permute)
                .def("expand", &MiniBuffer<float>::expand)
                .def("pad", &MiniBuffer<float>::pad)
                .def("shrink", &MiniBuffer<float>::shrink)
                // Utility
                .def("__len__", &MiniBuffer<float>::len)
                .def("is_scalar", &MiniBuffer<float>::is_scalar)
                .def("is_square", &MiniBuffer<float>::is_square)
                .def("__repr__", &MiniBuffer<float>::to_string);
                
        // Int MiniBuffer
        pybind11::class_<MiniBuffer<int>>(m, "MiniBufferI32")
                // Ctor
                .def(pybind11::init<const std::vector<int>&, const std::vector<int>&>())
                .def(pybind11::init<const std::vector<int>&, const std::vector<int>&, const std::vector<int>&>())
                // Getters
                .def("get_data", &MiniBuffer<int>::get_data)
                .def("get_shape", &MiniBuffer<int>::get_shape)
                .def("get_strides", &MiniBuffer<int>::get_strides)
                .def("get_rank", &MiniBuffer<int>::get_rank)
                // Generation methods
                .def("arange", &MiniBuffer<int>::arange)
                .def("fill", &MiniBuffer<int>::fill)
                .def("replace", &MiniBuffer<int>::replace)
                .def("full_like", &MiniBuffer<int>::full_like)
                .def("masked_fill", &MiniBuffer<int>::masked_fill)
                .def("tril", &MiniBuffer<int>::tril)
                // Unary
                .def(-pybind11::self)
                .def("log", &MiniBuffer<int>::log)
                .def("log2", &MiniBuffer<int>::log2)
                // Binary
                .def(pybind11::self + pybind11::self)
                .def(pybind11::self - pybind11::self)
                .def(pybind11::self * pybind11::self)
                .def(pybind11::self / pybind11::self)
                .def("pow", &MiniBuffer<int>::pow)
                .def("max", &MiniBuffer<int>::max)
                .def(pybind11::self == pybind11::self)
                .def(pybind11::self == int())
                .def(pybind11::self <  int())
                .def(pybind11::self >  int())
                // Reduction
                .def("sum", static_cast<MiniBuffer<int> (MiniBuffer<int>::*)()>(&MiniBuffer<int>::sum))
                .def("sum", static_cast<MiniBuffer<int> (MiniBuffer<int>::*)(int)>(&MiniBuffer<int>::sum))
                // Mutation
                .def("reshape", &MiniBuffer<int>::reshape)
                .def("flatten", &MiniBuffer<int>::flatten)
                .def("permute", &MiniBuffer<int>::permute)
                .def("expand", &MiniBuffer<int>::expand)
                .def("pad", &MiniBuffer<int>::pad)
                .def("shrink", &MiniBuffer<int>::shrink)
                // Utility
                .def("__len__", &MiniBuffer<int>::len)
                .def("is_scalar", &MiniBuffer<int>::is_scalar)
                .def("is_square", &MiniBuffer<int>::is_square)
                .def("__repr__", &MiniBuffer<int>::to_string);
    }
}
