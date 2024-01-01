#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <stack>

#include "Buffer.h"

namespace minitorch
{
    std::vector<float> row_sum(const std::vector<float>& data,
                               const std::vector<int>& shape,
                               const std::vector<int>& strides)
    {
        int depth_idx = 0;
        int current_position = 0;
        std::stack<std::pair<int, int>> stack{};
        std::vector<float> result_data{};
        result_data.reserve(data.size() / shape.back());
    
        while (true) 
        {
            if (depth_idx == shape.size() - 1) 
            {
                float sum = 0.0f;

                for (int val_idx = 0; val_idx < shape[depth_idx]; ++val_idx) 
                {
                    int val_pos = current_position + val_idx * strides[depth_idx];
                    sum += data[val_pos];
                }

                result_data.push_back(sum);
            }
            else 
            {
                for (int dim_idx = shape[depth_idx] - 1; dim_idx >= 0; --dim_idx) 
                {
                    int next_pos = current_position + dim_idx * strides[depth_idx];
                    stack.push(std::make_pair(depth_idx + 1, next_pos));
                }
            }
    
            if (stack.empty()) 
            {
                break;
            }
    
            std::pair<int, int> top = stack.top();
            stack.pop();
            depth_idx = top.first;
            current_position = top.second;
        }
    
        return result_data;
    }
    
    std::vector<float> collect_data(const std::vector<float>& data,
                                    const std::vector<int>& shape,
                                    const std::vector<int>& strides) 
    {
        int depth_idx = 0;
        int current_position = 0;
        std::stack<std::pair<int, int>> stack{};
        std::vector<float> result_data{};
        result_data.reserve(data.size());
    
        while (true) 
        {
            if (depth_idx == shape.size() - 1) 
            {
                for (int val_idx = 0; val_idx < shape[depth_idx]; ++val_idx) 
                {
                    int val_pos = current_position + val_idx * strides[depth_idx];
                    result_data.push_back(data[val_pos]);
                }
            }
            else 
            {
                for (int dim_idx = shape[depth_idx] - 1; dim_idx >= 0; --dim_idx) 
                {
                    int next_pos = current_position + dim_idx * strides[depth_idx];
                    stack.push(std::make_pair(depth_idx + 1, next_pos));
                }
            }
    
            if (stack.empty()) 
            {
                break;
            }
    
            std::pair<int, int> top = stack.top();
            stack.pop();
            depth_idx = top.first;
            current_position = top.second;
        }
    
        return result_data;
    }
    
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
    
        m.def("row_sum", &row_sum, "Sum a contiguous array along its rows");
        m.def("collect_data", &collect_data, "Collect provided data into a contiguous array");
    }
}