#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stack>

std::vector<float> row_sum(const std::vector<float>& data,
                           const std::vector<int>& shape,
                           const std::vector<int>& strides)
{
    int depth_idx = 0;
    int current_position = 0;
    std::vector<float> out_data;
    std::stack<std::pair<int, int>> stack;

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
            out_data.push_back(sum);
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

    return out_data;
}

std::vector<float> collect_data(const std::vector<float>& data,
                                const std::vector<int>& shape,
                                const std::vector<int>& strides) 
{
    int depth_idx = 0;
    int current_position = 0;
    std::vector<float> out_data;
    std::stack<std::pair<int, int>> stack;

    while (true) 
    {
        if (depth_idx == shape.size() - 1) 
        {
            for (int val_idx = 0; val_idx < shape[depth_idx]; ++val_idx) 
            {
                int val_pos = current_position + val_idx * strides[depth_idx];
                out_data.push_back(data[val_pos]);
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

    return out_data;
}

PYBIND11_MODULE(cpp_backend, m)
{
    m.doc() = "C++ backend for the minitorch library.";

    m.def("row_sum", &row_sum, "Sum a contiguous array along its rows");
    m.def("collect_data", &collect_data, "Collect provided data into a contiguous array");
}