#include <cmath>
#include <numeric>
#include <functional>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <iomanip>

#include "Buffer.h"

namespace minitorch
{
    MiniBuffer::MiniBuffer(const std::vector<float>& data, const std::vector<int>& shape)
        : m_Data(data), m_Shape(shape)
    {
        m_Strides = get_strides_from_shape(shape);
        m_Rank = shape.size();
    }

    MiniBuffer::MiniBuffer(const std::vector<float>& data, const std::vector<int>& shape, const std::vector<int>& strides)
        : m_Data(data), m_Shape(shape), m_Strides(strides)
    {
        m_Rank = shape.size();
    }

    std::vector<int> MiniBuffer::get_strides_from_shape(const std::vector<int>& shape)
    {
        std::vector<int> strides{};
        strides.reserve(shape.size());

        for (int i = 1; i <= shape.size(); i++)
        {
            if (i == shape.size())
            {
                strides.push_back(1);
            }
            else
            {
                strides.push_back(std::accumulate(shape.begin() + i, shape.end(), 1, std::multiplies<int>()));
            }
        }

        return strides;
    }

    MiniBuffer MiniBuffer::arange(int start, int end)
    {
        std::vector<float> data{};
        data.reserve(end-start);

        for (int i = start; i < end; i++)
        {
            data.push_back(float(i));
        }

        return MiniBuffer(data, std::vector<int>{1, end-start});
    }

    MiniBuffer MiniBuffer::fill(const std::vector<int>& shape, float value)
    {
        int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

        std::vector<float> data{};
        data.reserve(size);

        for (int i = 0; i < size; i++)
        {
            data.push_back(value);
        }

        return MiniBuffer(data, shape);
    }

    MiniBuffer MiniBuffer::replace(const MiniBuffer& input, float target, float value)
    {
        const auto& input_data = input.get_data();
        int input_size = static_cast<int>(input_data.size());

        std::vector<float> data{};
        data.reserve(input_size);

        for (int i = 0; i < input_size; i++)
        {
            if (input_data[i] == target)
            {
                data.push_back(value);
            }
            else
            {
                data.push_back(input_data[i]);
            }
        }

        return MiniBuffer(data, input.get_shape(), input.get_strides());
    }

    MiniBuffer MiniBuffer::full_like(const MiniBuffer& input, float value)
    {
        return MiniBuffer::fill(input.get_shape(), value);
    }

    MiniBuffer MiniBuffer::masked_fill(const MiniBuffer& input, const std::vector<bool> mask, float value)
    {
        const auto& input_data = input.get_data();
        int input_size = static_cast<int>(input_data.size());

        std::vector<float> data{};
        data.reserve(input_size);

        for (int i = 0; i < input_size; i++)
        {
            if (mask[i])
            {
                data.push_back(value);
            }
            else
            {
                data.push_back(input_data[i]);
            }
        }

        return MiniBuffer(data, input.get_shape(), input.get_strides());
    }

    MiniBuffer MiniBuffer::tril2(int diagonal) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;
        int input_size = static_cast<int>(input_data.size());

        std::vector<float> data{};
        data.reserve(input_size);

        int current_pos = 0;

        for (int i = 0, tril_cursor = diagonal + 1; i < input_shape[0]; i++, tril_cursor++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                if (j < tril_cursor)
                {
                    data.push_back(input_data[current_pos]);
                }
                else
                {
                    data.push_back(0.0);
                }
                
                current_pos++;
            }
        }

        return MiniBuffer(data, input_shape, input_strides);
    }

    MiniBuffer MiniBuffer::tril3(int diagonal) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;
        int input_size = static_cast<int>(input_data.size());

        std::vector<float> data{};
        data.reserve(input_size);

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0, tril_cursor = diagonal + 1; j < input_shape[1]; j++, tril_cursor++)
            {
                for (int k = 0; k < input_shape[2]; k++)
                {
                    if (k < tril_cursor)
                    {
                        data.push_back(input_data[current_pos]);
                    }
                    else
                    {
                        data.push_back(0.0);
                    }

                    current_pos++;
                }
            }
        }

        return MiniBuffer(data, input_shape, input_strides);
    }
    
    MiniBuffer MiniBuffer::tril4(int diagonal) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;
        int input_size = static_cast<int>(input_data.size());

        std::vector<float> data{};
        data.reserve(input_size);

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                for (int k = 0, tril_cursor = diagonal + 1; k < input_shape[2]; k++, tril_cursor++)
                {
                    for (int l = 0; l < input_shape[3]; l++)
                    {
                        if (l < tril_cursor)
                        {
                            data.push_back(input_data[current_pos]);
                        }
                        else
                        {
                            data.push_back(0.0);
                        }

                        current_pos++;
                    }
                }
            }
        }

        return MiniBuffer(data, input_shape, input_strides);
    }

    MiniBuffer MiniBuffer::tril(const MiniBuffer& input, int diagonal)
    {
        size_t rank = input.m_Rank;

        switch (rank)
        {
            case 2:
            {
                return input.tril2(diagonal);
            }
            break;
            case 3:
            {
                return input.tril3(diagonal);
            }
            break;
            case 4:
            {
                return input.tril4(diagonal);
            }
            break;
            default:
                assert(false && "Cannot tril MiniBuffer, invalid rank provided.");
                // Only for compiler warnings, will not reach
                return input;
                break;
        }
    }

    MiniBuffer MiniBuffer::operator-() const
    {
        std::vector<float> result;
        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(-this->m_Data[i]);
        }

        return MiniBuffer(result, this->m_Shape, this->m_Strides);
    } 


    MiniBuffer MiniBuffer::log() const
    {
        std::vector<float> result;
        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(std::log(this->m_Data[i]));
        }

        return MiniBuffer(result, this->m_Shape, this->m_Strides);
    }

    MiniBuffer MiniBuffer::log2() const
    {
        std::vector<float> result;
        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(std::log2(this->m_Data[i]));
        }

        return MiniBuffer(result, this->m_Shape, this->m_Strides);
    }

    MiniBuffer MiniBuffer::operator+(const MiniBuffer& other) const 
    {
        std::vector<float> result;

        if (this->m_Shape != other.m_Shape)
        {
            assert(false && "Cannot perform addition, shapes do not match.");
        }

        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] + other.m_Data[i]);
        }

        return MiniBuffer(result, this->m_Shape, this->m_Strides);
    }

    MiniBuffer MiniBuffer::operator-(const MiniBuffer& other) const 
    {
        std::vector<float> result;

        if (this->m_Shape != other.m_Shape)
        {
            assert(false && "Cannot perform addition, shapes do not match.");
        }

        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] - other.m_Data[i]);
        }

        return MiniBuffer(result, this->m_Shape, this->m_Strides);
    }

    MiniBuffer MiniBuffer::operator*(const MiniBuffer& other) const 
    {
        std::vector<float> result;

        if (this->m_Shape != other.m_Shape)
        {
            assert(false && "Cannot perform addition, shapes do not match.");
        }

        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] * other.m_Data[i]);
        }

        return MiniBuffer(result, this->m_Shape, this->m_Strides);
    }

    MiniBuffer MiniBuffer::operator/(const MiniBuffer& other) const 
    {
        std::vector<float> result;

        if (this->m_Shape != other.m_Shape)
        {
            assert(false && "Cannot perform addition, shapes do not match.");
        }

        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] / other.m_Data[i]);
        }

        return MiniBuffer(result, this->m_Shape, this->m_Strides);
    }

    MiniBuffer MiniBuffer::pow(const MiniBuffer& other) const
    {
        std::vector<float> result{};
        result.reserve(this->m_Data.size());
    
        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(std::pow(this->m_Data[i], other.m_Data[i]));
        }

        return MiniBuffer(result, this->m_Shape, this->m_Strides);
    }

    MiniBuffer MiniBuffer::max(const MiniBuffer& other) const
    {
        std::vector<float> result{};
        result.reserve(this->m_Data.size());
    
        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(std::max(this->m_Data[i], other.m_Data[i]));
        }

        return MiniBuffer(result, this->m_Shape, this->m_Strides);
    }

    bool MiniBuffer::operator==(const MiniBuffer& other) const
    {
        for (int i = 0; i < this->m_Data.size(); i++)
        {
            if (this->m_Data[i] != other.m_Data[i])
            {
                return false;
            }
        }

        return true;
    }

    std::vector<bool> MiniBuffer::operator==(float other) const
    {
        std::vector<bool> result{};
        result.reserve(this->m_Data.size());
    
        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] == other);
        }

        return result;
    }

    std::vector<bool> MiniBuffer::operator<(float other) const
    {
        std::vector<bool> result{};
        result.reserve(this->m_Data.size());
    
        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] < other);
        }

        return result;
    }

    std::vector<bool> MiniBuffer::operator>(float other) const
    {
        std::vector<bool> result{};
        result.reserve(this->m_Data.size());
    
        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] > other);
        }

        return result;
    }
    
    MiniBuffer MiniBuffer::sum2() const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        std::vector<float> data{};
        data.reserve(std::accumulate(input_shape.begin(), input_shape.end() - 1, 1, std::multiplies<int>()));

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            float row_sum = 0.0;

            for (int j = 0; j < input_shape[1]; j++)
            {
                row_sum += input_data[current_pos];
                current_pos++;
            }

            data.push_back(row_sum);
        }

        std::vector<int> output_shape(input_shape.begin(), input_shape.end() - 1);
        output_shape.push_back(1);

        return MiniBuffer(data, output_shape);
    }

    MiniBuffer MiniBuffer::sum3() const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        std::vector<float> data{};
        data.reserve(std::accumulate(input_shape.begin(), input_shape.end() - 1, 1, std::multiplies<int>()));

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                float row_sum = 0.0;

                for (int k = 0; k < input_shape[2]; k++)
                {
                    row_sum += input_data[current_pos];
                    current_pos++;
                }

                data.push_back(row_sum);
            }
        }

        std::vector<int> output_shape(input_shape.begin(), input_shape.end() - 1);
        output_shape.push_back(1);

        return MiniBuffer(data, output_shape);
    }

    MiniBuffer MiniBuffer::sum4() const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        std::vector<float> data{};
        data.reserve(std::accumulate(input_shape.begin(), input_shape.end() - 1, 1, std::multiplies<int>()));

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                for (int k = 0; k < input_shape[2]; k++)
                {
                    float row_sum = 0.0;

                    for (int l = 0; l < input_shape[3]; l++)
                    {
                        row_sum += input_data[current_pos];
                        current_pos++;
                    }

                    data.push_back(row_sum);
                }
            }
        }

        std::vector<int> output_shape(input_shape.begin(), input_shape.end() - 1);
        output_shape.push_back(1);

        return MiniBuffer(data, output_shape);
    }

    MiniBuffer MiniBuffer::sum()
    {
        const auto& input_data = this->m_Data;
        std::vector<int> new_shape(this->m_Rank, 1);

        float sum = std::accumulate(input_data.begin(), input_data.end(), 0.0F);

        return MiniBuffer(std::vector<float>{sum}, new_shape);
    }

    MiniBuffer MiniBuffer::sum(int axis)
    {
        size_t rank = this->m_Rank;

        if (rank == 1)
        {
            // No need to sum a scalar
            return *this;
        }

        MiniBuffer x = this->swap_nth_axis_with_last(axis);

        switch (rank)
        {
            case 2:
            {
                x = x.sum2();
            }
            break;
            case 3:
            {
                x = x.sum3();
            }
            break;
            case 4:
            {
                x = x.sum4();
            }
            break;
            default:
                assert(false && "Cannot sum MiniBuffer, invalid rank provided.");
                break;
        }
        
        return x.swap_nth_axis_with_last(axis);
    }

    MiniBuffer MiniBuffer::reshape(const std::vector<int> new_shape) const
    {
        return MiniBuffer(this->m_Data, new_shape);
    }

    MiniBuffer MiniBuffer::flatten() const
    {
        const auto& data = this->m_Data;
        int element_cnt = static_cast<int>(data.size());
        std::vector<int> flat_shape{1, element_cnt};

        return MiniBuffer(data, flat_shape); 
    } 

    MiniBuffer MiniBuffer::permute(const std::vector<int> order) const
    {
        size_t rank = this->m_Rank;
        std::vector<int> new_shape{};
        std::vector<int> new_strides{};
        new_shape.reserve(rank);
        new_strides.reserve(rank);

        for (int i = 0; i < rank; i++)
        {
            new_shape.push_back(this->m_Shape[order[i]]);
            new_strides.push_back(this->m_Strides[order[i]]);
        }

        return MiniBuffer(this->m_Data, new_shape, new_strides).contiguous();
    }
    
    MiniBuffer MiniBuffer::expand(int axis, int expanded_size) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;
        int input_size = static_cast<int>(input_data.size());

        int new_size = expanded_size * input_size;
        std::vector<float> data{};
        data.reserve(new_size);

        for (int i = 0; i < expanded_size; i++)
        {
            data.insert(data.end(), input_data.begin(), input_data.end());
        }

        //? NOTE: Mirko, 24. 12. 2023 
        // Since we're just multiplying the data array by
        // expanded_size, somehow we need to preserve the
        // meaning of the original shape and strides. Other-
        // wise, expanding a 1x3 and a 3x1 tensor would res-
        // ult in the same output, which is wrong.
        // The correct strides for the output are same as
        // the input strides, with the stride at position
        // pos=expansion_axis being the product of all dims
        // of the original shape except the one we're expan-
        // ding along. This makes sense because we expand by
        // simply duplicating the original data expanded_size
        // times.

        std::vector<int> new_shape = input_shape;
        std::vector<int> new_strides = input_strides;
        // Temporarily set to 1 in order to calculate
        // the new stride along axis
        new_shape[axis] = 1;
        new_strides[axis] = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
        new_shape[axis] = expanded_size;

        MiniBuffer result = MiniBuffer(data, new_shape, new_strides);
        return result.contiguous();
    }

    MiniBuffer MiniBuffer::pad1(const std::tuple<int, int> pad_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_pad_before = std::get<0>(pad_sizes);
        int n_pad_after  = std::get<1>(pad_sizes);

        std::vector<float> data{};
        int new_row_size = n_pad_before + input_shape.back() + n_pad_after;
        size_t new_size = std::accumulate(input_shape.begin(), input_shape.end() - 1, new_row_size, std::multiplies<int>()); 
        data.reserve(new_size);

        int current_pos = 0;

        std::vector<float> new_row{};
        new_row.reserve(new_row_size);
        
        for (int pad = 0; pad < n_pad_before; pad++)
        {
            new_row.push_back(0.0);
        }

        for (int j = 0; j < input_shape[1]; j++)
        {
            new_row.push_back(input_data[current_pos]);
            current_pos++;
        }

        for (int pad = 0; pad < n_pad_after; pad++)
        {
            new_row.push_back(0.0);
        }

        data.insert(data.end(), new_row.begin(), new_row.end());

        std::vector<int> output_shape(input_shape.begin(), input_shape.end() - 1);
        output_shape.push_back(new_row_size);

        return MiniBuffer(data, output_shape);
    }

    MiniBuffer MiniBuffer::pad2(const std::tuple<int, int> pad_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_pad_before = std::get<0>(pad_sizes);
        int n_pad_after  = std::get<1>(pad_sizes);

        std::vector<float> data{};
        int new_row_size = n_pad_before + input_shape.back() + n_pad_after;
        size_t new_size = std::accumulate(input_shape.begin(), input_shape.end() - 1, new_row_size, std::multiplies<int>()); 
        data.reserve(new_size);

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            std::vector<float> new_row{};
            new_row.reserve(new_row_size);
            
            for (int pad = 0; pad < n_pad_before; pad++)
            {
                new_row.push_back(0.0);
            }

            for (int j = 0; j < input_shape[1]; j++)
            {
                new_row.push_back(input_data[current_pos]);
                current_pos++;
            }

            for (int pad = 0; pad < n_pad_after; pad++)
            {
                new_row.push_back(0.0);
            }

            data.insert(data.end(), new_row.begin(), new_row.end());
        }

        std::vector<int> output_shape(input_shape.begin(), input_shape.end() - 1);
        output_shape.push_back(new_row_size);

        return MiniBuffer(data, output_shape);
    }

    MiniBuffer MiniBuffer::pad3(const std::tuple<int, int> pad_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_pad_before = std::get<0>(pad_sizes);
        int n_pad_after  = std::get<1>(pad_sizes);

        std::vector<float> data{};
        int new_row_size = n_pad_before + input_shape.back() + n_pad_after;
        size_t new_size = std::accumulate(input_shape.begin(), input_shape.end() - 1, new_row_size, std::multiplies<int>()); 
        data.reserve(new_size);

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                std::vector<float> new_row{};
                new_row.reserve(new_row_size);

                for (int pad = 0; pad < n_pad_before; pad++)
                {
                    new_row.push_back(0.0);
                }

                for (int k = 0; k < input_shape[2]; k++)
                {
                    new_row.push_back(input_data[current_pos]);
                    current_pos++;
                }

                for (int pad = 0; pad < n_pad_after; pad++)
                {
                    new_row.push_back(0.0);
                }

                data.insert(data.end(), new_row.begin(), new_row.end());
            }
        }

        std::vector<int> output_shape(input_shape.begin(), input_shape.end() - 1);
        output_shape.push_back(new_row_size);

        return MiniBuffer(data, output_shape);
    }

    MiniBuffer MiniBuffer::pad4(const std::tuple<int, int> pad_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_pad_before = std::get<0>(pad_sizes);
        int n_pad_after  = std::get<1>(pad_sizes);

        std::vector<float> data{};
        int new_row_size = n_pad_before + input_shape.back() + n_pad_after;
        size_t new_size = std::accumulate(input_shape.begin(), input_shape.end() - 1, new_row_size, std::multiplies<int>()); 
        data.reserve(new_size);

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                for (int k = 0; k < input_shape[2]; k++)
                {
                    std::vector<float> new_row{};
                    new_row.reserve(new_row_size);

                    for (int pad = 0; pad < n_pad_before; pad++)
                    {
                        new_row.push_back(0.0);
                    }

                    for (int l = 0; l < input_shape[3]; l++)
                    {
                        new_row.push_back(input_data[current_pos]);
                        current_pos++;
                    }

                    for (int pad = 0; pad < n_pad_after; pad++)
                    {
                        new_row.push_back(0.0);
                    }

                    data.insert(data.end(), new_row.begin(), new_row.end());
                }
            }
        }

        std::vector<int> output_shape(input_shape.begin(), input_shape.end() - 1);
        output_shape.push_back(new_row_size);

        return MiniBuffer(data, output_shape);
    }

    MiniBuffer MiniBuffer::pad(int axis, const std::tuple<int, int> pad_sizes)
    {
        size_t rank = this->m_Rank;

        MiniBuffer x = this->swap_nth_axis_with_last(axis);

        switch (rank)
        {
            case 1:
            {
                x = x.pad1(pad_sizes);
            }
            break;
            case 2:
            {
                x = x.pad2(pad_sizes);
            }
            break;
            case 3:
            {
                x = x.pad3(pad_sizes);
            }
            break;
            case 4:
            {
                x = x.pad4(pad_sizes);
            }
            break;
            default:
                assert(false && "Cannot pad MiniBuffer, invalid rank provided.");
                // Only for compiler warnings, will not reach
                return *this;
                break;
        }
    
        return x.swap_nth_axis_with_last(axis);
    }

    MiniBuffer MiniBuffer::shrink2(const std::tuple<int, int> shrink_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_shrink_before = std::get<0>(shrink_sizes);
        int n_shrink_after  = std::get<1>(shrink_sizes);

        std::vector<float> data{};
        int new_row_size = input_shape.back() - n_shrink_before - n_shrink_after;
        size_t new_size = std::accumulate(input_shape.begin(), input_shape.end() - 1, new_row_size, std::multiplies<int>()); 
        data.reserve(new_size);

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            std::vector<float> new_row{};
            new_row.reserve(new_row_size);

            for (int j = 0; j < input_shape[1]; j++)
            {
                if (j >= n_shrink_before && j < input_shape[1] - n_shrink_after)
                {
                    new_row.push_back(input_data[current_pos]);
                }
                
                current_pos++;
            }
            
            data.insert(data.end(), new_row.begin(), new_row.end());
        }

        std::vector<int> output_shape(input_shape.begin(), input_shape.end() - 1);
        output_shape.push_back(new_row_size);

        return MiniBuffer(data, output_shape);
    }
    
    MiniBuffer MiniBuffer::shrink3(const std::tuple<int, int> shrink_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_shrink_before = std::get<0>(shrink_sizes);
        int n_shrink_after  = std::get<1>(shrink_sizes);

        std::vector<float> data{};
        int new_row_size = input_shape.back() - n_shrink_before - n_shrink_after;
        size_t new_size = std::accumulate(input_shape.begin(), input_shape.end() - 1, new_row_size, std::multiplies<int>()); 
        data.reserve(new_size);

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                std::vector<float> new_row{};
                new_row.reserve(new_row_size);

                for (int k = 0; k < input_shape[2]; k++)
                {
                    if (k >= n_shrink_before && k < input_shape[2] - n_shrink_after)
                    {
                        new_row.push_back(input_data[current_pos]);
                    }
                    
                    current_pos++;
                }

                data.insert(data.end(), new_row.begin(), new_row.end());
            }
        }

        std::vector<int> output_shape(input_shape.begin(), input_shape.end() - 1);
        output_shape.push_back(new_row_size);

        return MiniBuffer(data, output_shape);
    }
    
    MiniBuffer MiniBuffer::shrink4(const std::tuple<int, int> shrink_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_shrink_before = std::get<0>(shrink_sizes);
        int n_shrink_after  = std::get<1>(shrink_sizes);

        std::vector<float> data{};
        int new_row_size = input_shape.back() - n_shrink_before - n_shrink_after;
        size_t new_size = std::accumulate(input_shape.begin(), input_shape.end() - 1, new_row_size, std::multiplies<int>()); 
        data.reserve(new_size);

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                for (int k = 0; k < input_shape[2]; k++)
                {
                    std::vector<float> new_row{};
                    new_row.reserve(new_row_size);

                    for (int l = 0; l < input_shape[3]; l++)
                    {
                        if (l >= n_shrink_before && l < input_shape[3] - n_shrink_after)
                        {
                            new_row.push_back(input_data[current_pos]);
                        }

                        current_pos++;
                    }

                    data.insert(data.end(), new_row.begin(), new_row.end());
                }
            }
        }

        std::vector<int> output_shape(input_shape.begin(), input_shape.end() - 1);
        output_shape.push_back(new_row_size);

        return MiniBuffer(data, output_shape);
    }

    MiniBuffer MiniBuffer::shrink(int axis, const std::tuple<int, int> shrink_sizes)
    {
        size_t rank = this->m_Rank;
        MiniBuffer x = this->swap_nth_axis_with_last(axis);

        switch (rank)
        {
            case 2:
            {
                x = x.shrink2(shrink_sizes);
            }
            break;
            case 3:
            {
                x = x.shrink3(shrink_sizes);
            }
            break;
            case 4:
            {
                x = x.shrink4(shrink_sizes);
            }
            break;
            default:
                assert(false && "Cannot shrink MiniBuffer, invalid rank provided.");
                // Only for compiler warnings, will not reach
                return *this;
                break;
        }

        return x.swap_nth_axis_with_last(axis);
    }

    MiniBuffer MiniBuffer::contiguous2() const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;

        std::vector<float> data{};
        data.reserve(std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>()));

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                int current_pos = i * input_strides[0] + j * input_strides[1]; 
                data.push_back(input_data[current_pos]);
            }
        }

        return MiniBuffer(data, this->m_Shape);
    }

    MiniBuffer MiniBuffer::contiguous3() const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;

        std::vector<float> data{};
        data.reserve(std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>()));

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                for (int k = 0; k < input_shape[2]; k++)
                {
                    int current_pos = i * input_strides[0] + j * input_strides[1] + k * input_strides[2]; 
                    data.push_back(input_data[current_pos]);
                }
            }
        }

        return MiniBuffer(data, this->m_Shape);
    }

    MiniBuffer MiniBuffer::contiguous4() const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;

        std::vector<float> data{};
        data.reserve(std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>()));

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                for (int k = 0; k < input_shape[2]; k++)
                {
                    for (int l = 0; l < input_shape[3]; l++)
                    {
                        int current_pos = i * input_strides[0] + j * input_strides[1] + 
                                            k * input_strides[2] + l * input_strides[3]; 
                        data.push_back(input_data[current_pos]);
                    }
                }
            }
        }

        return MiniBuffer(data, this->m_Shape);
    }

    MiniBuffer MiniBuffer::contiguous() const
    {
        size_t rank = this->m_Rank;

        switch (rank)
        {
            case 1:
            {
                // Scalar is already contiguous
                return *this;
            }
            case 2:
            {
                return this->contiguous2();
            }
            break;
            case 3:
            {
                return this->contiguous3();
            }
            break;
            case 4:
            {
                return this->contiguous4();
            }
            break;
            default:
                assert(false && "Cannot create a contiguous MiniBuffer, invalid rank provided.");
                // Only for compiler warnings, will not reach
                return *this;
                break;
        }
    }

    bool MiniBuffer::is_scalar() const
    {
        return this->m_Data.size() == 1;
    }

    bool MiniBuffer::is_square() const
    {
        int dim_cnt = static_cast<int>(this->m_Shape.size());
        return this->m_Shape[dim_cnt - 1] == this->m_Shape[dim_cnt - 2];
    }

    int MiniBuffer::len() const
    {
        return static_cast<int>(this->m_Data.size());
    }
    
    MiniBuffer MiniBuffer::swap_nth_axis_with_last(int n)
    {
        size_t rank = this->m_Rank;
        
        std::vector<int> order{};
        order.reserve(rank);

        for (int i = 0; i < rank; i++)
        {
            order.push_back(i);
        }

        int tmp = order[n];
        order[n] = order.back();
        order[rank - 1] = tmp;

        return this->permute(order);
    }

    std::string MiniBuffer::to_string() const
    {
        size_t rank = this->m_Rank;
        std::stringstream repr;
        repr << std::setprecision(4) << std::fixed;
        repr <<  "[";

        switch (rank)
        {
            case 1:
            {
                repr << this->to_string1();
            }
            break;
            case 2:
            {
                repr << this->to_string2();
            }
            break;
            case 3:
            {
                repr << this->to_string3();
            }
            break;
            case 4:
            {
                repr << this->to_string4();
            }
            break;
            default:
                assert(false && "Cannot convert MiniBuffer to string, invalid rank provided.");
                break;
        }

        repr << "]";
        return repr.str();
    }

    std::string MiniBuffer::to_string1() const
    {
        std::stringstream repr;
        repr << std::setprecision(4) << std::fixed;
        const auto& data = this->m_Data;
        const auto& shape = this->m_Shape;

        int current_pos = 0;
        
        for (int i = 0; i < shape[0]; i++)
        {
            if (i == (shape[0] - 1))
            {
                repr << data[current_pos];
            }
            else
            {
                repr << data[current_pos] << ", ";
            }
            current_pos++;
        }

        return repr.str();
    }

    std::string MiniBuffer::to_string2() const
    {
        std::stringstream repr;
        repr << std::setprecision(4) << std::fixed;
        const auto& data = this->m_Data;
        const auto& shape = this->m_Shape;

        int current_pos = 0;

        for (int i = 0; i < shape[0]; i++)
        {
            if (i == 0)
            {
                repr << "[";
            }
            else
            {
                repr << "          [";
            }

            for (int j = 0; j < shape[1]; j++)
            {
                if (j == (shape[1] - 1))
                {
                    repr << data[current_pos];
                }
                else
                {
                    repr << data[current_pos] << ", ";
                }

                current_pos++;
            }

            if (i == (shape[0] - 1))
            {
                repr << "]";
            }
            else
            {
                repr << "],\n";
            }
        }

        return repr.str();
    }

    std::string MiniBuffer::to_string3() const
    {
        std::stringstream repr;
        repr << std::setprecision(4) << std::fixed;
        const auto& data = this->m_Data;
        const auto& shape = this->m_Shape;

        int current_pos = 0;

        for (int i = 0; i < shape[0]; i++)
        {
            if (i == 0)
            {
                repr << "[";
            }
            else
            {
                repr << "         [";
            }

            for (int j = 0; j < shape[1]; j++)
            {
                if (j == 0)
                {
                    repr << "[";
                }
                else
                {
                    repr << "           [";
                }

                for (int k = 0; k < shape[2]; k++)
                {
                    if (k == (shape[2] - 1))
                    {
                        repr << data[current_pos];
                    }
                    else
                    {
                        repr << data[current_pos] << ", ";
                    }

                    current_pos++;
                }
                
                if (j == (shape[1] - 1))
                {
                    repr << "]";
                }
                else
                {
                    repr << "],\n";
                }
            }

            if (i == (shape[0] - 1))
            {
                repr << "]";
            }
            else
            {
                repr << "],\n";
            }
        }

        return repr.str();
    }

    std::string MiniBuffer::to_string4() const
    {
        std::stringstream repr;
        repr << std::setprecision(4) << std::fixed;
        const auto& data = this->m_Data;
        const auto& shape = this->m_Shape;

        int current_pos = 0;

        for (int i = 0; i < shape[0]; i++)
        {
            if (i == 0)
            {
                repr << "[";
            }
            else
            {
                repr << "         [";
            }

            for (int j = 0; j < shape[1]; j++)
            {
                if (j == 0)
                {
                    repr << "[";
                }
                else
                {
                    repr << "          [";
                }

                for (int k = 0; k < shape[2]; k++)
                {
                    if (k == 0)
                    {
                        repr << "[";
                    }
                    else
                    {
                        repr << "            [";
                    }

                    for (int l = 0; l < shape[3]; l++)
                    {
                        if (l == (shape[3] - 1))
                        {
                            repr << data[current_pos];
                        }
                        else
                        {
                            repr << data[current_pos] << ", ";
                        }

                        current_pos++;
                    }
                
                    if (k == (shape[2] - 1))
                    {
                        repr << "]";
                    }
                    else
                    {
                        repr << "],\n";
                    }
                }
                
                if (j == (shape[1] - 1))
                {
                    repr << "]";
                }
                else
                {
                    repr << "],\n";
                }
            }

            if (i == (shape[0] - 1))
            {
                repr << "]";
            }
            else
            {
                repr << "],\n\n";
            }
        }

        return repr.str();
    }
}
