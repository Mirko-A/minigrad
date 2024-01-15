#pragma once

#include <cmath>
#include <numeric>
#include <functional>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace minitorch
{
    template<class T>
    class MiniBuffer
    {
    public:
        MiniBuffer(const std::vector<T>& data, const std::vector<int>& shape);
        MiniBuffer(const std::vector<T>& data, const std::vector<int>& shape, const std::vector<int>& strides);

        inline const std::vector<T>& get_data()  const { return m_Data;    }
        inline const std::vector<int>& get_shape()   const { return m_Shape;   }
        inline const std::vector<int>& get_strides() const { return m_Strides; }
        inline const size_t get_rank() const { return m_Rank; }

        static MiniBuffer<T> arange(int start, int end);
        static MiniBuffer<T> fill(const std::vector<int>& shape, T value);
        static MiniBuffer<T> replace(const MiniBuffer<T>& input, T target, T value);
        static MiniBuffer<T> full_like(const MiniBuffer<T>& input, T value);
        static MiniBuffer<T> masked_fill(const MiniBuffer<T>& input, const std::vector<bool> mask, T value);
        static MiniBuffer<T> tril(const MiniBuffer<T>& input, int diagonal = 0);

        // Unary operations
        MiniBuffer<T> operator-() const;
        MiniBuffer<T> log() const;
        MiniBuffer<T> log2() const;

        // Binary operations
        MiniBuffer<T> operator+(const MiniBuffer<T>& other) const;
        MiniBuffer<T> operator-(const MiniBuffer<T>& other) const;
        MiniBuffer<T> operator*(const MiniBuffer<T>& other) const;
        MiniBuffer<T> operator/(const MiniBuffer<T>& other) const;
        MiniBuffer<T> pow(const MiniBuffer<T>& other) const;
        MiniBuffer<T> max(const MiniBuffer<T>& other) const;

        bool operator==(const MiniBuffer<T>& other) const;
        std::vector<bool> operator==(T other) const;
        std::vector<bool> operator<(T other) const;
        std::vector<bool> operator>(T other) const;

        // Reduce operations
        MiniBuffer<T> sum();
        MiniBuffer<T> sum(int axis);

        // Mutate operations
        MiniBuffer<T> reshape(const std::vector<int> new_shape) const;
        MiniBuffer<T> flatten() const; 
        MiniBuffer<T> permute(const std::vector<int> order) const;
        MiniBuffer<T> expand(int axis, int expanded_size) const;
        // TODO: Mirko, 30. 12. 2023
        // Could also add different pad types but padding
        // with zeros is enough for now
        MiniBuffer<T> pad(int axis, const std::tuple<int, int> pad_sizes);
        MiniBuffer<T> shrink(int axis, const std::tuple<int, int> shrink_sizes);

        // Utility
        MiniBuffer<T> contiguous() const;

        bool is_scalar() const;
        bool is_square() const;
        int len() const;

        std::string to_string() const;

    private:
        std::vector<int> get_strides_from_shape(const std::vector<int>& shape);

        MiniBuffer<T> tril2(int diagonal) const;
        MiniBuffer<T> tril3(int diagonal) const;
        MiniBuffer<T> tril4(int diagonal) const;

        MiniBuffer<T> sum2() const;
        MiniBuffer<T> sum3() const;
        MiniBuffer<T> sum4() const;

        MiniBuffer<T> pad1(const std::tuple<int, int> pad_sizes) const;
        MiniBuffer<T> pad2(const std::tuple<int, int> pad_sizes) const;
        MiniBuffer<T> pad3(const std::tuple<int, int> pad_sizes) const;
        MiniBuffer<T> pad4(const std::tuple<int, int> pad_sizes) const;

        MiniBuffer<T> shrink2(const std::tuple<int, int> shrink_sizes) const;
        MiniBuffer<T> shrink3(const std::tuple<int, int> shrink_sizes) const;
        MiniBuffer<T> shrink4(const std::tuple<int, int> shrink_sizes) const;

        MiniBuffer<T> contiguous2() const;
        MiniBuffer<T> contiguous3() const;
        MiniBuffer<T> contiguous4() const;

        //? NOTE: Mirko, 30. 12. 2023.
        // Some axis-wise operations (sum, pad etc.) are easier to
        // perform if the axis on which they are performed is the
        // last one. This function permutes the buffer so that they
        // nth axis becomes the last one. Usually it will be called
        // once before the axis-wise operation and once more after
        // the operation is done, to bring back the original order.
        MiniBuffer<T> swap_nth_axis_with_last(int n);
        
        std::string to_string1() const;
        std::string to_string2() const;
        std::string to_string3() const;
        std::string to_string4() const;

    private:
        std::vector<T> m_Data;
        std::vector<int> m_Shape;
        std::vector<int> m_Strides;
        size_t m_Rank;
    };

    template<class T>
    MiniBuffer<T>::MiniBuffer(const std::vector<T>& data, const std::vector<int>& shape)
        : m_Data(data), m_Shape(shape)
    {
        m_Strides = get_strides_from_shape(shape);
        m_Rank = shape.size();
    }

    template<class T>
    MiniBuffer<T>::MiniBuffer(const std::vector<T>& data, const std::vector<int>& shape, const std::vector<int>& strides)
        : m_Data(data), m_Shape(shape), m_Strides(strides)
    {
        m_Rank = shape.size();
    }

    template<class T>
    std::vector<int> MiniBuffer<T>::get_strides_from_shape(const std::vector<int>& shape)
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

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::arange(int start, int end)
    {
        std::vector<T> data{};
        data.reserve(end-start);

        for (int i = start; i < end; i++)
        {
            data.push_back(T(i));
        }

        return MiniBuffer<T>(data, std::vector<int>{end-start});
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::fill(const std::vector<int>& shape, T value)
    {
        int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

        std::vector<T> data{};
        data.reserve(size);

        for (int i = 0; i < size; i++)
        {
            data.push_back(value);
        }

        return MiniBuffer<T>(data, shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::replace(const MiniBuffer& input, T target, T value)
    {
        const auto& input_data = input.get_data();
        int input_size = static_cast<int>(input_data.size());

        std::vector<T> data{};
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

        return MiniBuffer<T>(data, input.get_shape(), input.get_strides());
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::full_like(const MiniBuffer& input, T value)
    {
        return MiniBuffer<T>::fill(input.get_shape(), value);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::masked_fill(const MiniBuffer& input, const std::vector<bool> mask, T value)
    {
        const auto& input_data = input.get_data();
        int input_size = static_cast<int>(input_data.size());

        std::vector<T> data{};
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

        return MiniBuffer<T>(data, input.get_shape(), input.get_strides());
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::tril2(int diagonal) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;
        int input_size = static_cast<int>(input_data.size());

        std::vector<T> data{};
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

        return MiniBuffer<T>(data, input_shape, input_strides);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::tril3(int diagonal) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;
        int input_size = static_cast<int>(input_data.size());

        std::vector<T> data{};
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

        return MiniBuffer<T>(data, input_shape, input_strides);
    }
    
    template<class T>
    MiniBuffer<T> MiniBuffer<T>::tril4(int diagonal) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;
        int input_size = static_cast<int>(input_data.size());

        std::vector<T> data{};
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

        return MiniBuffer<T>(data, input_shape, input_strides);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::tril(const MiniBuffer& input, int diagonal)
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

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::operator-() const
    {
        std::vector<T> result;
        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(-this->m_Data[i]);
        }

        return MiniBuffer<T>(result, this->m_Shape, this->m_Strides);
    } 

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::log() const
    {
        std::vector<T> result;
        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(std::log(this->m_Data[i]));
        }

        return MiniBuffer<T>(result, this->m_Shape, this->m_Strides);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::log2() const
    {
        std::vector<T> result;
        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(std::log2(this->m_Data[i]));
        }

        return MiniBuffer<T>(result, this->m_Shape, this->m_Strides);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::operator+(const MiniBuffer& other) const 
    {
        std::vector<T> result;

        if (this->m_Shape != other.m_Shape)
        {
            assert(false && "Cannot perform addition, shapes do not match.");
        }

        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] + other.m_Data[i]);
        }

        return MiniBuffer<T>(result, this->m_Shape, this->m_Strides);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::operator-(const MiniBuffer& other) const 
    {
        std::vector<T> result;

        if (this->m_Shape != other.m_Shape)
        {
            assert(false && "Cannot perform addition, shapes do not match.");
        }

        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] - other.m_Data[i]);
        }

        return MiniBuffer<T>(result, this->m_Shape, this->m_Strides);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::operator*(const MiniBuffer& other) const 
    {
        std::vector<T> result;

        if (this->m_Shape != other.m_Shape)
        {
            assert(false && "Cannot perform addition, shapes do not match.");
        }

        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] * other.m_Data[i]);
        }

        return MiniBuffer<T>(result, this->m_Shape, this->m_Strides);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::operator/(const MiniBuffer& other) const 
    {
        std::vector<T> result;

        if (this->m_Shape != other.m_Shape)
        {
            assert(false && "Cannot perform addition, shapes do not match.");
        }

        result.reserve(this->m_Data.size());

        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] / other.m_Data[i]);
        }

        return MiniBuffer<T>(result, this->m_Shape, this->m_Strides);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::pow(const MiniBuffer& other) const
    {
        std::vector<T> result{};
        result.reserve(this->m_Data.size());
    
        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(std::pow(this->m_Data[i], other.m_Data[i]));
        }

        return MiniBuffer<T>(result, this->m_Shape, this->m_Strides);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::max(const MiniBuffer& other) const
    {
        std::vector<T> result{};
        result.reserve(this->m_Data.size());
    
        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(std::max(this->m_Data[i], other.m_Data[i]));
        }

        return MiniBuffer<T>(result, this->m_Shape, this->m_Strides);
    }

    template<class T>
    bool MiniBuffer<T>::operator==(const MiniBuffer& other) const
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

    template<class T>
    std::vector<bool> MiniBuffer<T>::operator==(T other) const
    {
        std::vector<bool> result{};
        result.reserve(this->m_Data.size());
    
        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] == other);
        }

        return result;
    }

    template<class T>
    std::vector<bool> MiniBuffer<T>::operator<(T other) const
    {
        std::vector<bool> result{};
        result.reserve(this->m_Data.size());
    
        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] < other);
        }

        return result;
    }

    template<class T>
    std::vector<bool> MiniBuffer<T>::operator>(T other) const
    {
        std::vector<bool> result{};
        result.reserve(this->m_Data.size());
    
        for (int i = 0; i < this->m_Data.size(); i++)
        {
            result.push_back(this->m_Data[i] > other);
        }

        return result;
    }
    
    template<class T>
    MiniBuffer<T> MiniBuffer<T>::sum2() const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        std::vector<T> data{};
        data.reserve(std::accumulate(input_shape.begin(), input_shape.end() - 1, 1, std::multiplies<int>()));

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            T row_sum{};

            for (int j = 0; j < input_shape[1]; j++)
            {
                row_sum += input_data[current_pos];
                current_pos++;
            }

            data.push_back(row_sum);
        }

        std::vector<int> output_shape(input_shape.begin(), input_shape.end() - 1);
        output_shape.push_back(1);

        return MiniBuffer<T>(data, output_shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::sum3() const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        std::vector<T> data{};
        data.reserve(std::accumulate(input_shape.begin(), input_shape.end() - 1, 1, std::multiplies<int>()));

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                T row_sum{};

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

        return MiniBuffer<T>(data, output_shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::sum4() const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        std::vector<T> data{};
        data.reserve(std::accumulate(input_shape.begin(), input_shape.end() - 1, 1, std::multiplies<int>()));

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                for (int k = 0; k < input_shape[2]; k++)
                {
                    T row_sum{};

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

        return MiniBuffer<T>(data, output_shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::sum()
    {
        const auto& input_data = this->m_Data;
        std::vector<int> new_shape(this->m_Rank, 1);

        T sum = std::accumulate(input_data.begin(), input_data.end(), 0);

        return MiniBuffer<T>(std::vector<T>{sum}, new_shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::sum(int axis)
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

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::reshape(const std::vector<int> new_shape) const
    {
        return MiniBuffer<T>(this->m_Data, new_shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::flatten() const
    {
        const auto& data = this->m_Data;
        int element_cnt = static_cast<int>(data.size());
        std::vector<int> flat_shape{1, element_cnt};

        return MiniBuffer<T>(data, flat_shape); 
    } 

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::permute(const std::vector<int> order) const
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

        return MiniBuffer<T>(this->m_Data, new_shape, new_strides).contiguous();
    }
    
    template<class T>
    MiniBuffer<T> MiniBuffer<T>::expand(int axis, int expanded_size) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;
        int input_size = static_cast<int>(input_data.size());

        int new_size = expanded_size * input_size;
        std::vector<T> data{};
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

        MiniBuffer result = MiniBuffer<T>(data, new_shape, new_strides);
        return result.contiguous();
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::pad1(const std::tuple<int, int> pad_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_pad_before = std::get<0>(pad_sizes);
        int n_pad_after  = std::get<1>(pad_sizes);

        std::vector<T> data{};
        int new_row_size = n_pad_before + input_shape.back() + n_pad_after;
        size_t new_size = std::accumulate(input_shape.begin(), input_shape.end() - 1, new_row_size, std::multiplies<int>()); 
        data.reserve(new_size);

        int current_pos = 0;

        std::vector<T> new_row{};
        new_row.reserve(new_row_size);
        
        for (int pad = 0; pad < n_pad_before; pad++)
        {
            new_row.push_back(0.0);
        }

        for (int i = 0; i < input_shape[0]; i++)
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

        return MiniBuffer<T>(data, output_shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::pad2(const std::tuple<int, int> pad_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_pad_before = std::get<0>(pad_sizes);
        int n_pad_after  = std::get<1>(pad_sizes);

        std::vector<T> data{};
        int new_row_size = n_pad_before + input_shape.back() + n_pad_after;
        size_t new_size = std::accumulate(input_shape.begin(), input_shape.end() - 1, new_row_size, std::multiplies<int>()); 
        data.reserve(new_size);

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            std::vector<T> new_row{};
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

        return MiniBuffer<T>(data, output_shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::pad3(const std::tuple<int, int> pad_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_pad_before = std::get<0>(pad_sizes);
        int n_pad_after  = std::get<1>(pad_sizes);

        std::vector<T> data{};
        int new_row_size = n_pad_before + input_shape.back() + n_pad_after;
        size_t new_size = std::accumulate(input_shape.begin(), input_shape.end() - 1, new_row_size, std::multiplies<int>()); 
        data.reserve(new_size);

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                std::vector<T> new_row{};
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

        return MiniBuffer<T>(data, output_shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::pad4(const std::tuple<int, int> pad_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_pad_before = std::get<0>(pad_sizes);
        int n_pad_after  = std::get<1>(pad_sizes);

        std::vector<T> data{};
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
                    std::vector<T> new_row{};
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

        return MiniBuffer<T>(data, output_shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::pad(int axis, const std::tuple<int, int> pad_sizes)
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

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::shrink2(const std::tuple<int, int> shrink_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_shrink_before = std::get<0>(shrink_sizes);
        int n_shrink_after  = std::get<1>(shrink_sizes);

        std::vector<T> data{};
        int new_row_size = input_shape.back() - n_shrink_before - n_shrink_after;
        size_t new_size = std::accumulate(input_shape.begin(), input_shape.end() - 1, new_row_size, std::multiplies<int>()); 
        data.reserve(new_size);

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            std::vector<T> new_row{};
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

        return MiniBuffer<T>(data, output_shape);
    }
    
    template<class T>
    MiniBuffer<T> MiniBuffer<T>::shrink3(const std::tuple<int, int> shrink_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_shrink_before = std::get<0>(shrink_sizes);
        int n_shrink_after  = std::get<1>(shrink_sizes);

        std::vector<T> data{};
        int new_row_size = input_shape.back() - n_shrink_before - n_shrink_after;
        size_t new_size = std::accumulate(input_shape.begin(), input_shape.end() - 1, new_row_size, std::multiplies<int>()); 
        data.reserve(new_size);

        int current_pos = 0;

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                std::vector<T> new_row{};
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

        return MiniBuffer<T>(data, output_shape);
    }
    
    template<class T>
    MiniBuffer<T> MiniBuffer<T>::shrink4(const std::tuple<int, int> shrink_sizes) const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;

        int n_shrink_before = std::get<0>(shrink_sizes);
        int n_shrink_after  = std::get<1>(shrink_sizes);

        std::vector<T> data{};
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
                    std::vector<T> new_row{};
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

        return MiniBuffer<T>(data, output_shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::shrink(int axis, const std::tuple<int, int> shrink_sizes)
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

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::contiguous2() const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;

        std::vector<T> data{};
        data.reserve(std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>()));

        for (int i = 0; i < input_shape[0]; i++)
        {
            for (int j = 0; j < input_shape[1]; j++)
            {
                int current_pos = i * input_strides[0] + j * input_strides[1]; 
                data.push_back(input_data[current_pos]);
            }
        }

        return MiniBuffer<T>(data, this->m_Shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::contiguous3() const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;

        std::vector<T> data{};
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

        return MiniBuffer<T>(data, this->m_Shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::contiguous4() const
    {
        const auto& input_data = this->m_Data;
        const auto& input_shape = this->m_Shape;
        const auto& input_strides = this->m_Strides;

        std::vector<T> data{};
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

        return MiniBuffer<T>(data, this->m_Shape);
    }

    template<class T>
    MiniBuffer<T> MiniBuffer<T>::contiguous() const
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

    template<class T>
    bool MiniBuffer<T>::is_scalar() const
    {
        return this->m_Data.size() == 1;
    }

    template<class T>
    bool MiniBuffer<T>::is_square() const
    {
        int dim_cnt = static_cast<int>(this->m_Shape.size());
        return this->m_Shape[dim_cnt - 1] == this->m_Shape[dim_cnt - 2];
    }

    template<class T>
    int MiniBuffer<T>::len() const
    {
        return static_cast<int>(this->m_Data.size());
    }
    
    template<class T>
    MiniBuffer<T> MiniBuffer<T>::swap_nth_axis_with_last(int n)
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

    template<class T>
    std::string MiniBuffer<T>::to_string() const
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

    template<class T>
    std::string MiniBuffer<T>::to_string1() const
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

    template<class T>
    std::string MiniBuffer<T>::to_string2() const
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

    template<class T>
    std::string MiniBuffer<T>::to_string3() const
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

    template<class T>
    std::string MiniBuffer<T>::to_string4() const
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
