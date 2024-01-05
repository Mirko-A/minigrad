#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace minitorch
{
    class MiniBuffer
    {
    public:
        MiniBuffer(const std::vector<float>& data, const std::vector<int>& shape);
        MiniBuffer(const std::vector<float>& data, const std::vector<int>& shape, const std::vector<int>& strides);

        inline const std::vector<float>& get_data()  const { return m_Data;    }
        inline const std::vector<int>& get_shape()   const { return m_Shape;   }
        inline const std::vector<int>& get_strides() const { return m_Strides; }
        inline const size_t get_rank() const { return m_Rank; }

        static MiniBuffer arange(int start, int end);
        static MiniBuffer fill(const std::vector<int>& shape, float value);
        static MiniBuffer replace(const MiniBuffer& input, float target, float value);
        static MiniBuffer full_like(const MiniBuffer& input, float value);
        static MiniBuffer masked_fill(const MiniBuffer& input, const std::vector<bool> mask, float value);
        static MiniBuffer tril(const MiniBuffer& input, int diagonal = 0);

        // Unary operations
        MiniBuffer operator-() const;
        MiniBuffer log() const;
        MiniBuffer log2() const;

        // Binary operations
        MiniBuffer operator+(const MiniBuffer& other) const;
        MiniBuffer operator-(const MiniBuffer& other) const;
        MiniBuffer operator*(const MiniBuffer& other) const;
        MiniBuffer operator/(const MiniBuffer& other) const;
        MiniBuffer pow(const MiniBuffer& other) const;
        MiniBuffer max(const MiniBuffer& other) const;

        bool operator==(const MiniBuffer& other) const;
        std::vector<bool> operator==(float other) const;
        std::vector<bool> operator<(float other) const;
        std::vector<bool> operator>(float other) const;

        // Reduce operations
        MiniBuffer sum();
        MiniBuffer sum(int axis);

        // Mutate operations
        MiniBuffer reshape(const std::vector<int> new_shape) const;
        MiniBuffer flatten() const; 
        MiniBuffer permute(const std::vector<int> order) const;
        MiniBuffer expand(int axis, int expanded_size) const;
        // TODO: Mirko, 30. 12. 2023
        // Could also add different pad types but padding
        // with zeros is enough for now
        MiniBuffer pad(int axis, const std::tuple<int, int> pad_sizes);
        MiniBuffer shrink(int axis, const std::tuple<int, int> shrink_sizes);

        // Utility
        MiniBuffer contiguous() const;

        bool is_scalar() const;
        bool is_square() const;
        int len() const;

        std::string to_string() const;

    private:
        std::vector<int> get_strides_from_shape(const std::vector<int>& shape);

        MiniBuffer tril2(int diagonal) const;
        MiniBuffer tril3(int diagonal) const;
        MiniBuffer tril4(int diagonal) const;

        MiniBuffer sum2() const;
        MiniBuffer sum3() const;
        MiniBuffer sum4() const;

        MiniBuffer pad1(const std::tuple<int, int> pad_sizes) const;
        MiniBuffer pad2(const std::tuple<int, int> pad_sizes) const;
        MiniBuffer pad3(const std::tuple<int, int> pad_sizes) const;
        MiniBuffer pad4(const std::tuple<int, int> pad_sizes) const;

        MiniBuffer shrink2(const std::tuple<int, int> shrink_sizes) const;
        MiniBuffer shrink3(const std::tuple<int, int> shrink_sizes) const;
        MiniBuffer shrink4(const std::tuple<int, int> shrink_sizes) const;

        MiniBuffer contiguous2() const;
        MiniBuffer contiguous3() const;
        MiniBuffer contiguous4() const;

        //? NOTE: Mirko, 30. 12. 2023.
        // Some axis-wise operations (sum, pad etc.) are easier to
        // perform if the axis on which they are performed is the
        // last one. This function permutes the buffer so that they
        // nth axis becomes the last one. Usually it will be called
        // once before the axis-wise operation and once more after
        // the operation is done, to bring back the original order.
        MiniBuffer swap_nth_axis_with_last(int n);
        
        std::string to_string2() const;
        std::string to_string3() const;
        std::string to_string4() const;

    private:
        std::vector<float> m_Data;
        std::vector<int> m_Shape;
        std::vector<int> m_Strides;
        size_t m_Rank;
    };
}
