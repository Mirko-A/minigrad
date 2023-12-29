#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class TestBuffer
{
public:
    TestBuffer(const std::vector<float>& data, const std::vector<int>& shape);
    TestBuffer(const std::vector<float>& data, const std::vector<int>& shape, const std::vector<int>& strides);

    inline const std::vector<float>& get_data() const { return m_Data; }
    inline const std::vector<int>& get_shape() const { return m_Shape; }
    inline const std::vector<int>& get_strides() const { return m_Strides; }

    // Unary operators
    TestBuffer operator-() const;
    TestBuffer log() const;
    TestBuffer log2() const;

    // Binary operators
    TestBuffer operator+(const TestBuffer& other) const;
    TestBuffer operator-(const TestBuffer& other) const;
    TestBuffer operator*(const TestBuffer& other) const;
    TestBuffer operator/(const TestBuffer& other) const;

private:
    std::vector<int> get_strides_from_shape(const std::vector<int>& shape);

private:
    std::vector<float> m_Data;
    std::vector<int> m_Shape;
    std::vector<int> m_Strides;
};

