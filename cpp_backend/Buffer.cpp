#include "Buffer.h"

#include <cmath>
#include <cassert>

TestBuffer::TestBuffer(const std::vector<float>& data, const std::vector<int>& shape)
    : m_Data(data), m_Shape(shape)
{
    m_Strides = get_strides_from_shape(shape);
}

TestBuffer::TestBuffer(const std::vector<float>& data, const std::vector<int>& shape, const std::vector<int>& strides)
    : m_Data(data), m_Shape(shape), m_Strides(strides)
{}

std::vector<int> TestBuffer::get_strides_from_shape(const std::vector<int>& shape)
{
    std::vector<int> strides{};
    return strides;
}

TestBuffer TestBuffer::operator-() const
{
    std::vector<float> result_data;
    result_data.reserve(m_Data.size());
    
    for (size_t i = 0; i < this->m_Data.size(); i++)
    {
        result_data.push_back(-this->m_Data[i]);
    }
    
    return TestBuffer(result_data, this->m_Shape, this->m_Strides);
} 


TestBuffer TestBuffer::log() const
{
    std::vector<float> result_data;
    result_data.reserve(m_Data.size());
    
    for (size_t i = 0; i < this->m_Data.size(); i++)
    {
        result_data.push_back(std::log(this->m_Data[i]));
    }
    
    return TestBuffer(result_data, this->m_Shape, this->m_Strides);
}

TestBuffer TestBuffer::log2() const
{
    std::vector<float> result_data;
    result_data.reserve(m_Data.size());
    
    for (size_t i = 0; i < this->m_Data.size(); i++)
    {
        result_data.push_back(std::log2(this->m_Data[i]));
    }
    
    return TestBuffer(result_data, this->m_Shape, this->m_Strides);
}

TestBuffer TestBuffer::operator+(const TestBuffer& other) const 
{
    std::vector<float> result_data;

    if (this->m_Shape != other.m_Shape)
    {
        assert(false & "Cannot perform addition, shapes do not match.");
    }
    
    result_data.reserve(this->m_Data.size());

    for (size_t i = 0; i < this->m_Data.size(); i++)
    {
        result_data.push_back(this->m_Data[i] + other.m_Data[i]);
    }
    
    return TestBuffer(result_data, this->m_Shape, this->m_Strides);
}

TestBuffer TestBuffer::operator-(const TestBuffer& other) const 
{
    std::vector<float> result_data;
    
    if (this->m_Shape != other.m_Shape)
    {
        assert(false & "Cannot perform addition, shapes do not match.");
    }
    
    result_data.reserve(this->m_Data.size());

    for (size_t i = 0; i < this->m_Data.size(); i++)
    {
        result_data.push_back(this->m_Data[i] - other.m_Data[i]);
    }
    
    return TestBuffer(result_data, this->m_Shape, this->m_Strides);
}

TestBuffer TestBuffer::operator*(const TestBuffer& other) const 
{
    std::vector<float> result_data;
    
    if (this->m_Shape != other.m_Shape)
    {
        assert(false & "Cannot perform addition, shapes do not match.");
    }
    
    result_data.reserve(this->m_Data.size());

    for (size_t i = 0; i < this->m_Data.size(); i++)
    {
        result_data.push_back(this->m_Data[i] * other.m_Data[i]);
    }
    
    return TestBuffer(result_data, this->m_Shape, this->m_Strides);
}

TestBuffer TestBuffer::operator/(const TestBuffer& other) const 
{
    std::vector<float> result_data;
    
    if (this->m_Shape != other.m_Shape)
    {
        assert(false & "Cannot perform addition, shapes do not match.");
    }
    
    result_data.reserve(this->m_Data.size());

    for (size_t i = 0; i < this->m_Data.size(); i++)
    {
        result_data.push_back(this->m_Data[i] / other.m_Data[i]);
    }
    
    return TestBuffer(result_data, this->m_Shape, this->m_Strides);
}