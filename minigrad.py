import math

class Value:
    def __init__(self, data: float) -> None:
        self.data = data

    def item(self):
        return self.data

    def __add__(self, other):
        if isinstance(other, Value):
            out = Value(self.data + other.data)
        else:
            out = Value(self.data + other)
        
        return out
    
    def __radd__(self, other):
        out = Value(other + self.data)
        return out

    def __sub__(self, other):
        if isinstance(other, Value):
            out = Value(self.data - other.data)
        else:
            out = Value(self.data - other)
        
        return out
    
    def __rsub__(self, other):
        out = Value(other - self.data)
        return out

    def __neg__(self):
        out = Value(-self.data)
        return out

    def __mul__(self, other):
        if isinstance(other, Value):
            out = Value(self.data * other.data)
        else:
            out = Value(self.data * other)

        return out
    
    def __rmul__(self, other):
        out = Value(other * self.data)
        return out
    
    def __truediv__(self, other):
        if isinstance(other, Value):
            out = Value(self.data / other.data)
        else:
            out = Value(self.data / other)

        return out
    
    def __rtruediv__(self, other):
        out = Value(other / self.data)
        return out
    
    def __pow__(self, other):
        if isinstance(other, Value):
            out = Value(self.data ** other.data)
        else:
            out = Value(self.data ** other)
        
        return out
    
    def __rpow__(self, other):
        out = Value(other ** self.data)
        return out
    
    def exp(self):
        out = Value(math.e ** self.data)
        return out

    def sigmoid(self):
        out = 1 / (1 + (-self).exp())
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0)
        return out
    
    def __repr__(self) -> str:
        return f"Value: {self.data}"
    
a = Value(3.0)
b = Value(5.0)

c = a + b
d = c + 5.0
e = 10.0 + d
f = 15.0 + e
g = b ** a
h = b ** 2
i = a.relu()
print(i)
a = 3.0