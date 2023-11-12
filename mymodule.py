class Value:
    def __init__(self, val: float) -> None:
        self.val = val

    def __add__(self, other):
        if isinstance(other, Value):
            out = Value(self.val + other.val)
        else:
            out = Value(self.val + other)
        
        return out
    
    def __radd__(self, other):
        if isinstance(other, Value):
            out = Value(self.val + other.val)
        else:
            out = Value(self.val + other)
        
        return out
    
    def __pow__(self, other):
        if isinstance(other, Value):
            out = Value(self.val ** other.val)
        else:
            out = Value(self.val ** other)
        
        return out
    
    def relu(self):
        out = Value(self.val if self.val > 0 else 0)
        return out
    
    # TODO: ----------------------------------------
    # multiplication - Mire
    # exponentiation - Prki ✔
    # sigmoid - Mire 
    # relu - Prki ✔
    # ----------------------------------------------
    
    def __repr__(self) -> str:
        return f"Value: {self.val}"
    
a = Value(3.0)
b = Value(5.0)

c = a + b
d = c + 5.0
e = 10.0 + d
f = 150.0 + e
g = b ** a
h = b ** 2
i = a.relu()
print(i)
