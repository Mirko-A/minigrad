class Value:
    def __init__(self, val: float) -> None:
        self.val = val

    def __add__(self, other):
        if isinstance(other, Value):
            out = Value(self.val + other.val)
        else:
            assert isinstance(other, float)
            out = Value(self.val + other)
        
        return out
    
    def __radd__(self, other):
        if isinstance(other, Value):
            out = Value(self.val + other.val)
        else:
            assert isinstance(other, float)
            out = Value(self.val + other)
        
        return out
    
    # TODO: ----------------------------------------
    # multiplication
    # exponentiation
    # sigmoid
    # relu
    # ----------------------------------------------
    
    
    def __repr__(self) -> str:
        return f"Value: {self.val}"
    
a = Value(3.0)
b = Value(5.0)

c = a + b
d = c + 5.0
e = 10.0 + d
f = 150.0 + e
print(f)
