class Value:
    def __init__(self, val: float) -> None:
        self.val = val

    def __add__(self, other):
        return Value(self.val + other.val)
    
    def __repr__(self) -> str:
        return f"Value: {self.val}"
    
a = Value(3.0)
b = Value(5.0)

c = a + b
print(c)
