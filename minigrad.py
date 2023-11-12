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
    
    # TODO: ----------------------------------------
    # multiplication - Mire
    def __mul__(self, other):
        if isinstance(other, Value):
            out = Value(self.val * other.val)
        else:
            out = Value(self.val * other)

        return out
    
    def __rmul__(self, other):
        if isinstance(other, Value):
            out = Value(self.val * other.val)
        else:
            out = Value(self.val * other)

        return out
    
    # exponentiation - Prki
    # sigmoid - Mire
    # relu - Prki
    # ----------------------------------------------
    
    
    def __repr__(self) -> str:
        return f"Value: {self.val}"
