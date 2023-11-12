class Value:
    def __init__(self, val: float) -> None:
        self.val = val

    def __add__(self, other):
        return Value(self.val + other.val)
    
    def __repr__(self) -> str:
        return f"Value: {self.val}"