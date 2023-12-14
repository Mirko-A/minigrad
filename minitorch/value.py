from __future__ import annotations

import math

# NOTE: Children of a Value which is created as a result of commutative operations
#       might be swapped if it is called from r-operation. Potentailly the reverse
#       flag might be needed for those operations just to keep the children always
#       in the same order.

class Value:
    def __init__(self, data: float | int, requires_grad: bool = True, _children: tuple[Value, Value] | tuple[Value] | tuple[()] = ()) -> None:
        assert isinstance(data, (float, int)) and not isinstance(data, bool), f"Cannot construct Value object with type: {type(data)}. Expected float."
        if not isinstance(data, float): data = float(data)

        self.data = data
        self.grad = 0.0
        self.requires_grad = requires_grad
        self._children = _children
        self._backward = lambda: None
        # TODO: DEBUGGING ONLY
        self.backward_fn: str = ""

    # Operations

    def _add(self, other: Value | float, reverse: bool = False) -> Value:
        if not isinstance(other, Value): other = Value(other, requires_grad=False)
        x, y = self, other

        if reverse:
            x, y = y, x

        out = Value(x.data + y.data, _children = (x, y))
        
        def add_backward():
            if x.requires_grad:
                x.grad  += out.grad
            else:
                x.grad = None
            if y.requires_grad:
                y.grad += out.grad
            else:
                y.grad = None

        out._backward = add_backward
        out.backward_fn = "add"
        
        return out
    
    def _sub(self, other: Value | float, reverse: bool = False) -> Value:
        if not isinstance(other, Value): other = Value(other, requires_grad=False)
        x, y = self, other
        
        if reverse:
            x, y = y, x

        out = Value(x.data - y.data,  _children = (x, y))

        def sub_backward():
            if x.requires_grad:
                x.grad +=  out.grad
            else:
                x.grad = None
            if y.requires_grad:
                y.grad += -out.grad
            else:
                y.grad = None

        out._backward = sub_backward
        out.backward_fn = "sub"
        
        return out
    
    def _neg(self) -> Value:
        out = Value(-self.data)
        
        def neg_backward():
            self.grad += -out.grad

        out._backward = neg_backward

        return out
    
    def _mul(self, other: Value | float, reverse: bool = False) -> Value:
        if not isinstance(other, Value): other = Value(other, requires_grad=False)
        x, y = self, other

        if reverse:
            x, y = y, x

        out = Value(x.data * y.data, _children = (x, y))
        
        def mul_backward():
            if x.requires_grad:
                x.grad  += out.grad * y.data
            else:
                x.grad = None
            if y.requires_grad:
                y.grad += out.grad * x.data
            else:
                y.grad = None

        out._backward = mul_backward
        out.backward_fn = "mul"
        
        return out
    
    def _div(self, other: Value | float, reverse: bool = False) -> Value:
        if not isinstance(other, Value): other = Value(other, requires_grad=False)
        x, y = self, other
        
        if reverse:
            x, y = y, x

        out = Value(x.data / y.data, _children = (x, y))

        def div_backward():
            if x.requires_grad:
                x.grad += out.grad * 1 / y.data
            else:
                x.grad = None
            if y.requires_grad:
                y.grad += out.grad * (-1)*(x.data / (y.data ** 2))
            else:
                y.grad = None

        out._backward = div_backward
        out.backward_fn = "div"

        return out

    def _pow(self, other: Value | float, reverse: bool = False) -> Value:
        if not isinstance(other, Value): other = Value(other, requires_grad=False)
        x, y = self, other
        
        if reverse:
            x, y = y, x

        out = Value(x.data ** y.data, _children = (x, y))
        
        def pow_backward():
            if x.requires_grad:
                x.grad += out.grad * (y.data * (x.data ** (y.data - 1)))
            else:
                x.grad = None
            if y.requires_grad:
                y.grad += out.grad * ((x.data ** y.data) * math.log(x.data))
            else:
                y.grad = None
                
        out._backward = pow_backward
        out.backward_fn = "pow"

        return out
    
    def log(self, base: Value | float | int = math.e) -> Value:
        if not isinstance(base, Value): base = Value(base, requires_grad=False)
        arg = self
        
        out = Value(math.log(self.data, base.data), _children = (arg, ))
        
        def log_backward():
            if arg.requires_grad:
                arg.grad += out.grad * (1 / (arg.data * math.log(base.data)))
            else:
                arg.grad = None
            if base.requires_grad:
                base.grad += out.grad * (-(math.log(arg.data) / (base.data * math.log(base.data) ** 2)))
            else:
                base.grad = None

        out._backward = log_backward
        out.backward_fn = "log"
        
        return out

    def exp(self) -> Value:
        return math.e ** self

    # Operator magic methods

    def __add__(self, other): return self._add(other)
    def __radd__(self, other): return self._add(other, True)
    
    def __sub__(self, other): return self._sub(other)
    def __rsub__(self, other): return self._sub(other, True)

    def __neg__(self): return self._neg()

    def __mul__(self, other): return self._mul(other)
    def __rmul__(self, other): return self._mul(other, True)
    
    def __truediv__(self, other): return self._div(other)
    def __rtruediv__(self, other): return self._div(other, True)
    
    def __pow__(self, other): return self._pow(other)
    def __rpow__(self, other): return self._pow(other, True)

    # Activation funcs

    def sigmoid(self) -> Value:
        def sigmoid_impl(x):
            return 1 / (1 + math.exp(-x))
        
        out = Value(sigmoid_impl(self.data), _children = (self, ))

        def sigmoid_backward():
            self.grad += out.grad * (sigmoid_impl(self.data) * (1 - sigmoid_impl(self.data)))

        out._backward = sigmoid_backward

        return out

    def relu(self) -> Value:
        out = Value(self.data if self.data > 0 else 0, _children = (self, ))

        def relu_backward():
            relu_deriv = 1 if self.data > 0 else 0
            self.grad += out.grad * relu_deriv

        out._backward = relu_backward

        return out

    def tanh(self) -> Value:
        def tanh_impl(x):
            return (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        
        out = Value(tanh_impl(self.data), _children = (self, ))
        
        def tanh_backward():
            self.grad += out.grad * (1 - tanh_impl(self.data) ** 2)
        
        out._backward = tanh_backward
        
        return out
        
    # Backpropagation

    def backward(self, grad: float = 1.0) -> None:
        self.grad = grad

        nodes: list[Value] = []
        visited = set()
        def toposort(node: Value):
            if node not in visited:
                visited.add(node)

                for child in node._children:
                    toposort(child)

                nodes.append(node)

        toposort(self)

        for node in reversed(nodes):
            node._backward()

    # Utility

    def __repr__(self) -> str:
        return f"Value({self.data})"
