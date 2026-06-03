import math

class Value:
    """A node in a scalar-valued computation graph with automatic differentiation.

    Value wraps a scalar value and stores its gradient and a `_backward` function, which
    allows us to build a computation graph across various operations.

    Each Value contains a `_backward` closure that defines how gradients flow through
    each operation.

    Calling `backward()` on the output node creates a topological ordering of the Directed
    Acyclic Graph, on which `_backward()` is called recursively to accumulate the gradient
    of all ancestor nodes.
    """


    def __init__(self, data, _prev=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_prev)
        self._op = _op
        self.label = label


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        # Gradient of addition is 1 for both inputs, so we pass the gradient unchanged
        def _backward():
            # Multiple paths might contribute to the same node so we accumulate the gradients.
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out


    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    


    def __neg__(self):
        return self * -1


    def __radd__(self, other):
        return self + other


    def __sub__(self, other):
        return self + (-other)


    def __rsub__(self, other):
        return other + (-self)


    def __rmul__(self, other):
        return self * other


    def __truediv__(self, other):
        return self * other**-1


    def __rtruediv__(self, other):
        return other * self**-1


    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        assert self.data > 0, "log is only defined for positive values"
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out


    def backward(self):
        # Build topological order so we can traverse in reverse during backprop
        topo = []
        visited = set()
        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Set the gradient of the output node (usually last node or Loss) to 1 and run backprop
        # by traversing the graph in reverse dependency order.
        self.grad = 1
        for v in reversed(topo):
            v._backward()