import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph



def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot



class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label


    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def ____(self, other):
        out = Value(self.data - other.data, (self, other))
        return out

    def ____(self, other):
        out = Value(self.data / other.data, (self, other))
        return out


def main():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    f = Value(-2.0, label='f')

    f.grad = 4.0
    c.grad = -2.0
    a.grad = 6.0
    b.grad = -4.0

    a.data += 0.01 * a.grad
    b.data += 0.01 * b.grad
    c.data += 0.01 * c.grad
    f.data += 0.01 * f.grad


    e = a*b; e.label='e'
    d = e + c; d.label='d'
    L = d*f; L.label='L'
    d.grad = -2.0
    e.grad = -2.0
    L.grad = 1.0

    print(L.data)    

    dot = draw_dot(L)
    dot.render('graph', format='png', cleanup=True)



if __name__ == "__main__":
    main()