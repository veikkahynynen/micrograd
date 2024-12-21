import math
import numpy as np
import matplotlib.pyplot as plt





class Value:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Value(data={self.data}"





def f(x):
    return 3*x**2 - 4*x + 9


def main():
    x = 3.0
    print(f(x))
    h = 0.001
    print((f(x + h) - f(x)) / h)


if __name__ == "__main__":
    main()