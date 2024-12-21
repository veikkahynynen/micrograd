import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 3*x**2 - 4*x + 5



def main():
    xs = np.arange(-5, 5, 0.1)
    ys = f(xs)
    plt.plot(xs, ys)
    plt.show()



if __name__ == "__main__":
    main()