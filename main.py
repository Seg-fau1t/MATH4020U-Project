import numpy as np
from scipy.optimize import newton as newt

import cplot
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def f(x):
    """Real function f(x)."""
    return (x - 4) * (x + 1)


def fd(x):
    """Derivative of f(x)."""
    return 2 * x - 3


def g(x):
    """Real function g(x)."""
    return (x - 1) * (x + 3)


def gd(x):
    """Derivative of g(x)."""
    return 2 * x + 2


def h(x):
    """Real function h(x)."""
    return (x - 4) * (x - 1) * (x + 3)


def hd(x):
    """Derivative of h(x)."""
    return 3 * np.square(x) - 4 * x - 11


def c(z):
    """Complex function c(z)."""
    return np.power(z, 3) - 1


def cd(z):
    """Derivative of the complex function c(z)."""
    return 3 * np.square(z)

# ------------------------------------------------------------------------------


def f_zero(x):
    """Newton method applied to f(x)."""
    return newt(f, x, fprime=fd)


def g_zero(x):
    """Newton method applied to g(x)."""
    return newt(g, x, fprime=gd)


def h_zero(x):
    """Newton method applied to h(x)."""
    return newt(h, x, fprime=hd)


def c_zero(z):
    """Newton method applied to c(x)."""
    return newt(c, z, fprime=cd)

# ------------------------------------------------------------------------------


def real_graph(x, f, f_zero):
    """Plot real functions with color for roots found using newton method."""
    y = f(x)
    z = f_zero(x)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots()

    norm = plt.Normalize(z.min(), z.max())
    lc = LineCollection(segments, cmap='rainbow', norm=norm)

    lc.set_array(z)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs)

    axs.set_xlim(x.min() - 1, x.max() + 1)
    axs.set_ylim(y.min() - 10, y.max() + 10)

    axs.spines['left'].set_position('zero')
    axs.spines['right'].set_color('none')
    axs.spines['bottom'].set_position('zero')
    axs.spines['top'].set_color('none')

    axs.xaxis.set_ticks_position('bottom')
    axs.yaxis.set_ticks_position('left')

    plt.show()


def complex_graph(f, start=-1, stop=1, num=1000):
    """Plot a complex function."""
    plt = cplot.plot(f, (start, stop, num), (start, stop, num))
    plt.show()


def f_graph(x):
    """Quick way to plot f(x)."""
    real_graph(x, f, f_zero)


def g_graph(x):
    """Quick way to plot g(x)."""
    real_graph(x, g, g_zero)


def h_graph(x):
    """Quick way to plot h(x)."""
    real_graph(x, h, h_zero)


def c_graph(x, y):
    """Quick way to plot c(x) newton iterations."""
    complex_graph(c_zero, x, y)


if __name__ == "__main__":

    x = np.linspace(-5, 5, num=1000)

    f_graph(x)
    g_graph(x)
    h_graph(x)

    complex_graph(c, -5, 5)
    c_graph(-5, 5)
