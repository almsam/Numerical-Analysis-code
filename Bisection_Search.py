from numpy import arange
from sympy import lambdify, symbols

def bisection_search(func, range_start, range_end, tolerance=1e-6, max_iter=1000):
    x = symbols('x')
    f = lambdify(x, func)
    if f(range_start) * f(range_end) >= 0:
        return None # both range values have same sign, so no root. Better way to signify this?

    a = range_start
    b = range_end
    for _ in range(max_iter):
        m = (a+b)/2 # center of current interval

        if abs(f(m)) < tolerance or (b-a) / 2 < tolerance: # y-val close enough to 0, or interval small enough
            return m
        if f(a) * f(m) < 0:
            b = m
        else:
            a = m
    return (a+b) / 2

def bisection_all(func, range_start, range_end, step, tolerance=1e-6, max_iter=1000):
    roots = []
    x = symbols('x')
    f = lambdify(x, func)

    for x in arange(range_start, range_end, step):
        a = x
        b = x + step
        if f(a) * f(b) < 0:
            roots.append(bisection_search(func, a, b, tolerance, max_iter))
    
    return roots


