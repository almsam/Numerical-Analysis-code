from math import isinf, isnan
from numpy import arange
from sympy import lambdify, symbols

def shift_away_from_inf(f, x, step=1e-6, direction=1):
    while isinf(f(x)) or isnan(f(x)):
        x += step * direction
    return x

def bisection_search(func, range_start, range_end, tolerance=1e-6, max_iter=1000):
    x = symbols('x')
    f = lambdify(x, func)
    a = shift_away_from_inf(f, range_start, direction=1)
    b = shift_away_from_inf(f, range_end, direction=-1)
    if f(a) * f(b) >= 0:
        return None # both range values have same sign, so no root. Better way to signify this?
    for _ in range(max_iter):
        m = (a+b)/2 # center of current interval
        m = shift_away_from_inf(f, m)

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
        # a = x
        # b = x + step
        a = shift_away_from_inf(f, x, direction=1)
        b = shift_away_from_inf(f, x+step, direction=-1)
        if(f(a) == 0):
            if a not in roots:
                roots.append(a)
        elif(f(b) == 0):
            if b not in roots:
                roots.append(b)
        elif f(a) * f(b) < 0:
            root = bisection_search(func, a, b, tolerance, max_iter)
            if root not in roots:
                roots.append(root)
    
    # print(roots)
    return roots


