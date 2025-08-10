from sympy import symbols

# def central_difference_approx(func, x, h):
#     return (func(x+h) - func(x-h)) / (2*h)

def bisection_search(func, range_start, range_end, tolerance=1e-6, max_iter=1000):
    if func(range_start) * func(range_end) >= 0:
        return None

    x = symbols('x')
    a = range_start
    b = range_end
    for _ in range(max_iter):
        m = (a+b)/2 # center of current interval

        if abs(func(m)) < tolerance or (b-a) / 2 < tolerance: # y-val close enough to 0, or interval small enough
            return m
        if func(a) * func(m) < 0:
            b = m
        else:
            a = m
    return (a+b) / 2



