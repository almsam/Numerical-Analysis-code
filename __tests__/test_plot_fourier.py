import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Plot_Finder import find_fourier
import matplotlib.pyplot as plt

# Test Case 1: Quadratic + Linear + Sin
print("=== Test Case 1: y = x^2 + 2x + 0.01sin(x) ===")
x1 = np.linspace(0, 10, 100)
y1 = x1**2 + 2*x1 + 0.01*np.sin(x1)

# This will automatically plot the progressive approximation
funclist1, formula1 = find_fourier(x1, y1, Iterations=3, plot=True)

# Test Case 2: Linear + Sine with larger amplitude
print("\n=== Test Case 2: y = 3x + 5sin(2x) ===")
x2 = np.linspace(0, 4*np.pi, 150)
y2 = 3*x2 + 5*np.sin(2*x2)

funclist2, formula2 = find_fourier(x2, y2, Iterations=3, plot=True)

# Test Case 3: Exponential + Quadratic
print("\n=== Test Case 3: y = 0.5*exp(0.3x) + 0.1x^2 ===")
x3 = np.linspace(0, 5, 100)
y3 = 0.5*np.exp(0.3*x3) + 0.1*x3**2

funclist3, formula3 = find_fourier(x3, y3, Iterations=3, plot=True)

# Test Case 4: Complex multi-component
print("\n=== Test Case 4: y = x + sin(x) + 0.1x^2 + cos(2x) ===")
x4 = np.linspace(0, 10, 200)
y4 = x4 + np.sin(x4) + 0.1*x4**2 + np.cos(2*x4)

funclist4, formula4 = find_fourier(x4, y4, Iterations=4, plot=True)

# Print summary for each test
print("\n=== SUMMARY ===")
print(f"Test 1 found {len(funclist1)} components")
print(f"Test 2 found {len(funclist2)} components")
print(f"Test 3 found {len(funclist3)} components")
print(f"Test 4 found {len(funclist4)} components")

print(f"\nTest 1:")
print(f"Test 1-1 returned: {funclist1[0]}")
print(f"Test 1-2 returned: {funclist1[1]}")
print(f"Test 1-3 returned: {funclist1[2]}")
print(f"\nTest 2:")
print(f"Test 2-1 returned: {funclist2[0]}")
print(f"Test 2-2 returned: {funclist2[1]}")
print(f"Test 2-3 returned: {funclist2[2]}")
print(f"\nTest 3:")
print(f"Test 3-1 returned: {funclist3[0]}")
print(f"Test 3-2 returned: {funclist3[1]}")
print(f"Test 3-3 returned: {funclist3[2]}")
print(f"\nTest 4:")
print(f"Test 4-1 returned: {funclist4[0]}")
print(f"Test 4-2 returned: {funclist4[1]}")
print(f"Test 4-3 returned: {funclist4[2]}")
print(f"Test 4-4 returned: {funclist4[3]}")

# print(f"Test 2 found {funclist2} components")

# print(f"Test 3 found {funclist3} components")

# print(f"Test 4 found {funclist4} components")