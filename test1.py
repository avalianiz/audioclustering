import numpy as np
import matplotlib.pyplot as plt

# Define the function and its exact derivative
def f(x):
    return x**3 * np.exp(-x)

def f_prime_exact(x):
    return np.exp(-x) * (3*x**2 - x**3)

# Parameters
h = 0.05
x = np.arange(0, 5 + h, h)  # from 0 to 5

# Forward difference
fwd = (f(x + h) - f(x)) / h

# Backward difference
bwd = (f(x) - f(x - h)) / h

# Central difference
cent = (f(x + h) - f(x - h)) / (2*h)

# Exact derivative (for comparison)
exact = f_prime_exact(x)

# Plot results
plt.figure(figsize=(10,6))
plt.plot(x, exact, 'k-', label='Exact derivative')
plt.plot(x, fwd, 'r--', label='Forward difference')
plt.plot(x, bwd, 'b--', label='Backward difference')
plt.plot(x, cent, 'g-', label='Central difference')

plt.title("Numerical Derivative Approximations")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
plt.grid(True)
plt.show()
