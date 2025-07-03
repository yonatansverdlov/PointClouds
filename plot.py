import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(epsilon):
    term1 = 2 * (np.sqrt(1 + epsilon**2) - 1)
    term2 = 2 * np.sqrt(2 - (2 / np.sqrt(1 + epsilon**2)))
    term3 = -2 * epsilon
    return term1 + term2 + term3

# Define the range for epsilon
M = int(1e5)
epsilon_values = np.linspace(1e-5, 100, M)  # Range from 0 to 1
y_values = f(epsilon_values)
print(min(y_values))                                                    
# Plot the function                         
plt.figure(figsize=(8, 6))
plt.plot(epsilon_values, y_values, label=r"$2\cdot (\sqrt{1+\epsilon^2}-1) + 2\cdot \sqrt{2-\frac{2}{\sqrt{1+\epsilon^2}}} - 2\cdot\epsilon$")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8, label="y = 0")
plt.title("Plot of the Function", fontsize=14)
plt.xlabel(r"$\epsilon$", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.5)
plt.show()
