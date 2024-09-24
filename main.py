import numpy as np
import matplotlib.pyplot as plt

# Calculate step size to mimic np.linspace(0.001, 0.999, 100)
step_size = (0.999 - 0.001) / (100 - 1)  # Equivalent to np.linspace

# Define the sigmoid function output (which ranges between 0 and 1)
sigmoid_output = np.arange(0.001, 0.999 + step_size, step_size)  # Ensure last value is included

# Define the binary cross-entropy loss for both y = 0 and y = 1
def binary_cross_entropy_loss(y, sigmoid_output):
    return -(y * np.log(sigmoid_output) + (1 - y) * np.log(1 - sigmoid_output))

# Calculate loss for y = 0 and y = 1
loss_y0 = binary_cross_entropy_loss(0, sigmoid_output)
loss_y1 = binary_cross_entropy_loss(1, sigmoid_output)

# Plotting the loss functions
plt.figure(figsize=(8, 6))
plt.plot(sigmoid_output, loss_y0, label='y = 0', color='red')
plt.plot(sigmoid_output, loss_y1, label='y = 1', color='green')
plt.xlabel(r'Sigmoid Output (Predicted Probability) or $\text{sig[f[}(x, \phi)]]$')  # Use LaTeX syntax for phi
plt.ylabel(r'Binary Cross-Entropy Loss or $L$')  # Using LaTeX for 'L'
plt.title('Binary Cross-Entropy Loss as a Function of Sigmoid Output')
plt.legend()
plt.grid(True)
plt.show()