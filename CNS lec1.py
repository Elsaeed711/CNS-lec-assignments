def calculate_exponential(x, terms=10):
    result, factorial, power = 1, 1, 1
    for i in range(1, terms):
        factorial *= i
        power *= x
        result += power / factorial
    return result

def tanh_activation(x):
    return (calculate_exponential(x) - calculate_exponential(-x)) / (calculate_exponential(x) + calculate_exponential(-x))

# Given weights from the image
w1, w2 = 0.15, 0.20
w3, w4 = 0.25, 0.30
w5, w6 = 0.40, 0.45
w7, w8 = 0.50, 0.55

# Given biases from the image
b1, b2 = 0.35, 0.60

# Inputs and target outputs from the image
i1, i2 = 0.05, 0.10
target_o1, target_o2 = 0.01, 0.99

# Forward pass
h1 = tanh_activation(w1 * i1 + w2 * i2 + b1)
h2 = tanh_activation(w3 * i1 + w4 * i2 + b1)

o1 = tanh_activation(w5 * h1 + w6 * h2 + b2)
o2 = tanh_activation(w7 * h1 + w8 * h2 + b2)

# Calculate errors
E_o1 = 0.5 * (target_o1 - o1) ** 2
E_o2 = 0.5 * (target_o2 - o2) ** 2
total_loss = E_o1 + E_o2

# Print results
print(f"Hidden Outputs: h1 = {round(h1, 6)}, h2 = {round(h2, 6)}")
print(f"Output Values: o1 = {round(o1, 6)}, o2 = {round(o2, 6)}")
print(f"Losses: E_o1 = {round(E_o1, 6)}, E_o2 = {round(E_o2, 6)}")
print(f"Total Loss: {round(total_loss, 6)}")
