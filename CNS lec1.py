import math

def tanh_activation(x):
    return math.tanh(x)

# Weights and biases from the picture
w1, w2, w3, w4, w5, w6, w7, w8 = 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55
b1, b2 = 0.35, 0.60

inputs = [0.05, 0.10]
targets = [0.01, 0.99]

i1, i2 = inputs
target_o1, target_o2 = targets

# Forward pass
h1 = tanh_activation(w1 * i1 + w3 * i2 + b1)
h2 = tanh_activation(w2 * i1 + w4 * i2 + b1)

o1 = tanh_activation(w5 * h1 + w7 * h2 + b2)
o2 = tanh_activation(w6 * h1 + w8 * h2 + b2)

E_o1 = 0.5 * (target_o1 - o1) ** 2
E_o2 = 0.5 * (target_o2 - o2) ** 2
total_loss = E_o1 + E_o2

print(f"Hidden Outputs: h1 = {round(h1, 6)}, h2 = {round(h2, 6)}")
print(f"Output Values: o1 = {round(o1, 6)}, o2 = {round(o2, 6)}")
print(f"Losses: E_o1 = {round(E_o1, 6)}, E_o2 = {round(E_o2, 6)}")
print(f"Total Loss: {round(total_loss, 6)}")

# Backpropagation
delta_o1 = -(target_o1 - o1) * (1 - o1 ** 2)
delta_o2 = -(target_o2 - o2) * (1 - o2 ** 2)

delta_h1 = (delta_o1 * w5 + delta_o2 * w6) * (1 - h1 ** 2)
delta_h2 = (delta_o1 * w7 + delta_o2 * w8) * (1 - h2 ** 2)

grad_w1 = delta_h1 * i1
grad_w2 = delta_h2 * i1
grad_w3 = delta_h1 * i2
grad_w4 = delta_h2 * i2
grad_w5 = delta_o1 * h1
grad_w6 = delta_o1 * h2
grad_w7 = delta_o2 * h1
grad_w8 = delta_o2 * h2

grad_b1 = delta_h1 + delta_h2
grad_b2 = delta_o1 + delta_o2

learning_rate = 0.1

# Weight updates
w1 -= learning_rate * grad_w1
w2 -= learning_rate * grad_w2
w3 -= learning_rate * grad_w3
w4 -= learning_rate * grad_w4
w5 -= learning_rate * grad_w5
w6 -= learning_rate * grad_w6
w7 -= learning_rate * grad_w7
w8 -= learning_rate * grad_w8
b1 -= learning_rate * grad_b1
b2 -= learning_rate * grad_b2

print("\nUpdated Weights and Biases:")
print(f"w1 = {w1}, w2 = {w2}, w3 = {w3}, w4 = {w4}, w5 = {w5}, w6 = {w6}, w7 = {w7}, w8 = {w8}")
print(f"b1 = {b1}, b2 = {b2}")
