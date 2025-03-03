def generate_random(low, high, seed):
    seed = (seed * 9301 + 49297) % 233280
    return low + (seed / 233280.0) * (high - low)

def calculate_exponential(x, terms=10):
    result, factorial, power = 1, 1, 1
    for i in range(1, terms):
        factorial *= i
        power *= x
        result += power / factorial
    return result

def tanh_activation(x):
    return (calculate_exponential(x) - calculate_exponential(-x)) / (calculate_exponential(x) + calculate_exponential(-x))

seed = 42
weights = [generate_random(-0.5, 0.5, seed) for seed in [seed, seed+1, seed+2, seed+3, seed+4, seed+5]]
b1, b2 = 0.5, 0.7
inputs = [0.05, 0.1]
targets = [0.1, 0.99]

w1, w2, w3, w4, w5, w6 = weights
i1, i2 = inputs
target_o1, target_o2 = targets

# Forward pass
h1 = tanh_activation(w1 * i1 + w2 * i2 + b1)
h2 = tanh_activation(w3 * i1 + w4 * i2 + b1)

o1 = tanh_activation(w5 * h1 + w6 * h2 + b2)
o2 = tanh_activation(w5 * h1 + w6 * h2 + b2)


E_o1 = 0.5 * (target_o1 - o1) ** 2
E_o2 = 0.5 * (target_o2 - o2) ** 2
total_loss = E_o1 + E_o2

print(f"Hidden Outputs: h1 = {round(h1, 6)}, h2 = {round(h2, 6)}")
print(f"Output Values: o1 = {round(o1, 6)}, o2 = {round(o2, 6)}")
print(f"Losses: E_o1 = {round(E_o1, 6)}, E_o2 = {round(E_o2, 6)}")
print(f"Total Loss: {round(total_loss, 6)}")


delta_o1 = -(target_o1 - o1) * (1 - o1 ** 2)
delta_o2 = -(target_o2 - o2) * (1 - o2 ** 2)


delta_h1 = delta_o1 * w5 * (1 - h1 ** 2) + delta_o2 * w5 * (1 - h1 ** 2)
delta_h2 = delta_o1 * w6 * (1 - h2 ** 2) + delta_o2 * w6 * (1 - h2 ** 2)


grad_w1 = delta_h1 * i1 * (1 - h1 ** 2)
grad_w2 = delta_h1 * i2 * (1 - h1 ** 2)
grad_w3 = delta_h2 * i1 * (1 - h2 ** 2)
grad_w4 = delta_h2 * i2 * (1 - h2 ** 2)
grad_w5 = delta_o1 * h1 * (1 - o1 ** 2)
grad_w6 = delta_o2 * h2 * (1 - o2 ** 2)


grad_b1 = delta_h1 * (1 - h1 ** 2) + delta_h2 * (1 - h2 ** 2)
grad_b2 = delta_o1 + delta_o2


learning_rate = 0.1


w1 -= learning_rate * grad_w1
w2 -= learning_rate * grad_w2
w3 -= learning_rate * grad_w3
w4 -= learning_rate * grad_w4
w5 -= learning_rate * grad_w5
w6 -= learning_rate * grad_w6
b1 -= learning_rate * grad_b1
b2 -= learning_rate * grad_b2


print("\nUpdated Weights and Biases:")
print(f"w1 = {w1}, w2 = {w2}, w3 = {w3}, w4 = {w4}, w5 = {w5}, w6 = {w6}")
print(f"b1 = {b1}, b2 = {b2}")
