import random
import math

class SimpleNeuron:
    def __init__(self):
        self.bias = random.uniform(-0.1, 0.1)
        self.weights = []
        self.input_sum = 0
        self.activation = 0

    def forward(self, inputs):
        self.input_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.activation = SimpleNeuron.tanh(self.input_sum)
        return self.activation

    @staticmethod
    def tanh(x):
        x = max(min(x, 10), -10)
        return (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

    @staticmethod
    def tanh_derivative(output):
        return 1 - output * output

    @staticmethod
    def softmax(logits):
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        total = sum(exps)
        return [e / total for e in exps]

class RecurrentBlock:
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.neurons = [SimpleNeuron() for _ in range(hidden_dim)]
        for neuron in self.neurons:
            neuron.weights = [random.uniform(-0.1, 0.1) for _ in range(input_dim)]
        self.recurrent_weights = [[random.uniform(-0.1, 0.1) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.hidden_state = [0] * hidden_dim

    def forward(self, inputs):
        prev_state = self.hidden_state.copy()
        new_state = []
        for idx, neuron in enumerate(self.neurons):
            input_contrib = sum(w * i for w, i in zip(neuron.weights, inputs))
            recur_contrib = sum(w * h for w, h in zip(self.recurrent_weights[idx], prev_state))
            neuron.input_sum = input_contrib + recur_contrib + neuron.bias
            neuron.activation = SimpleNeuron.tanh(neuron.input_sum)
            new_state.append(neuron.activation)
        self.hidden_state = new_state
        return new_state

class OutputBlock:
    def __init__(self, hidden_dim, output_dim):
        self.neurons = [SimpleNeuron() for _ in range(output_dim)]
        for neuron in self.neurons:
            neuron.weights = [random.uniform(-0.1, 0.1) for _ in range(hidden_dim)]

    def forward(self, hidden_state):
        logits = [neuron.forward(hidden_state) for neuron in self.neurons]
        probabilities = SimpleNeuron.softmax([n.input_sum for n in self.neurons])
        for i, neuron in enumerate(self.neurons):
            neuron.activation = probabilities[i]
        return probabilities

class WordPredictorRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.hidden_layer = RecurrentBlock(input_size, hidden_size)
        self.output_layer = OutputBlock(hidden_size, output_size)
        self.lr = learning_rate

    def forward_pass(self, sequence):
        inputs, hidden_states, outputs = [], [], []
        self.hidden_layer.hidden_state = [0] * self.hidden_layer.hidden_dim
        for x in sequence:
            hidden = self.hidden_layer.forward(x)
            out = self.output_layer.forward(hidden)
            inputs.append(x)
            hidden_states.append(hidden)
            outputs.append(out)
        return inputs, hidden_states, outputs

    def backward_pass(self, inputs, hidden_states, outputs, targets):
        dW_input = [[0] * len(inputs[0]) for _ in range(self.hidden_layer.hidden_dim)]
        dW_recurrent = [[0] * self.hidden_layer.hidden_dim for _ in range(self.hidden_layer.hidden_dim)]
        db_hidden = [0] * self.hidden_layer.hidden_dim
        dW_output = [[0] * self.hidden_layer.hidden_dim for _ in range(len(outputs[0]))]
        db_output = [0] * len(outputs[0])
        dh_next = [0] * self.hidden_layer.hidden_dim
        for t in reversed(range(len(inputs))):
            dy = outputs[t][:]
            target_idx = targets[t].index(1)
            dy[target_idx] -= 1
            for o in range(len(dy)):
                for h in range(self.hidden_layer.hidden_dim):
                    dW_output[o][h] += dy[o] * hidden_states[t][h]
                db_output[o] += dy[o]
            dh = [0] * self.hidden_layer.hidden_dim
            for h in range(self.hidden_layer.hidden_dim):
                for o in range(len(dy)):
                    dh[h] += dy[o] * self.output_layer.neurons[o].weights[h]
            for h in range(self.hidden_layer.hidden_dim):
                dh[h] += dh_next[h]
            dtanh = [dh[h] * SimpleNeuron.tanh_derivative(hidden_states[t][h]) for h in range(self.hidden_layer.hidden_dim)]
            for h in range(self.hidden_layer.hidden_dim):
                db_hidden[h] += dtanh[h]
                for i in range(len(inputs[t])):
                    dW_input[h][i] += dtanh[h] * inputs[t][i]
                if t > 0:
                    for hh in range(self.hidden_layer.hidden_dim):
                        dW_recurrent[h][hh] += dtanh[h] * hidden_states[t-1][hh]
            dh_next = [0] * self.hidden_layer.hidden_dim
            for h in range(self.hidden_layer.hidden_dim):
                for hh in range(self.hidden_layer.hidden_dim):
                    dh_next[hh] += dtanh[h] * self.hidden_layer.recurrent_weights[h][hh]
        return dW_input, dW_recurrent, dW_output, db_hidden, db_output

    def update_parameters(self, dW_input, dW_recurrent, dW_output, db_hidden, db_output):
        for h, neuron in enumerate(self.hidden_layer.neurons):
            for i in range(len(neuron.weights)):
                neuron.weights[i] -= self.lr * dW_input[h][i]
            neuron.bias -= self.lr * db_hidden[h]
        for h in range(len(self.hidden_layer.recurrent_weights)):
            for hh in range(len(self.hidden_layer.recurrent_weights[h])):
                self.hidden_layer.recurrent_weights[h][hh] -= self.lr * dW_recurrent[h][hh]
        for o, neuron in enumerate(self.output_layer.neurons):
            for h in range(len(neuron.weights)):
                neuron.weights[h] -= self.lr * dW_output[o][h]
            neuron.bias -= self.lr * db_output[o]

    def train(self, data_seq, target_seq, epochs=1000):
        for epoch in range(epochs):
            inputs, hidden_states, outputs = self.forward_pass(data_seq)
            loss = sum(-math.log(outputs[t][target_seq[t].index(1)]) for t in range(len(data_seq)))
            dW_input, dW_recurrent, dW_output, db_hidden, db_output = self.backward_pass(inputs, hidden_states, outputs, target_seq)
            self.update_parameters(dW_input, dW_recurrent, dW_output, db_hidden, db_output)
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict(self, input_vector, prev_hidden=None):
        if prev_hidden:
            self.hidden_layer.hidden_state = prev_hidden
        hidden = self.hidden_layer.forward(input_vector)
        prediction = self.output_layer.forward(hidden)
        return prediction, hidden

if __name__ == "__main__":
    vocab = ["red", "green", "blue", "yellow"]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    def one_hot(index, size):
        vec = [0] * size
        vec[index] = 1
        return vec

    train_inputs = [
        one_hot(word2idx["red"], vocab_size),
        one_hot(word2idx["green"], vocab_size),
        one_hot(word2idx["blue"], vocab_size)
    ]

    train_targets = [
        one_hot(word2idx["green"], vocab_size),
        one_hot(word2idx["blue"], vocab_size),
        one_hot(word2idx["yellow"], vocab_size)
    ]

    model = WordPredictorRNN(input_size=vocab_size, hidden_size=8, output_size=vocab_size, learning_rate=0.01)

    print("\nTraining RNN to predict the 4th word...\n")
    model.train(train_inputs, train_targets, epochs=1000)

    print("\nTesting:")
    hidden = None
    words = ["red", "green", "blue"]
    for word in words:
        x = one_hot(word2idx[word], vocab_size)
        pred, hidden = model.predict(x, hidden)
        pred_idx = pred.index(max(pred))
        print(f"Input: {word} | Predicted Next: {idx2word[pred_idx]}")
