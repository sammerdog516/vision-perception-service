import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weight matrices
        self.w_input_hidden = np.random.normal(
            0.0, pow(self.input_size, -0.5),
            (self.hidden_size, self.input_size)
        )

        self.w_hidden_output = np.random.normal(
            0.0, pow(self.hidden_size, -0.5),
            (self.output_size, self.hidden_size)
        )

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, inputs):
        # Ensure column vector
        inputs = np.array(inputs, ndmin=2).T

        # Input -> hidden
        hidden_inputs = np.dot(self.w_input_hidden, inputs)
        hidden_outputs = self._sigmoid(hidden_inputs)

        # Hidden -> output
        final_inputs = np.dot(self.w_hidden_output, hidden_outputs)
        final_outputs = self._sigmoid(final_inputs)

        return hidden_outputs, final_outputs


    def backward(self, inputs, targets):
        # Forward pass
        hidden_outputs, final_outputs = self.forward(inputs)

        # Ensure column vectors
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        # Output layer error
        output_errors = targets - final_outputs
        
        # Hidden layer error
        hidden_errors = np.dot(self.w_hidden_output.T, output_errors)

        # Update hidden -> output weights
        self.w_hidden_output += self.learning_rate * np.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            hidden_outputs.T
        )

        # Update input -> hidden weights
        self.w_input_hidden += self.learning_rate * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            inputs.T
        )

    def predict(self, inputs):
        _, final_outputs = self.forward(inputs)
        return np.argmax(final_outputs)
    

nn = NeuralNetwork(784, 100, 10, 0.1)
dummy_input = np.zeros(784)
prediction = nn.predict(dummy_input)
print(prediction)