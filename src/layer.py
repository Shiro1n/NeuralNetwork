from random import shuffle
import numpy as np

class Layer:
    def __init__(self, input_size, output_size, alpha, bias=False, activation_function=None):
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.activation_function = activation_function

        self.weights = np.random.uniform(-1, 1, (self.input_size, self.output_size))
        self.bias = np.random.rand(self.output_size) if bias else None

        self.update_matrix = None
        self.current_inputs = None
        self.pre_activated_output = None

    def set_alpha(self, new_alpha):
        self.alpha = new_alpha

    def __call__(self, Layer_inputs):
        self.current_inputs = np.copy(Layer_inputs)
        layer_output = Layer_inputs @ self.weights
        if self.bias is not None:
            layer_output += self.bias
        if self.activation_function:
            self.pre_activated_output = np.copy(layer_output)
            layer_output = self.activation_function(layer_output)
        return layer_output

    def back(self, ret):
        if self.activation_function is not None:
            ret = self.activation_function.derivative(self.pre_activated_output, ret)
        self.update_matrix = self.current_inputs.T @ ret
        return ret @ self.weights.T

    def update(self):
        self.weights -= self.alpha * self.update_matrix
        if self.bias is not None:
            self.bias -= self.alpha * np.sum(self.pre_activated_output, axis=0)
        self.update_matrix = None
        self.current_inputs = None
        self.pre_activated_output = None


class LayerList:
    def __init__(self, *Layers):
        self.model = list(Layers)

    def append(self, *Layers):
        for layer in Layers:
            self.model.append(layer)

    def set_alpha(self, new_alpha):
        for layer in self.model:
            layer.set_alpha(new_alpha)

    def __call__(self, model_input):
        for layer in self.model:
            model_input = layer(model_input)
        return model_input

    def back(self, error):
        for layer in reversed(self.model):
            error = layer.back(error)

    def step(self):
        for layer in self.model:
            layer.update()

    @staticmethod
    def batch(input_data, expected, batch_size):
        indices = np.arange(input_data.shape[0])
        np.random.shuffle(indices)
        return (
            np.array_split(input_data[indices], len(indices) // batch_size),
            np.array_split(expected[indices], len(indices) // batch_size),
        )

    def fit(self, input_data, expected, batch_size, alpha, epochs, Loss_func, Loss_deriv_func):
        self.set_alpha(alpha)
        for e in range(epochs):
            total_loss = 0
            batched_input, batched_expected = LayerList.batch(input_data, expected, batch_size)
            for batch_inp, batch_exp in zip(batched_input, batched_expected):
                model_output = self(batch_inp)
                total_loss += Loss_func(model_output, batch_exp)
                self.back(Loss_deriv_func(model_output, batch_exp))
                self.step()
            print(f"Epoch {e + 1}/{epochs}, Loss: {total_loss / len(batched_input)}")

    def predict(self, inputs):
        return self(inputs)
