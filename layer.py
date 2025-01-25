import numpy as np

class Layer:
    def __init__(self, input_size, output_size, alpha, bias=False, activation_function=None):
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.activation_function = activation_function

        # Initialize weights
        self.weights = np.random.uniform(-1, 1, (self.input_size, self.output_size))

        # Initialize bias
        if bias:
            self.bias = np.random.rand(self.output_size)
        else:
            self.bias = None

    def set_alpha(self, new_alpha):
        self.alpha = new_alpha

    def __call__(self, Layer_inputs):
        # Linear transformation
        layer_output = Layer_inputs @ self.weights
        if self.bias is not None:
            layer_output += self.bias  # Add bias
        if self.activation_function:
            layer_output = self.activation_function(layer_output)  # Apply activation
        return layer_output


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
        intermediate_results = model_input
        for layer in self.model:
            intermediate_results = layer(intermediate_results)
        return intermediate_results
