from random import shuffle
import numpy as np

from src.optimizer import AdamOptimizer


class Layer:
    """
       Represents a single layer in a neural network.
       Performs forward and backward passes and updates weights and biases.
    """

    def __init__(self, input_size, output_size, alpha, bias=False, activation_function=None, optimizer=None,
                 lambda_=0.01, initialization="xavier"):
        """
              Initialize the layer with weights, biases, and other properties.

              Parameters:
                  input_size (int): Number of input neurons.
                  output_size (int): Number of output neurons.
                  alpha (float): Learning rate for weight updates.
                  bias (bool): Whether to include a bias term. Defaults to False.
                  activation_function (callable): Activation function to apply. Defaults to None.
                  optimizer (callable): Optimizer for updating weights. Defaults to AdamOptimizer.
                  lambda_ (float): L2 regularization strength. Defaults to 0.01.
                  initialization (str): Weight initialization method ('xavier', 'he', or 'uniform'). Defaults to 'xavier'.
              """
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.lambda_ = lambda_
        self.activation_function = activation_function

        # Validate initialization method
        valid_initializations = ["xavier", "he", "uniform"]
        assert initialization in valid_initializations, f"Invalid initialization method. Choose from {valid_initializations}."

        self.optimizer = optimizer if optimizer else AdamOptimizer(alpha=alpha)

        # Initialize weights
        if initialization == "xavier":
            self.weights = np.random.uniform(
                -np.sqrt(6 / (self.input_size + self.output_size)),
                np.sqrt(6 / (self.input_size + self.output_size)),
                (self.input_size, self.output_size),
            )
        elif initialization == "he":
            self.weights = np.random.uniform(
                -np.sqrt(6 / self.input_size),
                np.sqrt(6 / self.input_size),
                (self.input_size, self.output_size),
            )
        else:  # Default to uniform random initialization
            self.weights = np.random.uniform(-1, 1, (self.input_size, self.output_size))

        # Initialize bias
        self.bias = np.random.rand(self.output_size) if bias else None

        # Intermediate variables for backpropagation
        self.update_matrix = None
        self.current_inputs = None

    def set_alpha(self, new_alpha):
        """
        Update the learning rate (alpha) for the layer.

        Parameters:
            new_alpha (float): New learning rate value.
        """
        self.alpha = new_alpha

    def __call__(self, Layer_inputs):
        """
            Perform the forward pass for this layer.

            Parameters:
                Layer_inputs (np.ndarray): Input data of shape (batch_size, input_size).

            Returns:
                np.ndarray: Output of the layer after applying weights, bias, and activation.
        """
        self.current_inputs = np.copy(Layer_inputs)
        layer_output = Layer_inputs @ self.weights
        if self.bias is not None:
            layer_output += self.bias
        if self.activation_function:
            self.pre_activated_output = np.copy(layer_output)
            layer_output = self.activation_function(layer_output)
        return layer_output

    def back(self, ret):
        """
        Perform the backward pass for this layer.

        Parameters:
            ret (np.ndarray): Gradient from the subsequent layer.

        Returns:
            np.ndarray: Gradient propagated to the previous layer.
        """
        if self.activation_function is not None:
            ret = self.activation_function.derivative(self.pre_activated_output, ret)
        self.update_matrix = self.current_inputs.T @ ret
        return ret @ self.weights.T

    def update(self):
        """
        Update the weights and biases using the optimizer and L2 regularization.
        """
        if self.update_matrix is not None:
            reg_term = self.lambda_ * self.weights / self.current_inputs.shape[0]  # Scale by batch size
            self.weights = self.optimizer.update(self.weights, self.update_matrix + reg_term)
        if self.bias is not None:
            bias_gradient = np.sum(self.update_matrix, axis=0)
            self.bias = self.optimizer.update(self.bias, bias_gradient)
        # Reset intermediate values
        self.update_matrix = None
        self.current_inputs = None
        self.pre_activated_output = None


class LayerList:
    """
    Represents a collection of layers in a neural network.
    Manages forward and backward passes, as well as training.
    """

    def __init__(self, *Layers):
        """
        Initialize the LayerList with one or more layers.

        Parameters:
            *Layers: Variable number of Layer objects to include in the model.
        """
        self.model = list(Layers)

    def append(self, *Layers):
        """
        Add one or more layers to the model.

        Parameters:
            *Layers: Variable number of Layer objects to append.
        """
        for layer in Layers:
            self.model.append(layer)

    def set_alpha(self, new_alpha):
        """
        Update the learning rate (alpha) for all layers in the model.

        Parameters:
            new_alpha (float): New learning rate value.
        """
        for layer in self.model:
            layer.set_alpha(new_alpha)

    def __call__(self, model_input):
        """
        Perform the forward pass through all layers in the model.

        Parameters:
            model_input (np.ndarray): Input data for the model.

        Returns:
            np.ndarray: Final output of the model.
        """
        for layer in self.model:
            model_input = layer(model_input)
        return model_input

    def back(self, error):
        """
        Perform the backward pass through all layers in reverse order.

        Parameters:
            error (np.ndarray): Gradient from the output layer.
        """
        for layer in reversed(self.model):
            error = layer.back(error)

    def step(self):
        """
        Update weights and biases for all layers in the model.
        """
        for layer in self.model:
            layer.update()

    @staticmethod
    def batch(input_data, expected, batch_size):
        """
        Split input data and expected output into batches.

        Parameters:
            input_data (np.ndarray): Input data to split.
            expected (np.ndarray): Expected output to split.
            batch_size (int): Number of samples per batch.

        Returns:
            tuple: Two lists of batches (input_batches, expected_batches).
        """
        indices = np.arange(input_data.shape[0])
        np.random.shuffle(indices)
        return (
            np.array_split(input_data[indices], len(indices) // batch_size),
            np.array_split(expected[indices], len(indices) // batch_size),
        )

    def fit(self, input_data, expected, batch_size, alpha, epochs, Loss_func, Loss_deriv_func):
        """
        Train the model using the given data, loss function, and learning rate.

        Parameters:
            input_data (np.ndarray): Training input data.
            expected (np.ndarray): Training output data.
            batch_size (int): Number of samples per batch.
            alpha (float): Learning rate for training.
            epochs (int): Number of training epochs.
            Loss_func (callable): Loss function to compute the error.
            Loss_deriv_func (callable): Derivative of the loss function for backpropagation.
        """
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
        """
        Generate predictions for the given input data.

        Parameters:
            inputs (np.ndarray): Input data to predict.

        Returns:
            np.ndarray: Predicted output.
        """
        return self(inputs)
