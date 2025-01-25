import pickle
import numpy as np

from src.optimizer import AdamOptimizer


class Layer:
    """
       Represents a single layer in a neural network.
       Performs forward and backward passes and updates weights and biases.
    """

    def __init__(self, input_size, output_size, alpha, bias=False, activation_function=None, optimizer=None,
                 lambda_=0.0, initialization="xavier"):
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
        self.trainable = True
        self.optimizer = optimizer if optimizer else AdamOptimizer(alpha=alpha)

        if initialization == "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
        elif initialization == "he":
            limit = np.sqrt(6 / input_size)
        else:
            limit = 1.0

        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros((1, output_size)) if bias else None
        self.bias_optimizer = AdamOptimizer(alpha=alpha) if bias else None
        self.reset_state()

    def reset_state(self):
        self.update_matrix = None
        self.bias_grad = None
        self.current_inputs = None
        self.pre_activated_output = None

    def clip_gradients(self, max_norm):
        """
        Clip the gradients to a maximum norm.

        Parameters:
            max_norm (float): The maximum allowable norm for the gradients.
        """
        if self.update_matrix is not None:
            norm = np.linalg.norm(self.update_matrix)
            if norm > max_norm:
                self.update_matrix *= max_norm / norm

    def set_alpha(self, new_alpha):
        """
        Update the learning rate (alpha) for the layer.

        Parameters:
            new_alpha (float): New learning rate value.
        """
        self.alpha = new_alpha

    def __call__(self, inputs):
        """
            Perform the forward pass for this layer.

            Parameters:
                inputs (np.ndarray): Input data of shape (batch_size, input_size).

            Returns:
                np.ndarray: Output of the layer after applying weights, bias, and activation.
        """
        self.current_inputs = inputs
        output = np.dot(inputs, self.weights)
        if self.bias is not None:
            # Ensure bias has correct shape for broadcasting
            output = output + np.broadcast_to(self.bias, output.shape)
        if self.activation_function:
            self.pre_activated_output = output.copy()
            output = self.activation_function(output)
        return output

    def back(self, grad):
        """
        Perform the backward pass for this layer.

        Parameters:
            grad (np.ndarray): Gradient from the subsequent layer.

        Returns:
            np.ndarray: Gradient propagated to the previous layer.
        """
        if self.activation_function:
            grad = self.activation_function.derivative(self.pre_activated_output, grad)
        if self.trainable:
            self.update_matrix = np.dot(self.current_inputs.T, grad)
            if self.bias is not None:
                self.bias_grad = np.sum(grad, axis=0, keepdims=True)
            return np.dot(grad, self.weights.T)
        return grad

    def update(self, max_grad_norm=None):
        """
        Update the weights and biases using the optimizer and L2 regularization.

        Parameters:
            max_grad_norm (float, optional): Maximum norm for gradient clipping. If None, no clipping is applied.
        """
        if not self.trainable or self.update_matrix is None:
            return

        batch_size = float(self.current_inputs.shape[0])
        reg_term = (self.lambda_ * self.weights) / batch_size
        gradients = self.update_matrix / batch_size + reg_term

        if max_grad_norm:
            grad_norm = np.linalg.norm(gradients)
            if grad_norm > max_grad_norm:
                gradients *= max_grad_norm / grad_norm

        self.weights = self.optimizer.update(self.weights, gradients)

        if self.bias is not None and self.bias_grad is not None:
            bias_grad = self.bias_grad / batch_size
            if max_grad_norm:
                bias_norm = np.linalg.norm(bias_grad)
                if bias_norm > max_grad_norm:
                    bias_grad *= max_grad_norm / bias_norm
            self.bias = self.bias_optimizer.update(self.bias, bias_grad)

        self.reset_state()


class LayerList:
    """
    Represents a collection of layers in a neural network.
    Manages forward and backward passes, as well as training.
    """

    def __init__(self, *layers):
        """
        Initialize the LayerList with one or more layers.

        Parameters:
            *Layers: Variable number of Layer objects to include in the model.
        """
        self.model = list(layers)
        self._validate_layers()

    def _validate_layers(self):
        for i in range(1, len(self.model)):
            if self.model[i - 1].output_size != self.model[i].input_size:
                raise ValueError(f"Layer size mismatch at position {i}")

    def append(self, *layers):
        """
        Add one or more layers to the model.

        Parameters:
            *layers: Variable number of Layer objects to append.
        """
        for layer in layers:
            if self.model and self.model[-1].output_size != layer.input_size:
                raise ValueError(f"Layer size mismatch: expected {self.model[-1].output_size}, got {layer.input_size}")
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
        assert model_input.shape[1] == self.model[0].input_size, \
            f"Input data has {model_input.shape[1]} features, expected {self.model[0].input_size}."
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

    def step(self, max_grad_norm=None):
        """
        Update weights and biases for all layers in the model.

        Parameters:
            max_grad_norm (float): Maximum allowable norm for gradients. Defaults to None (no clipping).
        """
        for layer in self.model:
            layer.update(max_grad_norm=max_grad_norm)

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

    def fit(self, input_data, expected, batch_size, alpha, epochs, Loss_func, Loss_deriv_func, callbacks=None,
            metrics=None, lr_scheduler=None, verbose=True, max_grad_norm=None):
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
            callbacks (list): List of callback functions to execute at the end of each epoch.
            metrics (list): List of metric functions to evaluate during training.
            lr_scheduler (callable): Function to adjust the learning rate dynamically.
            verbose (bool): If True, print training progress. Defaults to True.
            max_grad_norm (float): Maximum allowable gradient norm for clipping. Defaults to None.
        """
        # Parameter validation
        assert input_data.shape[0] == expected.shape[0], "Input and expected data must have the same number of samples."
        assert input_data.shape[1] == self.model[0].input_size, \
            f"Expected input size {self.model[0].input_size}, but got {input_data.shape[1]}."
        assert batch_size > 0, "Batch size must be positive."
        assert callable(Loss_func), "Loss_func must be a callable function."
        assert callable(Loss_deriv_func), "Loss_deriv_func must be a callable function."

        self.set_alpha(alpha)
        callbacks = callbacks or []
        metrics = metrics or []

        for e in range(epochs):
            try:
                # Adjust learning rate at the start of the epoch
                if lr_scheduler:
                    alpha = lr_scheduler(e + 1, alpha)
                    self.set_alpha(alpha)

                total_loss = 0
                metric_results = {metric.__name__: 0 for metric in metrics}

                # Create batches
                batched_input, batched_expected = LayerList.batch(input_data, expected, batch_size)
                num_batches = len(batched_input)

                # Process each batch
                for i, (batch_inp, batch_exp) in enumerate(zip(batched_input, batched_expected)):
                    model_output = self(batch_inp)
                    batch_loss = Loss_func(model_output, batch_exp)

                    # Handle potential NaN or Inf in loss
                    if not np.isfinite(batch_loss):
                        raise ValueError(f"Non-finite loss detected at epoch {e + 1}, batch {i + 1}: {batch_loss}")

                    total_loss += batch_loss
                    self.back(Loss_deriv_func(model_output, batch_exp))
                    self.step(max_grad_norm=max_grad_norm)

                    # Track metrics
                    for metric in metrics:
                        metric_results[metric.__name__] += metric(model_output, batch_exp)

                    # Optionally log batch details
                    if verbose and verbose > 1:
                        print(f"  Batch {i + 1}/{num_batches}, Loss: {batch_loss:.4f}")

                # Normalize metrics
                metric_results = {k: v / num_batches for k, v in metric_results.items()}

                # Epoch summary
                if verbose:
                    print(f"Epoch {e + 1}/{epochs}, Loss: {total_loss / num_batches}, Metrics: {metric_results}")

                # Execute callbacks
                for callback in callbacks:
                    callback(epoch=e + 1, loss=total_loss / num_batches, metrics=metric_results)

            except Exception as err:
                print(f"Error during training at epoch {e + 1}: {err}")
                break

    def predict(self, inputs):
        """
        Generate predictions for the given input data.

        Parameters:
            inputs (np.ndarray): Input data to predict.

        Returns:
            list or np.ndarray: Predicted output(s).
        """
        output = self(inputs)
        return output if isinstance(output, np.ndarray) else list(output)

    def freeze_layer(self, index):
        """
        Freeze the layer at the specified index.
        """
        self.model[index].trainable = False

    def unfreeze_layer(self, index):
        """
        Unfreeze the layer at the specified index.
        """
        self.model[index].trainable = True

    def freeze_all(self):
        for layer in self.model:
            layer.trainable = False

    def unfreeze_all(self):
        for layer in self.model:
            layer.trainable = True

    def save(self, filepath):
        """Save the model parameters to a file."""
        params = []
        for layer in self.model:
            layer_params = {
                'input_size': layer.input_size,
                'output_size': layer.output_size,
                'alpha': layer.alpha,
                'bias': layer.bias is not None,
                'activation_function': layer.activation_function,
                'lambda_': layer.lambda_
            }
            # Save weights and bias separately
            layer_params['_weights'] = layer.weights
            if layer.bias is not None:
                layer_params['_bias_values'] = layer.bias

            # Save optimizer states if they exist
            if layer.optimizer:
                layer_params['_optimizer_state'] = layer.optimizer.__dict__
            if layer.bias_optimizer:
                layer_params['_bias_optimizer_state'] = layer.bias_optimizer.__dict__

            params.append(layer_params)

        with open(filepath, 'wb') as f:
            pickle.dump(params, f)

    @staticmethod
    def load(filepath):
        """Load model parameters from a file."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)

        layers = []
        for layer_params in params:
            # Extract weights and bias
            weights = layer_params.pop('_weights', None)
            bias_values = layer_params.pop('_bias_values', None)
            optimizer_state = layer_params.pop('_optimizer_state', None)
            bias_optimizer_state = layer_params.pop('_bias_optimizer_state', None)

            # Create layer
            layer = Layer(**layer_params)

            # Restore weights and bias
            if weights is not None:
                layer.weights = weights
            if bias_values is not None:
                layer.bias = bias_values

            # Restore optimizer states
            if optimizer_state:
                layer.optimizer = AdamOptimizer()
                layer.optimizer.__dict__.update(optimizer_state)
            if bias_optimizer_state:
                layer.bias_optimizer = AdamOptimizer()
                layer.bias_optimizer.__dict__.update(bias_optimizer_state)

            layers.append(layer)

        return LayerList(*layers)
