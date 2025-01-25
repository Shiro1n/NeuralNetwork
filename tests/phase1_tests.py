import numpy as np
from src.layer import Layer, LayerList
from src.optimizer import AdamOptimizer

# Mock activation function for testing
class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x, grad):
        return np.where(x > 0, 1, 0) * grad

# Mock loss functions for testing
def mean_squared_error(predicted, expected):
    return np.mean((predicted - expected) ** 2)

def mean_squared_error_derivative(predicted, expected):
    return 2 * (predicted - expected) / predicted.size

# Metric function
def accuracy(predicted, expected):
    predicted_classes = np.argmax(predicted, axis=1)
    expected_classes = np.argmax(expected, axis=1)
    return np.mean(predicted_classes == expected_classes)

# Callback function
def logging_callback(epoch, loss, metrics):
    print(f"[Callback] Epoch {epoch}: Loss={loss:.4f}, Metrics={metrics}")

# Learning rate scheduler
def lr_scheduler(epoch, current_lr):
    return current_lr * 0.9 if epoch % 5 == 0 else current_lr

# Test Layer functionality
def test_layer():
    print("=== Testing Layer ===")
    layer = Layer(input_size=5, output_size=3, alpha=0.01, activation_function=ReLU(), bias=True)
    inputs = np.random.rand(4, 5)  # Batch of 4 samples, each with 5 features
    outputs = layer(inputs)

    print("Forward Pass Output:")
    print(outputs)

    gradient = np.random.rand(4, 3)  # Mock gradient from the next layer
    backward_result = layer.back(gradient)
    print("Backward Pass Output (Gradient to Previous Layer):")
    print(backward_result)

    print("Weights Before Update:")
    print(layer.weights)

    layer.update()
    print("Weights After Update:")
    print(layer.weights)

# Test LayerList functionality
def test_layerlist():
    print("\n=== Testing LayerList ===")
    layer1 = Layer(input_size=5, output_size=4, alpha=0.01, activation_function=ReLU(), bias=True)
    layer2 = Layer(input_size=4, output_size=3, alpha=0.01, activation_function=ReLU(), bias=True)
    model = LayerList(layer1, layer2)

    inputs = np.random.rand(4, 5)  # Batch of 4 samples, each with 5 features
    outputs = model(inputs)
    print("Forward Pass Output:")
    print(outputs)

    gradient = np.random.rand(4, 3)  # Mock gradient from the next layer
    model.back(gradient)
    model.step()
    print("Weights Updated Successfully!")

# Test training functionality
def test_training():
    print("\n=== Testing Training ===")
    np.random.seed(42)
    input_data = np.random.rand(100, 5)  # 100 samples, 5 features
    expected = np.eye(3)[np.random.choice(3, 100)]  # 3 classes, one-hot encoded

    layer1 = Layer(input_size=5, output_size=4, alpha=0.01, activation_function=ReLU(), bias=True)
    layer2 = Layer(input_size=4, output_size=3, alpha=0.01, activation_function=ReLU(), bias=True)
    model = LayerList(layer1, layer2)

    model.fit(
        input_data,
        expected,
        batch_size=16,
        alpha=0.01,
        epochs=10,
        Loss_func=mean_squared_error,
        Loss_deriv_func=mean_squared_error_derivative,
        callbacks=[logging_callback],
        metrics=[accuracy],
        lr_scheduler=lr_scheduler,
        max_grad_norm=1.0,
    )

    # Assertions for model functionality
    test_input = np.random.rand(5, 5)  # 5 test samples
    test_output = model(test_input)
    assert test_output.shape == (5, 3), "Test output shape is incorrect."
    assert np.all(np.isfinite(test_output)), "Test output contains non-finite values."

    print("Training Completed Successfully!")


# Test save and load functionality
def test_save_load():
    print("\n=== Testing Save and Load ===")
    layer1 = Layer(input_size=5, output_size=4, alpha=0.01, activation_function=ReLU(), bias=True)
    layer2 = Layer(input_size=4, output_size=3, alpha=0.01, activation_function=ReLU(), bias=True)
    model = LayerList(layer1, layer2)

    # Save the model
    model.save("test_model.pkl")
    print("Model saved successfully.")

    # Load the model
    loaded_model = LayerList.load("test_model.pkl")
    print("Model loaded successfully.")

    # Verify the model is functional
    inputs = np.random.rand(4, 5)  # Batch of 4 samples, each with 5 features
    outputs = loaded_model(inputs)
    print("Loaded Model Forward Pass Output:")
    print(outputs)

# Run all tests
if __name__ == "__main__":
    test_layer()
    test_layerlist()
    test_training()
    test_save_load()