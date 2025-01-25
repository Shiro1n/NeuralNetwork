import numpy as np
from src.activations import ReLU, Sigmoid
from src.layer import Layer, LayerList


# Example Loss Functions and Gradients
def mean_squared_error(predicted, expected):
    return np.mean((predicted - expected) ** 2)


def mean_squared_error_derivative(predicted, expected):
    return 2 * (predicted - expected) / predicted.shape[0]


# Test Functionality
def test_layer():
    print("=== Testing Layer ===")
    input_data = np.array([[0.5, -0.5], [1.0, -1.0]])  # Input data
    layer = Layer(input_size=2, output_size=3, alpha=0.01, bias=True, activation_function=ReLU())

    # Forward Pass
    output = layer(input_data)
    print("Forward Output:\n", output)

    # Backward Pass
    gradient = np.ones_like(output)  # Dummy gradient for testing
    back_output = layer.back(gradient)
    print("Backward Output (Gradient to Previous Layer):\n", back_output)

    # Weight Update
    print("Weights Before Update:\n", layer.weights)
    layer.update()
    print("Weights After Update:\n", layer.weights)
    print("=" * 50)


def test_layer_list():
    print("=== Testing LayerList ===")
    # Create a simple network
    layer1 = Layer(input_size=2, output_size=3, alpha=0.01, bias=True, activation_function=ReLU())
    layer2 = Layer(input_size=3, output_size=1, alpha=0.01, bias=True, activation_function=Sigmoid())
    model = LayerList(layer1, layer2)

    # Forward Pass
    input_data = np.array([[0.5, -0.5], [1.0, -1.0]])  # Input data
    output = model(input_data)
    print("Forward Output:\n", output)

    # Backward Pass
    expected = np.array([[0.8], [0.2]])  # Example target output
    error = mean_squared_error_derivative(output, expected)
    model.back(error)

    # Weight Update
    print("Weights Before Update (Layer 1):\n", layer1.weights)
    print("Weights Before Update (Layer 2):\n", layer2.weights)
    model.step()
    print("Weights After Update (Layer 1):\n", layer1.weights)
    print("Weights After Update (Layer 2):\n", layer2.weights)
    print("=" * 50)


def test_training():
    print("=== Testing Training ===")
    # Create a simple network
    layer1 = Layer(input_size=2, output_size=3, alpha=0.01, bias=True, activation_function=ReLU())
    layer2 = Layer(input_size=3, output_size=1, alpha=0.01, bias=True, activation_function=Sigmoid())
    model = LayerList(layer1, layer2)

    # Generate synthetic data
    np.random.seed(42)
    input_data = np.random.rand(100, 2)  # 100 samples, 2 features
    expected = np.random.rand(100, 1)  # 100 target outputs

    # Training
    model.fit(
        input_data=input_data,
        expected=expected,
        batch_size=10,
        alpha=0.01,
        epochs=20,
        Loss_func=mean_squared_error,
        Loss_deriv_func=mean_squared_error_derivative,
    )

    # Predict
    predictions = model.predict(input_data[:5])  # Predict on first 5 samples
    print("Predictions on Test Data:\n", predictions)
    print("=" * 50)


# Run Tests
if __name__ == "__main__":
    test_layer()
    test_layer_list()
    test_training()
