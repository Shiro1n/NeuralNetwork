import numpy as np

from src.activations import ReLU, Sigmoid
from src.layer import Layer, LayerList


# 1. Test single layer
print("=== Single Layer Test ===")
layer = Layer(input_size=3, output_size=2, alpha=0.01, bias=True, activation_function=ReLU())
inputs = np.array([[1, 2, 3]])  # Shape: (1, 3)
print("Input:\n", inputs)
output = layer(inputs)
print("Output:\n", output)

# 2. Test multiple layers with LayerList
print("\n=== LayerList Test ===")
layer1 = Layer(input_size=3, output_size=4, alpha=0.01, bias=True, activation_function=ReLU())
layer2 = Layer(input_size=4, output_size=2, alpha=0.01, bias=True, activation_function=Sigmoid())
model = LayerList(layer1, layer2)

inputs = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3) (batch size = 2)
print("Input:\n", inputs)
output = model(inputs)
print("Model Output:\n", output)

# 3. Test bias inclusion/exclusion
print("\n=== Bias Test ===")
layer_no_bias = Layer(input_size=3, output_size=2, alpha=0.01, bias=False, activation_function=ReLU())
output_no_bias = layer_no_bias(inputs)
print("Output without Bias:\n", output_no_bias)

layer_with_bias = Layer(input_size=3, output_size=2, alpha=0.01, bias=True, activation_function=ReLU())
output_with_bias = layer_with_bias(inputs)
print("Output with Bias:\n", output_with_bias)

# 4. Test different activation functions
print("\n=== Activation Function Test ===")
relu_layer = Layer(input_size=3, output_size=2, alpha=0.01, bias=True, activation_function=ReLU())
sigmoid_layer = Layer(input_size=3, output_size=2, alpha=0.01, bias=True, activation_function=Sigmoid())

relu_output = relu_layer(inputs)
sigmoid_output = sigmoid_layer(inputs)

print("ReLU Output:\n", relu_output)
print("Sigmoid Output:\n", sigmoid_output)

# 5. Test dynamic input sizes
print("\n=== Dynamic Batch Size Test ===")
inputs_single = np.array([[1, 2, 3]])  # Shape: (1, 3)
inputs_batch = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Shape: (3, 3)

output_single = model(inputs_single)
output_batch = model(inputs_batch)

print("Single Input Output:\n", output_single)
print("Batch Input Output:\n", output_batch)
