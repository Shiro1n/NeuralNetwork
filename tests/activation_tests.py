from src.activations import *


def numerical_gradient(func, inputs, grad_so_far, epsilon=1e-5):
    """
    Compute numerical gradients for a given function to compare against analytical gradients.
    """
    grad_approx = np.zeros_like(inputs)
    for i in range(inputs.size):
        x_pos = inputs.copy()
        x_neg = inputs.copy()
        x_pos.flat[i] += epsilon
        x_neg.flat[i] -= epsilon
        grad_approx.flat[i] = (func(x_pos) - func(x_neg)) / (2 * epsilon)
    return grad_approx * grad_so_far


def test_activation(activation_class, inputs, grad_so_far):
    """
    Test an activation function and its derivative.
    """
    activation = activation_class()

    # Forward pass
    forward_output = activation(inputs)

    # Derivative
    analytical_gradient = activation.derivative(inputs, grad_so_far)
    numerical_grad = numerical_gradient(lambda x: activation(x).sum(), inputs, grad_so_far)

    print(f"Testing {activation_class.__name__}")
    print("Forward Output:\n", forward_output)
    print("Analytical Gradient:\n", analytical_gradient)
    print("Numerical Gradient:\n", numerical_grad)
    print("Gradient Difference:\n", np.abs(analytical_gradient - numerical_grad).max())
    print("=" * 50)


# Test Inputs
inputs = np.array([[0.5, -0.5], [1.0, -1.0]])
grad_so_far = np.ones_like(inputs)

# Test Each Activation Function
test_activation(ReLU, inputs, grad_so_far)
test_activation(LeakyReLU, inputs, grad_so_far)
test_activation(PReLU, inputs, grad_so_far)
test_activation(ELU, inputs, grad_so_far)
test_activation(GELU, inputs, grad_so_far)
test_activation(Sigmoid, inputs, grad_so_far)
test_activation(Swish, inputs, grad_so_far)
test_activation(Tanh, inputs, grad_so_far)

# Test Softmax separately due to batch processing
softmax = Softmax()
softmax_output = softmax(inputs)
softmax_grad = softmax.derivative(inputs, grad_so_far)
print("Testing Softmax")
print("Forward Output:\n", softmax_output)
print("Analytical Gradient:\n", softmax_grad)
print("=" * 50)

# Test Binary Step separately as its derivative is zero everywhere
binary_step = BinaryStep()
binary_output = binary_step(inputs)
binary_grad = binary_step.derivative(inputs, grad_so_far)
print("Testing BinaryStep")
print("Forward Output:\n", binary_output)
print("Derivative (Should be all zeros):\n", binary_grad)
print("=" * 50)
