import numpy as np


class ReLU:
    def __call__(self, pre_activated_output):
        return np.maximum(0, pre_activated_output)

    def derivative(self, pre_activated_output, grad_so_far):
        return np.where(pre_activated_output <= 0, 0, 1) * grad_so_far


class LeakyReLU:
    def __init__(self, alpha=0.01):  # Alpha is the slope for negative inputs
        self.alpha = alpha

    def __call__(self, pre_activated_output):
        return np.where(pre_activated_output > 0, pre_activated_output, self.alpha * pre_activated_output)

    def derivative(self, pre_activated_output, grad_so_far):
        return np.where(pre_activated_output > 0, 1, self.alpha) * grad_so_far


class PReLU:
    def __init__(self, alpha=0.25):  # Start with an initial alpha value
        self.alpha = alpha

    def __call__(self, pre_activated_output):
        return np.where(pre_activated_output > 0, pre_activated_output, self.alpha * pre_activated_output)

    def derivative(self, pre_activated_output, grad_so_far):
        return np.where(pre_activated_output > 0, 1, self.alpha) * grad_so_far


class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, pre_activated_output):
        return np.where(pre_activated_output > 0, pre_activated_output, self.alpha * (np.exp(pre_activated_output) - 1))

    def derivative(self, pre_activated_output, grad_so_far):
        elu_output = np.where(pre_activated_output > 0, 1, self.alpha * np.exp(pre_activated_output))
        return elu_output * grad_so_far


class GELU:
    def __call__(self, pre_activated_output):
        return 0.5 * pre_activated_output * (
                1 + np.tanh(np.sqrt(2 / np.pi) * (pre_activated_output + 0.044715 * pre_activated_output ** 3)))

    def derivative(self, pre_activated_output, grad_so_far):
        c = 0.0356774  # Constant for approximation
        gelu_tanh = np.tanh(0.79788456 * (pre_activated_output + c * pre_activated_output ** 3))
        gelu_derivative = 0.5 * (1 + gelu_tanh) + \
                          0.5 * pre_activated_output * (1 - gelu_tanh ** 2) * (
                                  0.79788456 + 3 * c * pre_activated_output ** 2)
        return gelu_derivative * grad_so_far


class Sigmoid:
    def __call__(self, pre_activated_output):
        pre_activated_output = np.clip(pre_activated_output, -500, 500)
        return 1 / (1 + np.exp(-pre_activated_output))

    def derivative(self, pre_activated_output, grad_so_far):
        sigmoid_output = 1 / (1 + np.exp(-pre_activated_output))
        return sigmoid_output * (1 - sigmoid_output) * grad_so_far


class Swish:
    def __call__(self, pre_activated_output):
        return pre_activated_output / (1 + np.exp(-pre_activated_output))

    def derivative(self, pre_activated_output, grad_so_far):
        sigmoid_output = 1 / (1 + np.exp(-pre_activated_output))
        swish_derivative = sigmoid_output + pre_activated_output * sigmoid_output * (1 - sigmoid_output)
        return swish_derivative * grad_so_far


class Softmax:
    def __call__(self, pre_activated_output):
        exp_shifted = np.exp(pre_activated_output - np.max(pre_activated_output, axis=1, keepdims=True))
        denominator = np.sum(exp_shifted, axis=1, keepdims=True)
        return exp_shifted / denominator

    def derivative(self, pre_activated_output, grad_so_far):
        output = self(pre_activated_output)
        batch_size, n_classes = output.shape

        jacobian = np.zeros((batch_size, n_classes, n_classes))
        for b in range(batch_size):
            out = output[b].reshape(-1, 1)
            jacobian[b] = np.diagflat(out) - np.dot(out, out.T)

        return np.einsum('bij,bj->bi', jacobian, grad_so_far)


class BinaryStep:
    def __call__(self, pre_activated_output):
        return np.where(pre_activated_output >= 0, 1, 0)

    def derivative(self, pre_activated_output, grad_so_far):
        return np.zeros_like(pre_activated_output)


class Tanh:
    def __call__(self, pre_activated_output):
        return np.tanh(pre_activated_output)

    def derivative(self, pre_activated_output, grad_so_far):
        tanh_output = np.tanh(pre_activated_output)
        return (1 - tanh_output ** 2) * grad_so_far
