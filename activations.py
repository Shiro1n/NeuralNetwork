import numpy as np


class ReLU:
    def __call__(self, pre_activated_output):
        return np.maximum(0, pre_activated_output)


class LeakyReLU:
    def __init__(self, alpha=0.01):  # Alpha is the slope for negative inputs
        self.alpha = alpha

    def __call__(self, pre_activated_output):
        return np.where(pre_activated_output > 0, pre_activated_output, self.alpha * pre_activated_output)


class PReLU:
    def __init__(self, alpha=0.25):  # Start with an initial alpha value
        self.alpha = alpha

    def __call__(self, pre_activated_output):
        return np.where(pre_activated_output > 0, pre_activated_output, self.alpha * pre_activated_output)


class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, pre_activated_output):
        return np.where(pre_activated_output > 0, pre_activated_output, self.alpha * (np.exp(pre_activated_output) - 1))


class GELU:
    def __call__(self, pre_activated_output):
        return 0.5 * pre_activated_output * (
                    1 + np.tanh(np.sqrt(2 / np.pi) * (pre_activated_output + 0.044715 * pre_activated_output ** 3)))


class Sigmoid:
    def __call__(self, pre_activated_output):
        pre_activated_output = np.clip(pre_activated_output, -500, 500)
        return 1 / (1 + np.exp(-pre_activated_output))


class Swish:
    def __call__(self, pre_activated_output):
        return pre_activated_output / (1 + np.exp(-pre_activated_output))


class Softmax:
    def __call__(self, pre_activated_output):
        exp_shifted = np.exp(pre_activated_output - np.max(pre_activated_output, axis=1, keepdims=True))
        denominator = np.sum(exp_shifted, axis=1, keepdims=True)
        return exp_shifted / denominator


class BinaryStep:
    def __call__(self, pre_activated_output):
        return np.where(pre_activated_output >= 0, 1, 0)


class Tanh:
    def __call__(self, pre_activated_output):
        return np.tanh(pre_activated_output)
