# Example Predictions and Ground Truth
from src.loss import *

predicted = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
expected = np.array([[1, 0, 0], [0, 0, 1]])

# Test Mean Squared Error
print("MSE Loss:", mean_squared_error(predicted, expected))

# Test Log Loss
binary_predicted = np.array([0.9, 0.2, 0.8])
binary_expected = np.array([1, 0, 1])
print("Log Loss:", log_loss(binary_predicted, binary_expected))

# Test Categorical Cross-Entropy Loss
print("CCE Loss:", categorical_cross_entropy_loss(predicted, expected))
