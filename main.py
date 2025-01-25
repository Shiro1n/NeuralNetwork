import numpy as np
from src.layer import Layer, LayerList
import matplotlib.pyplot as plt


# Mock activation function (ReLU) for layers
class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x, grad):
        return np.where(x > 0, 1, 0) * grad


# Loss function and its derivative
def mean_squared_error(predicted, expected):
    return np.mean((predicted - expected) ** 2)


def mean_squared_error_derivative(predicted, expected):
    return 2 * (predicted - expected) / predicted.size


# Metric function
def accuracy(predicted, expected):
    predicted_classes = np.argmax(predicted, axis=1)
    expected_classes = np.argmax(expected, axis=1)
    return np.mean(predicted_classes == expected_classes)


# Callback function for logging
def logging_callback(epoch, loss, metrics):
    print(f"[Callback] Epoch {epoch}: Loss={loss:.4f}, Metrics={metrics}")


# Learning rate scheduler
def lr_scheduler(epoch, current_lr):
    return current_lr * 0.9 if epoch % 5 == 0 else current_lr


# Visualization function for training results
def plot_training_results(losses, metrics_results):
    epochs = len(losses)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), losses, label="Loss", marker="o")
    for metric, values in metrics_results.items():
        plt.plot(range(1, epochs + 1), values, label=metric, marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Progress")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    # Generate synthetic data
    np.random.seed(42)
    input_data = np.random.rand(100, 5)  # 100 samples, 5 features each
    expected = np.eye(3)[np.random.choice(3, 100)]  # 3 classes, one-hot encoded

    # Define model layers
    layer1 = Layer(input_size=5, output_size=4, alpha=0.01, activation_function=ReLU(), bias=True)
    layer2 = Layer(input_size=4, output_size=3, alpha=0.01, activation_function=ReLU(), bias=True)
    model = LayerList(layer1, layer2)

    # Train the model
    print("Training the model...")
    losses = []
    metrics_results = {"accuracy": []}

    def training_callback(epoch, loss, metrics):
        losses.append(loss)
        for metric_name, metric_value in metrics.items():
            metrics_results[metric_name].append(metric_value)

    model.fit(
        input_data=input_data,
        expected=expected,
        batch_size=16,
        alpha=0.01,
        epochs=10,
        Loss_func=mean_squared_error,
        Loss_deriv_func=mean_squared_error_derivative,
        callbacks=[logging_callback, training_callback],
        metrics=[accuracy],
        lr_scheduler=lr_scheduler,
        verbose=True,
        max_grad_norm=1.0,  # Gradient clipping
    )

    # Visualize training progress
    plot_training_results(losses, metrics_results)

    # Save the model
    model_filepath = "demo_model.pkl"
    model.save(model_filepath)
    print(f"Model saved to '{model_filepath}'.")

    # Load the model
    loaded_model = LayerList.load(model_filepath)
    print("Model loaded successfully.")

    # Test predictions
    test_input = np.random.rand(5, 5)  # 5 samples, 5 features each
    test_output = loaded_model(test_input)
    print("Test Predictions:")
    print(test_output)


if __name__ == "__main__":
    main()
