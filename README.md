# Neural Network Implementation

A modular neural network framework with customizable layers, training support, and model persistence.

## Features

- Customizable layers with multiple activation functions and initialization methods
- Batch training with configurable loss functions and metrics  
- Gradient clipping and L2 regularization
- Model save/load functionality
- Learning rate scheduling
- Training callbacks

## Components

### `activations.py`
Activation functions:
- ReLU, LeakyReLU, PReLU
- ELU, GELU
- Sigmoid, Swish
- Softmax, BinaryStep, Tanh

### `layer.py`
- `Layer`: Core building block with forward/backward passes
- `LayerList`: Container for stacking layers

### `loss.py`
Loss functions:
- Mean Squared Error
- Log Loss
- Categorical Cross Entropy

### `optimizer.py`
- Adam optimizer implementation

## Usage

```python
from src.layer import Layer, LayerList
from src.activations import ReLU

# Create model
layer1 = Layer(input_size=5, output_size=4, alpha=0.01, activation_function=ReLU(), bias=True)
layer2 = Layer(input_size=4, output_size=3, alpha=0.01, activation_function=ReLU(), bias=True)
model = LayerList(layer1, layer2)

# Train
model.fit(
    input_data=X_train,
    expected=y_train, 
    batch_size=16,
    alpha=0.01,
    epochs=10,
    Loss_func=mean_squared_error,
    Loss_deriv_func=mean_squared_error_gradient,
    callbacks=[logging_callback],
    metrics=[accuracy],
    lr_scheduler=lr_scheduler,
    max_grad_norm=1.0
)

# Save/Load
model.save("model.pkl")
loaded_model = LayerList.load("model.pkl")
```

## Requirements

- NumPy
- Python 3.9+