---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 4.7: Deep Learning

### Module 4: Advanced ML

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Work with PyTorch tensors, autograd, and nn.Module
- Build CNNs with convolutional layers and ResBlocks
- Train deep learning models with proper data loading
- Export models via OnnxBridge for production serving

---

## Recap: Lesson 4.6

- Models decay as data distributions shift over time
- PSI measures distribution shift between reference and production data
- `DriftMonitor` automates feature, prediction, and performance monitoring
- Clear response strategies: ignore, investigate, or retrain

---

## When Deep Learning?

```
Classical ML (XGBoost/LightGBM):
  ✅ Tabular data (rows and columns)
  ✅ Small-medium datasets (< 1M rows)
  ✅ Fast training, interpretable

Deep Learning (PyTorch):
  ✅ Images, audio, video, long text
  ✅ Large datasets (millions of examples)
  ✅ Complex patterns (spatial, sequential)
  ❌ Tabular data (usually loses to gradient boosting)
```

Use the right tool for the data type.

---

## PyTorch Tensors

```python
import torch

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.zeros(3, 4)           # 3x4 matrix of zeros
z = torch.randn(3, 4)           # random normal values

# Operations
a = x + 1                       # element-wise add
b = torch.matmul(y, z.T)        # matrix multiplication

# GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_gpu = x.to(device)
```

Tensors are n-dimensional arrays with automatic differentiation.

---

## Autograd: Automatic Differentiation

```python
# Autograd tracks operations for gradient computation
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1     # y = x² + 2x + 1

y.backward()             # compute dy/dx
print(x.grad)            # 8.0  (dy/dx = 2x + 2 = 8)
```

```
Forward pass:  compute y from x
Backward pass: compute gradients dy/dx

This is how neural networks learn:
  1. Forward: compute predictions
  2. Loss: measure error
  3. Backward: compute gradients
  4. Update: adjust weights
```

---

## nn.Module: Building Networks

```python
import torch.nn as nn

class PricePredictor(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)

model = PricePredictor(n_features=25)
```

---

## Training Loop

```python
import torch.optim as optim

model = PricePredictor(n_features=25)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_tensor), y_val_tensor)
        print(f"Epoch {epoch}: train={loss:.0f}, val={val_loss:.0f}")
```

---

## DataLoader: Efficient Batching

```python
from torch.utils.data import DataLoader, TensorDataset

# Create dataset
dataset = TensorDataset(X_train_tensor, y_train_tensor)

# Create data loader (batches, shuffles, parallelises)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training with batches
for epoch in range(100):
    model.train()
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
```

---

## Convolutional Neural Networks (CNNs)

For image and spatial data: learn local patterns.

```
Input image    Conv filter    Feature map
┌─────────┐   ┌───┐         ┌───────┐
│ · · · · │   │1 0│         │ · · · │
│ · · · · │ * │0 1│    =    │ · · · │
│ · · · · │   └───┘         │ · · · │
│ · · · · │                  └───────┘
└─────────┘
  Detects edges, textures, shapes at different scales
```

---

## CNN Architecture

```python
class ImageClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

---

## ResBlock: Skip Connections

```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return nn.functional.relu(x + self.block(x))  # skip connection
```

```
Input ──→ Conv → BN → ReLU → Conv → BN ──→ Add → ReLU → Output
   │                                          ↑
   └──────────── skip connection ─────────────┘
```

Skip connections prevent gradient vanishing in deep networks.

---

## OnnxBridge: Export for Production

```python
from kailash_ml import OnnxBridge

# Export PyTorch model to ONNX format
bridge = OnnxBridge()
bridge.export(
    model=model,
    sample_input=torch.randn(1, 25),  # example input shape
    output_path="hdb_model.onnx",
)

# Load and run ONNX model (no PyTorch needed at inference)
onnx_model = bridge.load("hdb_model.onnx")
predictions = bridge.predict(onnx_model, new_data)
```

ONNX models run anywhere -- no PyTorch dependency in production.

---

## When to Use What

| Data Type              | Best Approach                              |
| ---------------------- | ------------------------------------------ |
| Tabular (structured)   | LightGBM / XGBoost via TrainingPipeline    |
| Images                 | CNN (PyTorch)                              |
| Text (short)           | TF-IDF + gradient boosting                 |
| Text (long)            | Transformers / LLMs (Module 5)             |
| Time series            | LSTM / Transformer or classical + features |
| Mixed (tabular + text) | Ensemble: boosting(tabular) + neural(text) |

---

## Exercise Preview

**Exercise 4.7: Deep Learning for Property Images**

You will:

1. Build a tabular neural network with nn.Module for price prediction
2. Implement a CNN for property image classification
3. Add ResBlocks and observe training improvement
4. Export the model with OnnxBridge for serving

Scaffolding level: **Light+ (~40% code provided)**

---

## Common Pitfalls

| Mistake                                 | Fix                                             |
| --------------------------------------- | ----------------------------------------------- |
| Using deep learning for tabular data    | XGBoost/LightGBM usually wins on tables         |
| Forgetting `model.eval()` for inference | Dropout/BatchNorm behave differently in eval    |
| Not normalising inputs                  | Neural networks are sensitive to input scale    |
| Learning rate too high                  | Start with 0.001 for Adam; decrease if unstable |
| No early stopping                       | Monitor validation loss; stop when it increases |

---

## Summary

- PyTorch: tensors for data, autograd for gradients, nn.Module for models
- CNNs learn spatial patterns via convolutional filters
- ResBlocks add skip connections to enable very deep networks
- OnnxBridge exports models for framework-independent production serving
- Use deep learning for images/text; gradient boosting for tabular data

---

## Next Lesson

**Lesson 4.8: Capstone — InferenceServer**

We will learn:

- Deploying models with `InferenceServer`
- End-to-end: training to serving
- Module 4 capstone project
