# Module 7: Deep Learning

**Kailash**: kailash-ml (OnnxBridge, InferenceServer, TrainingPipeline, ModelVisualizer, ModelRegistry) | **Scaffolding**: 30%

## Lecture (3h)

- **7A** Neural Network Foundations: linear regression as a neuron, hidden layers, XOR problem, depth vs width, hierarchical feature learning
- **7B** Architecture & Training: activations (sigmoid, ReLU, GELU), loss functions (MSE, CrossEntropy), weight initialization (Xavier, He), backpropagation, chain rule, vanishing gradients
- **7C** Optimisation & CNNs: SGD, momentum, Adam, learning rate scheduling, dropout, batch normalisation, convolution, pooling, CNN architectures, embeddings, ONNX export

## Lab (3h) — 8 Exercises

1. Linear regression as a single-neuron network (forward pass, MSE, gradient descent)
2. Hidden layers and the XOR problem (multi-layer perceptron, decision boundaries)
3. Activation functions comparison (sigmoid vs ReLU vs GELU, gradient flow analysis)
4. Loss functions and weight initialization (CrossEntropy, Xavier vs He init)
5. Backpropagation from scratch (chain rule, gradient checking, vanishing gradients)
6. Optimizers and learning rate scheduling (SGD vs momentum vs Adam, cosine annealing)
7. CNNs for image classification (convolution, pooling, dropout, OnnxBridge export)
8. Capstone: end-to-end DL pipeline (TrainingPipeline → ModelRegistry → OnnxBridge → InferenceServer)

## Datasets

MNIST sample, Fashion-MNIST sample, HDB resale prices, synthetic spirals
