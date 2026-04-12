# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP 5 — AI-Resilient Assessment Questions

Deep Learning: Architecture-Driven Feature Engineering
Covers: neural networks, hidden layers, activations, loss functions,
        backpropagation, optimizers, CNNs, OnnxBridge, InferenceServer
"""

QUIZ = {
    "module": "MLFP05",
    "title": "Deep Learning",
    "questions": [
        # ── Lesson 7.1: Linear regression as NN ─────────────────────────
        {
            "id": "7.1.1",
            "lesson": "7.1",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student implements gradient descent for linear regression as a "
                "single neuron. Training loss decreases for 3 epochs then explodes "
                "to infinity. The learning rate is 0.01. What is wrong?"
            ),
            "code": (
                "# Forward pass\n"
                "y_pred = w * x + b\n"
                "loss = sum((y_pred - y) ** 2) / n  # MSE\n"
                "\n"
                "# Backward pass\n"
                "dw = sum(2 * (y_pred - y) * x) / n\n"
                "db = sum(2 * (y_pred - y)) / n\n"
                "\n"
                "# Update (BUG)\n"
                "w = w + lr * dw\n"
                "b = b + lr * db\n"
            ),
            "options": [
                "A) MSE loss should divide by 2n, not n — the factor of 2 causes the gradient to be too large",
                "B) The update rule adds the gradient instead of subtracting it. Gradient descent MINIMIZES loss: w = w - lr * dw. Adding moves uphill.",
                "C) The learning rate 0.01 is too high for any gradient descent — reduce to 1e-6",
                "D) dw and db should use absolute value |y_pred - y| instead of (y_pred - y)",
            ],
            "answer": "B",
            "explanation": (
                "Gradient descent follows the NEGATIVE gradient direction to minimize loss. "
                "The update rule must be w = w - lr * dw (subtract). "
                "Adding the gradient moves parameters in the direction of INCREASING loss, "
                "causing divergence. The loss may decrease briefly due to stochastic noise "
                "before exploding. This is one of the most common DL bugs."
            ),
            "learning_outcome": "Implement correct gradient descent update rule for a single neuron",
        },
        {
            "id": "7.1.2",
            "lesson": "7.1",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 1, you implement y = wx + b as a neural network and compare "
                "gradient descent with the OLS closed-form solution. Your GD converges to "
                "w=0.482, b=3.21 while OLS gives w=0.485, b=3.19. The GD loss is 0.0312 "
                "while OLS loss is 0.0310. Should you worry about the difference?"
            ),
            "options": [
                "A) Yes — the difference indicates a bug in the backpropagation. GD should converge to the exact same solution as OLS.",
                "B) Yes — GD is always worse than OLS for linear regression and should never be used.",
                "C) No — GD converged close to the optimum. Small differences arise from learning rate, number of epochs, and floating-point precision. For linear regression, OLS is exact; GD is iterative and approximate. The 0.6% difference in w is negligible.",
                "D) No — the losses are identical (0.03), so the parameters must be equivalent.",
            ],
            "answer": "C",
            "explanation": (
                "GD is iterative — it approaches the optimum asymptotically. With finite epochs "
                "and a fixed learning rate, it may not reach the exact minimum. The 0.6% parameter "
                "difference and 0.6% loss difference confirm convergence to a near-optimal solution. "
                "For linear regression specifically, OLS computes the exact minimum analytically. "
                "GD's value is that it scales to non-linear models where no closed-form exists. "
                "The exercise demonstrates that DL starts from this familiar foundation."
            ),
            "learning_outcome": "Understand GD convergence properties vs analytical solutions",
        },
        # ── Lesson 7.2: Hidden layers ───────────────────────────────────
        {
            "id": "7.2.1",
            "lesson": "7.2",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "Your dataset has two features (x1, x2) and a binary label where the "
                "positive class forms a ring around the negative class (like a donut). "
                "A single-layer network achieves 50% accuracy. What change is needed?"
            ),
            "options": [
                "A) Increase the learning rate — the model needs more aggressive updates to find the decision boundary",
                "B) Add a hidden layer with non-linear activation. The ring decision boundary is not linearly separable — a hidden layer can learn a non-linear transformation that makes it separable (similar to XOR requiring a hidden layer).",
                "C) Add more features by computing x1² and x2² manually — neural networks cannot learn polynomial features",
                "D) Use MSE loss instead of cross-entropy — MSE works better for circular decision boundaries",
            ],
            "answer": "B",
            "explanation": (
                "A ring/donut pattern is not linearly separable — no single hyperplane can separate it. "
                "This is the same fundamental problem as XOR. A hidden layer with non-linear activation "
                "(e.g., ReLU) learns a feature transformation where the classes become separable. "
                "The universal approximation theorem guarantees a sufficiently wide single hidden layer "
                "can approximate any continuous function, but in practice, deeper networks learn such "
                "transformations more efficiently."
            ),
            "learning_outcome": "Identify when hidden layers are necessary for non-linear decision boundaries",
        },
        {
            "id": "7.2.2",
            "lesson": "7.2",
            "type": "output_interpretation",
            "difficulty": "advanced",
            "question": (
                "You visualize the decision boundary of a 2-hidden-layer network trained on "
                "the spiral dataset from Exercise 2. The boundary is very jagged with sharp "
                "corners that perfectly trace every training point. Test accuracy is 62% while "
                "training accuracy is 99%. What happened and what would you change?"
            ),
            "options": [
                "A) The model is underfitting — add more hidden layers to increase capacity",
                "B) The learning rate was too low — the model needs to explore more of the loss landscape",
                "C) The model is overfitting — the jagged boundary memorizes training noise. Add dropout (Exercise 7), reduce hidden layer width, or add L2 regularization to smooth the boundary.",
                "D) The spiral dataset is too hard — switch to a simpler dataset",
            ],
            "answer": "C",
            "explanation": (
                "99% train / 62% test is classic overfitting. The jagged boundary means the model "
                "memorized individual training points instead of learning the underlying spiral pattern. "
                "Solutions: (1) Dropout randomly zeros neurons during training, forcing redundancy. "
                "(2) Reducing width limits capacity to memorize. (3) L2 regularization penalizes large "
                "weights, encouraging smoother boundaries. The gap between train/test accuracy (37 points) "
                "is the overfitting signal."
            ),
            "learning_outcome": "Diagnose overfitting from train/test accuracy gap and boundary visualization",
        },
        # ── Lesson 7.3: DL power hierarchy ────────────────────────────
        {
            "id": "7.3.1",
            "lesson": "7.3",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You need to classify satellite images into 50 land-use categories. "
                "Each image is 256×256×3. A colleague proposes a wide single-hidden-layer "
                "network (1 layer, 8192 neurons). You propose a deep network (6 layers, "
                "512 neurons each). Both have ~100M parameters. Which is better and why?"
            ),
            "options": [
                "A) The wide network — universal approximation theorem guarantees a single hidden layer can represent any function, so depth adds no value",
                "B) The deep network. While a single wide layer CAN represent any function (UAT), it may need exponentially more neurons to do so. Depth enables hierarchical feature composition: layer 1 learns edges, layer 2 learns textures, layer 3 learns parts, layers 4-6 learn objects and scenes. This compositional hierarchy matches the structure of visual data — each layer reuses lower-level features, requiring far fewer total parameters for the same representational power.",
                "C) Both are equivalent — total parameter count is all that matters for network capacity",
                "D) The wide network — deep networks always suffer from vanishing gradients and cannot be trained",
            ],
            "answer": "B",
            "explanation": (
                "The depth vs width trade-off is central to the DL power hierarchy (Lesson 7.3). "
                "A single wide layer can theoretically represent any function (UAT), but may need "
                "exponentially many neurons. A deep network composes simple functions hierarchically: "
                "f(x) = f_6(f_5(f_4(f_3(f_2(f_1(x)))))). Each layer transforms representations, "
                "building from edges → textures → parts → objects. For 256×256 images with 50 "
                "categories, this compositionality is critical — a flat network must independently "
                "learn every pixel-to-category mapping, while a deep network reuses intermediate "
                "features. Modern techniques (ReLU, BatchNorm, skip connections) solve the vanishing "
                "gradient concern, making depth practical."
            ),
            "learning_outcome": "Choose depth over width for hierarchical feature learning in vision tasks",
        },
        # ── Lesson 7.4: Activations ────────────────────────────────────
        {
            "id": "7.4.1",
            "lesson": "7.4",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "A student trains a 5-layer network with sigmoid activations. After 100 "
                "epochs, layers 1-2 have gradient magnitudes of ~1e-8 while layers 4-5 "
                "have gradients of ~0.01. What is this problem called and what fix does "
                "Exercise 3 demonstrate?"
            ),
            "options": [
                "A) Exploding gradients — add gradient clipping to cap gradients at 1.0",
                "B) Dead neurons — the sigmoid output is stuck at 0; switch to Leaky ReLU",
                "C) Vanishing gradients — sigmoid squashes to (0,1) and its max derivative is 0.25. Across 5 layers: 0.25⁵ ≈ 0.001. Early layers barely update. Fix: replace sigmoid with ReLU (derivative = 1 for positive inputs) and use He initialization.",
                "D) Learning rate is too small — increase from 0.01 to 0.1 to compensate for small gradients",
            ],
            "answer": "C",
            "explanation": (
                "Vanishing gradients occur when activation derivatives are consistently < 1. "
                "Sigmoid's max derivative is 0.25 (at x=0), so gradients shrink by at least 4× "
                "per layer. After 5 layers: gradient ≈ 0.25⁵ = ~0.001× the output gradient. "
                "ReLU has derivative exactly 1 for positive inputs, allowing gradients to flow "
                "unchanged through layers. He initialization (scale = sqrt(2/fan_in)) ensures "
                "activations maintain variance across layers. Exercise 3 demonstrates this "
                "by comparing gradient magnitudes per layer with sigmoid vs ReLU."
            ),
            "learning_outcome": "Diagnose vanishing gradients from per-layer gradient magnitudes",
        },
        {
            "id": "7.4.2",
            "lesson": "7.4",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 3, you compare sigmoid, ReLU, and GELU on the same architecture. "
                "ReLU converges in 30 epochs. GELU converges in 25 epochs with slightly better "
                "final accuracy. Sigmoid converges in 80 epochs. Why is GELU faster than ReLU "
                "despite both solving the vanishing gradient problem?"
            ),
            "options": [
                "A) GELU has a larger maximum derivative than ReLU, allowing bigger gradient updates",
                "B) GELU is smoother than ReLU around x=0. ReLU has a hard kink (derivative jumps from 0 to 1), while GELU's smooth curve x·Φ(x) provides better gradient signals near zero, enabling finer-grained updates in the critical transition region.",
                "C) GELU uses less memory than ReLU because it doesn't need to store the activation mask",
                "D) GELU converges faster only on this specific dataset — in general, ReLU is always faster",
            ],
            "answer": "B",
            "explanation": (
                "GELU (Gaussian Error Linear Unit) is defined as x·Φ(x) where Φ is the Gaussian CDF. "
                "Unlike ReLU's hard threshold at x=0, GELU smoothly transitions, providing non-zero "
                "gradients for slightly negative inputs. This smooth approximation of a stochastic "
                "regularizer helps the optimizer navigate the loss landscape more efficiently. "
                "GELU is the default activation in modern transformers (BERT, GPT) precisely because "
                "this smoothness improves convergence on a wide range of tasks."
            ),
            "learning_outcome": "Compare activation function convergence properties from Exercise 3 results",
        },
        # ── Lesson 7.5: Loss functions ──────────────────────────────────
        {
            "id": "7.5.1",
            "lesson": "7.5",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "You are building a 10-class image classifier (Fashion-MNIST). A colleague "
                "suggests using MSE loss because 'it worked fine for regression.' Exercise 4 "
                "shows MSE achieves 85% accuracy while CrossEntropy achieves 92% on the same "
                "architecture. Why does CrossEntropy outperform MSE for classification?"
            ),
            "options": [
                "A) MSE requires one-hot encoded labels which are wasteful; CrossEntropy uses integer labels directly",
                "B) CrossEntropy's gradient is (predicted - target), which provides strong gradients when the model is confident but wrong. MSE's gradient (2×(pred-target)×sigmoid_derivative) includes sigmoid_derivative which shrinks near 0 and 1, slowing learning when the model is most wrong.",
                "C) MSE can only be used for binary classification, not multi-class",
                "D) CrossEntropy uses the log function which makes the loss landscape convex for any model",
            ],
            "answer": "B",
            "explanation": (
                "The key difference is gradient behavior when the model is wrong. "
                "CrossEntropy with softmax gives gradient = (predicted - target), which is LARGE "
                "when the model confidently predicts the wrong class. "
                "MSE with sigmoid gives gradient = 2(pred-target) × σ'(z), where σ'(z) is near 0 "
                "when the output is near 0 or 1. So MSE's gradient VANISHES precisely when the "
                "model is most wrong and needs the biggest correction. "
                "Exercise 4 demonstrates this: MSE loss plateaus early while CE continues improving."
            ),
            "learning_outcome": "Choose appropriate loss function based on gradient dynamics for classification",
        },
        {
            "id": "7.5.2",
            "lesson": "7.5",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "A student initializes a 10-layer ReLU network with all weights set to 0.01 "
                "(constant initialization). After 50 epochs, all neurons in each layer output "
                "identical values. What is this problem and what initialization from Exercise 4 "
                "fixes it?"
            ),
            "options": [
                "A) The learning rate is too small — increase to break symmetry",
                "B) Symmetry breaking failure. If all weights are identical, all neurons compute identical gradients and make identical updates forever. Use He initialization: w ~ N(0, sqrt(2/fan_in)) — random values break symmetry while the sqrt(2/fan_in) scale prevents vanishing/exploding activations in ReLU networks.",
                "C) Constant 0.01 is too small — use constant 1.0 instead to ensure activations are large enough",
                "D) The bias terms should be initialized to 1.0, not 0.0 — bias breaks the symmetry",
            ],
            "answer": "B",
            "explanation": (
                "Symmetry breaking is essential: if neurons in a layer start identical, they receive "
                "identical gradients and remain identical forever — effectively a single neuron "
                "replicated N times, wasting capacity. Random initialization breaks this symmetry. "
                "He initialization (scale = sqrt(2/fan_in)) is specifically designed for ReLU networks: "
                "the factor of 2 compensates for ReLU zeroing out negative inputs. "
                "Xavier initialization (sqrt(1/fan_in)) is optimal for sigmoid/tanh. "
                "Exercise 4 shows 10-layer networks diverge with constant init but converge with He."
            ),
            "learning_outcome": "Apply correct weight initialization strategy for deep ReLU networks",
        },
        # ── Lesson 7.6: Backpropagation ─────────────────────────────────
        {
            "id": "7.6.1",
            "lesson": "7.6",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "In Exercise 5, you implement gradient checking to verify your backprop. "
                "For most parameters, the relative error between analytical and numerical "
                "gradients is < 1e-5. But for one weight, the relative error is 0.23. "
                "What does this tell you?"
            ),
            "options": [
                "A) The numerical gradient is inaccurate — decrease epsilon from 1e-5 to 1e-10",
                "B) Floating point precision limits mean errors up to 0.5 are acceptable",
                "C) Relative error of 0.23 (23%) indicates a bug in the analytical gradient for that weight. The chain rule was likely applied incorrectly for that parameter's computation path. Since other gradients pass, the error is localized — check the backward pass for that specific layer/connection.",
                "D) The model has converged and gradients near zero cause unstable relative errors",
            ],
            "answer": "C",
            "explanation": (
                "Gradient checking compares: analytical gradient (from backprop) vs numerical gradient "
                "((f(x+ε) - f(x-ε)) / 2ε). Relative error > 1e-2 indicates a bug. At 0.23 (23%), "
                "the analytical gradient is clearly wrong for that parameter. Common causes: "
                "(1) forgot to apply chain rule through activation derivative, "
                "(2) transposed weight matrix in gradient computation, "
                "(3) wrong variable used in gradient update. "
                "Since other parameters pass, the issue is localized to one computation path."
            ),
            "learning_outcome": "Use gradient checking to identify backpropagation implementation errors",
        },
        {
            "id": "7.6.2",
            "lesson": "7.6",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 5 demonstrates vanishing gradients in a 10-layer sigmoid network. "
                "You measure gradient norms per layer: [0.0001, 0.0003, 0.001, 0.004, 0.015, "
                "0.06, 0.2, 0.8, 3.1, 12.0] (layer 1 to 10). After switching to ReLU + He init, "
                "the norms become [0.8, 0.9, 1.1, 0.95, 1.0, 1.05, 0.98, 1.02, 1.1, 1.0]. "
                "What pattern do you observe in each case?"
            ),
            "options": [
                "A) Sigmoid: gradients grow toward the output; ReLU: gradients are random",
                "B) Sigmoid: gradients decay exponentially toward input layers (factor ~4× per layer = 0.25 from sigmoid derivative). Layer 1 gets gradient 120,000× smaller than layer 10. ReLU + He init: gradients maintain roughly constant magnitude (~1.0) across all layers, confirming healthy gradient flow.",
                "C) Both cases show healthy training — gradient magnitudes don't matter, only their direction",
                "D) Sigmoid gradients are too small; ReLU gradients are too large — both need gradient clipping",
            ],
            "answer": "B",
            "explanation": (
                "Sigmoid case: gradient ratio layer10/layer1 = 12.0/0.0001 = 120,000×. "
                "The exponential decay (~4× per layer due to sigmoid max derivative 0.25) means "
                "early layers learn at 1/120,000th the speed of later layers. "
                "ReLU + He case: max/min ratio = 1.1/0.8 = 1.375×. Near-constant gradients mean "
                "all layers learn at similar speeds. He initialization (sqrt(2/fan_in)) maintains "
                "activation variance = 1.0 across layers, which maintains gradient variance too. "
                "This is the fundamental insight Exercise 5 demonstrates."
            ),
            "learning_outcome": "Interpret per-layer gradient norms to diagnose training health",
        },
        # ── Lesson 7.8-7.9: Optimizers ──────────────────────────────────
        {
            "id": "7.9.1",
            "lesson": "7.9",
            "type": "output_interpretation",
            "difficulty": "intermediate",
            "question": (
                "Exercise 6 compares SGD, SGD+Momentum, and Adam on MNIST. After 5 epochs: "
                "SGD loss=1.8, Momentum loss=0.9, Adam loss=0.4. The SGD loss curve oscillates "
                "wildly between epochs. What causes the oscillation and why does momentum fix it?"
            ),
            "options": [
                "A) SGD oscillates because the learning rate is too high; momentum slows down the effective learning rate",
                "B) SGD updates based on each mini-batch, causing noisy gradient estimates. In narrow ravines (common in loss landscapes), the gradient points across the ravine (oscillating) rather than along it. Momentum accumulates past gradients, canceling oscillations while reinforcing consistent direction — like a heavy ball rolling downhill.",
                "C) SGD uses the wrong loss function; Adam automatically selects the correct one",
                "D) The oscillation is normal and harmless — SGD will converge to the same loss given enough epochs",
            ],
            "answer": "B",
            "explanation": (
                "Mini-batch SGD computes gradients on a random subset, introducing noise. "
                "In narrow ravines of the loss landscape (common near saddle points), gradients "
                "oscillate perpendicular to the optimal path. Momentum (v = β×v + g; w -= lr×v) "
                "averages these oscillations: perpendicular components cancel, parallel components "
                "reinforce. The result is faster progress along the ravine. "
                "β=0.9 means 90% of the previous velocity carries forward, "
                "smoothing over ~1/(1-β)=10 past gradients."
            ),
            "learning_outcome": "Explain momentum's role in damping gradient oscillations",
        },
        {
            "id": "7.9.2",
            "lesson": "7.9",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "You are starting a new DL project with no prior hyperparameter knowledge. "
                "Exercise 6 shows Adam converges fastest with default hyperparameters. "
                "When should you prefer SGD+Momentum over Adam?"
            ),
            "options": [
                "A) Never — Adam is strictly better in all cases",
                "B) When you need reproducible results — SGD is deterministic while Adam is not",
                "C) When you have time for extensive hyperparameter tuning and want the best final performance. Research shows SGD+Momentum with carefully tuned learning rate + cosine schedule can generalize better than Adam on some tasks (image classification, language models). Adam converges faster but may find sharper minima.",
                "D) When your model has fewer than 1 million parameters — Adam's overhead is too high for small models",
            ],
            "answer": "C",
            "explanation": (
                "Adam is the best default optimizer: it adapts per-parameter learning rates and "
                "converges quickly with minimal tuning. However, multiple studies show SGD+Momentum "
                "with properly tuned learning rate + cosine annealing can achieve better generalization "
                "(flatter minima) on tasks like ImageNet classification and language model training. "
                "The trade-off: Adam = fast prototyping, minimal tuning. SGD+Momentum = potentially "
                "better final performance with more tuning effort. Start with Adam, switch to SGD "
                "if you need that last 0.5% accuracy."
            ),
            "learning_outcome": "Choose between Adam and SGD+Momentum based on project constraints",
        },
        # ── Lesson 7.11: CNNs ───────────────────────────────────────────
        {
            "id": "7.11.1",
            "lesson": "7.11",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student's CNN from Exercise 7 has conv(3×3, 4 filters, padding=0) → pool(2×2) "
                "on a 28×28 input. They set the fully-connected layer input size to 4×14×14=784. "
                "The model crashes with a dimension mismatch. What is the correct FC input size?"
            ),
            "options": [
                "A) 4 × 28 × 28 = 3136 — pooling doesn't change spatial dimensions",
                "B) 4 × 13 × 13 = 676. Without padding, conv(3×3) on 28×28 gives 26×26 (not 28×28). Pool(2×2) on 26×26 gives 13×13. With 4 filters: 4 × 13 × 13 = 676.",
                "C) 4 × 12 × 12 = 576 — conv reduces by 3 and pool by 2",
                "D) 1 × 14 × 14 = 196 — only one filter output is passed to the FC layer",
            ],
            "answer": "B",
            "explanation": (
                "Output size formula: out = (in - kernel + 2×padding) / stride + 1. "
                "Conv: (28 - 3 + 0) / 1 + 1 = 26. Pool: 26 / 2 = 13. "
                "With 4 filters: 4 × 13 × 13 = 676. "
                "The student assumed padding=1 (which would give 28→28→14), but the code has padding=0. "
                "This is the most common CNN dimension bug. "
                "Always trace spatial dimensions through each layer before writing the FC layer."
            ),
            "learning_outcome": "Calculate CNN output dimensions through conv and pooling layers",
        },
        {
            "id": "7.11.2",
            "lesson": "7.11",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 7 exports a CNN to ONNX via OnnxBridge and then serves it via "
                "InferenceServer in Exercise 8. Your model runs at 15ms/prediction in Python "
                "and 3ms/prediction via ONNX Runtime. A colleague asks: 'Why not just keep "
                "the Python model?' What is your answer?"
            ),
            "options": [
                "A) Python is fine for production — ONNX is only useful for non-Python environments",
                "B) ONNX Runtime applies graph-level optimizations (operator fusion, constant folding) and uses platform-native execution (CPU SIMD, GPU kernels) without Python interpreter overhead. The 5× speedup matters at scale: 1M daily predictions saves 3.3 hours of compute. Plus ONNX is portable — same model runs on mobile, browser, or server.",
                "C) ONNX reduces model size by 10× through automatic quantization during export",
                "D) Python models cannot be served over HTTP; only ONNX models work with InferenceServer",
            ],
            "answer": "B",
            "explanation": (
                "ONNX Runtime provides: (1) Graph optimizations — fuse adjacent operations like "
                "Conv+BatchNorm+ReLU into a single kernel. (2) Platform-native execution — "
                "vectorized CPU instructions or GPU kernels without Python GIL. (3) Portability — "
                "same .onnx file runs on any ONNX runtime (C++, JS, Java, mobile). "
                "The 5× speedup (15ms → 3ms) compounds: at 1M predictions/day, that's "
                "4.2 hours vs 0.8 hours of compute. OnnxBridge.export() handles the conversion; "
                "InferenceServer wraps it in an HTTP endpoint."
            ),
            "learning_outcome": "Justify ONNX export for production deployment via OnnxBridge",
        },
        # ── Lesson 7.10: Dropout ────────────────────────────────────────
        {
            "id": "7.10.1",
            "lesson": "7.10",
            "type": "output_interpretation",
            "difficulty": "intermediate",
            "question": (
                "After training a network with dropout=0.5, a student runs the same input "
                "through the model 10 times and gets 10 DIFFERENT predictions. They then "
                "switch to model.eval() mode and get the same prediction every time, but "
                "all predictions are ~50% smaller in magnitude than the training outputs. "
                "What explains both observations?"
            ),
            "options": [
                "A) The model has a bug — predictions should be identical in both training and eval mode",
                "B) During training, dropout randomly zeros 50% of neurons per forward pass, so each run activates a different random subset — producing different outputs. In eval mode, dropout is disabled (all neurons active), but each neuron's output is scaled by (1-p)=0.5 to compensate for the fact that twice as many neurons are now active. If the implementation uses inverted dropout (scale by 1/p during training), eval outputs are correct without scaling. The student's model likely uses standard dropout without inverted scaling.",
                "C) The 10 different predictions indicate the model has not converged — train for more epochs",
                "D) model.eval() disables gradient computation, which changes the forward pass outputs",
            ],
            "answer": "B",
            "explanation": (
                "Dropout has fundamentally different behavior at train vs test time: "
                "Training: each neuron is randomly kept with probability p (or zeroed with probability "
                "1-p). This means each forward pass uses a different random subnetwork — hence 10 "
                "different predictions for the same input. "
                "Eval/Test: all neurons are active. Expected activation doubles (since during training "
                "only 50% were active). Two approaches: (1) Standard dropout: multiply outputs by p=0.5 "
                "at test time (what the student sees). (2) Inverted dropout: multiply by 1/p=2.0 during "
                "training so test time needs no adjustment. Most frameworks use inverted dropout by "
                "default. The student's 50%-magnitude outputs confirm standard dropout without test-time "
                "scaling."
            ),
            "learning_outcome": "Explain dropout behavior difference between training and evaluation modes",
        },
    ],
}

# --- Merged from mlfp05 (NLP & Transformers) ---


NLP & Transformers
Covers: tokenization, BPE, TF-IDF, BM25, Word2Vec, RNN/LSTM,
        attention, transformers, BERT, ModelVisualizer, AutoMLEngine
"""

QUIZ = {
    "module": "MLFP05",
    "title": "NLP & Transformers",
    "questions": [
        # ── Section A: Text preprocessing ───────────────────────────────
        {
            "id": "8.A.1",
            "lesson": "8.A",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student's BPE tokenizer from Exercise 1 produces 12,000 unique tokens "
                "on a Singapore news corpus, but the vocabulary size was set to 5,000. "
                "The merge loop runs but the vocabulary keeps growing. What is wrong?"
            ),
            "code": (
                "# BPE merge step\n"
                "for i in range(num_merges):\n"
                "    pairs = get_pair_frequencies(tokens)\n"
                "    best_pair = max(pairs, key=pairs.get)\n"
                "    new_token = best_pair[0] + best_pair[1]\n"
                "    vocab.add(new_token)  # Bug: num_merges exceeds target vocab\n"
                "    tokens = merge_pair(tokens, best_pair, new_token)\n"
            ),
            "options": [
                "A) BPE should remove the individual tokens after merging — vocab.discard(best_pair[0]); vocab.discard(best_pair[1])",
                "B) The num_merges parameter should be set to 5000 - 256 (target vocab minus base characters). The vocabulary GROWS by design in BPE — each merge ADDS one token. Starting from 256 base characters, 4744 merges give exactly 5000 tokens. The bug is that num_merges is too large.",
                "C) get_pair_frequencies should only count pairs that appear more than 10 times",
                "D) BPE requires sorting the corpus alphabetically before merging",
            ],
            "answer": "B",
            "explanation": (
                "BPE starts with a base vocabulary (typically 256 byte-level tokens) and "
                "iteratively merges the most frequent pair into a new token. Each merge adds "
                "exactly one token to the vocabulary. To reach target_vocab_size, you need "
                "target_vocab_size - base_vocab_size merges. The code never removes old tokens "
                "because BPE doesn't remove them — subword units remain available. "
                "The fix is: num_merges = 5000 - len(base_vocab), not an arbitrary large number."
            ),
            "learning_outcome": "Implement BPE tokenization with correct merge count for target vocabulary",
        },
        {
            "id": "8.A.2",
            "lesson": "8.A",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "You are preprocessing Singapore Parliament speeches for classification. "
                "Should you apply stemming or lemmatization? The corpus contains legal "
                "terms like 'governing', 'governance', 'government', 'governed'."
            ),
            "options": [
                "A) Stemming — faster and simpler; all four words reduce to 'govern', which is sufficient for classification",
                "B) Lemmatization — preserves meaning distinctions. 'Governance' (noun: the act of governing) and 'government' (noun: the governing body) are different concepts in legal text. Stemming collapses them into 'govern', losing the distinction between policy concepts and institutions.",
                "C) Neither — modern tokenizers like BPE handle this automatically",
                "D) Both — apply stemming first, then lemmatization on the stems",
            ],
            "answer": "B",
            "explanation": (
                "In legal/parliamentary text, the distinction between 'governance' (process), "
                "'government' (institution), and 'governed' (past participle) carries meaning. "
                "Porter stemming reduces all to 'govern', losing these distinctions. "
                "Lemmatization maps to dictionary forms while preserving part-of-speech: "
                "governance→governance (noun), governed→govern (verb). "
                "For classification, these meaning differences affect which topic a speech belongs to. "
                "Note: BPE tokenizers in modern transformers largely bypass this choice, but for "
                "traditional NLP pipelines (Exercise 1), lemmatization preserves more signal."
            ),
            "learning_outcome": "Choose between stemming and lemmatization based on domain requirements",
        },
        # ── Section B: BoW / TF-IDF ────────────────────────────────────
        {
            "id": "8.B.1",
            "lesson": "8.B",
            "type": "output_interpretation",
            "difficulty": "intermediate",
            "question": (
                "Exercise 2 computes TF-IDF on Parliament speeches. The word 'Singapore' "
                "has high TF in every document but very low IDF (because it appears in 95% "
                "of documents). Its TF-IDF score is near zero. A colleague says 'Singapore "
                "must be important — it's everywhere!' Why is the low TF-IDF score correct?"
            ),
            "options": [
                "A) It's a bug — words that appear frequently must have high TF-IDF",
                "B) TF-IDF measures DISCRIMINATIVE power, not importance. 'Singapore' in Singapore Parliament speeches is like 'the' — it appears everywhere and distinguishes nothing. IDF = log(N/df) penalizes terms appearing in many documents. High TF × low IDF ≈ 0. Words like 'cryptocurrency' or 'housing' that appear in few documents have high IDF and actually distinguish topics.",
                "C) The IDF formula is wrong — it should not use logarithm",
                "D) 'Singapore' should be added to the stop word list and removed entirely",
            ],
            "answer": "B",
            "explanation": (
                "TF-IDF's purpose is retrieval and classification, which require DISCRIMINATIVE "
                "features. A word that appears in 95% of documents has IDF ≈ log(100/95) ≈ 0.05 — "
                "nearly zero. Multiplied by any TF, the score stays near zero. "
                "This is by design: ubiquitous terms don't help distinguish documents. "
                "'Cryptocurrency' appearing in 3/100 documents has IDF ≈ log(100/3) ≈ 3.5 — "
                "70× higher discriminative power. This is exactly what Exercise 2 demonstrates "
                "when comparing TF-IDF retrieval quality across terms."
            ),
            "learning_outcome": "Interpret TF-IDF scores as discriminative power, not importance",
        },
        {
            "id": "8.B.2",
            "lesson": "8.B",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 2 implements both TF-IDF and BM25 for document retrieval. "
                "On short queries (2-3 words), BM25 returns better results. On long queries "
                "(full sentences), TF-IDF performs similarly. Why does BM25 excel on short queries?"
            ),
            "options": [
                "A) BM25 uses a different tokenizer that handles short text better",
                "B) BM25 adds term frequency saturation: TF contribution approaches an asymptote as frequency increases (controlled by k1). A word appearing 20× vs 10× in a document gets diminishing extra credit. TF-IDF's linear TF scaling over-weights repeated terms. For short queries with few terms, BM25's saturation prevents any single matching term from dominating the score.",
                "C) BM25 uses word embeddings internally while TF-IDF uses only bag-of-words",
                "D) Short queries have too few terms for TF-IDF to work; BM25 has a minimum score floor",
            ],
            "answer": "B",
            "explanation": (
                "BM25's TF component: tf_bm25 = (k1+1)×tf / (k1×(1-b+b×dl/avgdl) + tf). "
                "As tf→∞, this approaches (k1+1), creating saturation. "
                "TF-IDF's TF is linear: a document with 'finance' 20 times scores 2× a document "
                "with it 10 times, even though both are clearly about finance. "
                "BM25 recognizes diminishing returns: 20× vs 10× barely matters. "
                "For short queries (2-3 terms), this prevents a single high-frequency match from "
                "dominating over documents that match ALL query terms moderately. "
                "The b parameter additionally normalizes for document length."
            ),
            "learning_outcome": "Explain BM25's term frequency saturation advantage over linear TF-IDF",
        },
        # ── Section C: Word embeddings ──────────────────────────────────
        {
            "id": "8.C.1",
            "lesson": "8.C",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 3 trains Word2Vec on Singapore news. You test analogies: "
                "vec('MAS') - vec('Singapore') + vec('USA') returns vec('Federal Reserve') "
                "as the closest match. But vec('HDB') - vec('Singapore') + vec('USA') returns "
                "vec('apartment') instead of the expected vec('HUD'). Why?"
            ),
            "options": [
                "A) The embedding dimensions are too small to capture housing concepts",
                "B) HDB is a Singapore-specific acronym that appears primarily in Singapore context. The model learned 'HDB' as associated with 'flat/apartment/housing' rather than as a government agency. MAS appears in international finance contexts alongside Fed/ECB, so its 'role' is better captured. Embeddings reflect co-occurrence patterns, not semantic taxonomy.",
                "C) Word2Vec cannot handle acronyms — only full words",
                "D) The analogy formula is wrong — it should be addition only, not subtraction",
            ],
            "answer": "B",
            "explanation": (
                "Word2Vec's distributional hypothesis: words that appear in similar contexts "
                "get similar vectors. MAS appears alongside 'central bank', 'monetary policy', "
                "'interest rates' — contexts shared with Federal Reserve. "
                "HDB appears alongside 'flat', 'resale', 'BTO', 'housing' — contexts more "
                "similar to 'apartment' than to 'HUD' (which rarely appears in the corpus). "
                "Analogies work when the relational pattern (institution→country) is consistently "
                "represented in training data. Singapore-specific entities may not have enough "
                "cross-national context for perfect analogies."
            ),
            "learning_outcome": "Interpret word embedding analogies as co-occurrence patterns",
        },
        {
            "id": "8.C.2",
            "lesson": "8.C",
            "type": "output_interpretation",
            "difficulty": "intermediate",
            "question": (
                "Exercise 3 visualizes Word2Vec embeddings with ModelVisualizer t-SNE. "
                "You see three clear clusters: one with financial terms, one with legal terms, "
                "one with technology terms. But 'fintech' appears between the financial and "
                "technology clusters. What does this mean?"
            ),
            "options": [
                "A) 'fintech' is an outlier and should be removed from the vocabulary",
                "B) The t-SNE perplexity parameter is set incorrectly — increase it to force 'fintech' into one cluster",
                "C) 'fintech' co-occurs with both financial AND technology terms in the corpus. Its embedding is a blend of both contexts, placing it between the two clusters in vector space. This is correct behavior — the embedding captures that fintech is genuinely at the intersection of finance and technology.",
                "D) t-SNE distorted the original distances — in the original high-dimensional space, 'fintech' is in the finance cluster",
            ],
            "answer": "C",
            "explanation": (
                "Word embeddings represent words as points in continuous space. Words used in "
                "multiple contexts get vectors that blend those contexts. 'Fintech' appears in "
                "sentences about both 'banking regulations' and 'machine learning', so its vector "
                "has components from both semantic neighborhoods. "
                "t-SNE preserves local structure: if 'fintech' is equidistant from 'banking' and "
                "'AI' in 300-dimensional space, it will appear between them in 2D. "
                "This is one of the most useful properties of embeddings — they capture "
                "semantic relationships that discrete categories cannot."
            ),
            "learning_outcome": "Interpret t-SNE embedding visualizations from ModelVisualizer",
        },
        # ── Section D: RNNs / LSTMs ─────────────────────────────────────
        {
            "id": "8.D.1",
            "lesson": "8.D",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "Exercise 4 implements a vanilla RNN for sentiment analysis. The model achieves "
                "98% accuracy on short reviews (< 20 words) but only 52% on long reviews "
                "(> 100 words). What architectural issue explains this and what fix does "
                "Exercise 4 demonstrate?"
            ),
            "options": [
                "A) Long reviews have more complex sentiment — add more hidden units",
                "B) The vanilla RNN suffers from vanishing gradients over long sequences. After 100 steps, gradient ≈ (W_hh)^100 which vanishes if max eigenvalue < 1. The model 'forgets' early words. Exercise 4 fixes this with LSTM: the forget/input/output gates and cell state provide a gradient highway that preserves information over long sequences.",
                "C) Long reviews should be truncated to 20 words to match the short review performance",
                "D) The embedding layer needs larger dimensions for longer texts",
            ],
            "answer": "B",
            "explanation": (
                "Vanilla RNN hidden state: h_t = tanh(W_hh × h_{t-1} + W_xh × x_t). "
                "Gradient through time: ∂h_T/∂h_1 = Π(W_hh × diag(tanh')). "
                "If ||W_hh|| < 1, gradients vanish exponentially with sequence length. "
                "After 100 steps, early words have near-zero influence on the final hidden state. "
                "LSTM's cell state c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t provides an additive "
                "gradient path. The forget gate f_t can be close to 1, passing gradients through "
                "unchanged — solving the vanishing gradient problem for sequences. "
                "Exercise 4 demonstrates: LSTM maintains 85%+ accuracy even on 100+ word reviews."
            ),
            "learning_outcome": "Diagnose RNN sequence length limitation and apply LSTM solution",
        },
        {
            "id": "8.D.2",
            "lesson": "8.D",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 4 builds a bidirectional LSTM. For the review 'The food was terrible "
                "but the service was excellent, so overall I recommend it', the unidirectional "
                "LSTM classifies as negative (attending to 'terrible') while the bidirectional "
                "LSTM classifies as positive. Why does bidirectionality help here?"
            ),
            "options": [
                "A) Bidirectional LSTMs use twice as many parameters, so they are always more accurate",
                "B) The forward LSTM processes left-to-right, heavily influenced by 'terrible' early in the sequence. By the time it reaches 'I recommend it', the signal is diluted. The backward LSTM processes right-to-left, starting from 'recommend it' — capturing the overall positive conclusion. Concatenating both directions captures the full sentiment arc.",
                "C) Bidirectional LSTMs can attend to any word in the sequence like attention mechanisms",
                "D) The unidirectional model is simply undertrained — more epochs would fix it",
            ],
            "answer": "B",
            "explanation": (
                "A forward-only LSTM at position t only knows words 1..t. By the final position, "
                "early strong signals ('terrible') may be diluted by subsequent words. "
                "The backward LSTM at position t knows words t..T, so at position 1 it has seen "
                "the entire review including the concluding 'recommend it'. "
                "Bidirectional concatenation [h_forward; h_backward] at each position gives the "
                "classifier access to both past and future context. "
                "For sentiment analysis, the concluding sentiment ('overall I recommend') often "
                "overrides earlier complaints, which the backward pass captures effectively."
            ),
            "learning_outcome": "Explain bidirectional LSTM advantage for sentiment with mixed signals",
        },
        # ── Section E: Attention ─────────────────────────────────────────
        {
            "id": "8.E.1",
            "lesson": "8.E",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "Exercise 5 implements scaled dot-product attention. The attention weights "
                "are all nearly uniform (1/n for n tokens) regardless of input. The model "
                "performs no better than averaging all token embeddings. What is missing?"
            ),
            "code": (
                "def attention(Q, K, V):\n"
                "    scores = matmul(Q, K.T)  # Bug: missing scaling\n"
                "    weights = softmax(scores)\n"
                "    return matmul(weights, V)\n"
            ),
            "options": [
                "A) The softmax temperature is too high — add a temperature parameter of 0.1",
                "B) Missing the scaling factor 1/√d_k. Without it, dot products grow proportionally to d_k (dimension). For d_k=512, scores can reach magnitude ~22 (√512). But softmax(22×[1,1,...]) ≈ softmax([22,22,...]) = uniform. Dividing by √d_k normalizes scores to unit variance, allowing softmax to differentiate: scores / √d_k.",
                "C) Q, K, V should be the same tensor — self-attention requires Q=K=V",
                "D) The attention is missing positional encoding — without positions, all tokens look identical",
            ],
            "answer": "B",
            "explanation": (
                "For random vectors of dimension d_k, the expected dot product has variance d_k. "
                "With d_k=512, dot products have std ≈ √512 ≈ 22.6. When all scores are "
                "uniformly large, softmax saturates to near-uniform distribution (all exp(22) "
                "are similar). Scaling by 1/√d_k gives variance 1, allowing meaningful differences: "
                "softmax([3.1, -0.5, 1.2]) → [0.72, 0.02, 0.26] — actual attention! "
                "This is why the mechanism is called 'Scaled Dot-Product Attention'. "
                "Exercise 5 demonstrates the difference: without scaling, attention degrades to "
                "mean pooling."
            ),
            "learning_outcome": "Implement scaled dot-product attention with correct √d_k scaling",
        },
        {
            "id": "8.E.2",
            "lesson": "8.E",
            "type": "output_interpretation",
            "difficulty": "advanced",
            "question": (
                "Exercise 5 visualizes attention weights. For 'The bank by the river', head 1 "
                "attends 'bank' → 'river' (0.8) while head 2 attends 'bank' → 'The' (0.6). "
                "For 'The bank approved the loan', head 1 attends 'bank' → 'loan' (0.7) while "
                "head 2 attends 'bank' → 'approved' (0.5). What do the heads learn?"
            ),
            "options": [
                "A) Head 1 always attends to the last noun; head 2 always attends to the first word",
                "B) Head 1 learns semantic disambiguation — attending to context words ('river' vs 'loan') that determine the meaning of 'bank'. Head 2 learns syntactic relationships — attending to grammatically related words ('The' as determiner, 'approved' as predicate). Multi-head attention captures different relationship types simultaneously.",
                "C) The two heads are redundant — one should be removed to save computation",
                "D) Head 1 and head 2 randomly attend to different words — there is no learned pattern",
            ],
            "answer": "B",
            "explanation": (
                "Multi-head attention allows the model to attend to different types of "
                "relationships simultaneously. Head 1 consistently attends to words that "
                "disambiguate meaning (semantic role), while head 2 attends to syntactically "
                "related words. This is emergent behavior — heads are not explicitly trained "
                "for specific roles, but the different Q/K/V projections learn complementary "
                "patterns. In transformers, different heads learn: positional, syntactic, "
                "semantic, and coreference relationships. This is why h=8 or h=16 heads "
                "are standard — each captures different aspects of the input."
            ),
            "learning_outcome": "Interpret multi-head attention patterns from ModelVisualizer output",
        },
        # ── Section F: Transformer architecture ─────────────────────────
        {
            "id": "8.F.1",
            "lesson": "8.F",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "Exercise 6 builds a transformer encoder. The model trains but performance "
                "is worse than the LSTM from Exercise 4. The student notices that shuffling "
                "the word order in input sentences doesn't change the model's predictions. "
                "What component is missing?"
            ),
            "options": [
                "A) The feed-forward network is too small — increase the hidden dimension",
                "B) Missing positional encoding. Self-attention is permutation-invariant — it has no notion of word order. Without positional encoding, 'dog bites man' and 'man bites dog' produce identical representations. Exercise 6 implements sinusoidal PE: PE(pos,2i) = sin(pos/10000^(2i/d)) which gives each position a unique signature.",
                "C) The model needs decoder layers, not just encoder layers",
                "D) LayerNorm should be applied before attention, not after (Pre-LN vs Post-LN)",
            ],
            "answer": "B",
            "explanation": (
                "Self-attention computes: Attention(Q,K,V) = softmax(QK^T/√d)V. "
                "If you permute the input tokens, Q, K, V are permuted identically, and "
                "the output is the same permutation of the original output. Word ORDER is lost. "
                "Positional encoding adds position-dependent vectors to token embeddings: "
                "x_i = embed(token_i) + PE(i). The sinusoidal formula creates unique patterns "
                "for each position and allows the model to learn relative positions "
                "(PE(pos+k) can be expressed as a linear function of PE(pos)). "
                "Without PE, the transformer degrades to a bag-of-words model with attention."
            ),
            "learning_outcome": "Identify missing positional encoding from order-invariant predictions",
        },
        {
            "id": "8.F.2",
            "lesson": "8.F",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 6 builds a transformer with 4 layers, d_model=256, and 4 attention "
                "heads. The model has 2.5M parameters and takes 45 minutes to train on your "
                "corpus of 10,000 documents. Your colleague suggests using 12 layers and "
                "d_model=768 (like BERT-base). Should you?"
            ),
            "options": [
                "A) Yes — larger models always perform better, and the training time increase is linear",
                "B) No — 12 layers with d_model=768 has ~110M parameters (44× more). On 10,000 documents, this will massively overfit. The model capacity should match data size. 4 layers is appropriate for 10K documents. Use transfer learning (Exercise 7 with AutoMLEngine) instead of training from scratch if you need BERT-level performance.",
                "C) Yes — but only if you also increase the number of attention heads to 12",
                "D) No — 12 layers cannot fit in GPU memory regardless of the data size",
            ],
            "answer": "B",
            "explanation": (
                "BERT-base (110M params) was trained on 3.3B words. With only 10K documents "
                "(perhaps 1M words), you have ~100 parameters per word — severe data starvation. "
                "The 4-layer, 2.5M param model has ~2.5 parameters per word, which is reasonable. "
                "If you need BERT-level representations, Exercise 7 demonstrates the right approach: "
                "AutoMLEngine with transfer learning fine-tunes a pre-trained BERT on your small "
                "dataset, leveraging the 3.3B-word pre-training while adapting to your domain. "
                "Training a 110M model from scratch on 10K documents is the wrong approach."
            ),
            "learning_outcome": "Match transformer model capacity to dataset size and choose transfer learning",
        },
        # ── Section G: Transfer learning ────────────────────────────────
        {
            "id": "8.G.1",
            "lesson": "8.G",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 7 uses AutoMLEngine with task='text_classification' on Singapore "
                "product reviews. The leaderboard shows: TF-IDF+LR (F1=0.82), "
                "TF-IDF+SVM (F1=0.84), Transformer+Classifier (F1=0.91). "
                "Why does the transformer model outperform TF-IDF approaches by 7+ points?"
            ),
            "options": [
                "A) Transformers use more parameters so they always achieve higher F1",
                "B) TF-IDF treats words independently — 'not good' has the same features as 'good not'. Transformer embeddings capture contextual meaning: 'good' in 'not good' gets a different representation than 'good' in 'very good'. This contextual understanding is critical for sentiment where negation, sarcasm, and hedging change meaning.",
                "C) AutoMLEngine allocated more training time to the transformer model",
                "D) TF-IDF cannot handle Singapore English (Singlish) vocabulary",
            ],
            "answer": "B",
            "explanation": (
                "TF-IDF is a bag-of-words model: word order and context are lost. "
                "'This movie is not good at all' and 'This movie is good, not bad at all' "
                "produce similar TF-IDF vectors despite opposite sentiment. "
                "Transformer embeddings are contextual: each word's representation depends "
                "on surrounding words. The pre-trained transformer has already learned these "
                "contextual patterns from billions of words, giving it a massive advantage "
                "even on small fine-tuning datasets. "
                "The 7-point F1 gap reflects the value of contextual vs bag-of-words features."
            ),
            "learning_outcome": "Explain transformer advantage over TF-IDF for text classification",
        },
        {
            "id": "8.G.2",
            "lesson": "8.G",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "Exercise 7 registers the best model in ModelRegistry. A teammate asks "
                "whether to use AutoMLEngine or TrainingPipeline for their next text "
                "classification task (5,000 labeled medical reports). What do you recommend?"
            ),
            "options": [
                "A) TrainingPipeline — AutoMLEngine is only for prototyping",
                "B) AutoMLEngine — it searches across model types (TF-IDF, transformer) and hyperparameters automatically. For a new task with unknown optimal approach, AutoML exploration is more efficient than manually configuring TrainingPipeline. Once AutoML identifies the best approach, you can switch to TrainingPipeline for fine-grained control in production.",
                "C) Neither — 5,000 samples is too few for any text classification model",
                "D) TrainingPipeline with model_type='bert' — always use the largest available model",
            ],
            "answer": "B",
            "explanation": (
                "AutoMLEngine's value is exploration: it tries multiple approaches (TF-IDF+LR, "
                "TF-IDF+SVM, transformer-based) with various hyperparameters in a time-bounded "
                "search. For a new task, you don't know if transformer fine-tuning will beat "
                "TF-IDF+SVM (on small datasets, simpler models sometimes win). "
                "AutoML discovers this empirically. Once the best approach is identified, "
                "TrainingPipeline offers more control for production tuning. "
                "5,000 samples is sufficient for text classification — transfer learning from "
                "pre-trained transformers requires as few as 100-1,000 labeled examples."
            ),
            "learning_outcome": "Choose AutoMLEngine for exploration vs TrainingPipeline for production",
        },
        # ── Section H: NLP Tasks & Decoding ────────────────────────────
        {
            "id": "8.H.1",
            "lesson": "8.H",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "A student generates text with beam search (num_beams=2) and the output "
                "keeps repeating: 'The company reported strong strong strong strong growth.' "
                "Increasing num_beams to 5 makes it worse: 'The company company company "
                "reported reported reported.' What is causing the repetition and what "
                "parameter fixes it?"
            ),
            "code": (
                "output = model.generate(\n"
                "    input_ids,\n"
                "    max_length=100,\n"
                "    num_beams=5,\n"
                "    # Bug: no repetition penalty\n"
                ")\n"
            ),
            "options": [
                "A) num_beams is too high — reduce to 1 (greedy decoding) to eliminate repetition",
                "B) Beam search maximizes total sequence probability. Repeated high-probability tokens compound: P('strong strong strong') can exceed P('strong quarterly growth') because 'strong' has high conditional probability given 'strong'. More beams explore more paths but converge on the same repetitive pattern. Fix: add repetition_penalty=1.2 (penalizes tokens already generated) or no_repeat_ngram_size=3 (blocks any 3-gram from repeating).",
                "C) The model vocabulary is too small — repeated tokens indicate missing words",
                "D) The input prompt is too short — longer prompts prevent repetition",
            ],
            "answer": "B",
            "explanation": (
                "Beam search selects the top-k most probable sequences at each step. "
                "Language models assign high probability to common n-grams, and once a word "
                "like 'strong' is generated, P('strong' | 'strong') remains high — creating "
                "a self-reinforcing loop. More beams makes this worse because all beams converge "
                "on the repetitive path (it's genuinely the highest probability sequence). "
                "Solutions: (1) repetition_penalty=1.2 divides the logit of any previously "
                "generated token by the penalty factor, reducing its probability. "
                "(2) no_repeat_ngram_size=3 hard-blocks any 3-token sequence from appearing twice. "
                "(3) Sampling with temperature (top_p, top_k) adds randomness that naturally "
                "avoids repetition but sacrifices determinism."
            ),
            "learning_outcome": "Diagnose beam search repetition and apply repetition penalty parameters",
        },
        {
            "id": "8.H.2",
            "lesson": "8.H",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 8 evaluates a summarization model. BLEU score is 0.32 and "
                "BERTScore F1 is 0.89. A colleague says 'BLEU is low — the model is bad.' "
                "You look at the outputs: the model paraphrases well but uses different words "
                "than the reference summaries. Which metric should you trust?"
            ),
            "options": [
                "A) BLEU — it is the gold standard for all text generation evaluation",
                "B) BERTScore. BLEU measures exact n-gram overlap between generated and reference text. A paraphrase like 'revenue increased 20%' vs reference 'sales grew by a fifth' scores low on BLEU (no shared n-grams) but high on BERTScore (semantic similarity via contextual embeddings). For summarization where paraphrasing is expected, BERTScore better captures output quality. BLEU is appropriate for translation where close lexical alignment is expected.",
                "C) Neither — use ROUGE instead, which is always correct for summarization",
                "D) Average the two scores: (0.32 + 0.89) / 2 = 0.605 for a balanced assessment",
            ],
            "answer": "B",
            "explanation": (
                "BLEU (Bilingual Evaluation Understudy) counts n-gram precision: how many "
                "n-grams in the output appear in the reference. 'Revenue increased 20%' vs "
                "'Sales grew by a fifth' shares zero 4-grams → BLEU-4 ≈ 0. "
                "BERTScore computes token-level cosine similarity using BERT embeddings: "
                "'revenue' ↔ 'sales' (similarity ~0.85), 'increased' ↔ 'grew' (~0.90). "
                "These semantic matches yield high BERTScore despite zero lexical overlap. "
                "For summarization, paraphrasing is desirable (summaries should compress, not copy). "
                "BLEU is designed for machine translation where the reference translation defines "
                "expected word choices. Using BLEU for summarization penalizes good paraphrasing."
            ),
            "learning_outcome": "Choose BERTScore over BLEU for evaluating paraphrastic text generation",
        },
    ],
}
