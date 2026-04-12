# Module 7: Deep Learning — Speaker Notes

Total time: ~180 minutes (3 hours)

---

## Slide 1: Deep Learning: Architecture-Driven Feature Engineering

**Time**: ~2 min
**Talking points**:

- Read the title slowly: "Architecture-Driven Feature Engineering." That phrase is the thesis of the whole module.
- Deep learning does not replace feature engineering — it moves it inside the model. The architecture IS the feature engineer.
- This is Module 7. Students have six modules of context. Respect that by connecting back: "In M3 you hand-crafted features. Today the network learns them."
- If beginners look confused: "Instead of us deciding which features matter, we build a network and let it figure that out."
- If experts look bored: "The key insight is representational learning — the hierarchy of transformations learned by the network is equivalent to a feature pipeline designed by a domain expert, except it is learned end-to-end."

**Transition**: "Before we go deep, let me remind you where you have been..."

---

## Slide 2: Recap: Your Journey So Far

**Time**: ~2 min
**Talking points**:

- Quick sweep: M1 data fluency, M2 statistics, M3 feature engineering, M4 supervised ML, M5 ML engineering, M6 unsupervised ML.
- Do not re-teach. This is an anchor, not a lecture.
- Point out the gap: "You can build a full ML pipeline, but the feature engineering step is still manual. Today we automate it."
- If beginners look confused: "Think of everything we have done as building the foundations. Today we build the tower."

**Transition**: "Here is what is new today..."

---

## Slide 3: What's New in Module 7

**Time**: ~2 min
**Talking points**:

- New concepts: neurons, layers, backpropagation, CNNs, embeddings, optimizers, regularization.
- New Kailash engines: OnnxBridge, InferenceServer, ModelVisualizer (plus TrainingPipeline and ModelRegistry from M4 used in new ways).
- This is the longest module — 12 sections. Reassure students: "We move from first principles to production in one session."
- If beginners look confused: "We start from linear regression — something you already know — and grow it into a full deep learning system step by step."

**Transition**: "Here is the full engine map so you can see how everything fits together..."

---

## Slide 4: Cumulative Kailash Engine Map

**Time**: ~2 min
**Talking points**:

- Show the map: M1 through M7 engines side by side.
- Highlight the M7 additions: OnnxBridge, InferenceServer, ModelVisualizer.
- "Every engine you learned before is still in play. We are adding the final layer: deep model export and serving."
- If experts look bored: "OnnxBridge is the bridge between training (PyTorch) and production (any runtime). This is the pattern you will use in the real world."

**Transition**: "Let me situate deep learning on the broader feature engineering spectrum..."

---

## Slide 5: The Feature Engineering Spectrum

**Time**: ~3 min
**Talking points**:

- Draw the spectrum on the whiteboard if possible: raw data → manual features → learned features (shallow) → learned features (deep).
- Traditional ML (M3-M4): we hand-craft features, feed them to a model.
- Unsupervised ML (M6): the model finds structure, but features are still mostly manual.
- Deep learning: the layers ARE the feature engineering pipeline — learned jointly with the task.
- If beginners look confused: "Imagine a recipe. Traditional ML: you choose all the ingredients. Deep learning: the chef figures out the best ingredients for your taste automatically."
- If experts look bored: "The spectrum maps to the bias-variance trade-off — manual features embed domain bias, learned features reduce it at the cost of needing more data and compute."

**Transition**: "Let me start with a concrete historical moment that makes this real..."

---

## Slide 6: Opening Case: The AlexNet Moment

**Time**: ~3 min
**Talking points**:

- ImageNet 2012. Top-5 error: second place 26.2%, AlexNet 15.3%.
- That 10-point gap was not incremental. It broke a decade-long plateau.
- The audience: everyone had been doing hand-crafted features (HOG, SIFT, SURF). AlexNet used none of them.
- If beginners look confused: "Imagine a competition to recognise cats in photos. For ten years, every team was hand-writing rules for what makes a cat. AlexNet said: show me a million photos and let me figure it out."
- If experts look bored: "The AlexNet gap was so large that the community initially thought it was a mistake. The second-place team was using an ensemble of HOG-based SVMs. The architecture and GPU training were the entire delta."

**Transition**: "So what did AlexNet actually do differently?"

---

## Slide 7: What AlexNet Actually Did

**Time**: ~3 min
**Talking points**:

- Five convolutional layers, three fully connected, ReLU activations, dropout, data augmentation, GPU training.
- The insight: each layer learns increasingly abstract representations. Layer 1 = edges. Layer 3 = textures. Layer 5 = object parts.
- "This is architecture-driven feature engineering in action. The architecture design IS the feature engineering decision."
- If beginners look confused: "The first layer learns to see lines and edges. The next learns corners and curves. The deepest layers learn faces and wheels. The model builds its own feature vocabulary."
- If experts look bored: "Zeiler and Fergus (2013) visualised AlexNet's learned filters. The Layer 1 Gabor-like filters emerged from random initialisation, which validated that convolution is the right inductive bias for images."

**Transition**: "Why does this matter for you specifically?"

---

## Slide 8: Why This Matters for You

**Time**: ~3 min
**Talking points**:

- Three industries where DL displaced manual feature engineering: vision (AlexNet), NLP (BERT), tabular (M8 preview).
- Practical takeaway: knowing when to use DL vs traditional ML is the professional skill.
- "DL is not always better. More data, more compute, less interpretability. Your job is to know when the trade-off is worth it."
- If beginners look confused: "Deep learning is a powerful tool, but it is not the only tool. We will teach you when to use it."
- If experts look bored: "The modern practitioner's dilemma: a 3-layer MLP on good features often beats a ResNet on raw data. The decision is data-regime dependent."

**Transition**: "Now let us build from the ground up. Section 7.1: linear regression as a neural network..."

---

## Slide 9: 7.1 Linear Regression as a Neural Network

**Time**: ~2 min
**Talking points**:

- Section marker. Pause. Let students know we are starting from something they already know.
- "This section has one goal: show you that you already understand neural networks. You just did not know it yet."
- If beginners look confused: "We learned linear regression in M2. Now we will see it is just the simplest possible neural network."

**Transition**: "Let me show you the connection directly..."

---

## Slide 10: Linear Regression IS a Neural Network

**Time**: ~4 min
**Talking points**:

- Draw the diagram: one input layer, one output neuron, no hidden layers, no activation function (or identity activation).
- y = Wx + b. That is linear regression. That is also a 1-layer neural network.
- The same computation, two names. This is not a metaphor — it is literally the same math.
- Walk through the notation: weights W = regression coefficients, bias b = intercept, output = prediction.
- If beginners look confused: "The line y = mx + b that you drew in school IS a neural network. The 'm' is the weight, the 'b' is the bias."
- If experts look bored: "The activation function is the identity — linear regression is a degenerate neural network with no non-linearity. Adding hidden layers with non-linear activations is what gives us the Universal Approximation property."

**Transition**: "How do we train this? With gradient descent..."

---

## Slide 11: Training: Gradient Descent

**Time**: ~4 min
**Talking points**:

- Loss surface: imagine a bowl. Gradient descent rolls down the bowl to find the minimum.
- The update rule: w_new = w_old - learning_rate \* gradient.
- Three elements: loss function (MSE for regression), gradient (direction of steepest ascent), learning rate (step size).
- Draw the 2D loss surface. Mark a random starting point. Show the path rolling down.
- If beginners look confused: "Imagine you are blindfolded on a hillside. To find the valley (minimum loss), you feel which direction is steepest downhill and take a small step. That is gradient descent."
- If experts look bored: "The loss surface for linear regression is convex — one global minimum, guaranteed convergence. Non-convex surfaces (deep networks) are why we need momentum, adaptive rates, and careful initialisation."

**Transition**: "But wait — we already have an exact solution for linear regression. Why use gradient descent at all?"

---

## Slide 12: From OLS to Gradient Descent: Why Bother?

**Time**: ~3 min
**Talking points**:

- OLS: closed-form solution, O(n^3) matrix inversion, exact answer.
- Gradient descent: iterative, O(n) per step, approximate, but scales to millions of parameters.
- The key insight: you cannot invert a 100-million-parameter matrix. Gradient descent is the only tractable option.
- "OLS is the sports car that only fits two people. Gradient descent is the bus that can carry everyone."
- If beginners look confused: "With 10 features you can solve the equation exactly. With 10 million features — like in a real neural network — you cannot. Gradient descent is the only way."
- If experts look bored: "The OLS normal equations require inverting X^T X — O(p^3) time and O(p^2) memory. With p = 10M parameters, that matrix would require petabytes of RAM. Gradient descent sidesteps this entirely."

**Transition**: "Let me show you what this looks like in code..."

---

## Slide 13: Code: Linear Regression as a Neural Network

**Time**: ~4 min
**Talking points**:

- Walk through the code step by step: import TrainingPipeline, define ModelSpec with a single linear layer, no activation, MSE loss.
- Emphasise: this is the same TrainingPipeline from M4. The engine is general — it handles everything from linear regression to ResNets.
- Show the connection: the Kailash layer definition maps directly to the mathematical notation on the previous slide.
- PAUSE. Ask: "What would we change to make this logistic regression?" (add sigmoid activation, change loss to BCE)
- If beginners look confused: "The code is just the math written in Python. Weight, bias, forward pass, loss, gradient step — each line has a direct mathematical counterpart."
- If experts look bored: "The TrainingPipeline abstracts away the gradient computation. Under the hood it is calling autograd. We will see exactly how that works in section 7.6."

**Transition**: "Linear regression can only learn linear relationships. What if the world is non-linear? Section 7.2..."

---

## Slide 14: 7.2 The Interaction Problem

**Time**: ~2 min
**Talking points**:

- Section marker. The problem we are solving: linear regression cannot capture interactions between features.
- Classic example: income and age interact. A 25-year-old earning $50k is unusual. A 50-year-old earning $50k is not. Linear models miss this.
- "In M3, we hand-crafted interaction terms. Hidden layers do this automatically."
- If beginners look confused: "Some patterns only appear when you look at two features together, not separately. Hidden layers learn those combinations automatically."

**Transition**: "Here is how hidden layers solve this..."

---

## Slide 15: Hidden Layers: Automatic Interaction Discovery

**Time**: ~4 min
**Talking points**:

- Walk through the architecture: input layer → hidden layer (with non-linear activation) → output layer.
- The hidden layer transforms the input space. Each hidden neuron learns a different linear combination of inputs, then applies a non-linearity.
- The non-linearity is essential. Without it, multiple linear layers collapse back to one linear layer (composition of linear functions is linear).
- Draw it: two input features, three hidden neurons, one output. Show how each hidden neuron looks at the inputs differently.
- If beginners look confused: "Each hidden neuron is asking a different question about your data. One asks 'is feature A high AND feature B low?' Another asks 'are both features moderate?' The final output combines all those answers."
- If experts look bored: "The hidden layer implements a non-linear basis expansion. This is equivalent to kernel methods — but learned rather than fixed. The Universal Approximation Theorem guarantees that one hidden layer with enough neurons can approximate any continuous function."

**Transition**: "The cleanest illustration of why non-linearity matters is the XOR problem..."

---

## Slide 16: The XOR Problem: Why Hidden Layers Matter

**Time**: ~4 min
**Talking points**:

- XOR truth table: (0,0) → 0, (0,1) → 1, (1,0) → 1, (1,1) → 0. Try to draw a straight line separating 0s from 1s. You cannot.
- This is linearly non-separable. A single-layer neural network (= linear regression/logistic regression) will fail.
- Historical moment: Minsky and Papert (1969) used XOR to argue that perceptrons were useless. It killed the first AI winter.
- Solution: add one hidden layer. Now the network can draw two lines and combine them.
- If beginners look confused: "XOR means 'one OR the other, but not both.' You cannot separate the true cases from false cases with a straight line. One hidden layer lets the network draw curved boundaries."
- If experts look bored: "XOR is the canonical example of a problem requiring at least one hidden layer. The hidden layer maps the input to a new space where the problem becomes linearly separable — this is the kernel trick intuition applied to learned representations."

**Transition**: "What does the hidden layer actually learn when it solves XOR?"

---

## Slide 17: What the Hidden Layer Actually Learns

**Time**: ~3 min
**Talking points**:

- Show the learned weights after training on XOR. The two hidden neurons have learned different views of the input space.
- Visualise the decision boundary: the network has transformed the input into a new space where XOR is linearly separable.
- Key insight: this transformation is learned, not hand-crafted. The network figured out the right transformation from data.
- "You just saw automatic feature engineering. The hidden layer is a learned feature extractor."
- If beginners look confused: "The hidden layer re-draws the picture so the classes become separable. Then the output layer just draws a straight line in the new picture."
- If experts look bored: "The hidden layer implements a learned manifold. For XOR, the optimal hidden representation is the NAND gate — which the network discovers without being told."

**Transition**: "Now that we understand hidden layers, let us ask: where does DL's real power come from?"

---

## Slide 18: 7.3 Where DL's Power Comes From

**Time**: ~2 min
**Talking points**:

- Section marker. Three sources of DL power: depth (hierarchical learning), width (parallel hypotheses), and scale (data + compute).
- Preview: this section explains WHY deep networks outperform shallow ones — not just empirically, but theoretically.
- If beginners look confused: "We know DL works. This section explains why."

**Transition**: "Let me start with the big picture — the power hierarchy..."

---

## Slide 19: The Power Hierarchy

**Time**: ~3 min
**Talking points**:

- The hierarchy: linear → shallow non-linear → deep → wide deep → scaled deep.
- Each step adds a qualitatively different capability, not just more parameters.
- The key insight: a deep network with fewer parameters can approximate functions that a shallow network requires exponentially more neurons to express.
- "Depth is not just more of the same. It is a qualitative leap in expressiveness per parameter."
- If beginners look confused: "Adding depth is like adding floors to a building. Each floor can see further. A 10-floor building is not just ten 1-floor buildings stacked — it changes what you can do."
- If experts look bored: "The depth separation theorem (Telgarsky 2016) formalises this: certain functions require exponentially many neurons to represent with one hidden layer but only polynomially many with k hidden layers."

**Transition**: "Let me make the hierarchy concrete with the concept of hierarchical feature learning..."

---

## Slide 20: Depth = Hierarchical Feature Learning

**Time**: ~4 min
**Talking points**:

- The canonical example: image recognition.
- Layer 1: pixel intensity patterns → edges.
- Layer 2: edges → corners and curves.
- Layer 3: corners and curves → textures and shapes.
- Layer 4: shapes → object parts (wheel, eye, wing).
- Layer 5: object parts → objects (car, face, bird).
- "The depth is the hierarchy. Each layer builds on the previous layer's vocabulary."
- This is why we call it "deep learning" — the depth enables this hierarchy.
- If beginners look confused: "Think of learning to read. Letters → syllables → words → sentences → meaning. Each level uses the previous level as building blocks. That is depth."
- If experts look bored: "This hierarchical decomposition is why deep networks generalise better — they learn compositional structure that mirrors the compositionality of the real world (Bengio et al., 2009)."

**Transition**: "But what about width? Is wider always better?"

---

## Slide 21: Width vs Depth

**Time**: ~4 min
**Talking points**:

- Width: more neurons per layer — more parallel hypotheses, more capacity.
- Depth: more layers — more hierarchical abstraction, better generalisation per parameter.
- The trade-off is empirical and task-dependent. For tabular data, sometimes wider is better. For images and text, deeper wins.
- Walk through the comparison table: params, expressiveness, training difficulty, inductive bias.
- PAUSE. Ask: "Given what you now know about images and hierarchical structure, why does depth beat width for vision tasks?"
- If beginners look confused: "Width makes the network smarter in parallel. Depth makes it smarter in stages. For some problems, stages are better."
- If experts look bored: "The theoretical analysis (Raghu et al., 2017) shows that deep ReLU networks have trajectory length that grows exponentially with depth but only polynomially with width. This is why depth generalises better on structured data."

**Transition**: "Now let us get precise. Section 7.4: the neuron..."

---

## Slide 22: 7.4 The Neuron

**Time**: ~2 min
**Talking points**:

- Section marker. We now formalise everything we have been saying informally.
- A neuron: takes a vector of inputs, computes a weighted sum plus bias, applies an activation function.
- z = w^T x + b, output = activation(z).
- If beginners look confused: "A neuron is just a little function. Inputs go in, it mixes them with weights, applies a squish function, and outputs a number."

**Transition**: "Now stack neurons into layers, and think about the matrix operations..."

---

## Slide 23: Layers as Matrix Operations

**Time**: ~4 min
**Talking points**:

- A layer with n inputs and m neurons: weight matrix W of shape (m, n), bias vector b of shape (m,).
- Forward pass: Z = W \* X + b, A = activation(Z).
- "Everything in deep learning is matrix multiplication. GPUs are fast because they are matrix multiplication machines."
- Walk through the shapes carefully. Dimension errors are the number-one debugging problem for students.
- If beginners look confused: "Think of the weight matrix as a scoreboard. Each row scores a different neuron. Matrix multiplication fills the scoreboard for all inputs at once."
- If experts look bored: "The batched forward pass is Z = XW^T + b — (batch x input) @ (input x output) = (batch x output). Getting comfortable with shape algebra is the single most important debugging skill in DL."

**Transition**: "How many parameters does a network actually have?"

---

## Slide 24: Parameter Count

**Time**: ~3 min
**Talking points**:

- Walk through the formula: each layer has (inputs x outputs) weights plus outputs biases.
- Example: input 784 → hidden 256 → hidden 128 → output 10.
- Layer 1: 784 x 256 + 256 = 200,960. Layer 2: 256 x 128 + 128 = 32,896. Layer 3: 128 x 10 + 10 = 1,290. Total: ~235K.
- Compare: GPT-2 has 117M, GPT-3 has 175B. Scale this intuition.
- PAUSE. Let students calculate a small network parameter count on paper. Takes 2 minutes but cements the concept.
- If beginners look confused: "For each connection between neurons, we have one number (weight). Plus one number per neuron (bias). Count the connections, add the neurons, that is your parameter count."
- If experts look bored: "Flop count scales differently from parameter count. For inference, parameters dominate memory; flops dominate latency. The parameter/flop ratio is an important architecture efficiency metric."

**Transition**: "Now the most important design decision in a layer: the activation function..."

---

## Slide 25: Activation Functions: The Complete Catalog

**Time**: ~3 min
**Talking points**:

- Overview slide: sigmoid, tanh, ReLU, Leaky ReLU, ELU, GELU, Softmax.
- "This is the menu. The next few slides explain when to order each dish."
- The rule of thumb: ReLU for hidden layers, GELU for transformers, Softmax for multi-class output.
- If beginners look confused: "The activation function is the squish. Without it, all layers collapse into one. Different squish functions have different properties."

**Transition**: "Start with the classics: sigmoid and tanh..."

---

## Slide 26: Sigmoid and Tanh: The Originals

**Time**: ~3 min
**Talking points**:

- Sigmoid: outputs (0, 1). Used in binary classification output layers and gates (LSTM).
- Tanh: outputs (-1, 1). Zero-centred — better gradient flow than sigmoid.
- The problem with both: saturation. At extreme inputs, gradients approach 0. This is vanishing gradients (preview of 7.7).
- "Sigmoid and tanh are not wrong — they are just slow. ReLU solves the speed problem."
- If beginners look confused: "Sigmoid squishes any number into 0 to 1 — good for probabilities. Tanh squishes to -1 to 1. Both flatten out at the extremes, which causes problems when training deep networks."
- If experts look bored: "The saturation problem is not just slow convergence — it is catastrophic for deep networks. The maximum derivative of sigmoid is 0.25. With 10 layers, gradient magnitude is bounded by 0.25^10 which is approximately 10^-6."

**Transition**: "The solution was ReLU..."

---

## Slide 27: ReLU and Variants

**Time**: ~4 min
**Talking points**:

- ReLU: max(0, x). Dead simple. Computationally trivial. Gradient is either 0 or 1 — no vanishing in the positive range.
- Why it works: sparse activations (roughly half the neurons are zero at any time), faster training, better generalisation.
- Variants: Leaky ReLU (small negative slope), ELU (smooth negative part), PReLU (learned slope).
- The tradeoff: dying ReLU problem. Neurons can get stuck in the zero region and never recover. (Preview of 7.7.)
- If beginners look confused: "ReLU says: if the input is negative, output 0. If positive, pass it through unchanged. That is it. But this simple rule makes training much faster."
- If experts look bored: "The sparsity induced by ReLU is a form of implicit regularisation — the network is forced to use a subset of neurons for any given input, which is equivalent to a learned ensemble."

**Transition**: "The most important modern variant: GELU..."

---

## Slide 28: GELU: The Transformer Activation

**Time**: ~3 min
**Talking points**:

- GELU: Gaussian Error Linear Unit. x \* Phi(x) where Phi is the Gaussian CDF.
- Smooth near zero (unlike ReLU which has a kink), stochastic interpretation (activates probabilistically based on magnitude).
- Used in BERT, GPT, and almost every modern transformer.
- "You will use GELU in M8. Understand it here."
- If beginners look confused: "GELU is a smoother version of ReLU. Instead of a hard cut at zero, it gradually fades. This smoothness helps transformers train more stably."
- If experts look bored: "The stochastic interpretation: GELU(x) = x \* P(X <= x) where X ~ N(0,1). This means GELU randomly gates a neuron proportional to its magnitude — a principled form of dropout baked into the activation."

**Transition**: "The output activation for multi-class problems: Softmax..."

---

## Slide 29: Softmax: The Output Activation

**Time**: ~3 min
**Talking points**:

- Softmax: converts a vector of real numbers into a probability distribution (sums to 1, all positive).
- Formula: softmax(z_i) = exp(z_i) / sum(exp(z_j)).
- Used as the final layer for multi-class classification.
- Numerical stability: always compute log-softmax in practice (avoids overflow in exp).
- If beginners look confused: "Softmax takes the network's raw scores for each class and converts them into percentages. The highest score gets the biggest percentage, but every class gets some."
- If experts look bored: "The connection to temperature: softmax(z/T). T approaching 0 gives argmax (hard), T approaching infinity gives uniform. Temperature scaling is used in knowledge distillation and sampling from language models."

**Transition**: "We have the architecture. Now we need to measure how wrong we are. Section 7.5: loss functions..."

---

## Slide 30: 7.5 Loss Functions for Regression

**Time**: ~4 min
**Talking points**:

- MSE: mean squared error. Penalises large errors quadratically. Sensitive to outliers.
- MAE: mean absolute error. Robust to outliers. Non-differentiable at zero (use Huber loss instead).
- Huber loss: MSE for small errors, MAE for large errors. Best of both.
- When to use: MSE default, Huber when you have outliers, MAE when outliers dominate.
- Show the curve shapes. Visual understanding is essential here.
- If beginners look confused: "MSE is the straightforward 'how wrong am I?' score. Big mistakes are punished much more than small ones. MAE treats all mistakes equally. Huber is a hybrid."
- If experts look bored: "The loss function defines the implicit noise model. MSE assumes Gaussian noise. MAE assumes Laplacian noise. Huber is a robustified Gaussian. Choosing the loss is choosing the probabilistic model."

**Transition**: "For classification, we need different losses..."

---

## Slide 31: Loss Functions for Classification

**Time**: ~4 min
**Talking points**:

- Binary cross-entropy: -[y log(p) + (1-y) log(1-p)]. For binary classification with sigmoid output.
- Categorical cross-entropy: -sum(y_i log(p_i)). For multi-class with softmax output.
- The intuition: cross-entropy penalises confident wrong predictions extremely heavily.
- Walk through a concrete example: true class = cat, predicted probability of cat = 0.01. Loss = -log(0.01) which is approximately 4.6. Very high.
- PAUSE. Ask: "Why is it catastrophic to be very confident and very wrong?"
- If beginners look confused: "Cross-entropy says: I don't care if you guess randomly, but if you are very sure and very wrong, I punish you heavily. This forces the model to be calibrated."
- If experts look bored: "Cross-entropy is equivalent to minimising KL divergence between the predicted and true distributions. MLE for classification with categorical likelihood IS cross-entropy minimisation."

**Transition**: "A preview of losses you will use in M8 and M9: embedding losses..."

---

## Slide 32: Embedding Losses (Preview)

**Time**: ~2 min
**Talking points**:

- Triplet loss: anchor, positive, negative. Pull matching pairs together, push non-matching pairs apart.
- Contrastive loss: similar pairs should have small embedding distance, dissimilar pairs large.
- InfoNCE: used in self-supervised learning (SimCLR, CLIP).
- "These are preview slides. We will implement them in M8. Right now, understand that they exist."
- If beginners look confused: "These losses teach the network to group similar things together in the hidden space. Useful for search, recommendation, and matching problems."
- If experts look bored: "InfoNCE is the contrastive learning workhorse. It is equivalent to cross-entropy on negative samples — understanding this connection makes CLIP's training objective trivial to derive."

**Transition**: "Before we can minimise the loss, we need to initialise the weights correctly..."

---

## Slide 33: Weight Initialisation: Why It Matters

**Time**: ~4 min
**Talking points**:

- Wrong init = broken training. Zero init: all neurons learn the same thing (symmetry problem). Too large: exploding gradients. Too small: vanishing gradients.
- Xavier/Glorot init: preserves variance across layers for tanh/sigmoid networks.
- He/Kaiming init: for ReLU networks. Accounts for the 50% zeroing. W ~ N(0, sqrt(2/n_in)).
- Rule of thumb: Xavier for tanh/sigmoid, He for ReLU/Leaky ReLU.
- If beginners look confused: "Think of initialisation like tuning a guitar. If the strings are too slack (small weights) or too tight (large weights), no sound comes out. We have formulas for the right tension."
- If experts look bored: "Xavier derivation: assume linear activations, iid inputs with variance 1, then Var(output) = n_in _ Var(W) _ Var(input). To preserve variance: Var(W) = 1/n_in. The harmonic mean of n_in and n_out gives the Glorot formula."

**Transition**: "For very deep networks, there is an even more precise approach..."

---

## Slide 34: Fixup Initialisation for Deep ResNets

**Time**: ~3 min
**Talking points**:

- Standard init breaks for networks with hundreds of layers. Fixup addresses this.
- Key idea: scale initialisation by L^{-1/(2m-2)} where L is depth and m is layers per residual block.
- Enables training very deep ResNets (1000+ layers) without batch normalisation.
- "This is an advanced topic. Flag it: if you are training vanilla networks, use He init. If you are doing research on very deep networks, study Fixup."
- If beginners look confused: "As networks get very deep, even careful initialisation is not enough. Fixup is a more precise formula for extreme depths."
- If experts look bored: "Fixup is important because it decouples the question of whether BatchNorm is needed for training stability from whether it is needed for generalisation — they are separate concerns."

**Transition**: "We have the network, the loss, and the init. Now: how do gradients actually flow backwards? Section 7.6: the chain rule..."

---

## Slide 35: 7.6 The Chain Rule

**Time**: ~3 min
**Talking points**:

- The chain rule from calculus: dz/dx = (dz/dy) \* (dy/dx).
- In neural networks: to compute how the loss changes with respect to the first layer's weights, we chain together all the derivatives along the path.
- "Backpropagation is just the chain rule applied systematically. That is all it is."
- Write it on the board: dL/dW1 = (dL/dA2) _ (dA2/dZ2) _ (dZ2/dA1) _ (dA1/dZ1) _ (dZ1/dW1).
- If beginners look confused: "The chain rule is like asking: if my speed affects my position, and my position affects my score, how does my speed affect my score? You multiply the two effects together."
- If experts look bored: "The computational efficiency of backprop comes from the reverse-mode autodiff approach. Reverse mode is efficient for scalar outputs (like loss) — it is O(1) backwards passes regardless of input dimension."

**Transition**: "Let me work through a concrete two-layer example..."

---

## Slide 36: Worked Example: 2-Layer Network

**Time**: ~5 min
**Talking points**:

- Walk through the forward pass step by step. Z1 = W1*X + b1, A1 = ReLU(Z1), Z2 = W2*A1 + b2, A2 = sigmoid(Z2), L = BCE(A2, y).
- Now the backward pass: dL/dA2, dA2/dZ2, dZ2/dW2, dZ2/dA1, dA1/dZ1, dZ1/dW1.
- DO NOT skip steps. Every line matters. This is the foundation of understanding autograd.
- PAUSE after forward pass. Let students verify they follow. Then do backward.
- If beginners look confused: "We just traced two paths: forward (compute prediction) and backward (compute blame). Backprop is just tracing blame backwards through the network."
- If experts look bored: "Note the reuse of computations: A1 computed in forward pass is reused in the backward pass. This is why the memory overhead of training (vs inference) is proportional to the number of activations — they must be kept for the backward pass."

**Transition**: "Now let us see all the gradients at once..."

---

## Slide 37: Backward Pass: Computing Every Gradient

**Time**: ~4 min
**Talking points**:

- Show the full gradient table: every parameter's gradient, written as a product of upstream gradients and local Jacobians.
- Point out the pattern: each layer's gradient is the upstream gradient times the local derivative.
- "The pattern is always: what comes back from above times what happens locally."
- PAUSE. Give students 2 minutes to identify which gradients would be zero if ReLU was negative.
- If beginners look confused: "Each gradient is a number telling us: increase this weight by 1, and the loss changes by this amount. Negative means increasing the weight reduces the loss — we should increase it."
- If experts look bored: "The computational graph perspective: backprop traverses the DAG in topological reverse order, accumulating gradients. PyTorch's autograd builds this DAG dynamically on every forward pass — that is what 'dynamic computational graph' means."

**Transition**: "Let us formalise this as the backpropagation algorithm..."

---

## Slide 38: Backpropagation: The Algorithm

**Time**: ~4 min
**Talking points**:

- Algorithm steps: (1) forward pass, store activations, (2) compute loss gradient, (3) backward pass, chain rule at each layer, (4) accumulate gradients, (5) update weights.
- The memory implication: we must store all intermediate activations during forward pass for use in backward. Training memory roughly doubles versus inference.
- Gradient accumulation: for large batches, accumulate gradients over mini-batches, update once.
- If beginners look confused: "Forward: make a prediction. Backward: figure out how each weight contributed to the error. Update: adjust each weight proportionally."
- If experts look bored: "Gradient checkpointing trades compute for memory: recompute activations during backward pass instead of storing them. Reduces memory from O(L) to O(sqrt(L)) at the cost of one extra forward pass. Essential for very large models."

**Transition**: "The chain rule also applies to vectors. Let us be precise about Jacobians..."

---

## Slide 39: Jacobians and Vector-Jacobian Products

**Time**: ~3 min
**Talking points**:

- Scalar function, vector input: the gradient is a vector (same shape as input).
- Vector function, vector input: the gradient is a Jacobian matrix.
- In backprop we always need vector-Jacobian products (VJPs), not full Jacobians. This is the efficiency key.
- PyTorch computes VJPs, not full Jacobians. That is why torch.autograd.grad(outputs, inputs, grad_outputs=v) exists.
- If beginners look confused: "When the function takes a list of numbers and outputs a list, the derivative is a whole grid of numbers — the Jacobian. In practice, we never build the full grid; we only need the product with an upstream vector."
- If experts look bored: "The VJP perspective formalises why reverse-mode autodiff is efficient: computing v^T J for any upstream vector v costs the same as one backward pass, regardless of the output dimension. This is why backprop scales to networks with billions of parameters."

**Transition**: "How do we know our gradients are correct? Gradient checking..."

---

## Slide 40: Gradient Checking: Debugging Backpropagation

**Time**: ~3 min
**Talking points**:

- Finite difference approximation: dL/dw is approximately (L(w + epsilon) - L(w - epsilon)) / (2\*epsilon).
- Compare analytical gradient (from backprop) to numerical gradient (from finite differences). Should match to 5+ decimal places.
- "If they do not match, your backprop has a bug. This is the gold standard debugging tool."
- Practical: torch.autograd.gradcheck() does this automatically.
- If beginners look confused: "We can check our math by approximating the derivative with a tiny change. If our formula gives the same answer as the approximation, our formula is correct."
- If experts look bored: "Finite difference only works for small networks because it requires 2 forward passes per parameter — O(n) cost for n parameters. For production debugging, use PyTorch anomaly detection mode: torch.autograd.set_detect_anomaly(True)."

**Transition**: "Now we know how gradients flow. What if they disappear? Section 7.7: vanishing gradients..."

---

## Slide 41: 7.7 Vanishing Gradients

**Time**: ~4 min
**Talking points**:

- The problem: in deep networks with sigmoid/tanh, gradients get multiplied by derivatives less than 1 at every layer. After 10 layers: 0.25^10 is approximately 10^-6.
- The symptom: first layers stop learning. Loss plateau. Monitoring early layer gradients shows near-zero values.
- The solutions: ReLU activations, batch normalisation, residual connections.
- Draw the gradient magnitude plot: x-axis = layer depth, y-axis = gradient magnitude. Show the exponential decay.
- If beginners look confused: "Imagine passing a message through 20 people, each one whispering it slightly quieter. By the time it reaches the last person, there is no message. Vanishing gradients are the same problem for training signals."
- If experts look bored: "The Lipschitz constant of the gradient is the key quantity. For sigmoid with derivative max 0.25, the product of 10 Lipschitz constants bounds the gradient magnitude. ResNets solve this by adding a skip connection with Lipschitz constant 1."

**Transition**: "The opposite problem: exploding gradients..."

---

## Slide 42: Exploding Gradients and Gradient Clipping

**Time**: ~3 min
**Talking points**:

- Exploding gradients: gradient magnitudes grow exponentially with depth. Causes NaN losses, unstable training.
- Common in RNNs (sequential multiplication of the same weight matrix).
- Solution: gradient clipping. If gradient norm exceeds threshold, scale it down. torch.nn.utils.clip*grad_norm*(params, max_norm).
- Practical value: max_norm = 1.0 is a common default for transformers.
- If beginners look confused: "If vanishing gradients are too quiet a signal, exploding gradients are too loud — they cause the training to overshoot in wild directions. Clipping puts a volume limit on the signal."
- If experts look bored: "Global gradient norm clipping vs per-layer clipping: global is the industry standard. The max_norm hyperparameter interacts with learning rate — reducing one often requires tuning the other."

**Transition**: "One more failure mode specific to ReLU networks: dead neurons..."

---

## Slide 43: Dead Neurons (Dying ReLU)

**Time**: ~3 min
**Talking points**:

- Dead neurons: ReLU outputs 0 for all inputs. Gradient is 0. Weights never update. Neuron is permanently dead.
- Cause: weights initialised too negative, learning rate too high (overshoot), large negative bias.
- Detection: monitor the fraction of zero activations per layer. Above 50% is a warning sign.
- Solutions: Leaky ReLU (never truly zero), ELU, careful initialisation, lower learning rate.
- If beginners look confused: "Imagine a neuron that has given up. It outputs zero for everything, learns nothing, and can never recover. The fix is to use a slightly different activation function that never truly turns off."
- If experts look bored: "Dead neuron fraction is a useful diagnostic. A healthy ReLU network has approximately 50% sparsity per layer — if you see 80%+ zeros in a layer, investigate. The dead fraction correlates with effective capacity loss."

**Transition**: "Now let us scale up. Section 7.8: how do we train on multiple GPUs?"

---

## Slide 44: 7.8 Parallelised Backpropagation

**Time**: ~3 min
**Talking points**:

- Section marker. The problem: modern models are too large and too data-hungry for a single GPU.
- Four parallelism strategies: data, model, pipeline, tensor.
- "We are not just scaling compute. We are re-thinking how the computation is distributed."
- If beginners look confused: "One GPU is like one worker. When the task is too big, you hire more workers. How you split the work is the strategy."
- If experts look bored: "The combination of all four strategies is what enabled training at the scale of the largest modern language models. Understanding the trade-offs between them is essential for anyone doing large-scale ML."

**Transition**: "Start with the simplest: data parallelism..."

---

## Slide 45: Data Parallelism in Detail

**Time**: ~4 min
**Talking points**:

- Split the batch across GPUs. Each GPU has a full model copy. Forward and backward passes happen in parallel.
- Gradient synchronisation: after backward pass, all-reduce gradients across GPUs, apply the same update to all model copies.
- PyTorch: torch.nn.parallel.DistributedDataParallel (DDP). Prefer over DataParallel (which has the GIL problem).
- Scaling: near-linear speedup up to the point where communication overhead dominates.
- If beginners look confused: "Give each worker a copy of the problem and a subset of the data. Workers solve their subset, share their findings, and all update their knowledge together."
- If experts look bored: "All-reduce efficiency: ring-allreduce (used by NCCL) achieves 2\*(N-1)/N bandwidth efficiency where N is the number of GPUs. Communication becomes the bottleneck at large N unless using NVLink or InfiniBand."

**Transition**: "What if the model itself does not fit on one GPU?"

---

## Slide 46: Model Parallelism and the Bubble Problem

**Time**: ~4 min
**Talking points**:

- Model parallelism: split the model across GPUs. GPU1 has layers 1-4, GPU2 has layers 5-8, etc.
- The bubble problem: while GPU2 is computing, GPU1 is idle. GPU utilisation is terrible.
- Naive model parallelism often has 50%+ idle time — worse than single GPU if not addressed.
- Show the timeline diagram: one row per GPU, time on x-axis. The bubbles (idle periods) are visible.
- If beginners look confused: "Imagine an assembly line where only one station works at a time. Everyone else waits. That is the bubble problem."
- If experts look bored: "The bubble fraction is (p-1)/p where p is pipeline stages. For 4-way pipeline parallelism, 75% of the time is wasted with naive scheduling. Micro-batching reduces but does not eliminate this."

**Transition**: "The solution is pipeline parallelism..."

---

## Slide 47: Pipeline Parallelism: Filling the Bubbles

**Time**: ~3 min
**Talking points**:

- Solution: split each mini-batch into micro-batches. While GPU2 processes micro-batch 1, GPU1 starts on micro-batch 2.
- This fills the pipeline — like a factory assembly line at full speed.
- GPipe (Google) and PipeDream (Microsoft) are the seminal implementations.
- Bubble fraction reduces from (p-1)/p to (p-1)/(p + m - 1) where m is the number of micro-batches.
- If beginners look confused: "Instead of waiting for the first batch to finish before sending the second, we overlap them. By the time GPU2 is processing piece 1, GPU1 is already working on piece 2."
- If experts look bored: "PipeDream-2BW (2-buffer-weight) uses double buffering to allow weight updates without pipeline flush, improving throughput by roughly 2x over GPipe's synchronous approach."

**Transition**: "The most fine-grained parallelism: tensor parallelism..."

---

## Slide 48: Tensor Parallelism

**Time**: ~3 min
**Talking points**:

- Split individual matrix multiplications across GPUs. Each GPU computes a slice of the weight matrix.
- Used for the widest layers — attention heads in transformers, for example.
- Megatron-LM (NVIDIA) pioneered this for transformer attention and FFN layers.
- The communication pattern: after each tensor-parallel operation, you need an all-reduce. More communication than data parallelism.
- If beginners look confused: "Tensor parallelism splits individual calculations, not just the data or the model layers. It is like having each worker compute a different column of a multiplication table simultaneously."
- If experts look bored: "Tensor parallelism works within a node (NVLink bandwidth). Pipeline and data parallelism work across nodes. The 3D parallelism used to train the largest models combined all three: TP within node, PP across nodes, DP across replica groups."

**Transition**: "A key memory optimisation: ZeRO..."

---

## Slide 49: ZeRO: Zero Redundancy Optimiser

**Time**: ~3 min
**Talking points**:

- ZeRO (DeepSpeed): eliminates redundancy in data parallelism. Each GPU stores only 1/N of the parameters, gradients, and optimizer states.
- Three stages: ZeRO-1 (partition optimizer states), ZeRO-2 (plus gradients), ZeRO-3 (plus parameters).
- ZeRO-3: memory per GPU = total model size / N GPUs. A 175B model across 1000 GPUs = 175M parameters per GPU.
- Trade-off: more communication bandwidth required as we go from ZeRO-1 to ZeRO-3.
- If beginners look confused: "Instead of every GPU holding a complete copy of the model, each GPU holds only its assigned piece. When needed, GPUs share their pieces with each other."
- If experts look bored: "The memory reduction is clean: ZeRO-1 gives 4x (optimizer state), ZeRO-2 gives 8x, ZeRO-3 gives roughly 64x on a very large model vs baseline DP. The communication volume for ZeRO-3 is 1.5x baseline DP — a small price for the memory savings."

**Transition**: "One more technique that touches both training speed and memory: mixed precision..."

---

## Slide 50: Mixed Precision Training

**Time**: ~3 min
**Talking points**:

- FP32 (32-bit): full precision, slow, large. FP16 (16-bit): half precision, fast, small, but range issues.
- Mixed precision: forward pass and gradients in FP16, master weights and optimizer states in FP32.
- Loss scaling: multiply loss by a large scalar before backward pass to prevent underflow in FP16 gradients.
- Speedup: 2-3x on modern GPUs (Tensor Cores are optimised for FP16/BF16 matrix multiply).
- BF16 (bfloat16): same range as FP32, lower precision. Preferred over FP16 for most modern hardware.
- If beginners look confused: "We do the heavy lifting in a compressed format (FP16) which is twice as fast, and only keep the exact numbers (FP32) where precision really matters."
- If experts look bored: "The numerical stability argument for BF16 over FP16: BF16 has the same 8 exponent bits as FP32, so the dynamic range is identical. FP16's limited range causes overflow in activations and gradients without loss scaling."

**Transition**: "We have the infrastructure for training at scale. Now let us choose how to descend the loss surface. Section 7.9: optimisers..."

---

## Slide 51: 7.9 SGD: Stochastic Gradient Descent

**Time**: ~4 min
**Talking points**:

- SGD vs gradient descent: use a random mini-batch instead of the full dataset. Noisy but much faster per update.
- Update rule: w = w - lr \* gradient(batch).
- The noise is actually beneficial: acts as regularisation, helps escape shallow local minima.
- Batch size trade-off: larger batch = less noise = more stable = better hardware utilisation = often worse generalisation.
- PAUSE. Ask: "Why might noisy gradients help generalisation?" Let students think.
- If beginners look confused: "Instead of looking at all our training data before each update, we look at a random sample. It is faster and, surprisingly, often finds better solutions."
- If experts look bored: "The sharp minima / flat minima connection (Keskar et al., 2017): large-batch training finds sharper minima with worse generalisation. The noise from small batches helps find flatter minima which generalise better. This explains why optimal batch sizes are often surprisingly small."

**Transition**: "SGD alone is too slow. We need momentum..."

---

## Slide 52: SGD with Momentum

**Time**: ~3 min
**Talking points**:

- Momentum: accumulate a velocity vector in directions of persistent gradient. v = beta*v + gradient, w = w - lr*v.
- beta = 0.9 is the standard choice.
- Physics analogy: momentum prevents a ball from stopping in a shallow valley and helps it roll through to the deeper one.
- Two effects: smoother updates (reduces oscillation), faster progress in consistent directions.
- If beginners look confused: "Momentum is like pushing a shopping cart. Even if there is a small bump, the cart keeps rolling. It smooths out the jerky movements of plain SGD."
- If experts look bored: "Nesterov momentum (NAG) computes the gradient at the lookahead position: v = beta*v + gradient_at(w - beta*v), w = w - lr\*v. This gives slightly better convergence rates and is the default when using SGD with momentum in modern practice."

**Transition**: "What about adapting the learning rate per parameter? AdaGrad and RMSProp..."

---

## Slide 53: AdaGrad and RMSProp

**Time**: ~4 min
**Talking points**:

- AdaGrad: accumulate squared gradients per parameter. Divide learning rate by sqrt of accumulated squared gradients. Rare parameters get larger updates.
- Problem: accumulated squared gradients grow monotonically. Learning rate approaches 0 over time. Training stops.
- RMSProp: exponential moving average of squared gradients instead of sum. Fixes the monotonic accumulation.
- RMSProp: S = beta*S + (1-beta)*gradient^2, w = w - lr \* gradient / sqrt(S + epsilon).
- If beginners look confused: "AdaGrad gives a smaller learning rate to parameters that have been updated a lot, and a larger rate to those updated rarely. The problem: eventually it stops updating everything. RMSProp fixes this with a forgetting mechanism."
- If experts look bored: "AdaGrad has a beautiful theoretical motivation: it adapts to the geometry of the loss surface by diagonally preconditioning the gradient. The issue is that the diagonal preconditioner grows without bound. RMSProp's exponential averaging effectively normalises the geometry locally."

**Transition**: "The combination of momentum and adaptive rates gives us Adam..."

---

## Slide 54: Adam: The Default Optimiser

**Time**: ~4 min
**Talking points**:

- Adam = RMSProp + momentum. Two moments: first moment (momentum), second moment (adaptive learning rate).
- m = beta1*m + (1-beta1)*gradient (first moment, beta1=0.9)
- v = beta2*v + (1-beta2)*gradient^2 (second moment, beta2=0.999)
- Bias correction: m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t).
- Update: w = w - lr \* m_hat / (sqrt(v_hat) + epsilon).
- Defaults: lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8.
- "Adam is your default. Use it unless you have a specific reason not to."
- If beginners look confused: "Adam has two memories: one for the direction of movement (momentum), one for how fast each parameter has been changing (adaptive rate). It uses both to make smart updates."
- If experts look bored: "The bias correction terms are important in early training — without them, the first few updates are dominated by the initialisation of m and v to zero, causing artificially small updates. This is especially noticeable with beta2=0.999."

**Transition**: "Adam has one known flaw. AdamW fixes it..."

---

## Slide 55: AdamW: Decoupled Weight Decay

**Time**: ~3 min
**Talking points**:

- The flaw in Adam: L2 regularisation (weight decay) is not equivalent to proper weight decay when using adaptive learning rates.
- In SGD: L2 reg in loss is equivalent to weight decay. In Adam: L2 reg in loss is not equal to weight decay (the adaptive scaling changes the effective magnitude).
- AdamW: decouple weight decay from the gradient update. Apply weight decay directly to weights, not via the gradient.
- Update: w = (1 - lr*lambda)*w - lr \* m_hat / (sqrt(v_hat) + epsilon).
- "AdamW is the correct implementation. It is the default for all transformer training."
- If beginners look confused: "AdamW is Adam but with regularisation done correctly. Just use AdamW instead of Adam."
- If experts look bored: "The decoupling matters empirically: Loshchilov and Hutter (2019) showed AdamW consistently outperforms Adam+L2 across architectures. The intuition: adaptive scaling in Adam makes L2 penalise rare features less than frequent ones, which is the wrong behaviour."

**Transition**: "A quick tour of cutting-edge optimisers..."

---

## Slide 56: Cutting-Edge Optimisers

**Time**: ~2 min
**Talking points**:

- Lion (ICML 2023): momentum only, no second moment. Memory efficient. 3x faster than AdamW in some benchmarks.
- Adan: adaptive Nesterov momentum estimation. Better on vision tasks.
- CAME: memory-efficient Adam variant using low-rank second moment approximation.
- "These are research options. Use AdamW for production. Follow these only if AdamW is a bottleneck."
- If experts look bored: "Lion was found via evolution strategy — the optimizer was literally evolved by a meta-learning system. This represents a new paradigm in optimizer design that sidesteps human intuition entirely."

**Transition**: "How do we change the learning rate during training?"

---

## Slide 57: Learning Rate Schedules

**Time**: ~4 min
**Talking points**:

- Constant LR: baseline. Usually suboptimal.
- Cosine annealing: decrease LR following a cosine curve from max to min. Most common in practice.
- Warmup: start with very low LR, increase linearly, then apply schedule. Essential for transformers (stabilises early training).
- Cyclical LR: oscillate between low and high LR. Can escape local minima.
- One-cycle policy: warmup + cosine decay in one cycle. Often achieves better results faster.
- Show the schedule plots. Students should be able to sketch these from memory.
- If beginners look confused: "The learning rate is like the stride of a person looking for their keys. Start with big strides to cover ground, then take small steps to search carefully near the likely spot."
- If experts look bored: "The warmup is theoretically motivated by the variance of the Adam estimators in early training: with t small, the bias-corrected second moment estimate has high variance, and large LR magnifies this. Warmup keeps the effective LR small until the estimates stabilise."

**Transition**: "We can train well. Now how do we prevent overfitting? Section 7.10: regularisation..."

---

## Slide 58: 7.10 Dropout

**Time**: ~4 min
**Talking points**:

- Dropout: randomly zero out neurons with probability p during training. p=0.5 is common for FC layers, p=0.1-0.2 for transformer attention.
- At inference: disable dropout, scale activations by (1-p). Or use inverted dropout (scale during training by 1/(1-p)).
- Interpretation 1: ensemble of 2^n sub-networks. Inference = averaging them.
- Interpretation 2: prevents co-adaptation. Each neuron must work independently.
- If beginners look confused: "During training, we randomly mute half the neurons on each pass. This forces the network to not rely on any single neuron — each must learn something useful on its own."
- If experts look bored: "The ensemble interpretation (Srivastava et al., 2014): geometric mean of 2^n networks with shared weights. Inference uses the full network as an approximation to the full ensemble. This connects dropout to Bayesian model averaging."

**Transition**: "Dropout is less common in CNNs. Batch normalisation is more important there..."

---

## Slide 59: Batch Normalisation

**Time**: ~4 min
**Talking points**:

- BatchNorm: normalise activations within a mini-batch (zero mean, unit variance), then scale and shift with learned parameters gamma and beta.
- Two benefits: (1) reduces internal covariate shift — each layer sees a stable distribution, (2) acts as regularisation (the batch statistics introduce noise).
- At inference: use running mean/variance computed during training (not batch statistics).
- Placement: after linear/conv layer, before activation. Or after activation — empirically both work, before is more common.
- If beginners look confused: "BatchNorm is like a thermostat for activations. Each layer's output gets normalised to a standard range before being passed on. This keeps the scale stable as gradients flow."
- If experts look bored: "The 'reduces internal covariate shift' explanation is contested (Santurkar et al., 2018 showed it does not actually reduce covariate shift). The actual mechanism appears to be smoothing the loss landscape — BatchNorm constrains the gradient norm, making the loss surface more Lipschitz."

**Transition**: "BatchNorm does not work for small batches or sequence models. LayerNorm does..."

---

## Slide 60: Layer Normalisation

**Time**: ~3 min
**Talking points**:

- LayerNorm: normalise across the feature dimension for each sample, not across the batch.
- Does not depend on batch size. Works identically at training and inference. No running statistics needed.
- Used in transformers (BERT, GPT, and almost all modern architectures).
- Comparison: BatchNorm = normalise each feature across the batch. LayerNorm = normalise all features for each sample.
- If beginners look confused: "LayerNorm normalises each sample's activations independently. It does not need to look at the whole batch, so it works even for batch size of 1 — important for inference."
- If experts look bored: "The architectural choice between BatchNorm and LayerNorm has practical implications: BatchNorm requires synchronisation across GPUs, while LayerNorm is trivially parallelisable. For distributed training, LayerNorm is strictly easier."

**Transition**: "Other regularisation techniques that every practitioner should know..."

---

## Slide 61: Other Regularisation Techniques

**Time**: ~4 min
**Talking points**:

- L2 weight decay (AdamW): penalises large weights. The most universal regulariser.
- Data augmentation: artificially expand the training set. Random crop, flip, colour jitter for images; synonym replacement for text.
- Early stopping: stop when validation loss stops improving. Simple and effective.
- Label smoothing: replace hard targets (0/1) with soft targets. Prevents overconfident predictions.
- Mixup: train on linear interpolations of training examples. Forces the model to learn smooth decision boundaries.
- "Regularisation is not one technique — it is a family. Stack them appropriately."
- If beginners look confused: "All regularisation techniques share one goal: prevent the model from memorising training data. They just do it in different ways — shrinking weights, augmenting data, or softening targets."
- If experts look bored: "Label smoothing has a calibration interpretation: it matches the confidence of the model's predictions to the actual accuracy. Hard labels push the model to be maximally confident, which leads to overconfidence. Smooth labels are the maximum entropy solution conditional on the correct class probability."

**Transition**: "Now the most important architecture for images. Section 7.11: convolutional neural networks..."

---

## Slide 62: 7.11 The Convolution Operation

**Time**: ~5 min
**Talking points**:

- Convolution: slide a small filter (kernel) over the input, computing dot products at each position.
- The key properties that make convolution right for images: (1) locality — nearby pixels are related, (2) translation equivariance — a cat in the top-left and a cat in the bottom-right should activate the same filters.
- Walk through the mechanics: 3x3 kernel, stride 1, on a 5x5 input gives 3x3 output. Show the sliding window.
- Output size formula: (W - K + 2P) / S + 1. Commit to memory.
- Compare to fully connected: a 28x28 image with an FC layer of 1000 neurons needs 784,000 parameters. The same with a 3x3 conv needs 9 parameters (shared). Orders of magnitude fewer.
- If beginners look confused: "A convolutional filter is like a detective looking for a specific pattern. It scans the whole image asking 'Is this pattern here?' at every position. Sharing the same filter across positions is why CNNs are so efficient."
- If experts look bored: "The translation equivariance of convolution is an inductive bias. It is the reason CNNs need 100x less data than FCNs for image tasks — the architecture encodes the symmetry of the problem. Vision transformers achieve similar performance without this inductive bias, but require more data or pretraining."

**Transition**: "The parameters of a convolutional layer..."

---

## Slide 63: Convolution Parameters

**Time**: ~3 min
**Talking points**:

- Kernel size: 3x3 (common), 5x5, 1x1 (pointwise — changes channels without spatial mixing).
- Stride: step size of the sliding window. Stride 2 halves the spatial dimensions.
- Padding: add zeros around the border. 'same' padding preserves spatial dimensions.
- Dilation: gaps between kernel elements. Increases receptive field without more parameters.
- Depth-wise separable convolutions: factored convolution (spatial + channel). Used in MobileNet — same expressiveness, 8-9x fewer FLOPs.
- If beginners look confused: "Kernel size = how big the detective's magnifying glass is. Stride = how far it jumps between positions. Padding = adding a border so the output is the same size as the input."
- If experts look bored: "Receptive field calculation: after L convolutional layers with kernel size K and stride 1, the receptive field is L*(K-1)+1. With dilation d, effective kernel size is d*(K-1)+1. Understanding receptive field is critical for debugging why a CNN cannot capture global structure."

**Transition**: "How do we build a CNN from these pieces?"

---

## Slide 64: CNN Architecture Components

**Time**: ~4 min
**Talking points**:

- Standard block: Conv → BatchNorm → ReLU → (optional Pool).
- Pooling: max pool (take the maximum in a window) or average pool. Reduces spatial dimensions, introduces translation invariance.
- Global average pooling: reduce entire spatial map to a single number per channel. Replaces the flatten-then-FC approach.
- Feature pyramid: early layers have high spatial resolution but few channels; deep layers have low resolution but many channels.
- Show the typical progression: 224x224x3 → 112x112x64 → 56x56x128 → 28x28x256 → 7x7x512 → 1x1x1000.
- If beginners look confused: "As we go deeper in a CNN, the image gets smaller but the number of filters grows. We trade spatial detail for richer feature abstractions."
- If experts look bored: "The information bottleneck at the end (global average pooling) is a form of invariance: the final representation is invariant to where in the image the feature appears, only encoding what features are present and their strength."

**Transition**: "How have CNN architectures evolved?"

---

## Slide 65: Classic Architectures: The Evolution

**Time**: ~3 min
**Talking points**:

- AlexNet (2012): 8 layers, dropout, data augmentation. The breakthrough.
- VGG (2014): all 3x3 convolutions, very deep (16/19 layers). Showed that depth matters more than filter size.
- Inception/GoogLeNet (2014): parallel paths with different kernel sizes. Multi-scale feature extraction.
- ResNet (2015): skip connections. Made very deep networks (50, 101, 152 layers) trainable.
- "Each generation solved a specific problem. AlexNet: can we train at all? VGG: does depth help? Inception: what kernel size? ResNet: how deep can we go?"
- If beginners look confused: "Think of this as an arms race. Each year, researchers found a new trick to make networks deeper, faster, or more accurate."
- If experts look bored: "The architectural innovations map directly to the theoretical problems: VGG addressed expressiveness via depth, Inception addressed multi-scale inductive bias, ResNet addressed the optimisation landscape."

**Transition**: "ResNet is the most important architecture in deep learning history. Let us go deep on skip connections..."

---

## Slide 66: ResNet: The Skip Connection Revolution

**Time**: ~5 min
**Talking points**:

- The problem ResNet solved: adding more layers made networks WORSE (not just overfitting — training error went up). Degradation problem.
- The insight: instead of learning H(x), learn F(x) = H(x) - x (the residual). Add the input x back directly.
- Skip connection: output = F(x) + x. The identity path provides a gradient highway.
- Why this works: (1) gradients can flow directly from loss to early layers, (2) layers only need to learn refinements, (3) the identity mapping is a trivial solution the network can fall back to.
- He et al. (2015): ResNet-152 won ImageNet with 3.57% top-5 error. Humans were at roughly 5%.
- If beginners look confused: "Instead of teaching each layer to do its job from scratch, ResNet says: just figure out what to add to what was already there. That is much easier to learn."
- If experts look bored: "The residual connection ensures gradient magnitude is at least 1 along the skip path. This gives a lower bound on the gradient, preventing vanishing. The effective depth of a ResNet is stochastic — some forward passes use many residual blocks, others skip most of them via the skip connections."

**Transition**: "A refinement that further improves training: pre-activation ResNets..."

---

## Slide 67: Pre-Activation ResNet

**Time**: ~2 min
**Talking points**:

- Original ResNet: Conv → BN → ReLU, then add skip.
- Pre-activation ResNet: BN → ReLU → Conv, then add skip.
- Benefit: the skip connection is a clean identity (no activation), providing an unobstructed gradient path.
- Used in ResNet-1001 (1000 layers). Pre-activation was essential for that depth.
- If beginners look confused: "By moving the normalisation before the convolution, the skip connection stays clean — the gradient flows straight through without being distorted."
- If experts look bored: "The theoretical advantage: the pre-activation arrangement means the residual path has normalised inputs, which prevents the scale explosion that can occur with post-activation when the residual outputs accumulate across many layers."

**Transition**: "CNNs are not just for classification. What else can they do?"

---

## Slide 68: Beyond Image Classification

**Time**: ~4 min
**Talking points**:

- Object detection: not just 'what' but 'where'. YOLO, SSD, Faster R-CNN.
- Semantic segmentation: classify every pixel. FCN, U-Net.
- Instance segmentation: detect and segment individual objects. Mask R-CNN.
- Medical imaging: U-Net is the standard for biomedical image segmentation. Critical real-world application.
- Key architecture: U-Net (encoder-decoder with skip connections). Show the architecture shape.
- If beginners look confused: "Classification says 'there is a cat in this photo.' Detection says 'there is a cat and it is in the top-right corner.' Segmentation says 'these exact pixels are the cat.' Each is progressively harder."
- If experts look bored: "U-Net's skip connections between encoder and decoder are different from ResNet's — they concatenate features rather than adding them. This preserves both high-level semantics (from the bottleneck) and low-level spatial detail (from the encoder). Essential for dense prediction tasks."

**Transition**: "We have deep networks. What do their intermediate representations look like? Section 7.12: embeddings..."

---

## Slide 69: 7.12 From Networks to Embeddings

**Time**: ~3 min
**Talking points**:

- Section marker. The hidden layers of a trained network contain rich representations.
- Embedding: a dense, low-dimensional vector representation of an object (image, text, user, product).
- "The last hidden layer of a trained network IS an embedding space."
- This section bridges Module 7 (architecture) to Module 8 (NLP/transformers) and Module 9 (RAG/agents).
- If beginners look confused: "An embedding is a way to describe something with a list of numbers. A trained network has learned the best list of numbers for each thing it has seen."

**Transition**: "What kinds of embeddings exist?"

---

## Slide 70: Types of Embeddings

**Time**: ~4 min
**Talking points**:

- Image embeddings: penultimate layer of a trained network. 2048-dimensional for ResNet-50.
- Word embeddings: Word2Vec, GloVe. 300-dimensional. Words with similar meanings are close in embedding space.
- Sentence embeddings: BERT [CLS] token. 768-dimensional. Used for semantic search.
- User/item embeddings: collaborative filtering. Matrix factorisation. Used for recommendation.
- Graph embeddings: Node2Vec, GraphSAGE. Encode graph structure.
- The unifying principle: similar things should have similar embeddings (close in Euclidean or cosine distance).
- If beginners look confused: "Imagine plotting every word on a map where similar words live close together. 'King' is close to 'queen', 'dog' is close to 'cat'. That map is the embedding space."
- If experts look bored: "The dimensionality of the embedding space controls the capacity to encode semantic distinctions. Too low: semantically different things map to the same vector. Too high: sample inefficiency. The intrinsic dimensionality of natural language appears to be in the low hundreds, which is why 256-768 dimensional embeddings work so well."

**Transition**: "How do we use these embeddings in downstream tasks?"

---

## Slide 71: Using Embeddings Downstream

**Time**: ~4 min
**Talking points**:

- Similarity search: find the k most similar items by embedding distance. Core of semantic search, recommendation.
- Transfer learning: use embeddings as features for a new task. Fine-tune the last layer or all layers.
- Clustering: embed all items, cluster in embedding space. K-means on embeddings is a powerful baseline.
- Anomaly detection: points far from the cluster centres in embedding space are anomalies.
- Kailash connection: FeatureStore (M3) + ModelRegistry (M4) together handle embedding storage and retrieval at scale.
- If beginners look confused: "Once you have embeddings, you can measure similarity by distance. Close = similar. Far = different. This powers search, recommendations, and anomaly detection."
- If experts look bored: "Approximate nearest neighbour (ANN) search — FAISS, ScaNN, HNSW — is the production-scale solution for embedding lookup. Exact cosine search is O(n\*d); ANN achieves O(log n) with small accuracy loss. This is the engineering backbone of RAG (Module 9)."

**Transition**: "A landmark in generative deep learning: GANs..."

---

## Slide 72: Generative Adversarial Networks (GANs)

**Time**: ~4 min
**Talking points**:

- GAN setup: Generator G tries to create realistic fake data. Discriminator D tries to distinguish real from fake.
- Min-max game: G tries to fool D, D tries to catch G. Equilibrium: G generates samples indistinguishable from real data.
- Applications: image synthesis (StyleGAN), super-resolution, data augmentation, domain adaptation.
- Training challenges: mode collapse (G produces only one type of output), oscillation (G and D never reach equilibrium).
- Historical significance: Goodfellow et al. (2014) paper described as "the most important idea in the last decade" by Yann LeCun.
- If beginners look confused: "Think of the generator as a counterfeiter and the discriminator as a detective. The counterfeiter gets better by learning from the detective's catches. Eventually the counterfeiter is so good the detective cannot tell the difference."
- If experts look bored: "The GAN objective is a Jensen-Shannon divergence minimisation (original GAN) or Wasserstein distance (WGAN). The Wasserstein formulation solves mode collapse by providing smoother gradients even when distributions have disjoint support."

**Transition**: "A related architecture: autoencoders..."

---

## Slide 73: Autoencoders

**Time**: ~3 min
**Talking points**:

- Autoencoder: encoder compresses input to a bottleneck embedding, decoder reconstructs the input.
- Loss: reconstruction loss (MSE). No labels needed — unsupervised.
- The bottleneck forces the network to learn a compact representation.
- Variational Autoencoder (VAE): bottleneck is a probability distribution (mean + variance). Can sample from the latent space to generate new data.
- Applications: anomaly detection, denoising, dimensionality reduction, generative modelling.
- If beginners look confused: "An autoencoder is taught to compress something and then uncompress it. The compressed form in the middle is the embedding. If it can compress and decompress well, the compressed form must contain the essential information."
- If experts look bored: "The ELBO objective for VAE: E[log p(x|z)] - KL(q(z|x) || p(z)). The first term is reconstruction quality, the second is a regularisation on the latent space. The reparameterisation trick makes the expectation differentiable with respect to the encoder parameters."

**Transition**: "For structured data: graph neural networks..."

---

## Slide 74: Graph Neural Networks

**Time**: ~3 min
**Talking points**:

- Graphs: nodes (entities) + edges (relationships). Social networks, molecules, knowledge graphs, supply chains.
- GNN: each node aggregates information from its neighbours, updates its own representation.
- GCN, GAT, GraphSAGE: different aggregation strategies (mean, attention-weighted, sampled).
- Applications: drug discovery (molecule property prediction), recommendation (user-item graphs), fraud detection (transaction graphs).
- If beginners look confused: "A GNN updates each node's understanding by looking at who it is connected to. In a social network, your recommendation score depends on what your friends have liked."
- If experts look bored: "The message-passing formalism unifies most GNN architectures: h_v = UPDATE(h_v, AGGREGATE({h_u : u in N(v)})). The expressiveness of GNNs is bounded by the Weisfeiler-Lehman graph isomorphism test — a fundamental theoretical result."

**Transition**: "A new paradigm replacing RNNs for sequences: state space models..."

---

## Slide 75: State Space Models

**Time**: ~3 min
**Talking points**:

- State space models (SSMs): represent sequences as linear dynamical systems. Mamba, S4, H3.
- Advantage over transformers: linear time complexity with sequence length (vs quadratic for attention).
- The core idea: a hidden state evolves over time, combining the current input with past context.
- Mamba (2023): selective SSM that adapts state transitions based on content. Close to transformer quality, much faster for long sequences.
- If beginners look confused: "SSMs are like a very fast summariser. They maintain a running state that captures the important information from everything they have seen so far, updating it efficiently as new input arrives."
- If experts look bored: "Mamba's selectivity mechanism is the key innovation: the discretisation parameters are input-dependent, giving the model content-aware gating similar to attention but at O(1) per step for autoregressive inference. This is why it is compelling for long-context tasks."

**Transition**: "How do we actually implement all of this? PyTorch essentials..."

---

## Slide 76: PyTorch Essentials

**Time**: ~4 min
**Talking points**:

- The core PyTorch objects: Tensor (data + gradients), nn.Module (layer/model definition), Optimizer, DataLoader.
- nn.Module anatomy: **init** (define layers), forward (define computation graph).
- The training loop skeleton: zero_grad, forward, loss, backward, step.
- Device management: .to(device) — move everything to the same device (CPU or CUDA).
- torch.no_grad() for inference; torch.enable_grad() inside custom contexts.
- "PyTorch is the tool. Understanding the training loop is the skill."
- If beginners look confused: "PyTorch is a Python library for building neural networks. The key concept: everything is a tensor (like a NumPy array but with automatic gradient tracking)."
- If experts look bored: "The compile API (torch.compile() in PyTorch 2.0) applies kernel fusion and graph-level optimisations transparently. On modern hardware, it gives 1.5-2x speedup with no code changes. Always apply it before production training runs."

**Transition**: "Let me show you the complete training loop in one slide..."

---

## Slide 77: The Complete Training Loop

**Time**: ~5 min
**Talking points**:

- Walk through every line: load batch, zero gradients (critical — they accumulate by default), forward pass, compute loss, backward pass, gradient clipping, optimizer step.
- The eval loop: model.eval(), torch.no_grad(), compute validation metrics.
- ModelVisualizer: use to log metrics, plot training curves, compare runs.
- PAUSE. Do not rush this slide. This loop is the foundation of everything.
- COMMON MISTAKE: forgetting optimizer.zero_grad(). Gradients accumulate across batches, causing bizarre training behaviour. Mention this explicitly.
- If beginners look confused: "This is the heartbeat of training. Predict, measure error, calculate blame, adjust. Repeat. Everything else is details."
- If experts look bored: "The canonical pattern is: gradient accumulation for large effective batch sizes, gradient clipping, and cosine LR with warmup. These three together cover 90% of practical training scenarios."

**Transition**: "When training goes wrong, how do we debug it?"

---

## Slide 78: Debugging DL Models

**Time**: ~4 min
**Talking points**:

- The debugging flowchart: (1) check loss is decreasing, (2) check gradient magnitudes per layer, (3) check activation statistics, (4) check data loading, (5) check training vs validation gap.
- Common pathologies and their signatures:
  - NaN loss: exploding gradients or division by zero
  - Loss plateaus immediately: dead neurons or too-small learning rate
  - Train loss low, val loss high: overfitting
  - Both losses not moving: learning rate too low or wrong loss for the task
  - Loss oscillates wildly: learning rate too high
- ModelVisualizer: visualise all of these with one call.
- If beginners look confused: "Think of debugging DL as diagnosing a patient. Each symptom (NaN, plateau, oscillation) points to a different root cause. Learn the symptoms and their causes."
- If experts look bored: "The loss landscape perspective: NaN = gradient overflow past the landscape boundary; plateau = stuck in a flat region or saddle; oscillation = too large step size causing bouncing. Each diagnosis has a precise geometric interpretation."

**Transition**: "Model trained. Now let us take it to production. OnnxBridge..."

---

## Slide 79: OnnxBridge: Export and Optimise

**Time**: ~5 min
**Talking points**:

- ONNX (Open Neural Network Exchange): framework-agnostic model format. Train in PyTorch, deploy anywhere.
- OnnxBridge handles: export, graph optimisation, quantisation (FP32 → INT8 → 4x speedup), validation.
- Quantisation types: post-training quantisation (easy), quantisation-aware training (better quality, more work).
- Deployment targets: ONNX Runtime (CPU/GPU/NPU), TensorRT (NVIDIA), CoreML (Apple), TFLite (mobile).
- Kailash pattern: TrainingPipeline trains → ModelRegistry stores → OnnxBridge exports → InferenceServer serves.
- PAUSE. Draw the full pipeline on the board: training → ONNX export → deployment.
- If beginners look confused: "ONNX is like a universal file format for AI models — like PDF for documents. OnnxBridge is the tool that converts your model to this format and optimises it for deployment."
- If experts look bored: "The ONNX graph optimisation passes include: constant folding, common subexpression elimination, operator fusion. For a ResNet-50, fusion of Conv+BN+ReLU reduces latency by roughly 30%. The optimisation graph is available via onnxruntime.backend.run."

**Transition**: "Once exported, how do we serve it?"

---

## Slide 80: InferenceServer: Serving Models

**Time**: ~4 min
**Talking points**:

- InferenceServer: wrap any model (ONNX, PyTorch, etc.) in a production-ready serving endpoint.
- Features: batching (combine multiple requests), caching (store frequent results), monitoring (latency, throughput, error rate).
- Dynamic batching: automatically group concurrent requests into batches. Critical for GPU efficiency.
- Scaling: horizontal scaling (multiple InferenceServer replicas), vertical scaling (larger GPU).
- Integration: expose via Nexus REST endpoint or gRPC for high-throughput.
- If beginners look confused: "InferenceServer is the waiter at a restaurant. It takes orders from multiple customers (requests), groups them into batches for the kitchen (GPU), and returns the results efficiently."
- If experts look bored: "The batching strategy is a queuing theory problem. Under Poisson arrival, the optimal dynamic batching policy depends on the distribution of request latency requirements. The trade-off: larger batch = higher throughput but higher P99 latency. Set batch size to match your SLA."

**Transition**: "How do we monitor training and diagnose model quality?"

---

## Slide 81: ModelVisualizer: Training History

**Time**: ~4 min
**Talking points**:

- ModelVisualizer: visual analysis of training runs. Plots loss curves, gradient norms, activation statistics, weight distributions.
- Integration with TrainingPipeline: automatic logging. No manual code.
- Key visualisations: (1) train/val loss curves — diagnose overfitting, (2) gradient flow plot — diagnose vanishing/exploding gradients, (3) activation heatmaps — diagnose dead neurons, (4) weight distribution histograms — diagnose initialisation issues.
- Comparison: overlay multiple runs for hyperparameter comparison.
- If beginners look confused: "ModelVisualizer shows you the inside of training. Instead of just watching the loss number change, you can see the health of every layer in real time."
- If experts look bored: "The gradient flow plot (gradient magnitude per layer) is the single most informative diagnostic for deep networks. A healthy network shows roughly uniform gradient magnitude across layers. Exponential decay = vanishing gradients. Exponential growth = exploding gradients."

**Transition**: "Let us wrap up. Module 7 summary by level..."

---

## Slide 82: Module 7 Summary by Level

**Time**: ~3 min
**Talking points**:

- Beginner takeaways: linear regression is a neural network, layers learn features automatically, backprop is the chain rule, use Adam, use ReLU, use OnnxBridge for deployment.
- Intermediate takeaways: initialisation matters, BatchNorm vs LayerNorm trade-offs, ResNet skip connections, parallelism strategies.
- Advanced takeaways: gradient dynamics, optimiser theory, tensor parallelism, ZeRO, architectural trade-offs.
- "Everyone got something different from today. That is by design."
- If beginners look worried about the advanced content: "Do not try to retain everything. Retain the practical pattern, and come back to the theory when you hit a real problem."

**Transition**: "Let me show you the complete feature engineering spectrum with DL added..."

---

## Slide 83: The Feature Engineering Spectrum: Complete

**Time**: ~3 min
**Talking points**:

- Return to the spectrum from Slide 5. Now fill in the DL quadrant.
- Manual features (M3) → learned shallow features (M4 via ensembles) → learned deep features (M7).
- The spectrum is not a ranking. Traditional ML with good features often beats DL on small tabular datasets.
- "Know where you are on the spectrum for each problem. Choose the right tool."
- If experts look bored: "The spectrum maps to the bias-variance trade-off and data regime. Manual features = high bias, low variance. Deep models = low bias, high variance. The cross-over point is approximately 10K-100K samples for tabular data."

**Transition**: "The theory-to-engine map..."

---

## Slide 84: Theory-to-Engine Map: Module 7

**Time**: ~2 min
**Talking points**:

- Walk through the mapping: architecture theory → nn.Module → TrainingPipeline; backprop → automatic gradient in TrainingPipeline; parallelism → ZeRO/DDP in TrainingPipeline; export → OnnxBridge; serving → InferenceServer; diagnostics → ModelVisualizer.
- "Every theory concept from today has a Kailash engine that handles the implementation. Your job is to configure, not implement."
- If beginners look confused: "This is the map from what we learned to what to type. Keep it as a reference."

**Transition**: "Now let us set up for the lab..."

---

## Slide 85: Lab Setup

**Time**: ~5 min
**Talking points**:

- Walk through the three formats: local Python, Jupyter notebook, Google Colab.
- Data loading: ASCENTDataLoader handles all three automatically.
- Lab exercises: (1) linear regression → MLP, (2) CNN on image data, (3) embedding extraction and downstream task, (4) full pipeline: TrainingPipeline → OnnxBridge → InferenceServer.
- Encourage pair work. DL debugging is faster with two pairs of eyes.
- Check: does everyone have their environment set up? Run the setup check cell first.
- If students have CUDA: "If you have a GPU, set device='cuda'. If not, CPU is fine for the lab exercises — they are designed to run in under 10 minutes."
- If experts want to go further: "The bonus exercise is training a ResNet with full mixed-precision, gradient clipping, cosine LR schedule, and ModelVisualizer integration."

**Transition**: "While you set up, let me give you some discussion questions to think about..."

---

## Slide 86: Discussion Prompts

**Time**: ~5 min
**Talking points**:

- Prompt 1: "When would you choose a traditional ML model (M4) over a deep network (M7)? Give three criteria."
- Prompt 2: "Your training loss is 0.02 but validation loss is 0.85. What are the three most likely causes and what do you check first?"
- Prompt 3: "You need to deploy a model to a mobile phone with a 50ms latency requirement. Walk through your pipeline from training to deployment."
- Let pairs discuss for 3 minutes before opening to the full room.
- Do not answer too quickly. Silence is productive here.
- If no one responds: "Start with Prompt 2 — overfitting is something everyone has seen."

**Transition**: "Here is what I will assess on..."

---

## Slide 87: Assessment Preview

**Time**: ~3 min
**Talking points**:

- Assessment structure: theory questions (30%), code debugging (30%), architecture design (40%).
- Theory: derive the backprop equations for a 2-layer network (work from first principles, not memorisation).
- Code debugging: given a broken training loop, identify and fix five bugs. Use gradient checking to validate.
- Architecture design: given a problem description, choose architecture, justify activation function, initialisation, optimiser, and regularisation choices.
- "The assessment is designed so that copying from the slides does not help. You need to understand, not memorise."
- If students look anxious: "The code debugging section is the easiest if you have done the labs. The bugs in the assessment are the same patterns as the bugs in the exercises."

**Transition**: "One last look ahead. Module 8 builds directly on everything we did today..."

---

## Slide 88: Preview: Module 8 — NLP & Transformers

**Time**: ~3 min
**Talking points**:

- Module 8 takes the DL foundations from M7 and applies them to text.
- The transformer is a deep network with a specific architecture: attention mechanisms instead of convolutions.
- GELU activation (learned in 7.4), LayerNorm (7.10), embeddings (7.12), and AdamW (7.9) all reappear in M8.
- New engines: AutoMLEngine for text, ModelVisualizer for attention maps.
- "Everything from today was not just DL theory. It was the foundation for transformers. M8 is where it pays off."
- If beginners look excited: "The attention mechanism is the innovation that made BERT and modern language models possible. We will derive it from scratch in M8."
- If experts are eager: "The transformer can be understood as a graph neural network on a fully-connected graph — a framing that unifies convolution, attention, and message-passing under one theoretical framework."

**Transition**: "Thank you. Module 7 is complete."

---

## Slide 89: Deep Learning: Architecture-Driven Feature Engineering (Back Cover)

**Time**: ~1 min
**Talking points**:

- Close the loop: "We opened with AlexNet and the question of why architecture drives feature learning. You can now answer that question."
- Three takeaways to leave the room with: (1) depth = hierarchical feature learning, (2) backpropagation = the chain rule, (3) OnnxBridge + InferenceServer = your path to production.
- Point to the next session: "Module 8, NLP and Transformers. See you there."
- Leave room visible. Do not advance past this slide.

---
