# MODULE 5: Deep Learning and Machine Learning Mastery in Vision and Transfer Learning

**Description**: Every major DL architecture. One paradigm per lesson. All implemented. Following R5 Deck 6A (comprehensive architecture coverage) + PCML6 notebooks (crown jewel implementations).

**Module Learning Objectives**: By the end of M5, students can:

- Build and train autoencoders (vanilla, denoising, VAE, convolutional)
- Implement and train CNNs with modern enhancements (ResNet, SE blocks, mixed precision)
- Build LSTM and GRU networks with attention mechanisms
- Derive self-attention from scratch and fine-tune transformers
- Implement GANs (DCGAN, WGAN) and understand diffusion model basics
- Apply GNNs to graph-structured data
- Transfer pre-trained models to new tasks (CV and NLP)
- Implement RL algorithms (DQN, DDPG, SAC, A2C, PPO) for business applications

**Kailash Engines**: ModelVisualizer, OnnxBridge, InferenceServer, RLTrainer

## Compute Backend — Auto-Detected (Apple MPS / NVIDIA CUDA / CPU)

Every M5 lesson selects the best available compute backend through a single
canonical detector (`kailash_ml.device()`), threaded into PyTorch via
`shared.kailash_helpers.get_device()` and into Lightning via the
`PRECISION` / `ACCELERATOR` constants in `shared.mlfp05.ex_2`.

What this means in practice:

| Hardware                        | Backend       | Precision    |
| ------------------------------- | ------------- | ------------ |
| Apple Silicon (M1/M2/M3/M4)     | `mps` (Metal) | `16-mixed`   |
| NVIDIA Ampere+ (RTX 30/40, A/H) | `gpu` (CUDA)  | `16-mixed`   |
| NVIDIA Turing- (older Tensor)   | `gpu` (CUDA)  | `32`         |
| AMD ROCm                        | `gpu` (ROCm)  | per detector |
| Intel Xe-HPC (via IPEX)         | `xpu`         | per detector |
| No GPU                          | `cpu`         | `32`         |

There is NO env var, NO `--device` flag, NO `if torch.cuda.is_available()`
branch in any M5 lesson. Students get the right backend automatically. The
`ACCELERATOR` and `PRECISION` constants in lesson helpers are derived from
`kailash_ml.device()` so the choice the lessons report agrees with what the
underlying `MLEngine()` actually picks.

For a Lightning training step inside a lesson:

```python
from shared.mlfp05.ex_2 import ACCELERATOR, PRECISION  # auto MPS/CUDA/CPU
trainer = lightning.Trainer(
    accelerator=ACCELERATOR,
    precision=PRECISION,
    max_epochs=3,
)
```

For raw PyTorch tensor placement:

```python
from shared.kailash_helpers import get_device
device = get_device()       # torch.device('mps') on Mac, 'cuda' on NVIDIA
model.to(device)
```

The kailash-ml 0.12 `MLEngine()` constructor (`accelerator='auto'` default)
exercises the same detector, so anything routed through the engine inherits
the same answer without further wiring.

---

## Lesson 5.1: Autoencoders

**Prerequisites**: M4.8 (neural networks, training toolkit)
**Spectrum Position**: Unsupervised DL — learning compressed representations

**DL Toolkit Refresher** (30 min opening): "In M4.8 you learned forward pass, backprop, gradient descent, dropout, batch norm, optimisers. Quick exercise: build a 2-layer classifier. Good — now we modify this architecture for UNSUPERVISED learning. That's an autoencoder."

**Topics**:

- Autoencoder concept: encoder (compress) → latent space → decoder (reconstruct). Minimise reconstruction error.
- **Deep dive** (4 variants with full implementations):
  - Vanilla autoencoder: simplest form, undercomplete
  - Denoising autoencoder (DAE): corrupt input, learn to reconstruct clean version
  - Variational autoencoder (VAE): ELBO, reparameterisation trick, latent space is a probability distribution. Generate NEW data by sampling from latent space.
  - Convolutional autoencoder: use conv layers for image data
- **Survey** (5 additional variants as reference):
  - Sparse autoencoder (L1 penalty on activations)
  - Contractive autoencoder (penalty on Jacobian)
  - Stacked autoencoder (progressively deeper)
  - Recurrent autoencoder (for sequences)
  - CVAE (Contractive + Variational)

**Key Formulas**:

- Reconstruction loss: L = ||x - decoder(encoder(x))||^2
- VAE ELBO: L = E_q[log p(x|z)] - KL(q(z|x) || p(z))
- Reparameterisation: z = mu + sigma \* epsilon, epsilon ~ N(0,1)

**Learning Objectives**: Students can:

- Implement vanilla, denoising, VAE, and convolutional autoencoders
- Explain the VAE reparameterisation trick and why it enables gradient flow
- Generate new data by sampling from VAE latent space
- Know when to use each variant

**Exercise**: Implement 4 autoencoder variants on MNIST/Fashion-MNIST. Visualise latent spaces. Generate new images from VAE. Compare reconstruction quality.

**Assessment Criteria**: 4 variants implemented and trained. VAE generates plausible new images. Latent space visualisation shows meaningful structure.

**R5 Source**: Deck 6A (9 variants) + PCML6-1 (10+ variants, implementations)

---

## Lesson 5.2: CNNs and Computer Vision

**Prerequisites**: 5.1 (autoencoder training experience)
**Spectrum Position**: Spatial feature learning — extracting patterns from grid data

**Topics**:

- **CNN fundamentals** (from Deck 6A):
  - Convolution operation: filters, stride, padding, feature maps
  - Pooling: max pooling, average pooling — reduce spatial dimensions
  - Normalisation layers: batch norm (between conv and pooling, from M4.8 toolkit)
  - Earlier layers → low-level features (edges, textures). Later layers → high-level features (objects, faces).
- **Architecture history** (from Deck 6A):
  - LeNet-5: earliest CNN (handwritten digits)
  - AlexNet: deeper, ReLU, dropout
  - VGGNet: very small (3x3) filters, depth
  - GoogLeNet/Inception: multiple filter sizes in parallel
  - **ResNet**: residual connections (skip connections) solving vanishing gradients
- **Modern training enhancements** (from PCML6-2):
  - SE blocks (Squeeze-and-Excitation): channel recalibration
  - Kaiming initialisation (proper for ReLU)
  - Mixed precision training (FP16/FP32)
  - Mixup augmentation (smooth decision boundaries)
  - Label smoothing (prevent overconfident predictions)
  - Gradient flow analysis
- **Vision Transformers (ViT)**: brief intro — applying transformers to image patches (connects to M5.4)

**Key Formulas**:

- Conv output size: (W - F + 2P) / S + 1
- ResNet: H(x) = F(x) + x (skip connection)
- SE block: s = sigmoid(W2 _ ReLU(W1 _ GAP(x)))

**Learning Objectives**: Students can:

- Build CNNs with convolution, pooling, and normalisation layers
- Implement ResNet with skip connections
- Apply modern training enhancements (SE blocks, mixed precision, Mixup)
- Explain why ResNet solves the vanishing gradient problem

**Exercise**: Build CNN for image classification (Fashion-MNIST or mask detection). Start simple, add ResBlock, add SE block. Compare training curves. Export to ONNX with OnnxBridge.

**Assessment Criteria**: Architecture progressively improved. Training enhancements measurably help. ONNX export successful.

**R5 Source**: Deck 6A (architecture history) + PCML6-2 (44MB, advanced enhancements)

---

## Lesson 5.3: RNNs and Sequence Models

**Prerequisites**: 5.2 (training experience, batch norm, gradient concepts)
**Spectrum Position**: Temporal feature learning — extracting patterns from sequences

**Topics**:

- **RNN fundamentals**: directed cycles, hidden state memory, vanishing gradient problem
- **LSTM** (from Deck 6A + PCML6-3):
  - 4 components: cell state, forget gate, input gate, output gate
  - All 6 gate equations
  - Why LSTM solves vanishing gradients (cell state highway)
- **GRU**: simplified LSTM (update gate, reset gate). Fewer parameters, faster.
- **Multi-layer with residual connections** (from PCML6-3)
- **Attention mechanisms** (from PCML6-3):
  - Temporal attention: focus on important time steps
  - Spatial attention: feature relationships via multi-headed attention
  - Connects to M5.4 (Transformers replace RNNs with pure attention)
- **Performance metrics** (from Deck 6A): perplexity, BLEU score, sequence accuracy, cross-entropy loss
- **Applications** (from PCML6-3): financial time series prediction (technical indicators: RSI, MACD, Bollinger Bands), Shakespeare text generation
- **Gradient clipping** (from M4.8 toolkit): essential for RNNs to prevent exploding gradients

**Key Formulas**:

- LSTM forget gate: f*t = sigma(W_f \* [h*{t-1}, x_t] + b_f)
- LSTM input gate: i*t = sigma(W_i \* [h*{t-1}, x_t] + b_i)
- LSTM cell update: C*t = f_t \* C*{t-1} + i*t * tanh(W_C * [h*{t-1}, x_t] + b_C)
- LSTM output gate: o*t = sigma(W_o \* [h*{t-1}, x_t] + b_o)
- LSTM hidden state: h_t = o_t \* tanh(C_t)
- Perplexity: PP = exp(-1/N \* Sum(log P(w_i)))

**Learning Objectives**: Students can:

- Implement LSTM and GRU networks
- Explain all LSTM gate equations and their purpose
- Apply temporal attention to sequence models
- Train RNNs for time-series prediction and text generation

**Exercise**: Build LSTM for Singapore stock price prediction with technical indicators. Add attention layer. Compare LSTM vs GRU. Implement text generation with character-level LSTM.

**Assessment Criteria**: LSTM gates implemented correctly. Attention improves prediction. Text generation produces coherent output.

**R5 Source**: Deck 6A (LSTM/GRU theory, metrics) + PCML6-3 (10MB, attention, financial prediction)

---

## Lesson 5.4: Transformers

**Prerequisites**: 5.3 (attention mechanisms, sequence models)
**Spectrum Position**: Attention-based feature learning — processing sequences in parallel

**Topics**:

- **Self-attention** (derive from scratch):
  - Query, Key, Value matrices
  - Attention(Q, K, V) = softmax(Q _ K^T / sqrt(d_k)) _ V
  - Why divide by sqrt(d_k): prevents softmax from saturating (dot products grow with dimension)
  - Multi-head attention: multiple attention heads capture different relationships
- **Positional encoding**: sinusoidal or learned embeddings (transformers have no inherent position sense)
- **Encoder-decoder architecture** (from Deck 6A):
  - Encoder: multi-head self-attention → feed-forward → layer norm + residual
  - Decoder: same + masked self-attention + cross-attention to encoder
- **Transformer variants** (from Deck 6A):
  - BERT: bidirectional, masked language modelling, NLU tasks
  - GPT: autoregressive decoder, next token prediction, generation
  - T5: text-to-text unified framework
  - Transformer-XL: segment-level recurrence for long contexts
  - Reformer/Longformer: efficient long-sequence handling
- **Vision Transformers (ViT)**: split image into patches → treat as sequence → transformer encoder. Dominant for image classification 2024+.
- **Layer normalisation** (vs batch norm from M4.8): why transformers use layer norm (sequence length varies)
- **BERT fine-tuning** (applied exercise from PCML6-4): fine-tune pre-trained BERT for text classification

**Key Formulas**:

- Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
- Multi-head attention: MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
- Positional encoding: PE(pos, 2i) = sin(pos / 10000^{2i/d}), PE(pos, 2i+1) = cos(pos / 10000^{2i/d})

**Consolidation segment** (30 min): After Transformers, students have seen the four major paradigms (AE, CNN, RNN, Transformer). Compare: "Which architecture for which data type?" Table: images→CNN/ViT, sequences→RNN/Transformer, graphs→GNN (M5.6), generation→VAE/GAN (M5.5).

**Learning Objectives**: Students can:

- Derive scaled dot-product attention from scratch
- Explain why dividing by sqrt(d_k) is necessary
- Fine-tune BERT for a downstream NLP task
- Compare transformer variants and select appropriate one
- Explain ViT and how vision tasks use transformers

**Exercise**: Derive self-attention from scratch (pen-and-paper + code). Fine-tune BERT for text classification (from PCML6-4 TREC6 dataset). Compare with LSTM baseline (M5.3).

**Assessment Criteria**: Self-attention implemented correctly. BERT fine-tuning produces good classification. Comparison with LSTM quantified (accuracy, training time).

**R5 Source**: Deck 6A (architecture + model variants) + PCML6-4 (BERT fine-tuning)

---

## Lesson 5.5: Generative Models — GANs and Diffusion

**Prerequisites**: 5.1 (autoencoders, VAE), 5.2 (CNNs)
**Spectrum Position**: Generative modelling — learning to create new data

**Topics**:

- **GAN fundamentals** (from Deck 6A):
  - Generator vs Discriminator: zero-sum game
  - Adversarial loss: binary cross-entropy
  - Training dynamics: alternating optimisation
- **GAN variants** (from Deck 6A, all covered):
  - **DCGAN**: convolutional generator/discriminator, no max pooling or FC layers, batch norm
  - **Conditional GAN (cGAN)**: condition on class labels for controlled generation
  - **WGAN**: Wasserstein distance instead of JS divergence, gradient penalty, prevents mode collapse
  - **CycleGAN**: unpaired image-to-image translation
  - **StyleGAN**: style-based generation, progressive growing, high-resolution
- **Training challenges**: mode collapse, training instability, evaluation difficulty
- **Evaluation**: FID (Frechet Inception Distance), IS (Inception Score)
- **Diffusion models** (brief, from completeness audit):
  - DDPM (Denoising Diffusion Probabilistic Models): add noise progressively, learn to reverse
  - More stable than GANs, better diversity
  - Stable Diffusion as practical application
  - When to use GANs vs diffusion vs VAE (from Deck 6A generation guide): images→GAN/diffusion, text→transformers, time-series→VAE/LSTM
- **Data generation applications** from Deck 6A: synthetic data for privacy, augmentation, simulation

**Key Formulas**:

- GAN minimax: min_G max_D [E[log D(x)] + E[log(1 - D(G(z)))]]
- WGAN: min_G max_D [E[D(x)] - E[D(G(z))]]
- FID: ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r * Sigma_g)^{1/2})

**Learning Objectives**: Students can:

- Implement DCGAN and WGAN with training loops
- Explain mode collapse and how WGAN addresses it
- Compare GAN, VAE, and diffusion models for different generation tasks
- Evaluate generated data quality using FID

**Exercise**: Implement DCGAN for image generation. Implement WGAN and compare training stability. Evaluate with FID. Discuss when to use GANs vs diffusion.

**Assessment Criteria**: DCGAN generates recognisable images. WGAN more stable (demonstrated). FID computed. Generation model selection guide understood.

**R5 Source**: Deck 6A (6 GAN variants + generation model guide) + PCML6-6 (expanded)

---

## Lesson 5.6: Graph Neural Networks

**Prerequisites**: 5.4 (attention mechanisms)
**Spectrum Position**: Graph feature learning — patterns in connected data

**Topics**:

- Graph data: nodes, edges, adjacency matrix. Applications: social networks, knowledge graphs, molecular structures.
- **GNN architectures** (from Deck 6A):
  - **GCN (Graph Convolutional Networks)**: spectral methods, message passing, aggregation
  - **GraphSAGE**: sampling + aggregating from local neighbourhood, inductive (handles unseen nodes)
  - **GAT (Graph Attention Networks)**: attention weights on neighbours (connects to M5.4 attention)
  - **GIN (Graph Isomorphism Networks)**: captures graph structure more effectively
- **Tasks**: node classification, graph classification, link prediction
- torch_geometric library: `GCNConv`, `global_mean_pool`, DataLoader for graphs

**Key Formulas**:

- GCN: H^(l+1) = sigma(D^{-1/2} A D^{-1/2} H^(l) W^(l))
- GAT attention: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])

**Learning Objectives**: Students can:

- Build GCNs for node and graph classification
- Explain message passing and neighbourhood aggregation
- Compare GCN, GraphSAGE, GAT for different tasks
- Use torch_geometric for graph ML

**Exercise**: Graph classification on TUDataset using GCN. Compare GCN vs GAT. Visualise learned node embeddings.

**Assessment Criteria**: GCN implemented correctly. GAT comparison shows attention weights. Node embeddings visualised.

**R5 Source**: Deck 6A (4 GNN architectures) + PCML6-5

---

## Lesson 5.7: Transfer Learning

**Prerequisites**: 5.2 (CNNs), 5.4 (Transformers/BERT)
**Spectrum Position**: Knowledge transfer — leveraging pre-trained representations

**Topics**:

- Transfer learning concept: pre-trained on large dataset, fine-tune on small target dataset
- **CV Transfer Learning** (from PCML6-8):
  - ResNet fine-tuning: freeze early layers, train later layers + new classifier head
  - Data augmentation for small datasets
  - Applications: mask detection (from PCML6-8), MNIST classification
- **NLP Transfer Learning** (from PCML6-9):
  - BERT fine-tuning: HuggingFace Pipeline API, Lightning-based training
  - Adapter modules as a concept (bottleneck layers between transformer layers)
  - Connects to M6.2 (LoRA + Adapters for LLM fine-tuning)
- **ONNX export**: OnnxBridge for portable model deployment
- **InferenceServer**: predict, predict_batch, warm_cache, PredictionResult
- **Architecture Selection Guide** (consolidation):
  | Data Type | Best Architecture | When to Transfer |
  |---|---|---|
  | Images | CNN/ViT | Always (ImageNet pre-trained) |
  | Text | Transformer | Always (BERT/GPT pre-trained) |
  | Sequences | LSTM/Transformer | Sometimes (domain-specific) |
  | Graphs | GNN | Rarely (task-specific) |
  | Tabular | Gradient boosting | Never (train from scratch) |

**Learning Objectives**: Students can:

- Fine-tune pre-trained vision and NLP models for new tasks
- Apply proper transfer learning techniques (freeze/unfreeze layers)
- Export models to ONNX for deployment
- Select the right architecture for a given problem

**Exercise**: Fine-tune ResNet for mask detection (PCML6-8 task). Fine-tune BERT for text classification (PCML6-9 task). Export both to ONNX. Deploy with InferenceServer.

**Assessment Criteria**: Both models fine-tuned and outperform training from scratch. ONNX export successful. InferenceServer serves predictions.

**R5 Source**: Deck 6A + PCML6-8 (CV transfer) + PCML6-9 (NLP transfer)

---

## Lesson 5.8: Reinforcement Learning

**Prerequisites**: 5.2-5.4 (neural network training experience)
**Spectrum Position**: Learning from interaction — policies learned through environment feedback

**Bridge**: "All DL so far learns from static data (images, text, sequences). RL learns from INTERACTION with an environment. The agent takes actions, receives rewards, and learns a policy that maximises cumulative reward."

**Topics**:

- **RL fundamentals**:
  - Agent, environment, state, action, reward
  - Episode: sequence of (state, action, reward) until termination
  - Policy: mapping from states to actions
  - Value function: expected cumulative reward from a state
- **Bellman equations**: expectation + optimality
  - V(s) = E[R + gamma * V(s')]
  - Q(s,a) = E[R + gamma * max_{a'} Q(s', a')]
- **5 algorithms, 5 business applications** (from PCML6-13):
  - **DQN** (Deep Q-Network): customer churn prevention. Discrete actions.
  - **DDPG** (Deep Deterministic Policy Gradient): manufacturing control. Continuous actions.
  - **SAC** (Soft Actor-Critic): dynamic pricing. Handles uncertainty.
  - **A2C** (Advantage Actor-Critic): resource allocation. Variance reduction via baseline.
  - **PPO** (Proximal Policy Optimization): supply chain optimisation. Clipped objective prevents large updates.
- Custom Gymnasium environments for each use case
- Connection to M6: "RLHF uses PPO to align LLMs with human preferences. DPO achieves the same goal without the reward model."

**Key Formulas**:

- Bellman expectation: V(s) = E[R_{t+1} + gamma * V(S_{t+1}) | S_t = s]
- Bellman optimality: Q*(s,a) = E[R\_{t+1} + gamma * max*{a'} Q\*(S*{t+1}, a') | S_t = s, A_t = a]
- PPO clipped objective: L^{CLIP} = E[min(r_t * A_t, clip(r_t, 1-epsilon, 1+epsilon) * A_t)]
- DQN loss: L = E[(r + gamma * max_{a'} Q(s', a'; theta^-) - Q(s, a; theta))^2]

**Learning Objectives**: Students can:

- Explain the Bellman equations and what they represent
- Implement DQN for a discrete action problem
- Implement PPO for a continuous action problem
- Create custom Gymnasium environments for business applications
- Explain how RL connects to RLHF for LLM alignment (bridge to M6)

**Exercise**: Implement DQN for customer churn prevention. Implement PPO for supply chain optimisation. Create custom environments. Compare performance.

**Assessment Criteria**: Both algorithms converge. Custom environments correctly implement reward functions. PPO→RLHF connection articulated.

**R5 Source**: PCML6-13 (5 algorithms, advanced implementations). Note: RL needs new DECK content — R5 has notebook only.

**End of Module Assessment**: Quiz + DL architecture project (choose a problem, select architecture, train, evaluate, deploy with ONNX).
