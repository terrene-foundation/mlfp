# Module 5 — Deep Learning: Architectures for Vision, Sequence, and Generation

> _"Every architecture is a hypothesis about the structure of the world."_

This chapter is where the training toolkit from Lesson 4.8 meets specialised neural architectures. In Module 4 you built a feedforward network from scratch and understood that hidden layers are automated feature engineering with error feedback. Now you will see how different architectures impose different structural biases on that feature learning — biases that make learning dramatically more efficient for specific data types.

A convolutional neural network assumes spatial locality: nearby pixels are more related than distant ones. A recurrent neural network assumes temporal dependency: the meaning of a word depends on the words before it. A transformer assumes that any element can attend to any other element, weighted by relevance. A graph neural network assumes that information flows along edges. Each assumption is a hypothesis about the data's structure, and when the hypothesis is correct, the architecture learns faster and generalises better than a generic feedforward network.

Every architecture in this chapter is implemented in PyTorch. You will write `nn.Module` subclasses, define `forward()` methods, configure `torch.optim` optimisers, and train with gradient descent. The DL training toolkit — dropout, batch normalisation, learning rate scheduling, gradient clipping, early stopping — applies uniformly across all architectures. What changes is the architecture; what remains constant is the training methodology.

By the end of this chapter you will have implemented autoencoders, CNNs, RNNs, transformers, GANs, GNNs, and RL agents. You will know which architecture to use for which data type, how to transfer pre-trained models to new tasks, and how to export models for production deployment.

---

## Learning Outcomes

By the end of this chapter you will be able to:

- Build and train autoencoders (vanilla, denoising, variational, convolutional, and the contractive, sparse and β variants) and generate new data from VAE latent spaces by deriving the ELBO and reparameterisation trick.
- Implement CNNs with modern enhancements (ResNet skip connections, SE blocks, Kaiming initialisation, mixed precision, Mixup and label smoothing) and explain the convolution output size formula and the degradation problem.
- Build LSTM and GRU networks, write all six LSTM equations, apply temporal and spatial attention, and train sequence models for time-series prediction (judged against a no-change baseline) and character-level text generation.
- Derive scaled dot-product self-attention from scratch, explain the $\sqrt{d_k}$ normalisation, implement multi-head, masked and cross-attention, explain ViT, and fine-tune pre-trained BERT for downstream tasks.
- Implement DCGAN and WGAN with gradient penalty, explain mode collapse and how Wasserstein distance addresses it, evaluate generative quality with FID and Inception Score, and explain why synthetic data is not private by default.
- Build GCN, GraphSAGE, GAT and GIN models for node and graph classification, implement message passing, evaluate link prediction without leakage, and use torch_geometric.
- Fine-tune pre-trained vision and NLP models using transfer learning and adapters, export models with OnnxBridge, and serve them from the ModelRegistry with InferenceServer.
- Implement DQN and PPO (including continuous actions), describe A2C, DDPG and SAC, create custom Gymnasium environments, and explain how RL connects to RLHF for LLM alignment.

---

## Prerequisites

**Module 4 complete.** Specifically, Lesson 4.8 is non-negotiable — this chapter assumes you can:

- Build a neural network from scratch (forward pass, backprop, gradient descent).
- Implement and explain dropout, batch normalisation, weight initialisation, Adam, and learning rate scheduling.
- Read training curves and diagnose overfitting, underfitting, vanishing gradients, and exploding gradients.
- Explain representation learning: hidden layers discover features guided by a loss function.

**PyTorch basics.** All code in this chapter uses PyTorch (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`). If you have not used PyTorch before, spend one hour on the official "60 Minute Blitz" tutorial before starting. The core concepts map directly from Lesson 4.8: `nn.Linear` replaces your manual weight matrices, `nn.ReLU` replaces your `relu()` function, `loss.backward()` replaces your manual backpropagation, and `optimizer.step()` replaces your manual weight update.

**Notation:**

- $\mathbf{W}^{(l)}$ is the weight matrix for layer $l$.
- $\odot$ is element-wise (Hadamard) product.
- $\sigma$ is the sigmoid function unless otherwise noted.
- $\text{softmax}(\mathbf{z})_i = e^{z_i} / \sum_j e^{z_j}$.
- $\mathcal{N}(\mu, \sigma^2)$ is a Gaussian with mean $\mu$ and variance $\sigma^2$.
- $\text{KL}(q \| p)$ is the Kullback-Leibler divergence from $p$ to $q$.

---

## How to Read This Chapter

Same structure as all previous modules: Why This Matters, Core Concepts, Mathematical Foundations, Kailash Engine, Worked Example, Try It Yourself (5+ drills), Cross-References, Reflection.

The three-layer depth markers continue:

| Marker           | Audience             | How to Read It                                                   |
| ---------------- | -------------------- | ---------------------------------------------------------------- |
| **FOUNDATIONS:** | Practitioner with M4 | Architecture intuition, PyTorch code, practical advice.          |
| **THEORY:**      | Intermediate         | Full derivations, loss function analysis, convergence arguments. |
| **ADVANCED:**    | Masters / researcher | Paper references, frontier results, open problems.               |

**Estimated reading time per lesson:**

| Lesson | Title                                  | Reading | Exercise | Total   |
| ------ | -------------------------------------- | ------- | -------- | ------- |
| 5.1    | Autoencoders                           | 110 min | 70 min   | ~3h     |
| 5.2    | CNNs and Computer Vision               | 120 min | 75 min   | ~3h 15m |
| 5.3    | RNNs and Sequence Models               | 120 min | 70 min   | ~3h 10m |
| 5.4    | Transformers                           | 130 min | 80 min   | ~3h 30m |
| 5.5    | Generative Models — GANs and Diffusion | 120 min | 70 min   | ~3h 10m |
| 5.6    | Graph Neural Networks                  | 100 min | 60 min   | ~2h 40m |
| 5.7    | Transfer Learning                      | 100 min | 65 min   | ~2h 45m |
| 5.8    | Reinforcement Learning                 | 130 min | 80 min   | ~3h 30m |

Total: roughly 25 hours. Lesson 5.4 (Transformers) and 5.8 (Reinforcement Learning) are the densest.

---

# Lesson 5.1: Autoencoders

## Why This Matters

In Module 4, PCA compressed data into a lower-dimensional linear subspace. The reconstruction was limited to linear combinations of the original features. What if the data lies on a curved manifold — like a Swiss roll or a nonlinear blend of facial features? Linear PCA cannot capture that curvature. An autoencoder can.

An autoencoder is a neural network that learns to reconstruct its input through a bottleneck. The encoder compresses the input to a low-dimensional latent representation. The decoder reconstructs the original input from that representation. By minimising reconstruction error, the network learns a compressed representation that captures the most important features of the data — just like PCA, but non-linear.

The Variational Autoencoder (VAE) goes further: it makes the latent space a probability distribution, enabling you to generate entirely new data by sampling from it. VAEs are foundational to modern generative AI, and the ELBO (Evidence Lower Bound) objective you will derive in this lesson reappears in diffusion models, variational inference, and Bayesian deep learning.

## Core Concepts

### FOUNDATIONS: The autoencoder architecture

An autoencoder consists of two parts:

**Encoder** $q_\phi$: maps input $\mathbf{x}$ to latent representation $\mathbf{z}$: $\mathbf{z} = q_\phi(\mathbf{x})$

**Decoder** $p_\theta$: maps latent representation $\mathbf{z}$ back to reconstructed input $\hat{\mathbf{x}}$: $\hat{\mathbf{x}} = p_\theta(\mathbf{z})$

The loss is the reconstruction error:

$$\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

The bottleneck (latent dimension $d_z < d_x$) forces the network to learn a compressed representation. If $d_z \geq d_x$, the network could learn the identity function, which is useless.

### FOUNDATIONS: Four variants

**Vanilla autoencoder.** The simplest form. Encoder and decoder are fully connected layers. Latent dimension is a hyperparameter. The learned representation is deterministic.

```python
import torch
import torch.nn as nn

class VanillaAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
```

**Denoising autoencoder (DAE).** Corrupt the input by adding noise (Gaussian noise, random zeroing, or salt-and-pepper), then train the network to reconstruct the clean version. This forces the network to learn robust features that are not sensitive to small perturbations.

**Convolutional autoencoder.** Replace fully connected layers with convolutional layers (encoder) and transposed convolutional layers (decoder). Ideal for image data because convolutions respect spatial locality.

**Variational autoencoder (VAE).** The encoder outputs parameters of a distribution (mean $\mu$ and log-variance $\log \sigma^2$) rather than a deterministic point. The latent representation is sampled from this distribution. This makes the latent space smooth and continuous, enabling generation of new data by sampling.

### THEORY: The VAE ELBO derivation

We want to maximise the marginal log-likelihood of the data:

$$\log p(\mathbf{x}) = \log \int p(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z}) \, d\mathbf{z}$$

This integral is intractable (we cannot compute it analytically). Instead, we introduce an approximate posterior $q_\phi(\mathbf{z} \mid \mathbf{x})$ and derive a lower bound.

Start with:

$$\log p(\mathbf{x}) = \log \int p(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z}) \, d\mathbf{z}$$

Multiply and divide by $q_\phi(\mathbf{z} \mid \mathbf{x})$:

$$= \log \int q_\phi(\mathbf{z} \mid \mathbf{x}) \frac{p(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})} \, d\mathbf{z}$$

By Jensen's inequality ($\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$ for concave $\log$):

$$\geq \int q_\phi(\mathbf{z} \mid \mathbf{x}) \log \frac{p(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})} \, d\mathbf{z}$$

$$= \mathbb{E}_{q_\phi}[\log p(\mathbf{x} \mid \mathbf{z})] - \text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))$$

This is the **ELBO** (Evidence Lower BOund):

$$\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi}[\log p(\mathbf{x} \mid \mathbf{z})]}_{\text{Reconstruction term}} - \underbrace{\text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))}_{\text{Regularisation term}}$$

The reconstruction term encourages accurate reconstruction. The KL term encourages the approximate posterior to be close to the prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$, keeping the latent space well-structured.

### THEORY: The reparameterisation trick

To backpropagate through the sampling step $\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$, we cannot differentiate through a random sample directly. The reparameterisation trick separates the randomness:

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Now $\mathbf{z}$ is a deterministic function of $\boldsymbol{\mu}$, $\boldsymbol{\sigma}$, and $\boldsymbol{\epsilon}$. Since $\boldsymbol{\epsilon}$ does not depend on the model parameters, gradients flow through $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ as usual. This is what makes VAE training possible with standard backpropagation.

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

def vae_loss(x, x_hat, mu, logvar):
    recon = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl
```

### ADVANCED: Additional autoencoder variants

Exercise 1 builds ten variants on Fashion-MNIST. The four above are the core; the rest are worth knowing by name and by the one idea each adds:

- **Sparse autoencoder:** adds an L1 penalty on hidden activations, encouraging most neurons to be inactive. Learns sparse, interpretable features.
- **Contractive autoencoder** (Rifai et al., 2011): adds the per-sample penalty $\lambda \, \|\partial \mathbf{z} / \partial \mathbf{x}\|_F^2$ — the squared Frobenius norm of the encoder's Jacobian _at that input_. It makes the code insensitive to small input perturbations. Because the Jacobian depends on which ReLUs are active for this particular image, it is **not** the same as L2 weight decay on the encoder weights (a squared-weight sum does not depend on the input at all).
- **Stacked autoencoder:** several encoder/decoder layers, historically trained one layer at a time; today simply a deep autoencoder trained end to end.
- **Recurrent autoencoder:** an LSTM/GRU encoder summarises a sequence into a vector and an RNN decoder reconstructs it — the same bottleneck idea for sequences (Lesson 5.3).
- **Contractive VAE:** a VAE with the contractive Jacobian penalty added to the ELBO. Do not abbreviate it "CVAE" — that acronym conventionally means _Conditional_ VAE (a VAE whose encoder and decoder also receive a class label).
- **$\beta$-VAE:** scales the KL term by $\beta > 1$ to encourage disentangled latent factors — each latent dimension tends to capture a single factor of variation.

The contractive penalty is cheap to compute exactly with `torch.func`: `jacrev` differentiates the encoder for one sample, and `vmap` does it for every sample in the batch (this is how Exercise 1's contractive variant computes it):

```python
import torch
from torch.func import jacrev, vmap

def contractive_penalty(encoder, xb):
    """Mean over the batch of ||dz/dx||_F^2, the encoder Jacobian at each input."""
    jac = vmap(jacrev(encoder))(xb)          # (batch, latent_dim, input_dim)
    return jac.pow(2).sum(dim=(1, 2)).mean()

ae = VanillaAutoencoder(input_dim=784, latent_dim=16)
xb = torch.rand(8, 784)
loss = nn.functional.mse_loss(ae(xb), xb) + 1e-3 * contractive_penalty(ae.encoder, xb)
loss.backward()   # the penalty is differentiable, so it trains the encoder
```

| Variant                 | Use it when                                                                   |
| ----------------------- | ----------------------------------------------------------------------------- |
| Vanilla / undercomplete | You want a compact non-linear code (non-linear PCA).                          |
| Denoising               | Inputs are noisy, or you want features robust to corruption.                  |
| Sparse                  | You want a few interpretable, active features per input.                      |
| Contractive             | Similar inputs must map to similar codes (smooth latent space).               |
| Convolutional           | The data is an image — keep spatial structure.                                |
| VAE / $\beta$-VAE       | You need to _generate_ new samples or want a smooth, sampleable latent space. |

## Mathematical Foundations

### THEORY: KL divergence between two Gaussians

For the VAE regularisation term with $q = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ and $p = \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$\text{KL}(q \| p) = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$

This has a closed-form solution, so no sampling is needed for the KL term — only the reconstruction term requires sampling (via reparameterisation).

## The Kailash Engine: ModelVisualizer (training curves and latent plots)

`ModelVisualizer` is kailash-ml's plotting engine. Module 5 uses two of its methods, both of which return an interactive Plotly figure:

- `viz.training_history(metrics, x_label="Epoch", y_label="Value")` — `metrics` is a dict of metric name → list of per-epoch values. Use it for every loss curve in this module.
- `viz.scatter(data, x, y, color=None, title=None)` — `data` is a polars DataFrame. Use it for latent spaces and embeddings.

It has no heatmap, image-grid or "latent scatter" method; for images use matplotlib's `imshow`. The worked example below ends with both calls.

## Worked Example: A VAE on Fashion-MNIST (the Exercise 1 data)

Exercise 1 trains its autoencoders on Fashion-MNIST with a 16-dimensional latent space for 10 epochs; this example uses the same data folder and settings. `get_device()` picks Apple MPS, CUDA or CPU automatically — there is no `torch.cuda.is_available()` branch anywhere in this module.

```python
import torch
import torch.nn as nn
import polars as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from kailash_ml import ModelVisualizer
from shared.kailash_helpers import get_device

device = get_device()
DATA_DIR = "data/mlfp05/fashion_mnist"           # the folder Exercise 1 downloads into
to_tensor = transforms.ToTensor()                 # pixels in [0, 1]: matches Sigmoid + BCE
train_data = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=to_tensor)
test_data = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=to_tensor)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=512)
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

LATENT_DIM, EPOCHS = 16, 10
vae = VAE(input_dim=784, latent_dim=LATENT_DIM).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

history = []
for epoch in range(EPOCHS):
    vae.train()
    total = 0.0
    for batch, _ in train_loader:
        batch = batch.view(-1, 784).to(device)
        x_hat, mu, logvar = vae(batch)
        loss = vae_loss(batch, x_hat, mu, logvar)   # summed over the batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    history.append(total / len(train_data))         # negative ELBO per image, in nats
    print(f"epoch {epoch + 1}: -ELBO per image = {history[-1]:.1f}")

# Generate NEW images: decode samples from the prior N(0, I)
vae.eval()
with torch.no_grad():
    z = torch.randn(16, LATENT_DIM, device=device)
    generated = vae.decoder(z).view(-1, 1, 28, 28).cpu()
print(generated.shape)                              # torch.Size([16, 1, 28, 28])

# Latent space of 512 test images (first two of the 16 dimensions)
with torch.no_grad():
    xb, yb = next(iter(test_loader))
    mu, _ = vae.encode(xb.view(-1, 784).to(device))
latent_df = pl.DataFrame({
    "z1": mu[:, 0].cpu().numpy(),
    "z2": mu[:, 1].cpu().numpy(),
    "label": [CLASSES[i] for i in yb.tolist()],
})
viz = ModelVisualizer()
fig_loss = viz.training_history({"-ELBO per image": history}, x_label="Epoch", y_label="nats")
fig_latent = viz.scatter(latent_df, x="z1", y="z2", color="label",
                         title="VAE latent space (Fashion-MNIST test images)")
```

What to look for: the per-image negative ELBO falls steeply in the first epoch and then flattens; the decoded prior samples look like blurry but recognisable garments (VAE samples are blurrier than GAN samples — Lesson 5.5); and in the latent scatter, visually distinct classes such as trousers and footwear occupy their own regions while shirts, pullovers and coats overlap. Two of sixteen dimensions show only part of the structure — Drill 3 trains a 2-D latent space so you can see all of it.

## Try It Yourself

The drills reuse `device`, `train_loader`, `test_loader`, `EPOCHS`, `VAE`, `vae_loss` and `ModelVisualizer` from the worked example.

**Drill 1.** Implement a vanilla autoencoder with latent dimension 32 and train it on Fashion-MNIST. Compute reconstruction error on the test set. Visualise 10 original images alongside their reconstructions.

**Solution:**

```python
import matplotlib.pyplot as plt

def add_noise(x, sigma):
    return torch.clamp(x + sigma * torch.randn_like(x), 0, 1) if sigma else x

def train_ae(model, epochs=EPOCHS, noise=0.0):
    """Train a flat autoencoder with MSE; with noise > 0 it becomes a denoising AE."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for batch, _ in train_loader:
            clean = batch.view(-1, 784).to(device)
            loss = nn.functional.mse_loss(model(add_noise(clean, noise)), clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def test_mse(model, noise=0.0):
    """Mean squared error per pixel against the CLEAN test images."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch, _ in test_loader:
            clean = batch.view(-1, 784).to(device)
            total += nn.functional.mse_loss(model(add_noise(clean, noise)), clean,
                                            reduction="sum").item()
            n += clean.numel()
    return total / n

ae = train_ae(VanillaAutoencoder(784, 32))
print(f"test MSE per pixel: {test_mse(ae):.4f}")

xb, _ = next(iter(test_loader))
with torch.no_grad():
    recon = ae(xb[:10].view(-1, 784).to(device)).view(-1, 28, 28).cpu()
fig, axes = plt.subplots(2, 10, figsize=(12, 2.6))
for i in range(10):
    axes[0, i].imshow(xb[i, 0], cmap="gray")
    axes[1, i].imshow(recon[i], cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].axis("off")
axes[0, 0].set_title("original", loc="left")
axes[1, 0].set_title("reconstruction", loc="left")
plt.show()
```

Reconstructions keep each garment's silhouette and overall brightness but lose fine texture (prints, stitching): an MSE-trained 32-number code averages away detail it cannot store.

**Drill 2.** Implement a denoising autoencoder. Add Gaussian noise ($\sigma = 0.3$) to the input during training. Compare reconstruction quality with the vanilla autoencoder. Does the DAE produce sharper reconstructions?

**Solution:**

```python
dae = train_ae(VanillaAutoencoder(784, 32), noise=0.3)   # noisy input, clean target
for name, model in [("vanilla", ae), ("denoising", dae)]:
    print(f"{name:>9}: clean input {test_mse(model):.4f} | "
          f"noisy input (sigma=0.3) {test_mse(model, noise=0.3):.4f}")
```

On noisy inputs the DAE's error is lower than the vanilla AE's — the vanilla model faithfully reconstructs much of the noise it was never taught to remove. On clean inputs the vanilla AE is clearly better, because it trained on exactly that input distribution. So the honest answer to "sharper?" is no: both are trained with MSE and both blur. The DAE's gain is robustness — its code ignores perturbations that do not change the garment.

**Drill 3.** Train a VAE with latent dimension 2. Visualise the 2D latent space, colouring each point by its Fashion-MNIST label. Do the classes separate? Generate images by traversing the latent space in a grid from $(-3, -3)$ to $(3, 3)$.

**Solution:**

```python
vae_2d = VAE(784, 2).to(device)
opt = torch.optim.Adam(vae_2d.parameters(), lr=1e-3)
for _ in range(EPOCHS):
    vae_2d.train()
    for batch, _ in train_loader:
        batch = batch.view(-1, 784).to(device)
        loss = vae_loss(batch, *vae_2d(batch))      # vae_2d returns (x_hat, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()

vae_2d.eval()
mus, labels = [], []
with torch.no_grad():
    for batch, y in test_loader:
        mu, _ = vae_2d.encode(batch.view(-1, 784).to(device))
        mus.append(mu.cpu())
        labels.extend(CLASSES[i] for i in y.tolist())
mus = torch.cat(mus)
df_2d = pl.DataFrame({"z1": mus[:, 0].numpy(), "z2": mus[:, 1].numpy(), "label": labels})
fig = ModelVisualizer().scatter(df_2d, x="z1", y="z2", color="label",
                                title="2-D VAE latent space (10,000 test images)")

# Latent traversal: decode a 15 x 15 grid of z values from (-3, -3) to (3, 3)
grid = torch.linspace(-3, 3, 15)
z = torch.cartesian_prod(grid, grid).to(device)               # (225, 2)
with torch.no_grad():
    tiles = vae_2d.decoder(z).view(15, 15, 28, 28).cpu()
canvas = tiles.permute(0, 2, 1, 3).reshape(15 * 28, 15 * 28)  # rows: z1, columns: z2
plt.figure(figsize=(7, 7))
plt.imshow(canvas, cmap="gray")
plt.axis("off")
plt.show()
```

The classes separate only partly. Trousers and the three footwear classes (sandal, sneaker, boot) form fairly distinct regions; pullovers, coats and especially shirts overlap heavily, because two numbers cannot hold everything that distinguishes them. The traversal shows smooth morphing between neighbouring garment types — the KL term is what makes every point in the grid decode to something plausible.

**Drill 4.** Implement a convolutional autoencoder using `nn.Conv2d` and `nn.ConvTranspose2d`, with the same 16-number bottleneck as the worked example. Compare its reconstruction error with a fully connected autoencoder of the same latent size.

**Solution:**

```python
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),    # 16 x 14 x 14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 32 x 7 x 7
            nn.Flatten(), nn.Linear(32 * 7 * 7, latent_dim),         # the bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

conv_ae = ConvAutoencoder().to(device)
print(conv_ae(torch.rand(2, 1, 28, 28, device=device)).shape)   # torch.Size([2, 1, 28, 28])

opt = torch.optim.Adam(conv_ae.parameters(), lr=1e-3)
for _ in range(EPOCHS):
    conv_ae.train()
    for batch, _ in train_loader:
        batch = batch.to(device)                     # keep the (B, 1, 28, 28) image shape
        loss = nn.functional.mse_loss(conv_ae(batch), batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

conv_ae.eval()
with torch.no_grad():
    sq = sum(nn.functional.mse_loss(conv_ae(b.to(device)), b.to(device), reduction="sum").item()
             for b, _ in test_loader)
fc_ae = train_ae(VanillaAutoencoder(784, 16))
n_params = lambda m: sum(p.numel() for p in m.parameters())
print(f"conv AE: {n_params(conv_ae):,} params, test MSE per pixel {sq / (len(test_data) * 784):.4f}")
print(f"FC AE:   {n_params(fc_ae):,} params, test MSE per pixel {test_mse(fc_ae):.4f}")
```

Keep the bottleneck equal when you compare: without the `Linear` layer the code would be $32 \times 7 \times 7 = 1{,}568$ numbers — more than the 784 input pixels — and the "autoencoder" could simply copy its input. At equal latent size the two reach similar error (in a short run the fully connected model can even be slightly ahead), but the convolutional model does it with 61,329 parameters against the fully connected model's 410,912 — about 7× fewer — because its filters are shared across positions. That parameter efficiency is what lets convolutional models scale to larger images.

**Drill 5.** Implement the $\beta$-VAE variant. Train with $\beta = 1, 4, 10$ and observe the effect on the latent space and the reconstructions.

**Solution:**

```python
def beta_vae_loss(x, x_hat, mu, logvar, beta=4.0):
    recon = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl

for beta in [1.0, 4.0, 10.0]:
    model = VAE(784, LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        for batch, _ in train_loader:
            batch = batch.view(-1, 784).to(device)
            loss = beta_vae_loss(batch, *model(batch), beta=beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        xb = next(iter(test_loader))[0].view(-1, 784).to(device)
        x_hat, mu, logvar = model(xb)
        recon = nn.functional.binary_cross_entropy(x_hat, xb, reduction="sum").item() / len(xb)
        kl_per_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)
    active = int((kl_per_dim > 0.05).sum())
    print(f"beta={beta:>4}: recon {recon:.1f} nats/image, active latent dims {active}/{LATENT_DIM}")
```

As $\beta$ grows, reconstruction error rises (blurrier images) and the number of *active* latent dimensions falls: dimensions whose KL is near zero have collapsed onto the prior and carry no information about the input. The dimensions that survive tend to encode broad factors (garment type, overall size and brightness) more independently — that is the disentanglement $\beta$-VAE trades reconstruction for. Higher $\beta$ does not make the class clusters "more separated"; it makes the code more compressed.

## Cross-References

- **Lesson 4.3** used PCA for linear dimensionality reduction. Autoencoders are the non-linear generalisation.
- **Lesson 4.8** introduced the forward pass, backpropagation, and training toolkit. All of that applies here unchanged.
- **Lesson 5.5** will use the VAE's generative capability alongside GANs and diffusion models.
- **Module 6, Lesson 6.2** will use adapter layers that share the bottleneck structure of autoencoders.

## Reflection

You should now be able to:

- Derive the VAE ELBO from the marginal log-likelihood using Jensen's inequality.
- Explain the reparameterisation trick and why it enables gradient flow through sampling.
- Implement all four autoencoder variants in PyTorch.
- Generate new data by sampling from a VAE's latent space.
- Explain why a denoising autoencoder learns more robust features than a vanilla autoencoder.

---

# Lesson 5.2: CNNs and Computer Vision

## Why This Matters

A feedforward network treats an image as a flat vector of pixels, ignoring the spatial structure entirely. A pixel in the upper-left corner is no more related to its neighbour than to a pixel in the lower-right corner. This is wasteful — images have strong spatial locality, and the patterns that matter (edges, textures, shapes) are local. A convolutional neural network exploits this structure by using small filters that slide across the image, detecting local patterns. Early layers learn edges and textures; later layers compose these into objects and scenes. This hierarchical feature learning is why CNNs dominate computer vision.

## Core Concepts

### FOUNDATIONS: The convolution operation

A convolution applies a small filter (also called a kernel) to every position of the input. The filter has learned weights. At each position, the filter's weights are multiplied element-wise with the input values in that region, and the results are summed to produce a single output value. Sliding the filter across the entire input produces a feature map.

Key parameters:

- **Filter size** $(F \times F)$: typically $3 \times 3$ or $5 \times 5$. Smaller filters detect finer patterns.
- **Stride** $(S)$: how many pixels the filter moves at each step. Stride 1 moves one pixel at a time; stride 2 moves two, halving the output size.
- **Padding** $(P)$: zero-valued pixels added around the input border. "Same" padding preserves the spatial dimensions.

### THEORY: CNN output size formula

The output spatial dimension after a convolution is:

$$\text{output} = \frac{W - F + 2P}{S} + 1$$

where $W$ is the input width, $F$ is the filter size, $P$ is the padding, and $S$ is the stride. This formula is essential for designing CNN architectures — you must ensure that spatial dimensions are consistent across layers.

Example: input $28 \times 28$ (Fashion-MNIST), filter $3 \times 3$, padding 1, stride 1: output $= (28 - 3 + 2)/1 + 1 = 28$. Same padding with stride 1 preserves dimensions.

With stride 2: output $= (28 - 3 + 2)/2 + 1 = 14$. The spatial dimension is halved.

### FOUNDATIONS: Pooling

Pooling reduces spatial dimensions by summarising local regions. **Max pooling** takes the maximum value in each window. **Average pooling** takes the mean. Pooling makes the representation more compact and slightly more invariant to small translations of the input.

### THEORY: ResNet skip connections and the degradation problem

Adding layers to a plain CNN should never hurt in principle — the extra layers could learn the identity. Yet He et al. (2015) found that a 56-layer plain network had **higher training error** than a 20-layer one. This is the **degradation problem**. It is not overfitting (the *training* error is worse), and the authors argued it is "unlikely to be caused by vanishing gradients": their plain networks used batch normalisation and the gradients they measured were healthy. It is an optimisation difficulty — solvers struggle to make a stack of non-linear layers approximate an identity mapping. ResNet's answer is the skip connection:

$$\mathbf{H}(\mathbf{x}) = \mathbf{F}(\mathbf{x}) + \mathbf{x}$$

where $\mathbf{F}(\mathbf{x})$ is the residual function learned by the convolutional layers and $\mathbf{x}$ is the identity shortcut. Two things follow:

1. **Identity is easy.** If the identity mapping is close to optimal, the block only has to push $\mathbf{F}(\mathbf{x})$ towards zero, which is far easier than fitting an identity with stacked non-linearities.
2. **An identity gradient path.** $\partial \mathbf{H} / \partial \mathbf{x} = \partial \mathbf{F} / \partial \mathbf{x} + \mathbf{I}$, so the gradient reaching earlier layers always includes an un-attenuated term, whatever the depth. This is why 50-, 101- and 152-layer ResNets train well.

```python
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # skip connection
        return torch.relu(out)
```

### FOUNDATIONS: A short history of CNN architectures

| Architecture | Year | Idea it introduced |
|---|---|---|
| LeNet-5 | 1998 | Convolution + subsampling + fully connected head, for handwritten digits. |
| AlexNet | 2012 | Much deeper, ReLU activations, dropout, GPU training; won ImageNet by a wide margin. |
| VGGNet | 2014 | Depth from uniform stacks of small $3 \times 3$ filters (two $3 \times 3$ layers see a $5 \times 5$ region with fewer parameters). |
| GoogLeNet / Inception | 2014 | Parallel $1 \times 1$, $3 \times 3$, $5 \times 5$ branches in one block; $1 \times 1$ convolutions to cut channels cheaply. |
| ResNet | 2015 | Skip connections; trainable at 152 layers. |

### FOUNDATIONS: SE blocks and modern training enhancements

**Squeeze-and-Excitation (SE) blocks** recalibrate channel-wise features by learning which channels are important:

$$\mathbf{s} = \sigma(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \text{GAP}(\mathbf{x})))$$

where GAP is Global Average Pooling (squeeze each channel to a scalar), and $\mathbf{W}_1 \in \mathbb{R}^{C/r \times C}$, $\mathbf{W}_2 \in \mathbb{R}^{C \times C/r}$ are small fully connected layers (excitation) with reduction ratio $r$. The output is the input scaled channel by channel by $\mathbf{s} \in (0, 1)^C$. An SE block is cheap: with biases it adds $2C^2/r + C/r + C$ parameters, which is $64 \cdot 4 + 4 + 4 \cdot 64 + 64 = 580$ for $C = 64$, $r = 16$. Across a whole SE-ResNet-50 the SE blocks add about 10% to the parameter count (Hu et al., 2018). Because every scale lies in $(0, 1)$, SE can only re-weight a channel — a channel whose ReLU output is zero stays zero.

**Kaiming (He) initialisation.** A ReLU zeroes half of its inputs on average, so to keep activation variance constant through depth the weights need variance $2/n_{\text{in}}$: $W \sim \mathcal{N}(0, 2/n_{\text{in}})$, i.e. standard deviation $\sqrt{2/n_{\text{in}}}$. (Glorot/Xavier uses $2/(n_{\text{in}} + n_{\text{out}})$ for tanh/sigmoid; $1/n_{\text{in}}$ is LeCun initialisation.) PyTorch's default for `nn.Conv2d` is a scaled uniform rule, so set Kaiming explicitly when you want it:

```python
def init_kaiming(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(module.bias)

# model.apply(init_kaiming)   # visits every sub-module once
```

**Mixed precision training** runs most of the forward and backward pass in 16-bit floats (faster, half the activation memory) while keeping an FP32 master copy of the weights. `torch.autocast(device_type=...)` picks the 16-bit operations; with float16, a gradient scaler multiplies the loss before `backward()` so small gradients do not underflow. The pattern works on CUDA, Apple MPS and CPU (use `torch.bfloat16` on CPU):

```python
from shared.kailash_helpers import get_device

device = get_device()
amp_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

def amp_step(model, images, labels, criterion, optimizer):
    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        loss = criterion(model(images), labels)
    optimizer.zero_grad()
    scaler.scale(loss).backward()   # no-op scaling when the scaler is disabled
    scaler.step(optimizer)
    scaler.update()
    return loss.item()
```

**Mixup augmentation** creates training examples by linearly interpolating between pairs of images and their labels: $\tilde{x} = \lambda x_i + (1 - \lambda) x_j$, $\tilde{y} = \lambda y_i + (1 - \lambda) y_j$, where $\lambda \sim \text{Beta}(\alpha, \alpha)$. With cross-entropy this is the same as weighting the loss on the two original labels by $\lambda$ and $1 - \lambda$. It smooths decision boundaries and discourages over-confident predictions.

**Label smoothing** replaces the one-hot target with $(1 - \varepsilon)$ on the true class and $\varepsilon / K$ spread over all $K$ classes. The model can no longer drive one logit to infinity to reach zero loss, which improves calibration. In PyTorch it is one argument: `nn.CrossEntropyLoss(label_smoothing=0.1)`.

**Gradient flow analysis.** After `loss.backward()`, the per-layer gradient norm `p.grad.norm()` shows whether signal reaches the early layers. Healthy networks have norms within an order of magnitude or two across depth; norms that shrink by many orders of magnitude towards the input mean the early layers are barely learning (Drill 2 prints them).

### ADVANCED: Vision Transformers (ViT)

Vision Transformers split an image into fixed-size patches (e.g., $16 \times 16$), embed each patch as a token, and feed the sequence to a transformer *encoder*. They need transformer machinery, so they are covered properly in Lesson 5.4 (with code); the point to take from this lesson is that a ViT has much weaker built-in spatial assumptions than a CNN and therefore needs far more pre-training data to match it.

## Mathematical Foundations

### THEORY: Why convolutions detect patterns

A convolution $(\mathbf{x} * \mathbf{w})[i,j] = \sum_{m,n} \mathbf{x}[i+m, j+n] \cdot \mathbf{w}[m,n]$ is a template-matching operation (strictly a cross-correlation, which is what deep-learning libraries compute). When the input patch matches the filter, the dot product is large. The filter is learned, so the network discovers which templates (edges, textures, shapes) are useful for the task. Weight sharing (the same filter applied everywhere) dramatically reduces the number of parameters and makes the layer translation **equivariant**: shift the input and the feature map shifts the same way. Pooling then adds a little translation *invariance*.

### THEORY: Parameter count comparison

For a $28 \times 28$ greyscale image:

- Fully connected layer to 256 outputs: $784 \times 256 + 256 = 200{,}960$ parameters.
- Convolutional layer with 32 filters of size $3 \times 3$: $32 \times 1 \times 3 \times 3 + 32 = 320$ parameters.

The convolutional layer has about $630\times$ fewer parameters — and it still produces a richer output ($32 \times 28 \times 28$ values against 256). Fewer parameters plus the right structural assumption (locality and equivariance) is why CNNs learn from images with far less data than a fully connected network would need.

## The Kailash Engine: OnnxBridge (model export)

`OnnxBridge` exports a trained model to ONNX, a portable graph format that ONNX Runtime can serve without PyTorch. Two calls matter:

- `bridge.export(model, "torch", output_path=Path(...), sample_input=x)` traces the model with `sample_input` and writes the file. The framework string is `"torch"`; pass `output_path` as a `pathlib.Path`; trace with **at least two rows**, because a batch-of-one trace can fix the batch size at 1. It returns an `OnnxExportResult` — check `.success` (and `.error_message`); export does **not** validate the graph and does not raise on failure.
- `bridge.validate(model, onnx_path, sample_input)` runs the native model and ONNX Runtime on the same rows and returns `.valid`, `.max_diff`, `.mean_diff`. It calls `model.predict(X)` with a NumPy array, so wrap a PyTorch module in a small adapter that has a `predict()` method.

The worked example ends with both calls.

## Worked Example: A residual CNN on CIFAR-10 (the Exercise 2 data)

Exercise 2 classifies CIFAR-10 (60,000 colour images, $3 \times 32 \times 32$, 10 classes) and stores it in `data/mlfp05/cifar10`. This example builds a small residual CNN on the same data, trains it with AdamW and cosine annealing, measures held-out accuracy, and exports it with OnnxBridge.

```python
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from kailash_ml import ModelVisualizer, OnnxBridge
from shared.kailash_helpers import get_device

device = get_device()
DATA_DIR = "data/mlfp05/cifar10"
to_tensor = transforms.ToTensor()
train_data = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=to_tensor)
test_data = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=to_tensor)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=512)

class CifarCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),   # 32 x 32 x 32
            nn.MaxPool2d(2),                                                 # 32 x 16 x 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),  # 64 x 16 x 16
            nn.MaxPool2d(2),                                                 # 64 x 8 x 8
            ResBlock(64),                                                    # 64 x 8 x 8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

EPOCHS = 10
model = CifarCNN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    scheduler.step()
    print(f"epoch {epoch + 1}: loss={total_loss / total:.3f}  train acc={correct / total:.3f}  "
          f"test acc={evaluate(model, test_loader):.3f}")

# Export with OnnxBridge, then check the ONNX graph against PyTorch
class TorchPredictor:
    """Adapter: OnnxBridge.validate calls predict(X) with a NumPy array."""
    def __init__(self, net):
        self.net = net.eval()

    def predict(self, X):
        with torch.no_grad():
            return self.net(torch.as_tensor(np.asarray(X), dtype=torch.float32)).numpy()

model_cpu = model.cpu().eval()
onnx_path = Path("outputs") / "cifar_cnn.onnx"
onnx_path.parent.mkdir(parents=True, exist_ok=True)
sample = torch.stack([test_data[i][0] for i in range(2)])          # 2 rows: dynamic batch
bridge = OnnxBridge()
result = bridge.export(model_cpu, "torch", output_path=onnx_path, sample_input=sample)
assert result.success, result.error_message

rows = torch.stack([test_data[i][0] for i in range(100)]).numpy()
check = bridge.validate(TorchPredictor(model_cpu), onnx_path, rows, tolerance=1e-3)
print(f"ONNX export ok: {result.success}; parity on 100 test images: "
      f"valid={check.valid}, max |diff| = {check.max_diff:.1e}")
```

What to expect: test accuracy climbs fastest in the first epochs and is far above the 10% chance level after one. For a model this small (94,346 parameters), a figure around 70% after ten epochs is typical — treat that as an illustrative ballpark, not a measured result; large pre-trained ResNets reach the mid-90s (Lesson 5.7). The ONNX graph reproduces the PyTorch logits to within about $10^{-6}$ (we measured a maximum difference of $2 \times 10^{-6}$). Recent PyTorch exporters may print an ONNX version-conversion traceback during export; that is log noise — `result.success` is the signal to trust.

## Try It Yourself

The drills reuse `device`, `train_loader`, `test_loader`, `test_data`, `CifarCNN`, `ResBlock`, `evaluate`, `TorchPredictor` and `EPOCHS` from the worked example.

**Drill 1.** Compute the output size at each layer of `CifarCNN` using the formula. Verify by printing tensor shapes during a forward pass.

**Solution:**

```python
probe = CifarCNN()
x = torch.randn(1, 3, 32, 32)
for layer in probe.features:
    x = layer(x)
    print(f"{layer.__class__.__name__:>12}: {tuple(x.shape)}")
```

Each $3 \times 3$ convolution with padding 1 and stride 1 keeps the size: $(32 - 3 + 2)/1 + 1 = 32$. Each $2 \times 2$ max pool with stride 2 halves it: $32 \to 16 \to 8$. The residual block keeps $64 \times 8 \times 8$ (its input and output shapes must match for the addition), and `AdaptiveAvgPool2d(1)` then reduces each of the 64 channels to one number.

**Drill 2.** Add an SE block after the residual block. Compare training curves with and without it, and print the per-layer gradient norms after one backward pass.

**Solution:**

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction), nn.ReLU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        return x * scale

def train_curve(model, epochs=EPOCHS):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    accs = []
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            loss = nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        accs.append(evaluate(model, test_loader))
    return accs

se_model = CifarCNN()
se_model.features.append(SEBlock(64))
print("SE parameters:", sum(p.numel() for p in se_model.features[-1].parameters()))   # 580
curves = {"plain": train_curve(CifarCNN()), "with SE": train_curve(se_model)}
fig = ModelVisualizer().training_history(curves, x_label="Epoch", y_label="Test accuracy")

images, labels = next(iter(train_loader))
loss = nn.functional.cross_entropy(se_model(images.to(device)), labels.to(device))
se_model.zero_grad()
loss.backward()
for name, p in se_model.named_parameters():
    if p.grad is not None and name.endswith("weight") and p.dim() > 1:
        print(f"{name:>28}: grad norm {p.grad.norm():.2e}")
```

On a network this shallow, one SE block changes final accuracy by at most a point or so — within run-to-run noise. Its value grows with depth and channel count, which is why the published gains are reported on ResNet-50-scale models. The gradient norms of a healthy residual CNN stay within a couple of orders of magnitude from the first convolution to the classifier.

**Drill 3.** Implement Mixup and label smoothing. Train with and without them for the same number of epochs. Compare test accuracy and calibration with a reliability diagram.

**Solution:**

```python
def mixup(x, y, alpha=0.2):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def train_regularised(model, epochs=EPOCHS, use_mixup=True, smoothing=0.1):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if use_mixup:
                x_mix, y_a, y_b, lam = mixup(images, labels)
                out = model(x_mix)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                loss = criterion(model(images), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def confidence_and_correct(model):
    model.eval()
    conf, correct = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu()
            top_p, pred = probs.max(dim=1)
            conf.append(top_p)
            correct.append((pred == labels).float())
    return torch.cat(conf).numpy(), torch.cat(correct).numpy()

viz = ModelVisualizer()
for tag, model in [("baseline", train_regularised(CifarCNN(), use_mixup=False, smoothing=0.0)),
                   ("mixup + smoothing", train_regularised(CifarCNN()))]:
    conf, correct = confidence_and_correct(model)
    print(f"{tag:>18}: test acc {correct.mean():.3f}, mean confidence {conf.mean():.3f}")
    fig = viz.calibration_curve(correct, conf, n_bins=10)   # reliability diagram
```

A model is calibrated when its mean confidence matches its accuracy. Without regularisation the network's confidence tends to run above its accuracy (over-confidence), and the gap widens with longer training. Label smoothing and Mixup pull confidence down towards accuracy; on a short run the accuracy change is small, so judge them mainly on the reliability diagram.

**Drill 4.** Export the trained model to ONNX and load it back. Verify that predictions match between the PyTorch model and the ONNX model on 100 test samples.

**Solution:**

```python
import onnxruntime as ort

session = ort.InferenceSession(str(onnx_path))
input_name = session.get_inputs()[0].name
onnx_logits = session.run(None, {input_name: rows})[0]         # rows: 100 test images
with torch.no_grad():
    torch_logits = model_cpu(torch.from_numpy(rows)).numpy()
print("same predicted class on all 100:",
      bool((onnx_logits.argmax(1) == torch_logits.argmax(1)).all()))
print(f"max |logit difference|: {np.abs(onnx_logits - torch_logits).max():.1e}")
```

A batch of 100 runs even though the model was traced with two rows — the exported batch dimension is dynamic. `OnnxBridge.validate` in the worked example performs the same comparison for you.

**Drill 5.** Visualise the learned filters of the first convolutional layer. What patterns do they detect? Compare filters from the trained model with a randomly initialised one.

**Solution:**

```python
import matplotlib.pyplot as plt

def show_filters(conv, title):
    w = conv.weight.detach().cpu()                    # (32, 3, 3, 3)
    w = (w - w.min()) / (w.max() - w.min())           # rescale to [0, 1] for display
    fig, axes = plt.subplots(4, 8, figsize=(8, 4))
    for ax, f in zip(axes.flat, w):
        ax.imshow(f.permute(1, 2, 0))                 # 3 x 3 RGB patch
        ax.axis("off")
    fig.suptitle(title)
    plt.show()

show_filters(model_cpu.features[0], "trained first-layer filters")
show_filters(CifarCNN().features[0], "random initial filters")
```

Trained $3 \times 3$ filters show structure — light/dark edges at different orientations and colour-opponent patterns (for example, more red than green) — whereas random filters are unstructured noise. Filters this small are hard to read; the edge and colour detectors become obvious in networks with $7 \times 7$ or $11 \times 11$ first layers (AlexNet's first-layer filters are the classic picture).

## Cross-References

- **Lesson 4.8** provided the training toolkit (batch norm, dropout, Adam, LR scheduling). All of that is used here.
- **Lesson 5.1** used fully connected autoencoders. Convolutional autoencoders combine this lesson with 5.1.
- **Lesson 5.4** introduces Vision Transformers, an alternative to CNNs for image tasks.
- **Lesson 5.7** applies transfer learning with pre-trained ResNets.

## Reflection

You should now be able to:

- Implement a CNN with convolution, pooling, batch normalisation, and skip connections.
- Compute output dimensions using the formula $(W - F + 2P)/S + 1$.
- Explain why ResNet skip connections make very deep networks trainable: the degradation problem and the identity gradient path.
- Apply SE blocks, Kaiming initialisation, mixed precision, Mixup and label smoothing as modern enhancements.
- Export a model to ONNX with OnnxBridge and verify parity with `validate()`.

---

# Lesson 5.3: RNNs and Sequence Models

## Why This Matters

Language is a sequence. Stock prices are a sequence. Musical notes are a sequence. A feedforward network or CNN processes each input independently — it has no memory of previous inputs. A recurrent neural network (RNN) maintains a hidden state that is updated at each time step, allowing it to model dependencies across time. But vanilla RNNs suffer from vanishing gradients when sequences are long. LSTMs solve this with a gating mechanism that controls what information to remember and what to forget.

## Core Concepts

### FOUNDATIONS: The vanilla RNN

At each time step $t$, the RNN takes the current input $\mathbf{x}_t$ and the previous hidden state $\mathbf{h}_{t-1}$, and produces a new hidden state:

$$\mathbf{h}_t = \tanh(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h)$$

The hidden state $\mathbf{h}_t$ is a summary of all inputs up to time $t$. For long sequences, the gradient of $\mathbf{h}_T$ with respect to $\mathbf{h}_1$ involves $T$ matrix multiplications, causing the gradient to either vanish (if the eigenvalues of $\mathbf{W}_{hh}$ are less than 1) or explode (if greater than 1).

### THEORY: LSTM — all six gate equations

The Long Short-Term Memory network introduces a cell state $\mathbf{C}_t$ — a highway for information that flows through the sequence with only element-wise, mostly additive modifications. An LSTM has **three gates (forget, input, output) plus a candidate**, and six equations in total:

**Forget gate** — what to discard from the cell state:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

**Input gate** — what new information to store:
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

**Candidate cell state** — what the new information looks like:
$$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C)$$

**Cell state update** — forget old + add new:
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$

**Output gate** — what to expose from the cell state:
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

**Hidden state** — filtered cell state:
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

Why this helps gradients: along the direct cell-state path, $\partial \mathbf{C}_t / \partial \mathbf{C}_{t-1} = \text{diag}(\mathbf{f}_t)$ (holding the gates fixed). So the gradient of $\mathbf{C}_T$ with respect to $\mathbf{C}_1$ along that path is a product of forget-gate values, not of weight matrices — and **when $\mathbf{f}_t \approx 1$ the gradient passes through nearly unchanged**. The network *learns* when to keep memory (forget gate near 1) and when to reset it (near 0). The full gradient also flows through the gates and the hidden state, so LSTMs greatly reduce vanishing gradients rather than abolish them; exploding gradients are still possible, which is why RNNs are trained with gradient clipping (below).

**GRU** (Gated Recurrent Unit) simplifies LSTM to two gates (update and reset) and no separate cell state, so it has three weight blocks where an LSTM has four — about 25% fewer parameters at the same size:

$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

(Conventions differ: PyTorch's `nn.GRU` writes the same update with the roles of $\mathbf{z}_t$ and $1 - \mathbf{z}_t$ swapped. The model class is identical.)

### FOUNDATIONS: Gradient clipping

Because backpropagation through time multiplies many Jacobians, an RNN's gradient norm occasionally spikes by orders of magnitude, and one such step can wreck the weights. Gradient clipping rescales the whole gradient vector whenever its norm exceeds a threshold: if $\|\mathbf{g}\| > \tau$, set $\mathbf{g} \leftarrow \tau \, \mathbf{g} / \|\mathbf{g}\|$. The direction is preserved; only the step size is capped. In PyTorch, call `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` between `loss.backward()` and `optimizer.step()`; it returns the norm *before* clipping, which is worth logging.

### THEORY: Perplexity

Perplexity measures how well a language model predicts a sequence. For a sequence of $N$ words:

$$\text{PP} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i \mid w_1, \ldots, w_{i-1})\right)$$

Lower perplexity means the model is less "perplexed" by the data — it assigns higher probability to the observed words. A perplexity of 1 means perfect prediction; a perplexity of $V$ (vocabulary size) is what a model that spreads probability uniformly over the vocabulary gets. Perplexity is simply $\exp$ of the mean cross-entropy loss (in nats), so you get it for free from the training loss; for a character-level model the "words" are characters and $V$ is the alphabet size (Drill 1).

Other sequence metrics you will meet: **sequence accuracy** (fraction of sequences predicted exactly right — harsh, useful for short outputs such as codes), and **BLEU** (n-gram overlap between a generated and a reference text, the classic machine-translation score; it rewards matching wording, not meaning).

### FOUNDATIONS: Temporal attention

Attention allows the model to focus on specific time steps when making a prediction. A score is computed for every hidden state, normalised with a softmax over time, and used to weight the hidden states:

$$e_t = \mathbf{w}^\top \mathbf{h}_t, \qquad \alpha_t = \frac{\exp(e_t)}{\sum_{s=1}^{T} \exp(e_s)}, \qquad \mathbf{c} = \sum_{t=1}^{T} \alpha_t \mathbf{h}_t$$

where $\mathbf{c}$ is the context vector passed to the prediction head. (Encoder–decoder RNNs score each encoder state against the decoder's current state instead, $e_t = \mathbf{s}^\top \mathbf{h}_t$; same idea.) The weights $\alpha_t$ are readable: they say which days or words the prediction leaned on. This mechanism is the precursor to the full self-attention of transformers (Lesson 5.4).

### THEORY: Stacked LSTMs with residual connections

Stacking LSTM layers (`nn.LSTM(..., num_layers=2)`) lets the lower layer track short patterns and the upper layer combine them. Deep stacks run into the same optimisation trouble as deep CNNs, and the same fix applies: add each layer's input to its output, $\mathbf{H}^{(l+1)} = \text{LSTM}^{(l)}(\mathbf{H}^{(l)}) + \mathbf{H}^{(l)}$, whenever the shapes match (Lesson 5.2's skip connection, applied across layers instead of across convolutions).

```python
import torch
import torch.nn as nn

class ResidualLSTM(nn.Module):
    """Stack of single-layer LSTMs with a skip connection around every layer after the first."""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.LSTM(input_dim if i == 0 else hidden_dim, hidden_dim, batch_first=True)
            for i in range(num_layers)
        )

    def forward(self, x):
        h = x
        for i, lstm in enumerate(self.layers):
            out, _ = lstm(h)
            h = out if i == 0 else out + h      # first layer changes the width, so no skip
        return h                                # (batch, seq_len, hidden_dim)
```

### THEORY: Spatial (feature) attention with multi-head attention

Temporal attention asks *which days matter*. Spatial attention asks *which features matter, and how they relate* — for a price series, how the RSI reading should modify the meaning of a Bollinger reading on the same day. Treat the $F$ features of one time step as $F$ tokens, embed each scalar into a small vector, and let multi-head attention (Lesson 5.4) mix them; each head can learn a different feature relationship. The attended features then feed the LSTM:

```python
class SpatialAttention(nn.Module):
    """Multi-head attention ACROSS the features of each time step."""
    def __init__(self, n_features, d_embed=16, n_heads=4):
        super().__init__()
        self.embed = nn.Linear(1, d_embed)                         # each scalar -> vector
        self.feature_id = nn.Parameter(torch.randn(n_features, d_embed) * 0.02)
        self.mha = nn.MultiheadAttention(d_embed, n_heads, batch_first=True)
        self.out = nn.Linear(n_features * d_embed, n_features)

    def forward(self, x):                                          # x: (B, T, F)
        B, T, F = x.shape
        tokens = self.embed(x.reshape(B * T, F, 1)) + self.feature_id    # (B*T, F, d)
        mixed, weights = self.mha(tokens, tokens, tokens)               # weights: (B*T, F, F)
        y = self.out(mixed.reshape(B * T, -1)).reshape(B, T, F)
        return x + y, weights.reshape(B, T, F, F)                       # residual keeps raw features

sa = SpatialAttention(n_features=6)
x_feat, feat_weights = sa(torch.randn(2, 20, 6))
print(x_feat.shape, feat_weights.shape)      # (2, 20, 6) and (2, 20, 6, 6)
```

### FOUNDATIONS: Technical indicators for financial sequences

Raw prices are a weak input: they drift over years, so a model trained on 2012 levels sees unfamiliar numbers in 2023. Traders' technical indicators turn price history into bounded, comparable signals:

- **RSI (14-day Relative Strength Index):** $\text{RSI} = 100 - 100/(1 + \overline{\text{gain}}/\overline{\text{loss}})$, using Wilder's exponential average (weight $1/14$) of daily gains and losses. It lies in $[0, 100]$; readings above 70 are conventionally called overbought and below 30 oversold.
- **MACD:** the 12-day minus the 26-day exponential moving average of the close; its 9-day EMA is the *signal line*, and MACD minus signal is the *histogram* (momentum turning up or down).
- **Bollinger %B:** where the close sits inside a band of the 20-day mean $\pm$ 2 standard deviations; 0 is the lower band, 1 the upper band.

They are computed with polars expressions in the worked example. One caution: daily index *prices* behave close to a random walk, so "tomorrow's close ≈ today's close" is a strong baseline. Indicators help a model describe the recent past; they do not guarantee it predicts the future better than that baseline.

## The Kailash Engine: ModelVisualizer (training curves)

`viz.training_history(metrics, x_label, y_label)` draws one line per entry of a dict of per-epoch lists. Record the train and validation loss as you go, then plot them together — the gap between the lines is your overfitting signal:

```python
from kailash_ml import ModelVisualizer

history = {"train_loss": [0.92, 0.41, 0.30], "val_loss": [0.88, 0.52, 0.47]}   # illustrative values
fig = ModelVisualizer().training_history(history, x_label="Epoch", y_label="MSE")
```

The worked example below builds `history` from a real run.

## Worked Example: Forecasting the Straits Times Index with an attention LSTM

Exercise 3 forecasts the **next five closes of the Straits Times Index** from 20-day windows of real daily bars (`data/mlfp05/stocks/STI.parquet`, 2010–2024). This example uses the same file, window and horizon, with two design decisions worth copying:

1. **Predict the change, not the level.** The model outputs the percentage change of each of the next five closes relative to the last observed close; the forecast price is `last_close * (1 + change / 100)`. We first tried predicting z-scored price *levels*: training loss fell to 0.04, but validation MSE stayed near 0.65 — about 20× worse than simply repeating the last close (0.031). The validation years (2022–2024) reach index levels above anything in the training years (2010–2021: maximum 3,615 against 3,823), and a network with saturating units cannot extrapolate to inputs and targets outside the range it was trained on. Changes and indicators are stationary — their range does not drift with the index level.
2. **Split by time and fit every statistic on the training period only.** Shuffling windows would leak the future into training.

```python
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from kailash_ml import ModelVisualizer
from shared.kailash_helpers import get_device

device = get_device()
SEQ_LEN, HORIZON, EPOCHS = 20, 5, 15

df = pl.read_parquet("data/mlfp05/stocks/STI.parquet").sort("Date")
close = pl.col("Close")
delta = close.diff()
gain = delta.clip(lower_bound=0).ewm_mean(alpha=1 / 14, adjust=False)   # Wilder smoothing
loss = (-delta).clip(lower_bound=0).ewm_mean(alpha=1 / 14, adjust=False)
mid, sd = close.rolling_mean(20), close.rolling_std(20)
macd = close.ewm_mean(span=12, adjust=False) - close.ewm_mean(span=26, adjust=False)
df = df.with_columns(
    (100 * close.pct_change()).alias("ret_pct"),                       # daily return, %
    (100 * (pl.col("High") - pl.col("Low")) / close).alias("range_pct"),
    (100 - 100 / (1 + gain / loss)).alias("rsi_14"),
    (100 * (macd - macd.ewm_mean(span=9, adjust=False)) / close).alias("macd_hist_pct"),
    ((close - (mid - 2 * sd)) / (4 * sd)).alias("bb_pct_b"),
).drop_nulls()

FEATURES = ["ret_pct", "range_pct", "rsi_14", "macd_hist_pct", "bb_pct_b"]
feats = df.select(FEATURES).to_numpy().astype(np.float32)
closes = df["Close"].to_numpy().astype(np.float32)
split = int(0.8 * len(feats))                                     # first 80% of days = train
mean, std = feats[:split].mean(0), feats[:split].std(0) + 1e-8    # fit on train only
z = (feats - mean) / std

def windows(start, end, seq_len=SEQ_LEN):
    """seq_len days of features -> % change of the next HORIZON closes vs the last close."""
    idx = range(start, end - seq_len - HORIZON + 1)
    X = np.stack([z[i:i + seq_len] for i in idx])
    last = np.array([closes[i + seq_len - 1] for i in idx])[:, None]
    future = np.stack([closes[i + seq_len:i + seq_len + HORIZON] for i in idx])
    y = (100 * (future / last - 1)).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)

X_tr, y_tr = windows(0, split)
X_va, y_va = windows(split - SEQ_LEN, len(z))     # validation targets all lie after the split
train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
print(f"{len(df)} trading days -> train windows {tuple(X_tr.shape)}, validation {tuple(X_va.shape)}")

class StockLSTM(nn.Module):
    def __init__(self, n_features, hidden_dim=64, horizon=HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.attention = nn.Linear(hidden_dim, 1)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)                                # (B, T, H)
        weights = torch.softmax(self.attention(out), dim=1)  # (B, T, 1): one weight per day
        context = (weights * out).sum(dim=1)                 # (B, H)
        return self.head(context), weights.squeeze(-1)

torch.manual_seed(0)
model = StockLSTM(len(FEATURES)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
history = {"train_loss": [], "val_loss": []}
for epoch in range(EPOCHS):
    model.train()
    batch_losses = []
    for xb, yb in train_loader:
        pred, _ = model(xb.to(device))
        loss_value = nn.functional.mse_loss(pred, yb.to(device))
        optimizer.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_losses.append(loss_value.item())
    model.eval()
    with torch.no_grad():
        val_pred, _ = model(X_va.to(device))
        history["val_loss"].append(nn.functional.mse_loss(val_pred, y_va.to(device)).item())
    history["train_loss"].append(float(np.mean(batch_losses)))

baseline = (y_va ** 2).mean().item()      # random-walk forecast: "no change" for all 5 days
print(f"validation MSE (%^2): LSTM best {min(history['val_loss']):.3f} "
      f"(epoch {int(np.argmin(history['val_loss'])) + 1}), final {history['val_loss'][-1]:.3f} | "
      f"no-change baseline {baseline:.3f}")
fig = ModelVisualizer().training_history(history, x_label="Epoch", y_label="MSE (%^2)")
```

Read the result against the baseline, not in isolation. In our run the no-change baseline scored 1.686 and the LSTM's validation MSE moved between about 1.65 and 1.80: its best epoch beat the baseline by roughly 2% (and choosing that epoch *on the validation set* makes even that optimistic), while the training loss kept falling and the validation loss drifted up — the overfitting signal in the plot. That is a finding, not a bug: daily index prices behave close to a random walk, and an honest model shows it. In practice you would pick the epoch on a separate validation period and report the final score on a later test period.

## Try It Yourself

The drills reuse `device`, `FEATURES`, `z`, `split`, `windows`, `X_va`, `y_va`, `baseline`, `train_loader`, `model`, `StockLSTM`, `ResidualLSTM`, `HORIZON`, `SEQ_LEN` and `EPOCHS` from above.

**Drill 1.** Implement a character-level LSTM for text generation. Train it on the news text in `data/mlfp05/ag_news.parquet` (5,000 headlines with their first sentence), report validation perplexity, and generate text by sampling from the model's output distribution.

**Solution:**

```python
texts = pl.read_parquet("data/mlfp05/ag_news.parquet")["text"].to_list()
corpus = "\n".join(texts)
chars = sorted(set(corpus))
stoi = {c: i for i, c in enumerate(chars)}
encoded = torch.tensor([stoi[c] for c in corpus])
cut = int(0.9 * len(encoded))
train_ids, val_ids = encoded[:cut], encoded[cut:]
V, CTX = len(chars), 100
print(f"{len(corpus):,} characters, vocabulary of {V}")

class CharLSTM(nn.Module):
    def __init__(self, vocab, emb=64, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, idx, state=None):
        h, state = self.lstm(self.emb(idx), state)
        return self.out(h), state

def batch(ids, n=64):
    starts = torch.randint(0, len(ids) - CTX - 1, (n,))
    x = torch.stack([ids[s:s + CTX] for s in starts])
    y = torch.stack([ids[s + 1:s + CTX + 1] for s in starts])   # next character
    return x.to(device), y.to(device)

char_model = CharLSTM(V).to(device)
opt = torch.optim.Adam(char_model.parameters(), lr=2e-3)
for step in range(3000):
    x, y = batch(train_ids)
    logits, _ = char_model(x)
    loss_value = nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad()
    loss_value.backward()
    torch.nn.utils.clip_grad_norm_(char_model.parameters(), 1.0)
    opt.step()

char_model.eval()
with torch.no_grad():
    x, y = batch(val_ids, n=256)
    logits, _ = char_model(x)
    val_ce = nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).item()
print(f"validation perplexity: {np.exp(val_ce):.2f}  (uniform guessing: {V})")

def generate(prompt, n=200, temperature=0.8):
    idx = torch.tensor([[stoi[c] for c in prompt]], device=device)
    out, state = list(prompt), None
    with torch.no_grad():
        logits, state = char_model(idx, state)
        for _ in range(n):
            probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
            out.append(chars[nxt.item()])
            logits, state = char_model(nxt.view(1, 1), state)
    return "".join(out)

print(generate("Stocks "))
```

Perplexity starts near the vocabulary size (83 characters here) and falls as training proceeds — in our check it was already 8.3 after 300 steps and it keeps falling with the full 3,000, i.e. the model is choosing among a handful of plausible next characters instead of the whole alphabet. The samples learn spelling, spacing, capitalised headline words and news phrasing well before they make sense. Lower `temperature` gives safer, more repetitive text; higher gives more varied text with more misspellings.

**Drill 2.** Compare LSTM and GRU on the STI task. Which has more parameters? Which converges faster? Which achieves lower validation MSE?

**Solution:**

```python
class StockGRU(StockLSTM):
    def __init__(self, n_features, hidden_dim=64, horizon=HORIZON):
        super().__init__(n_features, hidden_dim, horizon)
        self.lstm = nn.GRU(n_features, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)

def fit(model, loader=None, epochs=EPOCHS, clip=1.0):
    """Train a forecaster; return per-epoch validation MSE and the pre-clipping gradient norms."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    curve, norms = [], []
    for _ in range(epochs):
        model.train()
        for xb, yb in loader or train_loader:
            loss_value = nn.functional.mse_loss(model(xb.to(device))[0], yb.to(device))
            opt.zero_grad()
            loss_value.backward()
            norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item())
            opt.step()
        model.eval()
        with torch.no_grad():
            curve.append(nn.functional.mse_loss(model(X_va.to(device))[0], y_va.to(device)).item())
    return curve, np.array(norms)

for name, cls in [("LSTM", StockLSTM), ("GRU", StockGRU)]:
    torch.manual_seed(0)
    m = cls(len(FEATURES))
    n_params = sum(p.numel() for p in m.parameters())
    curve, _ = fit(m)
    print(f"{name}: {n_params:,} parameters, best val MSE {min(curve):.3f} "
          f"at epoch {int(np.argmin(curve)) + 1} (baseline {baseline:.3f})")
```

The LSTM has 51,846 parameters and the GRU 38,982: their recurrent layers have four and three weight blocks respectively, so the GRU's recurrent part is exactly 25% smaller. Neither reliably wins. In our runs both reached best validation MSEs of about 1.64–1.65 against the baseline's 1.69, and which one was ahead changed with the random seed. Differences between seeds are as large as the difference between the architectures — so on small data, prefer the cheaper GRU unless you have evidence otherwise.

**Drill 3.** Monitor the gradient norm during training. How often does clipping at 1.0 activate? Does clipping change the final validation loss?

**Solution:**

```python
torch.manual_seed(0)
clipped_curve, norms = fit(StockLSTM(len(FEATURES)), clip=1.0)
torch.manual_seed(0)
free_curve, _ = fit(StockLSTM(len(FEATURES)), clip=float("inf"))   # inf = never clip
print(f"clipping fired on {(norms > 1.0).mean():.0%} of steps; "
      f"median norm {np.median(norms):.2f}, max {norms.max():.2f}")
print(f"final val MSE with clipping {clipped_curve[-1]:.3f}, without {free_curve[-1]:.3f}")
```

`clip_grad_norm_` returns the norm *before* clipping, so the fraction of steps above 1.0 is exactly how often clipping changed the update. In our runs it fired on a minority of steps (between 2% and 16% depending on seed and cell type) with a median norm around 0.3–0.5 — but the largest pre-clipping norm was 9.1, so occasional spikes do happen even on this short series. In the run above the clipped model finished at a validation MSE of 1.670 and the unclipped one at 1.746; one pair of runs is not proof, but it is the direction you expect. Clipping is insurance against the rare exploding step, and it costs nothing when it does not fire.

**Drill 4.** Visualise attention weights for a specific prediction. Which time steps does the model attend to most? Does the pattern make sense?

**Solution:**

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    _, w = model(X_va.to(device))          # (n_windows, SEQ_LEN)
w = w.cpu().numpy()
plt.bar(range(-SEQ_LEN + 1, 1), w[-1])
plt.xlabel("day relative to forecast date (0 = most recent)")
plt.ylabel("attention weight")
plt.show()
print(f"mean weight on the last 5 days: {w[:, -5:].sum(1).mean():.3f} | uniform would be {5 / SEQ_LEN:.3f}")
```

In our run the last five days received 32% of the weight on average, against 25% for uniform attention: a mild tilt towards recent days rather than a sharp focus. That is what you would expect when the target is close to unpredictable — recent days are slightly more informative, but no day is decisive. Treat attention weights as a description of what this model uses, not as proof of a mechanism — two models that predict equally well can spread their weights quite differently.

**Drill 5.** Compare `ResidualLSTM` (4 layers) with a plain 4-layer `nn.LSTM` on longer windows (`seq_len=100`). Does the residual connection help?

**Solution:**

```python
class Forecaster(nn.Module):
    def __init__(self, body, hidden_dim=64):
        super().__init__()
        self.body, self.head = body, nn.Linear(hidden_dim, HORIZON)

    def forward(self, x):
        h = self.body(x)
        h = h[0] if isinstance(h, tuple) else h        # nn.LSTM returns (output, state)
        return self.head(h[:, -1]), None

X_tr100, y_tr100 = windows(0, split, seq_len=100)
X_va, y_va = windows(split - 100, len(z), seq_len=100)   # fit() scores on X_va / y_va
loader100 = DataLoader(TensorDataset(X_tr100, y_tr100), batch_size=64, shuffle=True)
n_in = len(FEATURES)
for name, body in [("plain 4-layer", nn.LSTM(n_in, 64, num_layers=4, batch_first=True)),
                   ("residual 4-layer", ResidualLSTM(n_in, 64, num_layers=4))]:
    torch.manual_seed(0)
    curve, norms = fit(Forecaster(body), loader=loader100)
    print(f"{name}: best val MSE {min(curve):.3f}, median grad norm {np.median(norms):.2f}")
```

Look at two things: how quickly each model's validation loss settles, and the gradient norms. The skip connections give every layer an identity path — in our run the residual stack's median gradient norm was about twice the plain stack's (1.17 against 0.54), i.e. more signal reached the parameters, and its best validation MSE was slightly lower (1.624 against 1.645). But on a near-random-walk target both models end close to the no-change baseline — a better optimiser cannot extract signal that is not there. The residual design pays off on long sequences that *do* contain learnable structure (Drill 1's text is one).

**Drill 6.** Put the `SpatialAttention` module in front of the LSTM so the features of each day can inform one another before the sequence model sees them. Train it on the 20-day windows and inspect which feature pairs the heads link.

**Solution:**

```python
class SpatialStockLSTM(StockLSTM):
    def __init__(self, n_features, hidden_dim=64, horizon=HORIZON):
        super().__init__(n_features, hidden_dim, horizon)
        self.spatial = SpatialAttention(n_features)

    def forward(self, x):
        x, self.feature_weights = self.spatial(x)          # keep the (B, T, F, F) weights
        return super().forward(x)

X_va, y_va = windows(split - SEQ_LEN, len(z))             # back to 20-day windows
torch.manual_seed(0)
spatial_model = SpatialStockLSTM(len(FEATURES))
curve, _ = fit(spatial_model)
print(f"spatial + temporal attention: best val MSE {min(curve):.3f} (baseline {baseline:.3f})")

spatial_model.eval()
with torch.no_grad():
    spatial_model(X_va.to(device))
links = spatial_model.feature_weights.mean(dim=(0, 1)).cpu()      # average over windows and days
for i, name in enumerate(FEATURES):
    j = int(links[i].argmax())
    print(f"{name:>14} attends most to {FEATURES[j]:<14} (weight {links[i, j]:.2f})")
```

Expect no reliable accuracy gain on this near-random-walk target — the point of the drill is the mechanism. The averaged weight matrix shows, for each feature, which other features its updated value draws on; uniform rows (every weight near $1/5$) mean the module found nothing worth mixing. On problems with genuinely interacting inputs (sensor arrays, many related series) this is where spatial attention earns its place.

## Cross-References

- **Lesson 4.8** introduced backpropagation through layers. BPTT (Backpropagation Through Time) is the same algorithm unrolled through time steps.
- **Lesson 5.2** used CNNs for spatial data. RNNs handle temporal data. The two can be combined (ConvLSTM) for spatiotemporal data.
- **Lesson 5.4** will replace the RNN's sequential processing with parallel self-attention. The attention mechanism you just learned is the conceptual ancestor of the transformer.

## Reflection

You should now be able to:

- Write all six LSTM equations (three gates plus a candidate, the cell update and the hidden state) and explain each gate's role.
- Explain why the cell-state highway eases the vanishing gradient problem, and why gradient clipping is still needed.
- Compare LSTM and GRU in terms of parameters and performance.
- Implement temporal attention, residual LSTM stacks and multi-head spatial attention, and read attention weights critically.
- Build technical-indicator features with polars and judge a forecast against the no-change baseline.
- Train a character-level LSTM for text generation and report its perplexity.

---

# Lesson 5.4: Transformers

## Why This Matters

The transformer is the architecture behind GPT, BERT and essentially every major language model since 2017. It replaced RNNs for most sequence tasks because it processes all positions in parallel (instead of sequentially) and captures long-range dependencies through attention (instead of hoping information persists through gates).

In this lesson you will derive self-attention from scratch — starting from the question "how should a sequence element decide which other elements to pay attention to?" — and build up to the full transformer architecture. The $\sqrt{d_k}$ normalisation factor, multi-head attention, positional encoding, and the encoder-decoder structure will all be derived from first principles.

## Core Concepts

### THEORY: Self-attention from scratch

Consider a sequence of $n$ vectors $\mathbf{x}_1, \ldots, \mathbf{x}_n$ (e.g., word embeddings). We want to compute a new representation for each position that incorporates information from all positions, weighted by relevance.

For each position $i$, we compute three vectors from $\mathbf{x}_i$:

- **Query** $\mathbf{q}_i = \mathbf{W}_Q \mathbf{x}_i$ — "what am I looking for?"
- **Key** $\mathbf{k}_i = \mathbf{W}_K \mathbf{x}_i$ — "what do I contain?"
- **Value** $\mathbf{v}_i = \mathbf{W}_V \mathbf{x}_i$ — "what information should I contribute?"

The attention weight from position $i$ to position $j$ is the dot product of query $i$ with key $j$, normalised:

$$\alpha_{ij} = \text{softmax}_j\left(\frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}}\right)$$

The output for position $i$ is the weighted sum of all values:

$$\mathbf{o}_i = \sum_{j=1}^{n} \alpha_{ij} \mathbf{v}_j$$

In matrix form:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

### THEORY: Why divide by $\sqrt{d_k}$

The dot product $\mathbf{q}^T \mathbf{k}$ is the sum of $d_k$ terms. If the entries of $\mathbf{q}$ and $\mathbf{k}$ have zero mean and unit variance, the dot product has variance $d_k$ (by the properties of independent random variables). For large $d_k$, the dot products become large in magnitude, which pushes the softmax into saturated regions where the gradients are near zero. Dividing by $\sqrt{d_k}$ normalises the variance of the dot products back to approximately 1, keeping the softmax in its sensitive regime.

Concretely: if $d_k = 512$, the dot products have standard deviation $\sqrt{512} \approx 22.6$. Without scaling, many attention weights would be pushed to near 0 or near 1, and the gradient of softmax in those regions is negligible. With scaling, the standard deviation is reduced to approximately 1, and softmax produces meaningful (non-degenerate) distributions.

### THEORY: Multi-head attention

A single attention head captures one type of relationship. Multiple heads capture different types simultaneously:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$

where $\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_Q^i, \mathbf{K}\mathbf{W}_K^i, \mathbf{V}\mathbf{W}_V^i)$.

Each head operates in a lower-dimensional subspace of size $d_k = d_{\text{model}}/h$ (64 per head for $d_{\text{model}} = 512$, $h = 8$), so the total computation is about the same as a single head with full dimensionality.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, L, _ = Q.shape
        Q = self.W_Q(Q).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_O(out)
```

### THEORY: Positional encoding

Self-attention on its own is **permutation-equivariant**: shuffle the input tokens and the outputs are shuffled in exactly the same way, so the model is order-blind — "dog bites man" and "man bites dog" contain the same set of tokens. Unlike RNNs, which process sequentially, transformers therefore need position information added to the input embeddings:

$$\text{PE}(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$
$$\text{PE}(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

Each pair of dimensions is a sinusoid with angular frequency $\omega_i = 10000^{-2i/d}$. At $i = 0$ the frequency is 1 radian per position, the fastest; as $i$ approaches $d/2$ it falls towards $10^{-4}$, the slowest. So **low dimensions oscillate rapidly and encode fine, local position; high dimensions change slowly and encode coarse, global position** — like the second, minute and hour hands of a clock. Because $\sin(a)\sin(b) + \cos(a)\cos(b) = \cos(a - b)$, the dot product $\text{PE}(p) \cdot \text{PE}(p+k) = \sum_i \cos(\omega_i k)$ depends only on the offset $k$, which makes relative positions easy to attend to (Drill 5 verifies this). Learned positional embeddings are an alternative and often perform similarly; BERT and ViT use learned ones.

### FOUNDATIONS: Layer normalisation

Transformers use layer normalisation instead of batch normalisation because sequence lengths vary within a batch. Layer normalisation normalises across the feature dimension for each individual sample:

$$\hat{z}_i = \frac{z_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

where $\mu$ and $\sigma^2$ are computed over the feature dimension, not the batch dimension.

### THEORY: The encoder and decoder blocks

An **encoder layer** is two sub-layers, each wrapped in a residual connection and a layer norm: multi-head self-attention, then a position-wise feed-forward network (two linear layers with a ReLU or GELU between them, typically 4× wider than $d_{\text{model}}$):

$$\mathbf{Z} = \text{LayerNorm}(\mathbf{X} + \text{MHA}(\mathbf{X}, \mathbf{X}, \mathbf{X})), \qquad \mathbf{Y} = \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))$$

(The original paper normalises after the residual, as written; most modern models normalise *before* each sub-layer — "pre-norm" — which trains more stably when deep. PyTorch's `nn.TransformerEncoderLayer(norm_first=True)` is pre-norm.)

A **decoder layer** adds two things:

1. **Masked (causal) self-attention.** When generating token $t$, the decoder must not look at tokens $t+1, t+2, \ldots$ — they do not exist yet at inference time. A causal mask sets those attention scores to $-\infty$ before the softmax, so their weights are exactly zero.
2. **Cross-attention.** Queries come from the decoder; keys and values come from the encoder's output. This is how a translation model's decoder "looks at" the source sentence.

The `MultiHeadAttention` class above already supports both: pass a lower-triangular mask for causal attention, and pass different tensors for the query and the key/value for cross-attention.

```python
d_model, n_heads, L_dec, L_enc = 32, 4, 5, 7
self_attn = MultiHeadAttention(d_model, n_heads)
cross_attn = MultiHeadAttention(d_model, n_heads)

dec = torch.randn(2, L_dec, d_model)               # decoder tokens so far
enc = torch.randn(2, L_enc, d_model)               # encoder output (source sentence)
causal = torch.tril(torch.ones(L_dec, L_dec))      # 1 = may attend, 0 = future (masked)

h = self_attn(dec, dec, dec, mask=causal)          # masked self-attention
out = cross_attn(h, enc, enc)                      # queries from decoder, keys/values from encoder
print(h.shape, out.shape)                          # (2, 5, 32) and (2, 5, 32)

# The mask really hides the future: changing the LAST token leaves earlier outputs unchanged
dec2 = dec.clone()
dec2[:, -1] += 10.0
same = torch.allclose(self_attn(dec, dec, dec, mask=causal)[:, :-1],
                      self_attn(dec2, dec2, dec2, mask=causal)[:, :-1])
print("earlier positions unaffected by a future token:", same)   # True
```

PyTorch provides the same mask as `nn.Transformer.generate_square_subsequent_mask(L)` (a float mask of 0 and $-\infty$), and the full blocks as `nn.TransformerEncoderLayer` and `nn.TransformerDecoderLayer`.

### FOUNDATIONS: Transformer variants

| Model | Architecture | Pre-training / idea | Strength |
|---|---|---|---|
| **BERT** | Encoder only | Masked language modelling (bidirectional context) | Understanding tasks: classification, NER, extractive QA |
| **GPT** | Decoder only | Next-token prediction (causal) | Generation, few-shot learning |
| **T5** | Encoder–decoder | Every task cast as text-to-text | One model and one loss for all NLP tasks |
| **Transformer-XL** | Decoder with memory | Re-uses hidden states from the previous segment (segment-level recurrence) | Context longer than one segment |
| **Reformer / Longformer** | Efficient attention | Locality-sensitive hashing (Reformer) or sliding-window + a few global tokens (Longformer) instead of all-pairs attention | Long documents; attention cost grows roughly linearly, not quadratically |
| **ViT** | Encoder only | Image patches as tokens | Image classification at scale |

### THEORY: Vision Transformers (ViT)

A ViT (Dosovitskiy et al., 2021) turns an image into a sequence: cut it into $P \times P$ patches, flatten and linearly project each patch to a $d_{\text{model}}$-dimensional token (a `Conv2d` with kernel size and stride both $P$ does exactly this), prepend a learnable `[CLS]` token, add learned positional embeddings, and run a standard transformer **encoder**. The classifier reads the final `[CLS]` vector. A $224 \times 224$ image with $16 \times 16$ patches becomes $14 \times 14 = 196$ tokens.

The trade-off against CNNs is inductive bias. A CNN assumes locality and translation equivariance; a ViT assumes almost nothing — every patch can attend to every other from the first layer. That makes ViTs data-hungry: trained from scratch on a small dataset they usually trail a CNN, while with large-scale pre-training they match or beat CNNs, and pre-trained ViTs are now a default backbone for image classification. Lesson 5.7 shows how to reuse such pre-trained backbones.

```python
class TinyViT(nn.Module):
    """ViT for 32x32 RGB images (CIFAR-10 shape): 4x4 patches -> 64 tokens + [CLS]."""
    def __init__(self, img_size=32, patch=4, in_ch=3, d_model=64, depth=4, n_heads=4, n_classes=10):
        super().__init__()
        n_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)   # learned positions
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)            # (B, 64, d_model)
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1) + self.pos
        return self.head(self.encoder(tokens)[:, 0])                        # read [CLS]

vit = TinyViT()
print(vit(torch.randn(2, 3, 32, 32)).shape)                    # torch.Size([2, 10])
print(f"{sum(p.numel() for p in vit.parameters()):,} parameters")
```

You can train `TinyViT` with exactly the CIFAR-10 loop from Lesson 5.2's worked example; with no pre-training, expect it to trail the small residual CNN — the inductive-bias point above, made concrete.

## The Kailash Engine: ModelVisualizer and attention maps

Use `ModelVisualizer().training_history(...)` for the loss and accuracy curves of every model in this lesson, exactly as in Lessons 5.1–5.3. `ModelVisualizer` has no heatmap method, so draw attention maps with matplotlib's `imshow`. To read the attention weights of a **trained** `nn.TransformerEncoderLayer`, call its attention module yourself with `need_weights=True` — inside the layer's own forward pass PyTorch requests no weights (so a forward hook sees `None`):

```python
import matplotlib.pyplot as plt

layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True, norm_first=True)
layer.eval()
x = torch.randn(1, 10, 64)                       # in practice: your embedded, position-encoded tokens
with torch.no_grad():
    h = layer.norm1(x)                           # pre-norm layers attend over the normed input
    _, attn = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=False)
print(attn.shape)                                # (1, 4, 10, 10): batch, head, query, key
plt.imshow(attn[0, 0], cmap="viridis")
plt.xlabel("key position")
plt.ylabel("query position")
plt.colorbar()
plt.show()
```

Use the weights of the trained layer — a freshly constructed attention module has random projections, and its "patterns" mean nothing.

## Worked Example: Self-Attention from Scratch and BERT Fine-Tuning

Part A computes attention by hand. Part B fine-tunes pre-trained BERT on AG News topic classification (World, Sports, Business, Sci/Tech), the task of Exercise 4. Exercise 4 downloads the full 120,000-headline training set; this example uses the 5,000-row slice and 1,000-row test set bundled in `data/mlfp05/` so it runs in minutes.

```python
# Part A: Self-attention from scratch
import torch

d_model = 64
seq_len = 10
x = torch.randn(1, seq_len, d_model)  # 1 batch, 10 tokens, 64 dims

W_Q = torch.randn(d_model, d_model) * 0.1
W_K = torch.randn(d_model, d_model) * 0.1
W_V = torch.randn(d_model, d_model) * 0.1

Q = x @ W_Q
K = x @ W_K
V = x @ W_V

d_k = d_model
scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
attn_weights = torch.softmax(scores, dim=-1)
output = attn_weights @ V

print(f"Scores shape: {scores.shape}")          # (1, 10, 10)
print(f"Attention shape: {attn_weights.shape}") # (1, 10, 10)
print(f"Output shape: {output.shape}")          # (1, 10, 64)
print(attn_weights.sum(dim=-1))                 # every row sums to 1
```

```python
# Part B: fine-tune BERT for 4-class topic classification
import polars as pl
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertForSequenceClassification
from shared.kailash_helpers import get_device

device = get_device()
train_df = pl.read_parquet("data/mlfp05/ag_news.parquet")        # 5,000 rows: text, label
test_df = pl.read_parquet("data/mlfp05/ag_news_test.parquet")    # 1,000 held-out rows
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4).to(device)

def encode(df, max_len=64):
    enc = tokenizer(df["text"].to_list(), max_length=max_len, padding="max_length",
                    truncation=True, return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(df["label"].to_list()))

bert_train = DataLoader(encode(train_df), batch_size=32, shuffle=True)
bert_test = DataLoader(encode(test_df), batch_size=64)

# Freeze the embeddings and the lower 8 of 12 encoder layers, as Exercise 4 does
frozen = ("bert.embeddings.",) + tuple(f"bert.encoder.layer.{i}." for i in range(8))
for name, p in bert.named_parameters():
    p.requires_grad = not name.startswith(frozen)
trainable = [p for p in bert.parameters() if p.requires_grad]
print(f"trainable {sum(p.numel() for p in trainable):,} of {sum(p.numel() for p in bert.parameters()):,}")

def bert_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ids, mask, y in loader:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            correct += (logits.argmax(-1).cpu() == y).sum().item()
            total += len(y)
    return correct / total

optimizer = torch.optim.AdamW(trainable, lr=2e-5, weight_decay=0.01)
for epoch in range(3):
    bert.train()
    for ids, mask, y in bert_train:
        out = bert(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
        optimizer.zero_grad()
        out.loss.backward()            # the model computes cross-entropy when labels are passed
        optimizer.step()
    print(f"epoch {epoch + 1}: test accuracy {bert_accuracy(bert, bert_test):.3f}")
```

About 28.9M of BERT-base's 109.5M parameters are trained here (the top four layers, the pooler and the new 4-way head). Judge the result against two reference points measured on the same 1,000 test rows: the majority class is 27.4%, and a TF-IDF + logistic-regression bag-of-words model scores **85.5%**. Topic classification of news is a task where word choice alone carries most of the signal, so BERT has to beat a strong baseline, not a weak one. A learning rate around $2 \times 10^{-5}$ is standard when pre-trained layers are being updated; much larger rates destroy the pre-trained features in the first few hundred steps.

## Try It Yourself

The drills reuse `MultiHeadAttention`, `bert`, `bert_accuracy`, `encode`, `train_df`, `test_df`, `bert_test` and `device` from above.

**Drill 1.** Implement scaled dot-product attention from scratch (no PyTorch modules, just matrix operations). Verify the output shape is correct for batch size 4, sequence length 20, and dimension 128.

**Solution:**

```python
B, L, D = 4, 20, 128
x = torch.randn(B, L, D)
Q = x @ torch.randn(D, D)
K = x @ torch.randn(D, D)
V = x @ torch.randn(D, D)
scores = Q @ K.transpose(-2, -1) / (D ** 0.5)
attn = torch.softmax(scores, dim=-1)
out = attn @ V
assert out.shape == (B, L, D)
assert torch.allclose(attn.sum(-1), torch.ones(B, L))
```

**Drill 2.** Demonstrate the $\sqrt{d_k}$ effect empirically. Compute attention weights with and without scaling for $d_k = 512$. Show that without scaling the attention distribution is peakier (higher maximum, lower entropy).

**Solution:**

```python
torch.manual_seed(0)
Q, K = torch.randn(1, 10, 512), torch.randn(1, 10, 512)
scores_unscaled = Q @ K.transpose(-2, -1)
scores_scaled = scores_unscaled / (512 ** 0.5)

def entropy(p):
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean()

for name, s in [("unscaled", scores_unscaled), ("scaled", scores_scaled)]:
    a = torch.softmax(s, dim=-1)
    print(f"{name:>8}: score std {s.std():6.2f}, mean max weight {a.max(-1).values.mean():.3f}, "
          f"entropy {entropy(a):.3f} (uniform over 10 = {torch.log(torch.tensor(10.0)):.3f})")
```

The unscaled scores have a standard deviation near $\sqrt{512} \approx 22.6$ (25.0 in our run), so each row's softmax puts almost all its weight on one key — mean maximum weight 0.97, entropy 0.06. After scaling the standard deviation is about 1 and the weights spread over several keys — mean maximum 0.41, entropy 1.77, against 2.30 for uniform attention over 10 keys. A near one-hot softmax has near-zero gradients for every other key, which is why unscaled attention trains badly.

**Drill 3.** Fine-tune BERT with (a) only the classifier head trained, (b) the last 2 transformer layers also unfrozen. Compare accuracy.

**Solution:**

```python
def finetune(n_top_layers, lr, epochs=2, n_train=2000):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4).to(device)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("classifier.") or name.startswith("bert.pooler.") or any(
            name.startswith(f"bert.encoder.layer.{11 - k}.") for k in range(n_top_layers))
    loader = DataLoader(encode(train_df.head(n_train)), batch_size=32, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        model.train()
        for ids, mask, y in loader:
            loss_value = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                               labels=y.to(device)).loss
            opt.zero_grad()
            loss_value.backward()
            opt.step()
    return bert_accuracy(model, bert_test)

print(f"(a) head only:        {finetune(0, lr=1e-3):.3f}")
print(f"(b) + top 2 layers:   {finetune(2, lr=2e-5):.3f}")
```

Note the different learning rates: with the encoder frozen, the head is a small linear classifier on fixed features and needs a "normal" rate such as $10^{-3}$; once pre-trained layers are trainable, the rate drops to about $2 \times 10^{-5}$. Expect (b) to beat (a): BERT's frozen `[CLS]` features were trained for next-sentence prediction, not topic, and adapting even the top two layers re-shapes them for the task. Both should land far above the 27% majority rate; whether they clear the 85.5% bag-of-words baseline with only 2,000 training rows is exactly what this drill measures.

**Drill 4.** Compare the fine-tuned BERT with an LSTM text classifier trained from scratch on the same data (the model of Exercise 4's LSTM baseline). Report accuracy and training time.

**Solution:**

```python
import time
from collections import Counter

counts = Counter(w for t in train_df["text"].to_list() for w in t.lower().split())
vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(20000))}   # 0 = pad, 1 = unknown

def to_ids(texts, max_len=40):
    rows = [[vocab.get(w, 1) for w in t.lower().split()][:max_len] for t in texts]
    return torch.tensor([r + [0] * (max_len - len(r)) for r in rows])

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=128, n_classes=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, ids):
        h, _ = self.lstm(self.emb(ids))
        mask = (ids != 0).unsqueeze(-1).float()
        return self.out((h * mask).sum(1) / mask.sum(1).clamp(min=1))   # mean over real tokens

X_tr, y_tr = to_ids(train_df["text"].to_list()), torch.tensor(train_df["label"].to_list())
X_te, y_te = to_ids(test_df["text"].to_list()), torch.tensor(test_df["label"].to_list())
lstm_clf = LSTMClassifier(len(vocab) + 2).to(device)
opt = torch.optim.Adam(lstm_clf.parameters(), lr=1e-3)
start = time.perf_counter()
for _ in range(8):
    lstm_clf.train()
    for xb, yb in DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True):
        loss_value = nn.functional.cross_entropy(lstm_clf(xb.to(device)), yb.to(device))
        opt.zero_grad()
        loss_value.backward()
        opt.step()
lstm_clf.eval()
with torch.no_grad():
    lstm_acc = (lstm_clf(X_te.to(device)).argmax(-1).cpu() == y_te).float().mean().item()
print(f"LSTM from scratch: accuracy {lstm_acc:.3f}, {time.perf_counter() - start:.0f}s to train")
print(f"BERT (fine-tuned above): accuracy {bert_accuracy(bert, bert_test):.3f}")
```

Time the BERT epochs in the worked example the same way. The usual picture: the from-scratch LSTM trains in a fraction of BERT's time but, with only 5,000 labelled rows, learns weaker word representations than BERT brings from pre-training; BERT costs far more compute per example. Whether the accuracy gap justifies the compute is the real decision — and with a bag-of-words baseline at 85.5%, report all three numbers side by side.

**Drill 5.** Implement sinusoidal positional encoding from scratch. Visualise the encoding matrix as a heatmap. Verify that the dot product between positions $p$ and $p+k$ depends only on $k$, and confirm which dimensions oscillate fastest.

**Solution:**

```python
def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

pe = sinusoidal_pe(100, 64)
for k in [1, 5, 20]:
    dots = [float(pe[p] @ pe[p + k]) for p in [0, 10, 40, 70]]
    print(f"k={k:>2}: dot products at p=0,10,40,70 -> {[round(d, 4) for d in dots]}")

crossings = ((pe[1:] * pe[:-1]) < 0).sum(0)        # strict sign changes per dimension
print("zero crossings over 100 positions, dims 0, 1, 20, 62:", crossings[[0, 1, 20, 62]].tolist())

plt.imshow(pe.T, aspect="auto", cmap="RdBu")
plt.xlabel("position")
plt.ylabel("dimension")
plt.colorbar()
plt.show()
```

For each $k$ the four dot products are identical (up to floating-point rounding): the similarity between two positions depends only on how far apart they are. The zero-crossing counts confirm the frequencies: dimensions 0 and 1 change sign about 30 times in 100 positions, dimension 20 once, and dimension 62 not at all — low dimensions are the fast "second hand", high dimensions the slow "hour hand". In the heatmap this is the band of rapid stripes at the bottom (low dimensions) fading to smooth colour at the top.

## Cross-References

- **Lesson 5.3** introduced attention as a mechanism on top of RNNs. Self-attention removes the RNN entirely.
- **Module 4, Lesson 4.6** used word embeddings as features. Transformers compute contextualised embeddings — the same word gets different representations depending on context.
- **Lesson 5.2** introduced CNNs; ViT is the attention-based alternative for images.
- **Lesson 5.7** applies transfer learning with pre-trained models, including BERT with adapters and the HuggingFace pipeline API.
- **Module 6** builds extensively on transformers: LLM fundamentals (6.1), fine-tuning (6.2), and RAG (6.4).

## Reflection

You should now be able to:

- Derive scaled dot-product attention from first principles.
- Explain why dividing by $\sqrt{d_k}$ is necessary (prevent softmax saturation).
- Implement multi-head attention in PyTorch, including causal masking and cross-attention.
- Explain why self-attention is permutation-equivariant and how sinusoidal encodings (fast low dimensions, slow high dimensions) restore order.
- Explain how a ViT turns an image into tokens and why it needs more data than a CNN.
- Fine-tune BERT for a downstream classification task and judge it against a bag-of-words baseline.
- Compare BERT, GPT, T5 and the long-context variants and know when each is appropriate.

---

# Lesson 5.5: Generative Models — GANs and Diffusion

## Why This Matters

Autoencoders (Lesson 5.1) generate data by sampling from a latent space, but the samples are often blurry. GANs produce sharper, more realistic outputs through adversarial training — a generator tries to fool a discriminator that tries to distinguish real from fake. The tension between these two networks drives the generator to produce increasingly realistic data.

## Core Concepts

### THEORY: GAN minimax objective

The GAN training objective is a minimax game:

$$\min_G \max_D \left[ \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))] \right]$$

The discriminator $D$ maximises the objective by correctly classifying real data as real ($D(\mathbf{x}) \to 1$) and generated data as fake ($D(G(\mathbf{z})) \to 0$). The generator $G$ minimises the objective by producing data that the discriminator classifies as real ($D(G(\mathbf{z})) \to 1$). Both terms are binary cross-entropy, which is how the losses are implemented.

For a fixed $G$ the best discriminator is $D^*(\mathbf{x}) = p_{\text{data}}(\mathbf{x}) / (p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x}))$, and substituting it back gives $2\,\text{JS}(p_{\text{data}} \| p_g) - \log 4$: training $G$ against an optimal $D$ minimises the Jensen–Shannon divergence. At the equilibrium $p_g = p_{\text{data}}$ and $D$ outputs 0.5 everywhere. In practice, training oscillates and rarely reaches it.

**The non-saturating generator loss.** Early in training $D$ rejects fakes easily, $D(G(\mathbf{z})) \approx 0$, and the minimax term $\log(1 - D(G(\mathbf{z})))$ is flat there — its gradient with respect to the generator vanishes. So in practice $G$ minimises $-\log D(G(\mathbf{z}))$ instead (implemented as BCE against the label "real"). It has the same fixed point but a strong gradient exactly when $G$ is losing. All the code in this lesson and in Exercise 5 uses this non-saturating loss. It fixes the *saturation* problem; it does not fix the deeper problem below.

### FOUNDATIONS: Mode collapse

Mode collapse occurs when the generator learns to produce only a few types of outputs that fool the discriminator, ignoring the full diversity of the training data. For instance, a GAN trained on MNIST might generate only the digit 1 — the discriminator cannot tell these apart from real 1s, but the generator has stopped producing any other digit. Exercise 5 measures it directly: a classifier labels the generated digits, and the spread of predicted classes shows how many of the ten modes the generator covers.

### THEORY: WGAN and gradient penalty

Real images occupy a thin, low-dimensional set inside pixel space, and early in training the generator's samples occupy a different thin set. When the two supports do not overlap, the JS divergence is stuck at its maximum, the constant $\log 2$, whatever the distance between them — so it gives the generator no signal about which direction to move. That, not saturation, is why vanilla GAN training is unstable. The Wasserstein GAN (WGAN) replaces JS with the Wasserstein-1 (Earth Mover's) distance, which keeps growing smoothly with how far apart the distributions are:

$$\min_G \max_{D \in \text{1-Lip}} \left[ \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[D(\mathbf{x})] - \mathbb{E}_{\mathbf{z} \sim p_z}[D(G(\mathbf{z}))] \right]$$

The discriminator, now called a **critic**, outputs an unbounded score (no sigmoid) and must be 1-Lipschitz. **Gradient penalty** (WGAN-GP) enforces that softly by penalising the critic's input-gradient norm at random interpolates $\hat{\mathbf{x}}$ between real and generated samples:

$$\mathcal{L}_{\text{critic}} = \mathbb{E}[D(G(\mathbf{z}))] - \mathbb{E}[D(\mathbf{x})] + \lambda \, \mathbb{E}_{\hat{\mathbf{x}}}\left[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2\right], \qquad \mathcal{L}_G = -\mathbb{E}[D(G(\mathbf{z}))]$$

with $\lambda = 10$ as standard. The norm is taken over the *whole* input (flatten each sample first) — a per-pixel or per-channel norm is a different, wrong constraint. Batch normalisation should not be used in the critic, because the penalty is defined per sample.

**Reading the critic loss.** Without the penalty term, $-\mathcal{L}_{\text{critic}}$ is the critic's estimate of the Wasserstein distance. As the generator improves the distance shrinks, so the critic loss is negative and **rises towards 0** as quality improves. A critic loss that becomes more negative means the distributions are moving apart.

```python
import torch

def gradient_penalty(critic, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interp)
    gradients = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,                       # the penalty itself is trained through
    )[0]
    grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)   # one norm per sample
    return ((grad_norm - 1) ** 2).mean()
```

### FOUNDATIONS: GAN variants

- **DCGAN** (Radford et al., 2016): the recipe that made convolutional GANs train reliably — strided convolutions instead of pooling, transposed convolutions to upsample in the generator, batch norm, no fully connected hidden layers, ReLU in $G$ and LeakyReLU in $D$, Tanh output. The worked example builds one.
- **Conditional GAN (cGAN):** feed a class label to both $G$ and $D$, so you can ask for "a 7". (Drill 4.)
- **CycleGAN** (Zhu et al., 2017): *unpaired* image-to-image translation (photos ↔ paintings, summer ↔ winter) with no matched pairs. Two generators $G: X \to Y$ and $F: Y \to X$ each have an adversarial loss, plus a **cycle-consistency loss** $\|F(G(x)) - x\|_1 + \|G(F(y)) - y\|_1$: translating there and back must return the original, which stops $G$ from mapping every input to one convincing output.
- **StyleGAN** (Karras et al., 2019): a mapping network turns $\mathbf{z}$ into an intermediate latent $\mathbf{w}$ that controls each resolution of the generator through adaptive instance normalisation ("styles"), with per-layer noise for fine detail. Coarse layers set pose and shape, fine layers set texture and colour — and mixing styles from two latents mixes those attributes. It inherited progressive growing (train at low resolution, then add higher-resolution layers) from ProGAN; StyleGAN2 later replaced that with skip and residual connections. It produces high-resolution, photo-realistic faces.

### FOUNDATIONS: Evaluating generators — FID and Inception Score

Generated images have no ground-truth labels, so evaluation compares *distributions* of features from a pre-trained network.

**FID (Fréchet Inception Distance).** Fit a Gaussian to the features of real images ($\boldsymbol{\mu}_r, \boldsymbol{\Sigma}_r$) and of generated images ($\boldsymbol{\mu}_g, \boldsymbol{\Sigma}_g$) and compute the Fréchet distance between them:

$$\text{FID} = \|\boldsymbol{\mu}_r - \boldsymbol{\mu}_g\|^2 + \text{Tr}(\boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g - 2(\boldsymbol{\Sigma}_r \boldsymbol{\Sigma}_g)^{1/2})$$

Lower is better. FID is a **distribution-level** score: the mean term catches samples that look wrong on average (fidelity) and the covariance term catches missing variety (diversity, e.g. mode collapse). It says nothing about any single image. The standard version uses 2,048-dimensional InceptionV3 features of $299 \times 299$ RGB images, and published thresholds ("FID below 10") refer to that extractor. For $28 \times 28$ digits, Exercise 5 uses a small LeNet classifier trained on MNIST (64-dimensional features); those FIDs are comparable only with other FIDs from the same extractor.

**Inception Score (IS).** Classify each generated image with a pre-trained classifier and compute

$$\text{IS} = \exp\Big(\mathbb{E}_{\mathbf{x} \sim p_g}\, \text{KL}\big(p(y \mid \mathbf{x}) \,\|\, p(y)\big)\Big)$$

High when each image is classified confidently (sharp $p(y \mid \mathbf{x})$) *and* the predicted classes are spread out (broad marginal $p(y)$). Its maximum is the number of classes. IS never looks at real images, so it cannot tell whether the samples resemble the training data, and it rewards one perfect image per class; FID is generally preferred, with IS reported alongside. Drill 3 computes both with the course's LeNet classifier.

### ADVANCED: Diffusion models

Diffusion models (DDPM — Denoising Diffusion Probabilistic Models, Ho et al., 2020) define a **forward process** that adds a little Gaussian noise at each of $T$ steps (typically 1,000) until the data is pure noise. It has a closed form for any step, $\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\epsilon}$, so training is simple: pick a random $t$, noise a real image to $\mathbf{x}_t$, and train a network (usually a U-Net) to predict the noise $\boldsymbol{\epsilon}$ with an MSE loss. The **reverse process** starts from pure noise and removes the predicted noise step by step. The training objective is a simplified form of the ELBO from Lesson 5.1.

Diffusion models train stably (an MSE regression, no adversary) and cover the data distribution well (good diversity, little mode collapse), at the cost of slow sampling — many network evaluations per image, although modern samplers cut this to tens of steps. Stable Diffusion runs the process in the latent space of an autoencoder ("latent diffusion") to make it affordable. DALL-E 2 and DALL-E 3 use diffusion; the original DALL-E (2021) was an autoregressive transformer over discrete image tokens.

**Which generator for which job?**

| Data / need | First choice | Why |
|---|---|---|
| Images, highest quality and diversity | Diffusion | Stable training, excellent coverage; slow sampling is acceptable offline |
| Images, fast sampling or real-time | GAN (StyleGAN-type) | One forward pass per image |
| Unpaired image translation | CycleGAN | Cycle consistency needs no paired data |
| Text | Transformers (autoregressive) | Discrete tokens; Lesson 5.4 |
| Time series, smooth latent space, anomaly scores | VAE / LSTM | Explicit likelihood and a smooth latent space |

### FOUNDATIONS: Synthetic data — augmentation, simulation and privacy

Generative models produce **augmentation** data for rare classes (extra examples of an uncommon defect), **simulation** data for testing systems before real data exists, and candidate **privacy-preserving** releases. The last needs care. Synthetic data is **not private by default**: GANs and diffusion models can memorise training examples and reproduce them almost exactly — Carlini et al. (2023) extracted recognisable training images from deployed diffusion models. And FID cannot detect this: a model that copies its training set gets an *excellent* FID, because FID measures fidelity and diversity, not privacy. Sharing a generator or its samples in place of sensitive records (medical scans, customer data) requires formal guarantees such as differentially private training (DP-SGD) plus memorisation and membership-inference testing — and a legal review under the applicable data-protection law.

## Worked Example: DCGAN and WGAN-GP on MNIST (the Exercise 5 data)

Exercise 5 trains GANs on MNIST in `data/mlfp05/mnist`, scaled to $[-1, 1]$ to match the generator's Tanh output, with a 64-dimensional latent space. Its generators are fully connected; this example builds the convolutional DCGAN the spec calls for, on the same data. Scaling matters: if real images were in $[0, 1]$ while fakes were in $[-1, 1]$, the discriminator could separate them by value range alone.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from shared.kailash_helpers import get_device

device = get_device()
LATENT_DIM = 64
to_pm1 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # [-1, 1]
mnist = datasets.MNIST("data/mlfp05/mnist", train=True, download=True, transform=to_pm1)
loader = DataLoader(mnist, batch_size=128, shuffle=True, drop_last=True)

class Generator(nn.Module):
    """z (64) -> project to 128x7x7 -> upsample 14x14 -> 28x28, Tanh output in [-1, 1]."""
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(latent_dim, 128 * 7 * 7), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),  # 14x14
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1), nn.Tanh(),                        # 28x28
        )

    def forward(self, z):
        return self.net(self.project(z).view(-1, 128, 7, 7))

class Discriminator(nn.Module):
    """Strided convolutions, no pooling; outputs a LOGIT (no sigmoid)."""
    def __init__(self, batch_norm=True):
        super().__init__()
        norm = nn.BatchNorm2d(128) if batch_norm else nn.Identity()   # critics must not use BN
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),             # 14x14
            nn.Conv2d(64, 128, 4, stride=2, padding=1), norm, nn.LeakyReLU(0.2),     # 7x7
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.net(x)

G, D = Generator().to(device), Discriminator().to(device)
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()             # sigmoid + BCE in one numerically stable step
fixed_z = torch.randn(64, LATENT_DIM, device=device)
losses = {"D": [], "G": []}
snapshots = []                           # samples from fixed_z after each epoch

for epoch in range(5):
    for real, _ in loader:
        real = real.to(device)
        ones = torch.ones(real.size(0), 1, device=device)
        zeros = torch.zeros(real.size(0), 1, device=device)
        # Discriminator: real -> 1, fake -> 0
        fake = G(torch.randn(real.size(0), LATENT_DIM, device=device))
        loss_D = bce(D(real), ones) + bce(D(fake.detach()), zeros)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()
        # Generator, non-saturating: minimise -log D(G(z)), i.e. BCE against "real"
        loss_G = bce(D(fake), ones)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
        losses["D"].append(loss_D.item())
        losses["G"].append(loss_G.item())
    with torch.no_grad():
        G.eval()
        snapshots.append(G(fixed_z).cpu())   # the same 64 latent points every epoch
        G.train()
    print(f"epoch {epoch + 1}: D loss {sum(losses['D'][-100:]) / 100:.3f}, "
          f"G loss {sum(losses['G'][-100:]) / 100:.3f}")
```

Unlike a supervised loss, neither GAN loss should go to zero: a D loss near 0 means the discriminator has won and $G$ gets little useful signal; a healthy run keeps both losses moving within a band. Judge progress by the samples (`snapshots`, from fixed latent points, so you can watch the same "digit" sharpen across epochs) and by FID, not by the loss curves. After a few full epochs (468 batches each) the samples should be recognisable digits; if they are still grey blobs, check the $[-1, 1]$ scaling first. In a short check (300 batches) the discriminator briefly won outright (D loss 0.03 in the second epoch, G loss 4.5) and then the two losses settled into a band — normal GAN behaviour, not a bug.

## Try It Yourself

The drills reuse `device`, `loader`, `mnist`, `LATENT_DIM`, `Generator`, `Discriminator`, `gradient_penalty`, `G`, `losses` and `snapshots` from above.

**Drill 1.** Plot the generated images from the fixed latent points after epochs 1, 3 and 5 of the DCGAN loop, side by side with real digits. What changes between epochs?

**Solution:**

```python
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_grid(images, title):
    grid = make_grid(images[:64], nrow=8, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(title)
    plt.show()

real_batch, _ = next(iter(loader))
show_grid(real_batch, "real MNIST")
for epoch in [1, 3, 5]:
    show_grid(snapshots[epoch - 1], f"DCGAN samples after epoch {epoch}")
```

Early snapshots are blurry blobs with the right overall brightness; by the middle epochs strokes appear, and by the last epoch most samples are recognisable digits with occasional broken or merged strokes. Because the latent points are fixed, you can see each sample refine rather than jump between unrelated images.

**Drill 2.** Implement WGAN-GP with the same generator. Compare training stability with the DCGAN (plot the critic and generator losses).

**Solution:**

```python
G_w = Generator().to(device)
critic = Discriminator(batch_norm=False).to(device)        # unbounded score, no BN
opt_Gw = torch.optim.Adam(G_w.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_C = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
N_CRITIC, LAMBDA = 5, 10.0
w_losses = {"critic": [], "W estimate": []}

for epoch in range(5):
    for i, (real, _) in enumerate(loader):
        real = real.to(device)
        fake = G_w(torch.randn(real.size(0), LATENT_DIM, device=device)).detach()
        w_est = critic(real).mean() - critic(fake).mean()                # Wasserstein estimate
        loss_C = -w_est + LAMBDA * gradient_penalty(critic, real, fake)
        opt_C.zero_grad()
        loss_C.backward()
        opt_C.step()
        w_losses["critic"].append(loss_C.item())
        w_losses["W estimate"].append(w_est.item())
        if i % N_CRITIC == 0:                                            # G steps less often
            loss_Gw = -critic(G_w(torch.randn(real.size(0), LATENT_DIM, device=device))).mean()
            opt_Gw.zero_grad()
            loss_Gw.backward()
            opt_Gw.step()

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(losses["D"], label="D (BCE)")
ax[0].plot(losses["G"], label="G (non-saturating)")
ax[0].set_title("DCGAN")
ax[0].legend()
ax[1].plot(w_losses["W estimate"])
ax[1].set_title("WGAN-GP: critic's Wasserstein estimate")
plt.show()
```

The DCGAN losses oscillate and their level says little about image quality. The WGAN-GP critic's Wasserstein estimate is a usable progress signal. Early on it climbs while the critic learns to separate real from fake — in our short check (300 critic steps, 60 generator steps) it rose from about 3 to 19 and was still rising. Over a full run it should level off and then trend down as the generator closes the gap — equivalently, the critic loss rises towards 0. An estimate that keeps climbing for many epochs means the generator is falling behind the critic. The critic is updated five times per generator step, with Adam's $\beta_1 = 0$, the settings from the WGAN-GP paper.

**Drill 3.** Compute FID and Inception Score for the DCGAN with the course's feature extractor. Sanity-check FID by comparing two halves of the real data.

**Solution:**

```python
from shared.mlfp05.ex_5 import train_feature_extractor, compute_fid

X_real = torch.stack([mnist[i][0] for i in range(10000)]).to(device)        # in [-1, 1]
y_real = torch.tensor([mnist[i][1] for i in range(10000)], device=device)
extractor = train_feature_extractor(X_real, y_real, device, epochs=3)      # LeNet, 64-d features

with torch.no_grad():
    G.eval()
    fake = torch.cat([G(torch.randn(500, LATENT_DIM, device=device)) for _ in range(10)])
real01, fake01 = (X_real + 1) / 2, (fake + 1) / 2     # the extractor was trained on [0, 1] pixels

print(f"FID real-vs-real (two halves): {compute_fid(extractor, real01[:5000], real01[5000:]):.2f}")
print(f"FID real-vs-DCGAN:             {compute_fid(extractor, real01, fake01):.2f}")

def inception_score(classifier, images01):
    with torch.no_grad():
        p_yx = torch.softmax(classifier(images01), dim=1)              # p(y | x) per image
    p_y = p_yx.mean(dim=0, keepdim=True)                               # marginal p(y)
    kl = (p_yx * (torch.log(p_yx + 1e-12) - torch.log(p_y + 1e-12))).sum(dim=1)
    return torch.exp(kl.mean()).item()

print(f"IS real: {inception_score(extractor, real01):.2f} | "
      f"IS DCGAN: {inception_score(extractor, fake01):.2f} (maximum 10)")
```

The real-vs-real FID is the floor for this extractor and sample size — small but not zero, because two finite samples never have identical statistics. In a short check it was 1.06, against 40.2 for a DCGAN trained for only 300 batches. Read a generator's FID against that floor and against other generators scored with the same extractor; Inception-scale thresholds do not apply to 64-dimensional LeNet features. IS behaves as the formula predicts: real digits scored 7.2 (this quickly trained extractor is only 91% accurate, so even real digits are not classified with full confidence; a better classifier pushes the score towards 10), the undertrained DCGAN 4.0. A mode-collapsed generator would score low even if each image were sharp, because its predicted classes would not be spread out.

**Drill 4.** Implement conditional generation: given a class label, generate an image of that class.

**Solution:**

```python
class ConditionalGenerator(Generator):
    def __init__(self, n_classes=10):
        super().__init__(latent_dim=LATENT_DIM + n_classes)
        self.n_classes = n_classes

    def forward(self, z, labels):
        onehot = nn.functional.one_hot(labels, self.n_classes).float()
        return super().forward(torch.cat([z, onehot], dim=1))

class ConditionalDiscriminator(Discriminator):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net[0] = nn.Conv2d(1 + n_classes, 64, 4, stride=2, padding=1)   # label as extra channels
        self.n_classes = n_classes

    def forward(self, x, labels):
        maps = nn.functional.one_hot(labels, self.n_classes).float()[:, :, None, None]
        return super().forward(torch.cat([x, maps.expand(-1, -1, 28, 28)], dim=1))

cG, cD = ConditionalGenerator().to(device), ConditionalDiscriminator().to(device)
opt_cG = torch.optim.Adam(cG.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_cD = torch.optim.Adam(cD.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
for epoch in range(5):
    for real, labels in loader:
        real, labels = real.to(device), labels.to(device)
        ones = torch.ones(len(real), 1, device=device)
        zeros = torch.zeros(len(real), 1, device=device)
        fake = cG(torch.randn(len(real), LATENT_DIM, device=device), labels)
        loss_D = bce(cD(real, labels), ones) + bce(cD(fake.detach(), labels), zeros)
        opt_cD.zero_grad()
        loss_D.backward()
        opt_cD.step()
        loss_G = bce(cD(fake, labels), ones)
        opt_cG.zero_grad()
        loss_G.backward()
        opt_cG.step()

with torch.no_grad():
    cG.eval()
    sevens = cG(torch.randn(16, LATENT_DIM, device=device),
                torch.full((16,), 7, device=device, dtype=torch.long))
print(sevens.shape)   # torch.Size([16, 1, 28, 28]): sixteen different 7s
```

Both networks see the label: the generator so it can produce the requested class, the discriminator so it can reject "a good-looking 3 labelled 7". Varying $\mathbf{z}$ with a fixed label changes the style (slant, thickness) while the class stays fixed.

**Drill 5.** Create a comparison table: VAE vs DCGAN vs WGAN-GP vs diffusion. For each, report training stability, sample quality (FID), sample diversity, and training time. Which would you use to generate extra training images for a rare class — and what would you check before sharing a generator trained on sensitive data?

**Solution:** Fill the table from your own runs (Lesson 5.1's VAE, this lesson's DCGAN and WGAN-GP, all scored with the same LeNet-feature FID). Typical pattern: the VAE is the most stable and fastest but blurriest (worse FID); the DCGAN gives sharp samples quickly but its losses oscillate and it can drop modes; WGAN-GP trains more steadily with a meaningful loss curve at a higher cost per step (five critic updates per generator update); diffusion gives the best quality and diversity but samples slowly. For augmenting a rare class, a conditional GAN or a diffusion model conditioned on the class is the natural choice — and the augmented model must be validated on *real* held-out examples, never on synthetic ones. Before sharing a generator trained on sensitive records, test for memorisation (nearest-neighbour distance from each generated sample to the training set, membership-inference attacks) and use differentially private training if the data is personal; a good FID is not evidence of privacy.

## Cross-References

- **Lesson 5.1** introduced VAEs for generation. GANs produce sharper samples; VAEs train more stably and cover the data more evenly. The diffusion training objective is a simplified ELBO.
- **Lesson 5.2** provided the strided and transposed convolutions the DCGAN is built from.
- **Module 6, Lesson 6.3** uses preference alignment (DPO) — a different approach to steering generative models.

## Reflection

You should now be able to:

- Write the GAN minimax objective, explain why the non-saturating generator loss is used, and why JS divergence gives no signal when supports do not overlap.
- Implement a convolutional DCGAN and a WGAN-GP critic, and read the critic loss in the right direction.
- Explain mode collapse and how Wasserstein distance addresses it; describe cGAN, CycleGAN and StyleGAN in one sentence each.
- Evaluate generative quality with FID and Inception Score, and say what each cannot measure.
- Compare VAE, GAN, and diffusion models for different generation tasks, and explain why synthetic data is not private by default.

---

# Lesson 5.6: Graph Neural Networks

## Why This Matters

Social networks, molecular structures, supply chains, and knowledge graphs are naturally represented as graphs — nodes connected by edges. Standard neural networks cannot process graph-structured data directly. GNNs operate by message passing: each node aggregates information from its neighbours, then updates its representation. After several rounds of message passing, each node's representation captures information from its local neighbourhood.

## Core Concepts

### FOUNDATIONS: Graph data and the three graph tasks

A graph has $N$ nodes with feature vectors (stacked as $\mathbf{X} \in \mathbb{R}^{N \times F}$) and edges, stored as an adjacency matrix $\mathbf{A}$ ($A_{ij} = 1$ if $i$ and $j$ are connected) or, in torch_geometric, as an `edge_index` tensor of shape $(2, E)$ listing source and target nodes. Three tasks recur:

- **Node classification** — predict a label per node (the topic of a paper in a citation network, whether an account is fraudulent). Exercise 6 does this on Cora: 2,708 papers, 1,433 bag-of-words features, 7 topics, 5,278 undirected citation links.
- **Graph classification** — predict one label per graph (is this molecule mutagenic?). Needs a *readout* that pools node vectors into one graph vector.
- **Link prediction** — predict whether an edge exists (which papers should cite each other, which products are bought together).

**Transductive vs inductive.** In Cora's standard setting every node, including the test nodes, is in the graph during training; only their *labels* are hidden. That is transductive learning. Inductive learning must handle nodes or graphs never seen in training (new users, new molecules) — GraphSAGE, GAT and GIN can all do that, because they learn functions of a neighbourhood rather than an embedding per node.

### THEORY: GCN propagation rule

The Graph Convolutional Network (GCN) layer updates node features using the normalised adjacency matrix:

$$\mathbf{H}^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$

where $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ (adjacency matrix with self-loops), $\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{\mathbf{A}}_{ij}$ (degree matrix), $\mathbf{H}^{(l)}$ is the feature matrix at layer $l$, and $\mathbf{W}^{(l)}$ is the learnable weight matrix.

$\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2}$ is the **symmetric-normalised adjacency**: the message from $j$ to $i$ is weighted by $1/\sqrt{d_i d_j}$, so aggregated features stay on a comparable scale whatever a node's degree and high-degree nodes do not dominate. (The normalised graph *Laplacian* used in spectral clustering is $\mathbf{I}$ minus this matrix; GCN is a first-order approximation of a spectral filter on that Laplacian, which is where the name "graph convolution" comes from.)

### FOUNDATIONS: Message passing

The GCN propagation can be viewed as message passing:

1. **Message:** each node sends its current representation to all neighbours.
2. **Aggregate:** each node combines the messages from its neighbours (and itself, via the self-loop) — for GCN, a degree-normalised sum.
3. **Update:** the aggregated message is transformed by a linear layer and activation function.

After $L$ layers of message passing, each node's representation captures information from nodes up to $L$ hops away. Every GNN in this lesson is this template with a different aggregate and update.

**Over-smoothing.** Each layer mixes a node with its neighbours, so many layers make all node vectors converge towards the same thing and the classes blur together. Most GNNs therefore use 2–3 layers (Drill 3 measures this).

### THEORY: GraphSAGE — sample and aggregate

GraphSAGE (Hamilton et al., 2017) separates the node from its neighbourhood and **samples** a fixed number of neighbours per node, so the cost per node is bounded even in graphs with millions of edges:

$$\mathbf{h}_i' = \sigma\left(\mathbf{W} \left[\mathbf{h}_i \,\|\, \text{AGG}\left(\{\mathbf{h}_j : j \in \mathcal{S}(i)\}\right)\right]\right)$$

where $\mathcal{S}(i)$ is a sampled subset of $i$'s neighbours and AGG is a mean, max-pool or LSTM aggregator. Because it learns *how to aggregate* rather than a vector per node, a trained GraphSAGE model can embed nodes that were not in the training graph — it is inductive. In torch_geometric the layer is `SAGEConv` and neighbour sampling is done by `NeighborLoader`.

### THEORY: GAT attention weights

Graph Attention Networks (GAT) compute attention weights between neighbours:

$$e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j])$$
$$\alpha_{ij} = \text{softmax}_j(e_{ij})$$
$$\mathbf{h}_i' = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j\right)$$

This allows the model to learn which neighbours are more important for each node, rather than weighting them by degree as GCN does. Like Lesson 5.4, GAT usually runs several heads and concatenates them. GAT is inductive: the original paper evaluates it on unseen protein-interaction graphs.

### THEORY: GIN — the most expressive message-passing GNN

How well can a GNN tell two different graphs apart? Xu et al. (2019) showed that a message-passing GNN is at most as powerful as the Weisfeiler–Lehman (WL) graph-isomorphism test, and reaches that bound only if its aggregation is **injective** on the multiset of neighbour features. Mean and max are not injective: a node with neighbours $\{a, b\}$ and one with $\{a, a, b, b\}$ get the same mean, and $\{a, b\}$ and $\{a, b, b\}$ get the same max. Sum is injective (for suitable features). The Graph Isomorphism Network uses sum plus an MLP:

$$\mathbf{h}_i' = \text{MLP}\left((1 + \epsilon)\, \mathbf{h}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j\right)$$

For graph classification, GIN reads out each graph by **summing** its node vectors (again to keep counts), often from every layer. It is the standard strong baseline for molecule and graph classification; in torch_geometric it is `GINConv(mlp)` with `global_add_pool`.

| Architecture | Aggregation | Strength | Typical use |
|---|---|---|---|
| GCN | Degree-normalised sum | Simple, fast, strong on citation-style graphs | Node classification (transductive) |
| GraphSAGE | Sampled mean / max / LSTM | Scales to huge graphs; inductive | New nodes arriving (users, products) |
| GAT | Learned attention over neighbours | Neighbour importance varies | Heterogeneous, noisy neighbourhoods |
| GIN | Sum + MLP | Most expressive (WL-level) | Graph classification (molecules) |

## The Kailash Engine: ModelVisualizer (embeddings)

`ModelVisualizer().scatter(df, x, y, color=...)` shows learned node embeddings: project the hidden layer to 2-D (PCA or t-SNE), put the coordinates and the true labels in a polars DataFrame, and colour by label. If the GNN has learned the task, nodes of one class cluster together. The first worked example ends with this plot.

## Worked Example 1: Node classification on Cora (the Exercise 6 data)

Exercise 6 classifies Cora papers by topic and stores the dataset in `data/mlfp05/cora`. Cora ships with a standard split: 140 labelled training nodes (20 per class), 500 validation nodes and 1,000 test nodes. **Choose the epoch by validation accuracy and report test accuracy at that epoch** — picking the best test accuracy over epochs is selecting on the test set, and it inflates the result.

```python
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from kailash_ml import ModelVisualizer
from shared.kailash_helpers import get_device

device = get_device()
torch.manual_seed(0)
cora = Planetoid("data/mlfp05/cora", name="Cora")[0].to(device)
print(f"{cora.num_nodes} nodes, {cora.num_edges // 2} undirected edges, {cora.num_features} features; "
      f"train/val/test = {int(cora.train_mask.sum())}/{int(cora.val_mask.sum())}/{int(cora.test_mask.sum())}")

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, dropout=0.5):
        super().__init__()
        self.conv1, self.conv2 = GCNConv(in_dim, hidden), GCNConv(hidden, n_classes)
        self.dropout = dropout

    def embed(self, x, edge_index):
        x = nn.functional.dropout(x, self.dropout, self.training)
        return torch.relu(self.conv1(x, edge_index))

    def forward(self, x, edge_index):
        h = nn.functional.dropout(self.embed(x, edge_index), self.dropout, self.training)
        return self.conv2(h, edge_index)

def train_node_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    """Train on train_mask; return test accuracy at the epoch with the best VALIDATION accuracy."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, test_at_best, best_epoch = 0.0, 0.0, 0
    for epoch in range(epochs):
        model.train()
        out = model(data.x, data.edge_index)
        loss = nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(1)
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        if val_acc > best_val:
            best_val, test_at_best, best_epoch = val_acc, test_acc, epoch + 1
    return best_val, test_at_best, best_epoch

gcn = GCN(cora.num_features, 16, 7)
val_acc, test_acc, epoch = train_node_model(gcn, cora)
print(f"GCN: best val acc {val_acc:.3f} at epoch {epoch}; test acc there {test_acc:.3f}")

class NoGraphMLP(nn.Module):
    """Same features, edges ignored: the baseline that shows what the graph adds."""
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(hidden, n_classes))

    def forward(self, x, edge_index):
        return self.net(x)

_, mlp_test, _ = train_node_model(NoGraphMLP(cora.num_features, 16, 7), cora)
print(f"MLP (no edges): test acc {mlp_test:.3f}")

gcn.eval()
with torch.no_grad():
    hidden = gcn.embed(cora.x, cora.edge_index).cpu().numpy()      # (2708, 16)
xy = PCA(n_components=2).fit_transform(hidden)
topics = cora.y.cpu().tolist()
emb_df = pl.DataFrame({"pc1": xy[:, 0], "pc2": xy[:, 1], "topic": [f"topic {t}" for t in topics]})
fig = ModelVisualizer().scatter(emb_df, x="pc1", y="pc2", color="topic",
                                title="GCN hidden layer (PCA), coloured by true topic")
```

Read the two numbers together. In our run the GCN reached 0.804 test accuracy (validation-selected at epoch 24) and the MLP 0.540. The MLP sees exactly the same word features but not the citations; that 26-point gap is what the graph contributes. With only 140 labelled nodes, citations let the label information spread to unlabelled neighbours — papers mostly cite papers on the same topic.

## Worked Example 2: Graph classification on MUTAG with GIN

Graph classification needs one prediction per graph. MUTAG (from the TUDataset collection, downloaded to `data/mlfp05/tudataset`) has 188 small molecules — atoms are nodes with a one-hot atom type, bonds are edges — each labelled mutagenic or not. torch_geometric's `DataLoader` packs many small graphs into one big disconnected graph per batch and keeps a `batch` vector saying which graph each node belongs to; the readout pools by that vector.

```python
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as GraphLoader
from torch_geometric.nn import GINConv, global_add_pool

torch.manual_seed(0)
mutag = TUDataset("data/mlfp05/tudataset", name="MUTAG").shuffle()
print(f"{len(mutag)} graphs, {mutag.num_node_features} node features, "
      f"mean {sum(g.num_nodes for g in mutag) / len(mutag):.1f} nodes per graph, "
      f"{float(sum(g.y.item() for g in mutag)) / len(mutag):.0%} positive")
train_set, val_set, test_set = mutag[:120], mutag[120:150], mutag[150:]
train_graphs = GraphLoader(train_set, batch_size=32, shuffle=True)

class GIN(nn.Module):
    def __init__(self, in_dim, hidden=64, n_layers=3, n_classes=2):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            mlp = nn.Sequential(nn.Linear(in_dim if i == 0 else hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden), nn.ReLU())
            self.convs.append(GINConv(mlp, train_eps=True))      # (1 + eps) * h_i + sum of neighbours
        self.head = nn.Linear(hidden * n_layers, n_classes)

    def forward(self, x, edge_index, batch):
        readouts = []
        for conv in self.convs:
            x = conv(x, edge_index)
            readouts.append(global_add_pool(x, batch))           # sum readout per graph, per layer
        return self.head(torch.cat(readouts, dim=1))

def graph_accuracy(model, graphs):
    model.eval()
    correct = 0
    with torch.no_grad():
        for g in GraphLoader(graphs, batch_size=64):
            g = g.to(device)
            correct += (model(g.x, g.edge_index, g.batch).argmax(1) == g.y).sum().item()
    return correct / len(graphs)

gin = GIN(mutag.num_node_features).to(device)
opt = torch.optim.Adam(gin.parameters(), lr=0.01)
best_val, test_at_best = 0.0, 0.0
for epoch in range(100):
    gin.train()
    for g in train_graphs:
        g = g.to(device)
        loss = nn.functional.cross_entropy(gin(g.x, g.edge_index, g.batch), g.y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    val_acc = graph_accuracy(gin, val_set)
    if val_acc > best_val:
        best_val, test_at_best = val_acc, graph_accuracy(gin, test_set)
majority = max(float(sum(g.y.item() for g in test_set)), len(test_set) - float(sum(g.y.item() for g in test_set)))
print(f"GIN: best val acc {best_val:.3f}, test acc at that epoch {test_at_best:.3f} "
      f"(majority class {majority / len(test_set):.3f}, {len(test_set)} test graphs)")
```

With 38 test graphs every misclassified molecule moves accuracy by 2.6 points, so a single split is noisy; the published protocol for MUTAG is 10-fold cross-validation. Compare against the majority-class rate printed on the last line, not against 50%: in our run GIN scored 0.895 on the 38 test graphs against a majority rate of 0.658.

### FOUNDATIONS: Link prediction done properly

A link predictor embeds nodes with a GNN encoder and scores a candidate edge $(i, j)$, usually by the dot product $\mathbf{z}_i^\top \mathbf{z}_j$ passed through a sigmoid. Training uses known edges as positives and randomly sampled non-edges as negatives. The trap is leakage: if the edges you test on are also in the graph the encoder propagates over, the model has already seen the answer. The correct protocol is to **split the edges** into train/validation/test (for example 85/5/10), build the message-passing graph from the training edges only, sample fresh negatives for each split, and report AUC on the test edges. torch_geometric's `RandomLinkSplit` transform does exactly this (Drill 5).

## Try It Yourself

The drills reuse `device`, `cora`, `GCN`, `NoGraphMLP`, `train_node_model`, `gin`, `mutag`, `GIN` and `graph_accuracy` from the worked examples.

**Drill 1.** Compare GCN with GAT on Cora, using validation-based epoch selection for both. Does attention improve accuracy?

**Solution:**

```python
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_dim, hidden=8, heads=8, n_classes=7):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=0.6)            # 8 heads x 8 = 64
        self.conv2 = GATConv(hidden * heads, n_classes, heads=1, dropout=0.6)

    def forward(self, x, edge_index):
        x = nn.functional.dropout(x, 0.6, self.training)
        x = nn.functional.elu(self.conv1(x, edge_index))
        x = nn.functional.dropout(x, 0.6, self.training)
        return self.conv2(x, edge_index)

for name, make in [("GCN", lambda: GCN(cora.num_features, 16, 7)), ("GAT", lambda: GAT(cora.num_features))]:
    tests = []
    for seed in range(3):
        torch.manual_seed(seed)
        tests.append(train_node_model(make(), cora, lr=0.005 if name == "GAT" else 0.01)[1])
    print(f"{name}: test acc over 3 seeds {np.mean(tests):.3f} +/- {np.std(tests):.3f}")
```

On Cora the two land within about a point of each other — 0.810 ± 0.005 for GCN and 0.811 ± 0.009 for GAT over three seeds in our run — and the seed-to-seed spread is of the same size — report the mean and spread over seeds, not one run. Attention pays off on graphs where some neighbours are much more informative than others; Cora's citation neighbourhoods are fairly homogeneous, so the degree-based weighting of GCN is already a good guess.

**Drill 2.** Visualise the learned node embeddings from the GCN's hidden layer with t-SNE. Do nodes of the same class cluster together? Compare with t-SNE of the raw bag-of-words features.

**Solution:**

```python
from sklearn.manifold import TSNE

def tsne_df(features, labels):
    xy = TSNE(n_components=2, random_state=0, init="pca").fit_transform(features)
    return pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "topic": [f"topic {t}" for t in labels]})

labels = cora.y.cpu().tolist()
viz = ModelVisualizer()
fig_raw = viz.scatter(tsne_df(cora.x.cpu().numpy(), labels), x="x", y="y", color="topic",
                      title="t-SNE of raw word features")
fig_gcn = viz.scatter(tsne_df(hidden, labels), x="x", y="y", color="topic",
                      title="t-SNE of GCN hidden layer")
```

The raw features form a diffuse cloud with only faint class structure; the GCN embeddings form clear clusters, one per topic, with mixing at the boundaries. Remember that t-SNE distorts distances between clusters — read cluster membership, not the gaps.

**Drill 3.** Vary the number of GCN layers from 1 to 6. Does over-smoothing occur?

**Solution:**

```python
class DeepGCN(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, n_layers):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [n_classes]
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(nn.functional.dropout(x, 0.5, self.training), edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
        return x

for n_layers in range(1, 7):
    torch.manual_seed(0)
    _, test_acc, _ = train_node_model(DeepGCN(cora.num_features, 16, 7, n_layers), cora)
    print(f"{n_layers} layers: test acc {test_acc:.3f}")

# The mechanism, without any training: apply the propagation matrix k times
from torch_geometric.utils import to_dense_adj
A_tilde = to_dense_adj(cora.edge_index, max_num_nodes=cora.num_nodes)[0] + torch.eye(cora.num_nodes, device=device)
d_inv_sqrt = A_tilde.sum(1).pow(-0.5)
A_norm = d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]
H = cora.x
for k in range(1, 65):
    H = A_norm @ H
    if k in (1, 4, 16, 64):
        Hn = nn.functional.normalize(H, dim=1)
        print(f"after {k:>2} propagation steps: mean pairwise cosine similarity {(Hn @ Hn.T).mean():.3f}")
```

In our runs test accuracy peaked at two layers (0.804) and fell steadily to about 0.73–0.74 at six. The second loop shows why: repeated neighbourhood averaging makes node vectors more and more alike — the mean pairwise cosine similarity of the features rose from 0.06 for the raw features to 0.15 after one step, 0.35 after four, 0.62 after 16 and 0.82 after 64. That is over-smoothing: deep stacks pull every node in a connected region towards the same vector, so the classes blur. Residual connections, jumping-knowledge readouts (concatenating every layer, as the GIN example does) and normalisation layers are the standard counter-measures.

**Drill 4.** Implement GCN message passing from scratch (no torch_geometric layers). Verify it produces the same output as `GCNConv`.

**Solution:**

```python
conv = GCNConv(cora.num_features, 16).to(device)
N = cora.num_nodes
A = torch.zeros(N, N, device=device)
A[cora.edge_index[0], cora.edge_index[1]] = 1.0
A_tilde = A + torch.eye(N, device=device)                      # add self-loops
d_inv_sqrt = A_tilde.sum(1).pow(-0.5)
A_norm = d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]   # D^-1/2 (A + I) D^-1/2

with torch.no_grad():
    ours = A_norm @ (cora.x @ conv.lin.weight.T) + conv.bias    # aggregate (X W) over neighbours
    theirs = conv(cora.x, cora.edge_index)
print("max |difference|:", (ours - theirs).abs().max().item())
```

The difference is at floating-point rounding level (around $10^{-7}$ in our runs): `GCNConv` is exactly $\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{X}\mathbf{W} + \mathbf{b}$. The library version stores the graph sparsely as `edge_index` instead of a dense $N \times N$ matrix — essential once graphs have millions of nodes.

**Drill 5.** Train a link predictor on Cora with a proper edge split, and report test AUC.

**Solution:**

```python
from sklearn.metrics import roc_auc_score
from torch_geometric.transforms import RandomLinkSplit

torch.manual_seed(0)
split = RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True,
                        add_negative_train_samples=False)
train_data, val_data, test_data = split(Planetoid("data/mlfp05/cora", name="Cora")[0])

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.conv1, self.conv2 = GCNConv(in_dim, hidden), GCNConv(hidden, hidden)

    def forward(self, x, edge_index):
        return self.conv2(torch.relu(self.conv1(x, edge_index)), edge_index)

def score(z, pairs):
    return (z[pairs[0]] * z[pairs[1]]).sum(dim=1)             # dot-product decoder (a logit)

enc = Encoder(cora.num_features).to(device)
opt = torch.optim.Adam(enc.parameters(), lr=0.01)
train_data = train_data.to(device)
for epoch in range(200):
    enc.train()
    z = enc(train_data.x, train_data.edge_index)               # message passing on TRAIN edges only
    pos = train_data.edge_label_index
    neg = torch.randint(0, train_data.num_nodes, pos.shape, device=device)   # fresh negatives
    logits = torch.cat([score(z, pos), score(z, neg)])
    labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]).to(device)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()

enc.eval()
with torch.no_grad():
    z = enc(train_data.x, train_data.edge_index)
    test_scores = score(z, test_data.edge_label_index.to(device)).cpu()
print(f"test AUC on held-out edges: {roc_auc_score(test_data.edge_label.numpy(), test_scores.numpy()):.3f}")
```

`RandomLinkSplit` removes the validation and test edges from the message-passing graph and attaches labelled positive and negative pairs to each split, so the encoder never sees the edges it is scored on. Our runs reached a test AUC of about 0.89. The test AUC is therefore a genuine "predict a missing citation" score. If you instead score the training edges with the full graph, you will see a much higher number that measures memorisation, not prediction.

## Cross-References

- **Lesson 4.1** introduced spectral clustering, which uses the normalised graph Laplacian $\mathbf{I} - \tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$ — GCN propagates with the normalised adjacency inside it.
- **Lesson 5.4** introduced attention. GAT applies attention to graph neighbours.

## Reflection

You should now be able to:

- Explain message passing and how GCN, GraphSAGE, GAT and GIN differ in how they aggregate neighbours.
- Build GCNs for node classification and GIN for graph classification with torch_geometric (`GCNConv`, `GINConv`, `global_add_pool`, the graph `DataLoader`).
- Select epochs on validation data and compare against a no-graph baseline.
- Explain over-smoothing and why most GNNs are shallow.
- Evaluate link prediction on held-out edges without leakage.

---

# Lesson 5.7: Transfer Learning

## Why This Matters

Training a model from scratch on a small dataset often leads to overfitting. Transfer learning solves this by starting from a model pre-trained on a large dataset (ImageNet for vision, Wikipedia/BookCorpus for NLP) and fine-tuning it on your small target dataset. The pre-trained model has already learned general features (edges, textures for vision; grammar, semantics for NLP) that transfer to new tasks.

## Core Concepts

### FOUNDATIONS: The transfer learning recipe

1. **Load a pre-trained model** — e.g. ResNet-18/50 trained on ImageNet-1k (1.28 million labelled images, 1,000 classes), or BERT pre-trained on BookCorpus and English Wikipedia.
2. **Replace the task head** with a new one matching your number of classes.
3. **Freeze the backbone** at first — its early layers contain general features (edges, textures; syntax, word meaning) that transfer well — and train only the new head.
4. **Fine-tune later layers** with a smaller learning rate once the head is sensible ("discriminative learning rates": the deeper you go into the pre-trained network, the smaller the rate).
5. **Unfreeze more layers** only if you have enough data; otherwise you overfit and erase the pre-trained features ("catastrophic forgetting").

Lower layers transfer best because they are the most generic; the top layers are the most specific to the original task, which is why they are the first to be retrained.

### FOUNDATIONS: Data augmentation for small datasets

With little labelled data, every random transformation that preserves the label is free extra data: random crops, horizontal flips, small rotations, colour jitter for images; synonym replacement or back-translation for text. Augment the **training** set only, and choose transformations that keep the label true (a horizontal flip is fine for animals and vehicles, wrong for digits or text in images). Exercise 7's training transform is `Resize(96) → RandomHorizontalFlip → RandomCrop(96, padding=8) → ToTensor → Normalize(ImageNet mean/std)`.

### THEORY: Adapter modules

Full fine-tuning updates every parameter and needs a full model copy per task. **Adapters** (Houlsby et al., 2019) freeze the whole pre-trained network and insert small trainable bottleneck modules inside it: down-project to a small dimension $r$, apply a non-linearity, up-project back, and add the result to the input,

$$\text{Adapter}(\mathbf{h}) = \mathbf{h} + \mathbf{W}_{\text{up}}\, \text{ReLU}(\mathbf{W}_{\text{down}} \mathbf{h}).$$

Initialising $\mathbf{W}_{\text{up}}$ (and its bias) to zero makes every adapter the **identity at the start**, so training begins exactly from the pre-trained model and only learns a task-specific correction. Where the adapter sits matters: if it is applied to a summary of a block's output (as Exercise 7 does, on the channel-pooled features of a ResNet stage), add back only the adapter's *change*, `adapted - pooled`, or you add the pooled features a second time and perturb the backbone from step 0. With bottleneck 64 on ResNet-18's last two stages, about 1% of the parameters are trainable. Module 6's LoRA is the same idea with a low-rank update to the weight matrices themselves.

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BottleneckAdapter(nn.Module):
    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down, self.up = nn.Linear(dim, bottleneck), nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)           # zero-init: the adapter starts as the identity
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(torch.relu(self.down(x)))

class AdaptedStage(nn.Module):
    """Wrap a ResNet stage; adapt its channel-pooled output and add back only the change."""
    def __init__(self, stage, channels, bottleneck=64):
        super().__init__()
        self.stage, self.adapter = stage, BottleneckAdapter(channels, bottleneck)

    def forward(self, x):
        out = self.stage(x)
        pooled = out.mean(dim=(2, 3))                                  # (B, C)
        return out + (self.adapter(pooled) - pooled)[:, :, None, None]

backbone = resnet18(weights=ResNet18_Weights.DEFAULT)                 # ImageNet-1k weights
for p in backbone.parameters():
    p.requires_grad = False
x = torch.randn(2, 3, 96, 96)
backbone.eval()
with torch.no_grad():
    before = backbone(x)
backbone.layer3 = AdaptedStage(backbone.layer3, 256)
backbone.layer4 = AdaptedStage(backbone.layer4, 512)
with torch.no_grad():
    after = backbone(x)
print("identity at initialisation:", torch.allclose(before, after, atol=1e-5))     # True

backbone.fc = nn.Linear(512, 10)                                      # new, trainable head
trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
total = sum(p.numel() for p in backbone.parameters())
print(f"trainable {trainable:,} of {total:,} ({trainable / total:.1%})")
```

### FOUNDATIONS: The HuggingFace pipeline API

For NLP, the `transformers` library wraps tokenisation, the model call and post-processing in one object. Give the model readable label names and the pipeline returns them directly:

```python
from transformers import AutoTokenizer, BertForSequenceClassification, pipeline

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
clf_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=4,
    id2label=dict(enumerate(LABELS)), label2id={name: i for i, name in enumerate(LABELS)},
)
# ... fine-tune clf_model exactly as in Lesson 5.4's worked example, then:
classify = pipeline("text-classification", model=clf_model, tokenizer=tokenizer)
print(classify(["Central bank holds rates as inflation cools", "Striker scores twice in cup final"]))
```

Each result is a dict such as `{"label": "Business", "score": 0.93}`. Until the head is fine-tuned the labels are meaningless — the new classification layer is randomly initialised (the library warns about exactly this). The pipeline is for inference and quick demos; training stays in your own loop or the `Trainer` class.

### FOUNDATIONS: Architecture selection guide

| Data Type | Best Architecture | When to Transfer |
|---|---|---|
| Images | CNN / ViT | Almost always (ImageNet pre-trained) |
| Text | Transformer | Almost always (BERT/GPT-style pre-trained) |
| Sequences | LSTM / Transformer | Sometimes (domain-specific) |
| Graphs | GNN | Rarely (task-specific) |
| Tabular | Gradient boosting (Module 3) | Rarely — train from scratch |

## The Kailash Engines: OnnxBridge, ModelRegistry and InferenceServer

Deployment is three steps, each a Kailash engine:

1. **Export** with `OnnxBridge().export(model, "torch", output_path=Path(...), sample_input=x)` and check parity with `validate(...)` (Lesson 5.2).
2. **Register** the model in `ModelRegistry` and store the `.onnx` file as that version's `model.onnx` artifact.
3. **Serve** with `server = await InferenceServer.from_registry(name, registry=registry, version=v, runtime="onnx")`, then `await server.start()` and `await server.predict({"records": [...]})`, which returns a mapping with a `"predictions"` list. `from_registry`, `start`, `predict` and `stop` are all coroutines, so they run inside an `async` function.

Two practical details. InferenceServer turns each request record (a dict of named numbers) into **one row** of a 2-D float array, so an image model must be exported behind a small adapter that accepts flat pixel rows and reshapes them. And the PyTorch exporter OnnxBridge uses writes the weights to a side file (`<name>.onnx.data`) next to the graph; the registry stores a single file, so fold the weights into the `.onnx` file before registering it (`onnx.save_model(..., save_as_external_data=False)`), or the server loads a graph with no weights and every prediction fails. (There is no `InferenceServer(model_path=...)` constructor and no `predict_batch`, `warm_cache` or `PredictionResult` — older course material showed APIs that do not exist.)

## Worked Example: Fine-Tuning ResNet-18 on CIFAR-10 and serving it

Exercise 7 fine-tunes an ImageNet ResNet-18 on CIFAR-10 resized to $96 \times 96$ (at $32 \times 32$, ResNet's five stride-2 stages would shrink the image to $1 \times 1$ before the final pooling). This example follows the same recipe, then exports, registers and serves the model.

```python
import asyncio
import pickle
from pathlib import Path
import numpy as np
import onnx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from kailash.db import ConnectionManager
from kailash_ml import InferenceServer, ModelRegistry, OnnxBridge
from kailash_ml.engines.model_registry import LocalFileArtifactStore
from shared.kailash_helpers import get_device

device = get_device()
SIZE, MEAN, STD = 96, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]     # ImageNet statistics
train_tf = T.Compose([T.Resize((SIZE, SIZE)), T.RandomHorizontalFlip(), T.RandomCrop(SIZE, padding=8),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
test_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
train_data = datasets.CIFAR10("data/mlfp05/cifar10", train=True, download=True, transform=train_tf)
test_data = datasets.CIFAR10("data/mlfp05/cifar10", train=False, download=True, transform=test_tf)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

model = resnet18(weights=ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False                       # freeze the backbone
model.fc = nn.Linear(512, 10)                     # new head (trainable by default)
model = model.to(device)

def run_epoch(optimizer):
    model.train()
    for images, labels in train_loader:
        loss = nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_accuracy():
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            correct += (model(images.to(device)).argmax(1).cpu() == labels).sum().item()
    return correct / len(test_data)

# Stage 1: train the head only
run_epoch(torch.optim.Adam(model.fc.parameters(), lr=1e-3))
print(f"head only, 1 epoch: test acc {test_accuracy():.3f}")

# Stage 2: unfreeze the last stage with a 10x smaller learning rate (discriminative LRs)
for p in model.layer4.parameters():
    p.requires_grad = True
run_epoch(torch.optim.Adam([{"params": model.layer4.parameters(), "lr": 1e-4},
                            {"params": model.fc.parameters(), "lr": 1e-3}]))
print(f"+ layer4, 1 more epoch: test acc {test_accuracy():.3f}")

# Export, register, serve
IMAGE_SHAPE = (3, SIZE, SIZE)

class FlatImageModel(nn.Module):
    """Accept flat pixel rows (what InferenceServer sends) and reshape them to images."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.eval()   # the WRAPPER too: OnnxBridge puts a module that was training back into train mode

    def forward(self, rows):
        return self.net(rows.reshape(-1, *IMAGE_SHAPE))

    def predict(self, X):                         # OnnxBridge.validate calls this
        with torch.no_grad():
            return self.forward(torch.as_tensor(np.asarray(X), dtype=torch.float32)).numpy()

def to_records(images):
    rows = images.reshape(len(images), -1).tolist()
    return [{f"px{j:05d}": float(v) for j, v in enumerate(row)} for row in rows]

async def export_register_serve(net, name, images):
    flat = FlatImageModel(net.cpu())
    onnx_path = Path("outputs") / f"{name}.onnx"
    bridge = OnnxBridge()
    result = bridge.export(flat, "torch", output_path=onnx_path,
                           sample_input=images[:2].reshape(2, -1))      # 2 rows: dynamic batch
    assert result.success, result.error_message
    check = bridge.validate(flat, onnx_path, images.reshape(len(images), -1).numpy(), tolerance=1e-3)
    print(f"ONNX parity: valid={check.valid}, max |diff| {check.max_diff:.1e}")
    onnx.save_model(onnx.load(str(onnx_path)), str(onnx_path), save_as_external_data=False)

    store = LocalFileArtifactStore(".kailash_ml/artifacts")
    conn = ConnectionManager(f"sqlite:///{Path('mlfp05_serving.db').resolve()}")   # absolute path
    await conn.initialize()
    registry = ModelRegistry(conn, artifact_store=store)
    version = await registry.register_model(name, pickle.dumps(net.state_dict()))
    await store.save(name, version.version, onnx_path.read_bytes(), "model.onnx")

    server = await InferenceServer.from_registry(name, registry=registry,
                                                 version=version.version, runtime="onnx")
    await server.start()
    response = await server.predict({"records": to_records(images)})
    await server.stop()
    await conn.close()
    return np.asarray(response["predictions"], dtype=np.float32)

images, labels = next(iter(DataLoader(test_data, batch_size=8)))
served = asyncio.run(export_register_serve(model, "cifar10_resnet18", images))
with torch.no_grad():
    direct = model.cpu()(images).numpy()
print("served classes:", served.argmax(1).tolist(), "| true:", labels.tolist())
print(f"served vs PyTorch, max |logit difference|: {np.abs(served - direct).max():.1e}")
```

What to expect: the pre-trained features are strong enough that even the head-only stage gives a large jump over a from-scratch model trained for the same single epoch, and unfreezing `layer4` adds more; Exercise 7 trains longer and compares against a from-scratch baseline explicitly. The served predictions match PyTorch to floating-point precision — the same graph, executed by ONNX Runtime behind the server.

## Try It Yourself

The drills reuse the objects defined in the worked example and the adapter section.

**Drill 1.** Data efficiency: train (a) the residual CNN from Lesson 5.2 from scratch and (b) the frozen ResNet-18 with a new head, each on only 10% of the CIFAR-10 training images. Which wins, and by how much?

**Solution:**

```python
from torch.utils.data import Subset

small = Subset(train_data, range(0, len(train_data), 10))       # 5,000 images, all classes
small_loader = DataLoader(small, batch_size=128, shuffle=True)

def fit(net, params, epochs=3, lr=1e-3):
    net.to(device)
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        net.train()
        for images, labels in small_loader:
            loss = nn.functional.cross_entropy(net(images.to(device)), labels.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            correct += (net(images.to(device)).argmax(1).cpu() == labels).sum().item()
    return correct / len(test_data)

frozen = resnet18(weights=ResNet18_Weights.DEFAULT)
for p in frozen.parameters():
    p.requires_grad = False
frozen.fc = nn.Linear(512, 10)
scratch = resnet18(weights=None, num_classes=10)                 # same architecture, random weights
print(f"pre-trained, head only: {fit(frozen, frozen.fc.parameters()):.3f}")
print(f"from scratch, all layers: {fit(scratch, scratch.parameters()):.3f}")
```

With 5,000 labelled images the pre-trained model should win clearly, even though it trains only a 5,130-parameter head while the scratch model trains all 11 million parameters: the scratch model has to learn edges and textures from too little data. The gap narrows as the labelled set grows — which is exactly the trade-off Exercise 7's data-efficiency study plots.

**Drill 2.** Compare three ways of adapting the same backbone on the 10% subset: head only, adapters (the `AdaptedStage` model above), and full fine-tuning. Report trainable parameters and accuracy.

**Solution:**

```python
def adapter_model():
    net = resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in net.parameters():
        p.requires_grad = False
    net.layer3, net.layer4 = AdaptedStage(net.layer3, 256), AdaptedStage(net.layer4, 512)
    net.fc = nn.Linear(512, 10)
    return net

def full_model():
    net = resnet18(weights=ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(512, 10)
    return net

for name, net, lr in [("head only", frozen, 1e-3), ("adapters", adapter_model(), 1e-3),
                      ("full fine-tune", full_model(), 1e-4)]:
    params = [p for p in net.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    print(f"{name:>15}: {n:>10,} trainable params, test acc {fit(net, params, lr=lr):.3f}")
```

The trainable counts are exact: 5,130 for the head, 104,330 with the adapters (0.9% of the adapted model's 11,280,842) and all 11,181,642 for full fine-tuning. Adapters usually recover much of the gap between head-only and full fine-tuning at under 1% of the trainable parameters — and per task you store only those parameters, not a new copy of the network. Full fine-tuning needs the smaller learning rate, or it erases the pre-trained features.

**Drill 3.** Measure inference latency of the exported ONNX model against PyTorch, for a batch of 1 and a batch of 64.

**Solution:**

```python
import time
import onnxruntime as ort

session = ort.InferenceSession(str(Path("outputs") / "cifar10_resnet18.onnx"))
input_name = session.get_inputs()[0].name
flat_cpu = FlatImageModel(model.cpu())

def ms_per_call(fn, repeats=20):
    fn()                                                  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return 1000 * (time.perf_counter() - start) / repeats

for batch in [1, 64]:
    rows = torch.randn(batch, 3 * SIZE * SIZE)
    t_onnx = ms_per_call(lambda: session.run(None, {input_name: rows.numpy()}))
    with torch.no_grad():
        t_torch = ms_per_call(lambda: flat_cpu(rows))
    print(f"batch {batch:>2}: ONNX Runtime {t_onnx:6.1f} ms | PyTorch (CPU) {t_torch:6.1f} ms")
```

Both run on the CPU here, and timings are only meaningful on an otherwise idle machine. The usual pattern is that ONNX Runtime has its clearest advantage at batch size 1, where framework overhead dominates, and that the two converge at large batches, where the convolutions themselves dominate. On our heavily loaded workstation ONNX Runtime was faster at batch 1 (155 against 223 ms) and slightly slower at batch 64 — so measure on the hardware you will actually serve from. The more important win is operational: the server needs ONNX Runtime, not PyTorch.

**Drill 4.** Implement progressive unfreezing: start with only the head, then unfreeze one stage at a time (`layer4`, then `layer3`, then `layer2`) after each epoch, giving deeper stages smaller learning rates. Compare with unfreezing everything at once.

**Solution:**

```python
def fit_groups(net, param_groups, epochs=1):
    """Like fit(), but with one learning rate per parameter group."""
    net.to(device)
    opt = torch.optim.Adam(param_groups)
    for _ in range(epochs):
        net.train()
        for images, labels in small_loader:
            loss = nn.functional.cross_entropy(net(images.to(device)), labels.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            correct += (net(images.to(device)).argmax(1).cpu() == labels).sum().item()
    return correct / len(test_data)

net = full_model()
for p in net.parameters():
    p.requires_grad = False
param_groups = []
schedule = [(net.fc, 1e-3), (net.layer4, 1e-4), (net.layer3, 5e-5), (net.layer2, 2e-5)]
for stage, (module, lr) in enumerate(schedule):
    for p in module.parameters():
        p.requires_grad = True
    param_groups.append({"params": list(module.parameters()), "lr": lr})
    print(f"after unfreezing stage {stage}: test acc {fit_groups(net, param_groups):.3f}")

all_at_once = full_model()
print(f"everything unfrozen for 4 epochs: {fit(all_at_once, all_at_once.parameters(), epochs=4, lr=1e-4):.3f}")
```

Each epoch the optimiser is rebuilt with one more, lower-rate group, so the new head settles before the layers beneath it start to move. On small data, progressive unfreezing usually ends at least as high as unfreezing everything at once and is less sensitive to the learning rate, because the randomly initialised head never sends large, noisy gradients into layers that are still being trained. On larger datasets the difference shrinks.

**Drill 5.** Remove the training augmentation (use `test_tf` for training) and retrain the head-only model on the 10% subset. How much did augmentation contribute?

**Solution:**

```python
plain_small = Subset(datasets.CIFAR10("data/mlfp05/cifar10", train=True, transform=test_tf),
                     range(0, 50000, 10))
small_loader = DataLoader(plain_small, batch_size=128, shuffle=True)    # fit() reads small_loader
no_aug = resnet18(weights=ResNet18_Weights.DEFAULT)
for p in no_aug.parameters():
    p.requires_grad = False
no_aug.fc = nn.Linear(512, 10)
print(f"head only, no augmentation: {fit(no_aug, no_aug.fc.parameters()):.3f}")
```

With a frozen backbone and only three epochs, augmentation changes little: the head sees fixed features, and a few epochs are not enough to overfit 5,000 examples. Augmentation earns its keep when many parameters are trained for many epochs on little data — repeat the comparison with full fine-tuning for 10 epochs and the gap opens up.

## Cross-References

- **Lesson 5.2** built CNNs from scratch. Transfer learning reuses pre-trained CNNs.
- **Lesson 5.4** introduced BERT. Transfer learning with BERT is the practical application.
- **Module 6, Lesson 6.2** extends transfer learning to LoRA and adapter-based fine-tuning.

## Reflection

You should now be able to:

- Fine-tune a pre-trained model in stages: new head first, then deeper layers with smaller learning rates.
- Explain why lower layers transfer best and when to unfreeze more.
- Use augmentation correctly (training set only, label-preserving transformations).
- Build an adapter that starts as the identity, and explain why adapters train about 1% of the parameters.
- Run a fine-tuned text classifier through the HuggingFace `pipeline` API.
- Export with OnnxBridge, register in ModelRegistry, and serve with `InferenceServer.from_registry` — and say why the served model needs flat input rows and embedded weights.

---

# Lesson 5.8: Reinforcement Learning

## Why This Matters

All deep learning so far learns from static datasets — images, text, sequences. Reinforcement learning (RL) learns from interaction with an environment. An agent takes actions, receives rewards, and learns a policy that maximises cumulative reward. RL powers game-playing AI (AlphaGo), robotics, and — crucially — the alignment of large language models (RLHF, which you will study in Module 6).

## Core Concepts

### FOUNDATIONS: Agent, environment, reward

At each step $t$ the agent observes a **state** $s_t$, chooses an **action** $a_t$ from its **policy** $\pi(a \mid s)$, and the environment returns a **reward** $r_{t+1}$ and the next state. An **episode** runs until the task ends. The agent maximises the expected **return**, the discounted sum of future rewards $G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots$, where $\gamma \in [0, 1)$ is the discount factor — future rewards are worth less than immediate ones.

Gymnasium, the standard environment library, distinguishes two ways an episode stops: `terminated` (the task genuinely ended — the customer churned, the pole fell) and `truncated` (a time limit cut the episode short). The distinction matters for learning: after a termination the future value is zero, but after a truncation the state still had a future, so value estimates should keep bootstrapping. A loop that waits only for `terminated` never ends on a time-limited environment; use `done = terminated or truncated` to stop the loop, and `terminated` alone to zero the bootstrap target.

### THEORY: Bellman equations — expectation and optimality

The **state-value function** of a policy $\pi$ is the expected return from a state when following $\pi$. It satisfies the **Bellman expectation equation**:

$$V^\pi(s) = \mathbb{E}_\pi\left[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s\right]$$

and likewise for the **action-value function** $Q^\pi(s, a) = \mathbb{E}_\pi[R_{t+1} + \gamma Q^\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a]$. These describe *a given* policy.

The **optimal** action-value function $Q^*(s, a) = \max_\pi Q^\pi(s, a)$ satisfies the **Bellman optimality equation**, in which the next action is chosen greedily:

$$Q^*(s, a) = \mathbb{E}\left[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \mid S_t = s, A_t = a\right]$$

Once you have $Q^*$, the optimal policy is simply $\pi^*(s) = \arg\max_a Q^*(s, a)$. Value-based methods such as DQN learn $Q^*$ from the optimality equation; policy-gradient methods such as PPO learn $\pi$ directly and use $V^\pi$ as a baseline.

### THEORY: DQN (Deep Q-Network)

DQN approximates $Q^*(s, a)$ with a neural network $Q(s, a; \theta)$ that outputs one value per discrete action, and regresses it onto the optimality target:

$$\mathcal{L} = \mathbb{E}\left[\left(r + \gamma\,(1 - \text{terminated}) \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

Two stabilisers make this work. **Experience replay** stores transitions in a buffer and trains on random mini-batches, breaking the correlation between consecutive steps. The **target network** $\theta^-$ is a periodically refreshed copy of $\theta$, so the regression target does not move with every update. Exploration is **$\varepsilon$-greedy**: with probability $\varepsilon$ take a uniformly random action (any of the $|A|$ actions), otherwise the greedy one; $\varepsilon$ decays during training. DQN needs a discrete action space — the $\max_{a'}$ is a max over a list.

### THEORY: Policy gradients, actor-critic and A2C

Policy-gradient methods adjust a parameterised policy $\pi_\theta$ directly, increasing the log-probability of actions that turned out better than expected:

$$\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, \hat{A}_t\right]$$

The **advantage** $\hat{A}_t$ is how much better the action was than the state's average, $\hat{A}_t \approx G_t - V(s_t)$. Subtracting the baseline $V(s_t)$ does not change the expected gradient but greatly reduces its variance. An **actor-critic** learns both: the actor is $\pi_\theta$, the critic is $V_\phi$, trained by regression onto observed returns. **A2C (Advantage Actor-Critic)** is the synchronous version: collect a short rollout from one or several environment copies, compute advantages (often with Generalised Advantage Estimation, GAE, which blends one-step and multi-step estimates), take one gradient step, discard the data, repeat. It is on-policy and simple, and it works for discrete and continuous actions.

### THEORY: PPO (Proximal Policy Optimization)

A2C takes one step per batch of experience; taking several would be more data-efficient, but large policy changes based on stale data collapse performance. PPO allows several epochs of updates on each rollout while limiting how far the policy moves, through a clipped objective:

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ is the probability ratio and $\epsilon$ is typically 0.2. The clip is **asymmetric in effect**. When $\hat{A}_t > 0$ the objective stops rewarding increases of $r_t$ beyond $1 + \epsilon$ — the gradient there is zero — but if $r_t$ has fallen *below* $1 - \epsilon$ the unclipped term is the minimum and the gradient still pushes the probability back up. Mirror-image for $\hat{A}_t < 0$. So the clip only removes the incentive to move *further in the direction the advantage favours*; it never blocks corrections.

PPO works for discrete actions (a categorical policy, as in Exercise 8) and **continuous** actions (a Gaussian policy whose network outputs a mean, with a learned standard deviation) — the second worked example below.

### THEORY: DDPG and SAC — off-policy methods for continuous control

**DDPG (Deep Deterministic Policy Gradient)** extends DQN to continuous actions. Since you cannot take a max over infinitely many actions, an actor network $\mu_\theta(s)$ outputs the action directly and is trained to maximise the critic $Q_\phi(s, \mu_\theta(s))$; the critic is trained like DQN with target networks and a replay buffer. Exploration comes from adding noise to the actor's output. It is sample-efficient (off-policy, reuses old data) but brittle: the critic tends to over-estimate values and the actor exploits those errors. (TD3 adds twin critics and delayed actor updates to fix this.)

**SAC (Soft Actor-Critic)** maximises reward *plus* the entropy of the policy, $\mathbb{E}[\sum_t \gamma^t (r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t)))]$. The actor is stochastic, two critics are trained and the smaller estimate is used, and the temperature $\alpha$ is usually tuned automatically. The entropy bonus keeps exploring and makes it robust to hyperparameters, which is why SAC is a default choice for continuous control when environment interaction is expensive.

| Algorithm | Action space | On/off-policy | Key idea | Example application |
|---|---|---|---|---|
| DQN | Discrete | Off (replay) | Regress $Q$ onto the Bellman optimality target | Which retention offer to make to a customer |
| A2C | Discrete or continuous | On | Actor-critic with an advantage baseline | Allocating a budget across a few channels |
| PPO | Discrete or continuous | On | Clipped policy updates, several epochs per rollout | Re-order quantities in a supply chain |
| DDPG | Continuous | Off | Deterministic actor + Q critic | Setting a machine's continuous control inputs |
| SAC | Continuous | Off | Max-entropy actor + twin critics | Continuous price adjustments under uncertainty |

The applications are illustrative pairings, not prescriptions: the deciding questions are whether the actions are discrete or continuous, and whether interaction with the environment is cheap (on-policy is fine) or expensive (prefer off-policy).

## The Kailash Engine: RLTrainer (status in this course)

kailash-ml's RL engine lives at `kailash_ml.rl.RLTrainer` (there is no top-level `kailash_ml.RLTrainer`), with the functional entry point `kailash_ml.rl.rl_train(env, algo="ppo", total_timesteps=..., hyperparameters={...})`. Its training backend is Stable-Baselines3, which ships as the optional `kailash-ml[rl]` extra and is **not installed** in the course environment. Exercise 8 therefore hand-writes DQN and PPO in PyTorch, which is also the best way to learn what each line of the algorithms does. In a production setting with the extra installed, RLTrainer runs the same algorithms (PPO, A2C, DQN, DDPG, SAC) and records the results with the rest of the Kailash ML lifecycle. You can check what your environment has:

```python
import importlib.util
from kailash_ml.rl import RLTrainer, RLTrainingConfig, rl_train

config = RLTrainingConfig(algorithm="PPO", total_timesteps=50_000)   # what RLTrainer.train() takes
print("RL backend (stable-baselines3) installed:",
      importlib.util.find_spec("stable_baselines3") is not None)
```

## Worked Example 1: A custom environment and DQN for customer churn

The environment is a simplified version of Exercise 8's churn-prevention scenario (synthetic: the dynamics below are invented for teaching, not fitted to any company's data). Each episode is one customer over 30 days. The state is satisfaction, usage, the fraction of the month elapsed and open support tickets, all in $[0, 1]$; the actions are do nothing, offer a discount, or make a support call; churn probability rises as satisfaction falls and tickets pile up, and a customer retained to the end of the month earns a bonus. The state includes the time elapsed because the bonus depends on it — without it the environment would not be Markov (the same observation could be one day or twenty days from the bonus), and value learning would be much harder.

```python
import random
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

class ChurnEnv(gym.Env):
    """One customer, 30 daily decisions. Synthetic dynamics for teaching."""
    COST = {0: 0.0, 1: 1.0, 2: 0.5}               # do nothing, discount, support call

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.max_steps = 30

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)                   # seeds self.np_random
        satisfaction, usage, tickets = self.np_random.uniform(0.2, 0.8, size=3)
        self.state = np.array([satisfaction, usage, 0.0, tickets], dtype=np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        satisfaction, usage, _, tickets = self.state
        if action == 1:                            # discount: happier, uses more
            satisfaction, usage = satisfaction + 0.10, usage + 0.05
        elif action == 2:                          # support call: fewer open tickets
            tickets, satisfaction = tickets - 0.15, satisfaction + 0.05
        satisfaction += -0.02 + self.np_random.normal(0, 0.02)     # natural drift
        usage += -0.01 + self.np_random.normal(0, 0.02)
        tickets += 0.02 + self.np_random.normal(0, 0.01)
        self.steps += 1
        elapsed = self.steps / self.max_steps
        self.state = np.clip([satisfaction, usage, elapsed, tickets], 0.0, 1.0).astype(np.float32)

        churn_prob = max(0.0, 0.3 - 0.4 * self.state[0] + 0.3 * self.state[3])
        terminated = bool(self.np_random.random() < churn_prob)   # the customer left
        truncated = self.steps >= self.max_steps                  # the month is over
        reward = -5.0 if terminated else 1.0 - self.COST[int(action)]
        if truncated and not terminated:
            reward += 10.0                                         # retained for the month
        return self.state.copy(), reward, terminated, truncated, {}

check_env(ChurnEnv())          # Gymnasium's API checker: spaces, dtypes, reset/step contract

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, action_dim))

    def forward(self, x):
        return self.net(x)

def train_dqn(env, episodes=600, gamma=0.99, batch_size=64, lr=5e-4, target_every=100,
              reward_scale=0.1, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    n_actions = env.action_space.n
    q_net = DQN(env.observation_space.shape[0], n_actions)
    target = DQN(env.observation_space.shape[0], n_actions)
    target.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer, returns, step_count = deque(maxlen=20_000), [], 0

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        epsilon = max(0.05, 1.0 - episode / (0.6 * episodes))     # linear decay to 5%
        done, total = False, 0.0
        while not done:
            if random.random() < epsilon:
                action = random.randrange(n_actions)                # every action can be explored
            else:
                with torch.no_grad():
                    action = int(q_net(torch.as_tensor(state)).argmax())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.append((state, action, reward * reward_scale, next_state, float(terminated)))
            state, total, step_count = next_state, total + reward, step_count + 1

            if len(buffer) >= 1_000:
                s, a, r, s2, term = map(np.array, zip(*random.sample(buffer, batch_size)))
                s, s2 = torch.as_tensor(s), torch.as_tensor(s2)
                q = q_net(s).gather(1, torch.as_tensor(a).view(-1, 1)).squeeze(1)
                with torch.no_grad():   # bootstrap unless the episode TERMINATED (truncation still has a future)
                    y = torch.as_tensor(r, dtype=torch.float32) + gamma * (
                        1 - torch.as_tensor(term, dtype=torch.float32)) * target(s2).max(1).values
                loss = nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if step_count % target_every == 0:
                target.load_state_dict(q_net.state_dict())
        returns.append(total)
    return q_net, returns

def evaluate(env, policy, episodes=200, seed=10_000):
    totals = []
    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        done, total = False, 0.0
        while not done:
            state, reward, terminated, truncated, _ = env.step(policy(state))
            done = terminated or truncated
            total += reward
        totals.append(total)
    return float(np.mean(totals))

env = ChurnEnv()
q_net, returns = train_dqn(env)
greedy = lambda s: int(q_net(torch.as_tensor(s)).argmax())
rng = np.random.default_rng(0)
for name, policy in [("never intervene", lambda s: 0), ("random", lambda s: int(rng.integers(3))),
                     ("always discount", lambda s: 1), ("always call", lambda s: 2),
                     ("DQN (greedy)", greedy)]:
    print(f"{name:>16}: mean return {evaluate(env, policy):6.2f}")
```

Always evaluate a learned policy against simple fixed policies on the same seeds. In our run (200 evaluation customers) the fixed policies scored: never intervene $-2.4$, always discount $-4.6$ (the discount costs as much as a day's revenue), random $3.3$, and **always make a support call $10.4$**. DQN after 600 training episodes scored $11.0$: it had learned that discounts lose money and that calls help, but not yet a policy as good as the simplest sensible heuristic. With the target network refreshed only every 500 steps, or without scaling the rewards down, it did worse than random. That is a realistic picture of RL: it is sample-hungry and sensitive to settings, and a learned policy is only worth deploying once it beats the heuristics a domain expert would try first.

Three easy-to-miss details in the code: the bootstrap is masked by `terminated` only (a time-limit cut still has a future); exploration samples from **all** `n_actions`; and rewards are scaled by 0.1 for training, which keeps the regression targets near 1 without changing which policy is best.

## Worked Example 2: PPO for a continuous action problem

Pendulum-v1 is Gymnasium's standard continuous-control task: swing a pendulum upright and hold it there, choosing a torque in $[-2, 2]$ every step (200 steps per episode, reward between about $-16$ and 0 per step). The policy is a Gaussian: the actor network outputs the mean torque and a learned log standard deviation; actions are sampled, and clipped to the valid range only when sent to the environment.

```python
from torch.distributions import Normal

class GaussianActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                   nn.Tanh(), nn.Linear(hidden, act_dim))
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                    nn.Tanh(), nn.Linear(hidden, 1))    # separate network, no shared trunk

    def dist(self, obs):
        return Normal(self.actor(obs), self.log_std.exp())

def ppo_continuous(env_id="Pendulum-v1", updates=150, steps=2048, epochs=10, minibatch=64,
                   gamma=0.99, lam=0.95, clip_eps=0.2, lr=3e-4, seed=0):
    env = gym.make(env_id)
    torch.manual_seed(seed)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    low, high = env.action_space.low, env.action_space.high
    ac = GaussianActorCritic(obs_dim, act_dim)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    obs, _ = env.reset(seed=seed)
    episode_return, finished = 0.0, []

    for update in range(updates):
        buf = {k: [] for k in ("obs", "act", "logp", "rew", "val", "end", "boot")}
        for _ in range(steps):                                   # 1. collect a rollout
            o = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                d = ac.dist(o)
                act = d.sample()
                logp, val = d.log_prob(act).sum(), ac.critic(o).squeeze()
            obs, rew, term, trunc, _ = env.step(np.clip(act.numpy(), low, high))
            boot = 0.0                                            # value of the future after this step
            if trunc and not term:                                # time limit: the future still exists
                with torch.no_grad():
                    boot = ac.critic(torch.as_tensor(obs, dtype=torch.float32)).item()
            for k, v in zip(buf, (o, act, logp, rew, val, term or trunc, boot)):
                buf[k].append(v)
            episode_return += rew
            if term or trunc:
                finished.append(episode_return)
                obs, _ = env.reset()
                episode_return = 0.0

        with torch.no_grad():                                     # 2. advantages with GAE
            next_val = ac.critic(torch.as_tensor(obs, dtype=torch.float32)).squeeze()
            adv, gae = torch.zeros(steps), 0.0
            for t in reversed(range(steps)):
                if buf["end"][t]:                                 # episode boundary: no GAE across it
                    v_next, gae = buf["boot"][t], 0.0
                else:
                    v_next = next_val if t == steps - 1 else buf["val"][t + 1]
                delta = buf["rew"][t] + gamma * v_next - buf["val"][t]
                gae = delta + gamma * lam * gae
                adv[t] = gae
            values = torch.stack(buf["val"])
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        O, A, old_logp = torch.stack(buf["obs"]), torch.stack(buf["act"]), torch.stack(buf["logp"])

        for _ in range(epochs):                                    # 3. several clipped epochs
            for idx in torch.randperm(steps).split(minibatch):
                d = ac.dist(O[idx])
                ratio = (d.log_prob(A[idx]).sum(-1) - old_logp[idx]).exp()
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (ac.critic(O[idx]).squeeze(-1) - returns[idx]).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()
        if (update + 1) % 25 == 0:
            print(f"update {update + 1}: mean return of last 10 episodes {np.mean(finished[-10:]):.0f}")
    return ac, finished

ac, episode_returns = ppo_continuous()
```

A random policy on Pendulum scores roughly $-1{,}200$ per episode (we measured $-1{,}195$ over 20 episodes); a well-trained policy reaches a few hundred below zero or better, with the pendulum held upright. Learning on this task is slow at first — the agent must discover swinging up before holding steady. In a 40-update check (about 82,000 steps) the mean return of the last ten episodes moved from about $-1{,}270$ to $-870$, with plenty of noise along the way; the default 150 updates (about 300,000 steps) give it room to go much further. The exact path depends on the seed; compare runs on the mean of the last few episodes.

## Try It Yourself

The drills reuse `ChurnEnv`, `DQN`, `train_dqn`, `evaluate`, `GaussianActorCritic` and `ppo_continuous` from the worked examples.

**Drill 1.** Run DQN on Gymnasium's `CartPole-v1` (the environment Exercise 8 starts with). How many episodes until the agent regularly reaches the 500-step limit?

**Solution:**

```python
cartpole = gym.make("CartPole-v1")
cp_net, cp_returns = train_dqn(cartpole, episodes=300)
for start in range(0, 300, 50):
    print(f"episodes {start:>3}-{start + 49}: mean return {np.mean(cp_returns[start:start + 50]):.0f}")
print("greedy evaluation:", evaluate(cartpole, lambda s: int(cp_net(torch.as_tensor(s)).argmax()), episodes=20))
```

Returns stay low while $\varepsilon$ is high and climb as exploration decays. In our run the mean return per block of 50 episodes went 25, 54, 136, 129, 289, 325 — but the greedy policy then averaged only 130 over 20 evaluation episodes, well short of the 500-step cap. DQN on CartPole is famously unstable: performance can collapse and recover between nearby checkpoints, so evaluate checkpoints during training rather than trusting the last one, and expect to need longer training (and refinements such as Double DQN). CartPole is time-limited at 500 steps, so this is exactly the case where `truncated` must end the loop but must not zero the bootstrap.

**Drill 2.** Verify that PPO's clipping is active. Instrument the update to record the fraction of samples whose ratio left $[1 - \epsilon, 1 + \epsilon]$ and the fraction where the clipped term was the one selected.

**Solution:**

```python
def clip_stats(ratio, adv, clip_eps=0.2):
    outside = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean().item()
    surr1, surr2 = ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    clipped_active = (surr2 < surr1).float().mean().item()     # min() picked the clipped term
    return outside, clipped_active

# Inside ppo_continuous's minibatch loop, after computing ratio:
#     stats.append(clip_stats(ratio.detach(), adv[idx]))
ratio = torch.tensor([0.7, 0.95, 1.1, 1.3, 1.3, 0.7])
adv = torch.tensor([1.0, 1.0, -1.0, 1.0, -1.0, -1.0])
print(clip_stats(ratio, adv))   # (0.667, 0.333): 4 of 6 outside the range, only 2 actually clipped
```

The worked toy batch shows the asymmetry: of the four ratios outside $[0.8, 1.2]$, only two have their gradient cut — the ratio of 1.3 with a positive advantage (already pushed far enough up) and the ratio of 0.7 with a negative advantage (already pushed far enough down). The other two are outside the range in the direction that *undoes* a previous move, so the unclipped term is selected and their gradient survives. In a real run the clipped-and-active fraction typically sits in the low tens of percent and grows over the epochs of each update.

**Drill 3.** Create a custom Gymnasium environment for ride-hailing pricing with a continuous action: the state is (hour of day, local demand, available drivers), the action is a price multiplier in $[0.8, 2.0]$, and the reward is revenue minus a penalty for riders lost to high prices. Check it with `check_env`, then train it with `ppo_continuous`.

**Solution:**

```python
class SurgePricingEnv(gym.Env):
    """Synthetic ride-hailing market: one decision per 15 minutes for one day."""
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(0.8, 2.0, shape=(1,), dtype=np.float32)
        self.max_steps = 96

    def _obs(self):
        hour = (self.t / self.max_steps) % 1.0
        rush = np.exp(-((hour - 0.35) ** 2) / 0.005) + np.exp(-((hour - 0.75) ** 2) / 0.005)
        self.demand = float(np.clip(0.3 + 0.6 * rush + self.np_random.normal(0, 0.05), 0, 1))
        self.drivers = float(np.clip(0.5 + self.np_random.normal(0, 0.1), 0, 1))
        return np.array([hour, self.demand, self.drivers], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        price = float(np.clip(action[0], 0.8, 2.0))
        wanting = 100 * self.demand                                     # riders who want a trip
        riders = wanting * np.exp(-1.2 * (price - 1.0))                 # fewer accept a higher price
        served = min(riders, 100 * self.drivers)                        # limited by available drivers
        lost = wanting - served                                         # priced out + unserved
        reward = (served * price - 0.3 * lost) / 100                    # scaled revenue - goodwill cost
        self.t += 1
        return self._obs(), float(reward), False, self.t >= self.max_steps, {}

check_env(SurgePricingEnv())
gym.register(id="SurgePricing-v0", entry_point=SurgePricingEnv)
pricing_ac, pricing_returns = ppo_continuous("SurgePricing-v0", updates=40, steps=2048)
```

The environment never *terminates* (a day always runs its 96 steps) — it only truncates, which is the correct modelling choice for a fixed horizon. Inspect the trained policy by feeding it states across the day: a sensible policy raises the multiplier in the two rush-hour peaks, when demand exceeds the available drivers, and drops it towards 0.8–1.0 off-peak, where high prices just lose riders. All dynamics here are invented; a real pricing environment would be fitted to historical demand data, and the reward would include the business's own constraints (caps on surge multipliers, fairness rules).

**Drill 4.** Compare DQN and PPO on the same discrete-action environment (`ChurnEnv`). Which learns faster in environment steps, and which reaches a higher final return?

**Solution:** PPO needs a categorical policy for discrete actions — replace the Gaussian head with logits:

```python
from torch.distributions import Categorical

class CategoricalActorCritic(GaussianActorCritic):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__(obs_dim, n_actions, hidden)

    def dist(self, obs):
        return Categorical(logits=self.actor(obs))

# In a copy of ppo_continuous: build CategoricalActorCritic(obs_dim, env.action_space.n),
# send int(act) to env.step instead of the clipped vector, and drop the .sum(-1) on
# log_prob / entropy (a categorical log-probability is already one number per sample).
probe = CategoricalActorCritic(4, 3)
d = probe.dist(torch.rand(5, 4))
print(d.sample().shape, d.log_prob(d.sample()).shape)     # torch.Size([5]) torch.Size([5])
```

Train both for the same number of environment steps (count steps, not episodes — DQN updates every step, PPO once per rollout) and plot return against steps. DQN usually extracts more from each step because it replays old experience; PPO is usually more stable and less sensitive to hyperparameters. On a 3-action problem this small, both should beat the fixed policies of Worked Example 1, and the gap between them is often smaller than the variation across seeds.

**Drill 5.** Explain in a paragraph how PPO connects to RLHF for LLM alignment. What is the "environment"? What is the "reward"? What is the "policy"? (This connects to Module 6, Lesson 6.3.)

**Solution:** In RLHF the **policy** is the language model. The **state** is the prompt plus the tokens generated so far, and each **action** is the next token — a choice from a *discrete* vocabulary of tens of thousands of tokens, so the policy is categorical, exactly like Drill 4's. An episode is one complete response. The **reward** comes from a reward model trained on human preference comparisons ("response A is better than response B") and is given at the end of the response. PPO then updates the LLM to make high-reward responses more likely. Two separate mechanisms keep the update safe, and they should not be confused: PPO's **clipping** bounds each update relative to the policy that generated the current batch ($\pi_{\theta_{\text{old}}}$); a **KL penalty** added to the reward, $-\beta\,\text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$, keeps the model close to the original supervised fine-tuned *reference* model across the whole of training, so it does not drift into text that games the reward model. Module 6, Lesson 6.3 shows how DPO reaches the same preference objective without training a separate reward model or running PPO.

## Cross-References

- **Lesson 4.8** introduced gradient descent and loss functions. RL uses the same optimisation but with rewards instead of labels.
- **Lesson 5.4** built the transformer that RLHF fine-tunes; its output layer is the categorical policy over tokens.
- **Module 6, Lesson 6.3** covers DPO, which optimises the preference objective directly without a separate reward model or PPO loop, and GRPO, a PPO-style method that replaces the learned critic with group-relative advantages.

## Reflection

You should now be able to:

- Write the Bellman expectation and optimality equations and say which algorithm uses which.
- Implement DQN with experience replay, a target network, and correct `terminated`/`truncated` handling.
- Explain PPO's clipped objective, including when the clip does and does not remove the gradient, and implement PPO for a continuous action problem.
- Describe how A2C, DDPG and SAC differ (on/off-policy, deterministic/stochastic actor, entropy bonus).
- Create and check custom Gymnasium environments, and benchmark a learned policy against simple heuristics.
- Articulate the PPO-to-RLHF connection, keeping PPO clipping and the KL-to-reference penalty distinct.

---

# Chapter Summary

Module 5 covered the major deep learning architectures and two ways of putting them to work (transfer learning and reinforcement learning). Each exploits a different structural assumption about the data:

| Architecture      | Assumption              | Data Type               | Lesson |
| ----------------- | ----------------------- | ----------------------- | ------ |
| Autoencoder       | Compression             | Any (unsupervised)      | 5.1    |
| CNN               | Spatial locality        | Images, grids           | 5.2    |
| RNN/LSTM          | Temporal dependency     | Sequences, time series  | 5.3    |
| Transformer       | Any-to-any attention    | Sequences, text, images | 5.4    |
| GAN               | Adversarial competition | Generation              | 5.5    |
| GNN               | Graph structure         | Networks, molecules     | 5.6    |
| Transfer Learning | Reuse                   | Small datasets          | 5.7    |
| RL                | Interaction             | Decision-making         | 5.8    |

The common thread: every architecture learns features from data via backpropagation, using the DL training toolkit from Lesson 4.8. What changes is the structural bias — convolutions for spatial patterns, recurrence for temporal patterns, attention for relevance patterns, message passing for graph patterns.

## What Module 6 builds on

Module 6 is the capstone. It assumes you can:

- Fine-tune pre-trained models (BERT, ResNet) for new tasks.
- Implement and train any architecture from this chapter.
- Export models with OnnxBridge and serve them with InferenceServer.
- Explain how RL connects to LLM alignment.

Module 6 will take you from trained models to production LLM applications: prompt engineering, fine-tuning with LoRA, preference alignment with DPO, RAG systems, AI agents with ReAct, multi-agent orchestration, AI governance with PACT, and full production deployment with Nexus.

---

# Glossary

**A2C.** Advantage Actor-Critic. An on-policy RL algorithm with a policy (actor) and a value baseline (critic).

**Adapter.** A small bottleneck module inserted into a frozen pre-trained network and trained for a new task; zero-initialised so it starts as the identity.

**Advantage.** How much better an action was than the state's average, $A(s, a) = Q(s, a) - V(s)$.

**Attention.** A mechanism where each element in a sequence computes a weighted combination of all other elements, with weights based on relevance.

**Autoencoder.** A neural network that learns compressed representations by reconstructing its input through a bottleneck.

**Backpropagation Through Time (BPTT).** The application of backpropagation to recurrent networks by unrolling the computation graph through time steps.

**Batch normalisation.** Normalising layer inputs within each mini-batch to stabilise training.

**Bellman equation.** A recursive equation defining the value of a state (or state-action pair) as the immediate reward plus the discounted value of what follows. The _expectation_ form describes a given policy; the _optimality_ form (with a max over next actions) defines $Q^*$.

**BERT.** Bidirectional Encoder Representations from Transformers. A pre-trained transformer encoder for NLU tasks.

**Cell state.** The memory component of an LSTM that flows through time with only additive modifications.

**CNN.** Convolutional Neural Network. Processes grid-structured data using learned filters.

**Convolution.** A template-matching operation that slides a filter across an input, producing a feature map.

**Cosine annealing.** A learning rate schedule that follows a cosine curve.

**CycleGAN.** A GAN for unpaired image-to-image translation, trained with a cycle-consistency loss.

**DCGAN.** Deep Convolutional GAN. Uses strided and transposed convolutions in the discriminator and generator.

**DDPG.** Deep Deterministic Policy Gradient. An off-policy actor-critic for continuous actions with a deterministic actor.

**Decoder.** The component of an autoencoder or transformer that maps from latent space to output space.

**Diffusion model.** A generative model that learns to reverse a gradual noising process.

**Discount factor.** The weight $\gamma$ applied to future rewards in RL, controlling how much the agent values long-term versus immediate rewards.

**DQN.** Deep Q-Network. Approximates the Q-function with a neural network for discrete action RL.

**ELBO.** Evidence Lower BOund. The objective function for VAE training, consisting of a reconstruction term and a KL divergence term.

**Encoder.** The component that maps from input space to latent space.

**Experience replay.** Storing past transitions in a buffer and sampling from them for training, decorrelating sequential samples.

**Feature map.** The output of a convolutional filter applied to an input.

**FID.** Fréchet Inception Distance. A distribution-level metric comparing feature statistics of real and generated images (fidelity and diversity); it does not score individual images and does not measure privacy.

**Filter.** A small learnable matrix used in convolution to detect local patterns.

**Fine-tuning.** Adapting a pre-trained model to a new task by training on task-specific data.

**Forget gate.** The LSTM gate that controls what information to discard from the cell state.

**GAN.** Generative Adversarial Network. A generator and discriminator trained adversarially.

**GAT.** Graph Attention Network. A GNN that uses attention weights between neighbours.

**GCN.** Graph Convolutional Network. A GNN that aggregates neighbour features using the normalised adjacency matrix.

**GIN.** Graph Isomorphism Network. A GNN with sum aggregation and an MLP update, as expressive as the Weisfeiler–Lehman test.

**GraphSAGE.** An inductive GNN that samples a fixed number of neighbours and learns how to aggregate them.

**GELU.** Gaussian Error Linear Unit. Activation function used in BERT- and GPT-style transformers (PyTorch's `nn.TransformerEncoderLayer` defaults to ReLU unless you pass `activation="gelu"`).

**GPT.** Generative Pre-trained Transformer. An autoregressive decoder for text generation.

**Gradient penalty.** A regularisation term in WGAN that enforces the Lipschitz constraint on the critic.

**GRU.** Gated Recurrent Unit. A simplified RNN with update and reset gates.

**Hidden state.** The internal memory of an RNN at each time step.

**InferenceServer.** Kailash ML engine that loads a registered model version from the ModelRegistry (`from_registry`) and serves predictions with `await predict({"records": [...]})`.

**Inception Score (IS).** $\exp(\mathbb{E}_x \text{KL}(p(y \mid x) \| p(y)))$: high when generated images are classified confidently and their classes are diverse.

**Input gate.** The LSTM gate that controls what new information to store.

**Kaiming (He) initialisation.** Weight variance $2/n_{\text{in}}$, which keeps activation variance stable through ReLU layers.

**Label smoothing.** Training against targets of $1 - \varepsilon$ for the true class and $\varepsilon / K$ spread over all classes, to discourage over-confidence.

**Key.** One of the three projections (Q, K, V) in attention, representing what each element contains.

**Latent space.** The low-dimensional space learned by an autoencoder or VAE.

**Layer normalisation.** Normalising across the feature dimension for each sample, used in transformers.

**LSTM.** Long Short-Term Memory. An RNN variant with three gates and a cell state that greatly reduce (but do not abolish) vanishing gradients.

**Message passing.** The GNN mechanism where nodes exchange information along edges.

**Mixed precision.** Using FP16 for computation and FP32 for weight updates.

**Mixup.** Data augmentation by linearly interpolating training examples.

**Mode collapse.** A GAN failure where the generator produces only a limited variety of outputs.

**Multi-head attention.** Running multiple attention operations in parallel with different projections.

**OnnxBridge.** Kailash ML engine for exporting models to ONNX (`export(model, "torch", output_path=..., sample_input=...)`) and checking parity with the native model (`validate`).

**Output gate.** The LSTM gate that controls what to expose from the cell state.

**Over-smoothing.** The convergence of node representations as GNN layers are stacked, which limits useful GNN depth.

**Permutation equivariance.** Shuffling the inputs shuffles the outputs the same way; self-attention without positional encoding has this property.

**Perplexity.** A measure of language model quality; lower is better.

**Policy.** In RL, a mapping from states to actions.

**Positional encoding.** Sinusoidal or learned embeddings added to transformer inputs to provide position information.

**PPO.** Proximal Policy Optimization. An RL algorithm with clipped objectives for stable training.

**Q-function.** The expected cumulative reward for taking action $a$ in state $s$ and following the policy thereafter.

**Query.** One of the three projections (Q, K, V) in attention, representing what each element is looking for.

**Reparameterisation trick.** Expressing a random sample as a deterministic function of the mean, variance, and independent noise, enabling gradient flow.

**Residual connection.** A skip connection $H(x) = F(x) + x$ that facilitates gradient flow in deep networks.

**ResNet.** Residual Network. A CNN architecture with skip connections enabling very deep networks.

**Reward.** In RL, the scalar feedback signal the agent receives after taking an action.

**RLHF.** Reinforcement Learning from Human Feedback. Using RL to align LLMs with human preferences.

**RNN.** Recurrent Neural Network. Processes sequences by maintaining a hidden state across time steps.

**SE block.** Squeeze-and-Excitation block. Channel recalibration mechanism for CNNs.

**SAC.** Soft Actor-Critic. An off-policy, maximum-entropy actor-critic for continuous actions.

**Self-attention.** Attention where queries, keys, and values all come from the same sequence.

**Skip connection.** A shortcut that adds the input of a layer directly to its output.

**StyleGAN.** A GAN whose generator is controlled per resolution by a learned "style" vector, giving high-resolution, controllable images.

**Stride.** The step size of a convolutional filter as it slides across the input.

**Transfer learning.** Reusing a pre-trained model's features for a new task.

**Transformer.** An architecture based entirely on attention, processing all positions in parallel.

**Transposed convolution.** An upsampling operation used in decoders to increase spatial dimensions.

**Value.** One of the three projections (Q, K, V) in attention, representing the information to contribute.

**Value function.** In RL, the expected cumulative reward from a state.

**VAE.** Variational Autoencoder. An autoencoder with a probabilistic latent space, enabling generation.

**ViT.** Vision Transformer. A transformer encoder applied to a sequence of image-patch tokens.

**Wasserstein distance.** A distance metric between probability distributions used in WGAN.

**WGAN.** Wasserstein GAN. Uses Wasserstein distance for more stable training.

---

# Further Reading

**On autoencoders and VAEs**

- Kingma, D., and Welling, M. "Auto-Encoding Variational Bayes." _ICLR_, 2014. The original VAE paper.
- Doersch, C. "Tutorial on Variational Autoencoders." _arXiv:1606.05908_, 2016.
- Rifai, S., et al. "Contractive Auto-Encoders: Explicit Invariance During Feature Extraction." _ICML_, 2011.
- Higgins, I., et al. "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." _ICLR_, 2017.

**On CNNs**

- He, K., et al. "Deep Residual Learning for Image Recognition." _CVPR_, 2016. The ResNet paper.
- Hu, J., Shen, L., and Sun, G. "Squeeze-and-Excitation Networks." _CVPR_, 2018. The SE block paper.
- He, K., et al. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." _ICCV_, 2015. Kaiming initialisation.
- Zhang, H., et al. "mixup: Beyond Empirical Risk Minimization." _ICLR_, 2018.
- Szegedy, C., et al. "Rethinking the Inception Architecture for Computer Vision." _CVPR_, 2016. Introduces label smoothing.
- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words." _ICLR_, 2021. The Vision Transformer paper.

**On RNNs and LSTMs**

- Hochreiter, S., and Schmidhuber, J. "Long Short-Term Memory." _Neural Computation_, 1997. The original LSTM paper.
- Cho, K., et al. "Learning Phrase Representations using RNN Encoder-Decoder." _EMNLP_, 2014. The GRU paper.

**On transformers**

- Vaswani, A., et al. "Attention Is All You Need." _NeurIPS_, 2017. The original transformer paper.
- Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers." _NAACL_, 2019.
- Radford, A., et al. "Language Models are Unsupervised Multitask Learners." OpenAI, 2019. The GPT-2 paper.
- Dai, Z., et al. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context." _ACL_, 2019.
- Beltagy, I., Peters, M., and Cohan, A. "Longformer: The Long-Document Transformer." _arXiv:2004.05150_, 2020.

**On GANs and generative models**

- Goodfellow, I., et al. "Generative Adversarial Nets." _NeurIPS_, 2014. The original GAN paper.
- Arjovsky, M., Chintala, S., and Bottou, L. "Wasserstein GAN." _ICML_, 2017.
- Gulrajani, I., et al. "Improved Training of Wasserstein GANs." _NeurIPS_, 2017. WGAN-GP.
- Ho, J., Jain, A., and Abbeel, P. "Denoising Diffusion Probabilistic Models." _NeurIPS_, 2020.
- Radford, A., Metz, L., and Chintala, S. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." _ICLR_, 2016. DCGAN.
- Zhu, J.-Y., et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." _ICCV_, 2017. CycleGAN.
- Karras, T., Laine, S., and Aila, T. "A Style-Based Generator Architecture for Generative Adversarial Networks." _CVPR_, 2019. StyleGAN.
- Salimans, T., et al. "Improved Techniques for Training GANs." _NeurIPS_, 2016. Inception Score.
- Heusel, M., et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." _NeurIPS_, 2017. FID.
- Carlini, N., et al. "Extracting Training Data from Diffusion Models." _USENIX Security_, 2023.

**On GNNs**

- Kipf, T., and Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks." _ICLR_, 2017.
- Hamilton, W., Ying, R., and Leskovec, J. "Inductive Representation Learning on Large Graphs." _NeurIPS_, 2017. GraphSAGE.
- Velickovic, P., et al. "Graph Attention Networks." _ICLR_, 2018. GAT.
- Xu, K., et al. "How Powerful are Graph Neural Networks?" _ICLR_, 2019. GIN.

**On reinforcement learning**

- Sutton, R., and Barto, A. _Reinforcement Learning: An Introduction._ MIT Press, 2018. The definitive textbook. Free online at `incompleteideas.net/book/the-book.html`.
- Mnih, V., et al. "Human-level control through deep reinforcement learning." _Nature_, 2015. DQN.
- Schulman, J., et al. "Proximal Policy Optimization Algorithms." _arXiv:1707.06347_, 2017.
- Schulman, J., et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." _ICLR_, 2016. GAE.
- Mnih, V., et al. "Asynchronous Methods for Deep Reinforcement Learning." _ICML_, 2016. A3C/A2C.
- Lillicrap, T., et al. "Continuous Control with Deep Reinforcement Learning." _ICLR_, 2016. DDPG.
- Haarnoja, T., et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." _ICML_, 2018.
- Ouyang, L., et al. "Training Language Models to Follow Instructions with Human Feedback." _NeurIPS_, 2022. RLHF with PPO and a KL penalty.

**On transfer learning**

- Zhuang, F., et al. "A Comprehensive Survey on Transfer Learning." _Proceedings of the IEEE_, 109(1), 2021.
- Houlsby, N., et al. "Parameter-Efficient Transfer Learning for NLP." _ICML_, 2019. Adapter modules.
- Wolf, T., et al. "Transformers: State-of-the-Art Natural Language Processing." _EMNLP (System Demonstrations)_, 2020. The HuggingFace library.

---
