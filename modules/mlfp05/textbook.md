# Module 5 — Deep Learning: Architectures for Vision, Sequence, and Generation

> _"Every architecture is a hypothesis about the structure of the world."_

This chapter is where the training toolkit from Lesson 4.8 meets specialised neural architectures. In Module 4 you built a feedforward network from scratch and understood that hidden layers are automated feature engineering with error feedback. Now you will see how different architectures impose different structural biases on that feature learning — biases that make learning dramatically more efficient for specific data types.

A convolutional neural network assumes spatial locality: nearby pixels are more related than distant ones. A recurrent neural network assumes temporal dependency: the meaning of a word depends on the words before it. A transformer assumes that any element can attend to any other element, weighted by relevance. A graph neural network assumes that information flows along edges. Each assumption is a hypothesis about the data's structure, and when the hypothesis is correct, the architecture learns faster and generalises better than a generic feedforward network.

Every architecture in this chapter is implemented in PyTorch. You will write `nn.Module` subclasses, define `forward()` methods, configure `torch.optim` optimisers, and train with gradient descent. The DL training toolkit — dropout, batch normalisation, learning rate scheduling, gradient clipping, early stopping — applies uniformly across all architectures. What changes is the architecture; what remains constant is the training methodology.

By the end of this chapter you will have implemented autoencoders, CNNs, RNNs, transformers, GANs, GNNs, and RL agents. You will know which architecture to use for which data type, how to transfer pre-trained models to new tasks, and how to export models for production deployment.

---

## Learning Outcomes

By the end of this chapter you will be able to:

- Build and train autoencoders (vanilla, denoising, variational, convolutional) and generate new data from VAE latent spaces by deriving the ELBO and reparameterisation trick.
- Implement CNNs with modern enhancements (ResNet skip connections, SE blocks, mixed precision training, Mixup augmentation) and explain the convolution output size formula.
- Build LSTM and GRU networks, write all six LSTM gate equations, apply temporal attention, and train sequence models for time-series prediction and text generation.
- Derive scaled dot-product self-attention from scratch, explain the $\sqrt{d_k}$ normalisation, implement multi-head attention, and fine-tune pre-trained BERT for downstream tasks.
- Implement DCGAN and WGAN with gradient penalty, explain mode collapse and how Wasserstein distance addresses it, and evaluate generative quality with FID.
- Build graph convolutional networks for node and graph classification, implement message passing, and use torch_geometric.
- Fine-tune pre-trained vision and NLP models using transfer learning, export models to ONNX, and deploy with InferenceServer.
- Implement DQN and PPO reinforcement learning algorithms, create custom Gymnasium environments, and explain how RL connects to RLHF for LLM alignment.

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

The Long Short-Term Memory network introduces a cell state $\mathbf{C}_t$ — a highway for information that flows through the sequence with only additive modifications, avoiding the vanishing gradient problem.

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

The cell state $\mathbf{C}_t$ flows through time with only element-wise operations (multiply by forget gate, add input), which means the gradient of $\mathbf{C}_T$ with respect to $\mathbf{C}_1$ is a product of forget-gate values — not a product of weight matrices. Since forget-gate values are between 0 and 1, the gradient is bounded, and information can persist over long sequences.

**GRU** (Gated Recurrent Unit) simplifies LSTM to two gates (update and reset), using fewer parameters:

$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

### THEORY: Perplexity

Perplexity measures how well a language model predicts a sequence. For a sequence of $N$ words:

$$\text{PP} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i \mid w_1, \ldots, w_{i-1})\right)$$

Lower perplexity means the model is less "perplexed" by the data — it assigns higher probability to the observed words. A perplexity of 1 means perfect prediction; a perplexity of $V$ (vocabulary size) means random guessing.

### FOUNDATIONS: Temporal attention

Attention allows the model to focus on specific time steps when making a prediction. For each output step, an attention weight is computed over all input hidden states:

$$\alpha_t = \text{softmax}(\mathbf{h}_t^T \mathbf{H})$$
$$\mathbf{c}_t = \alpha_t \mathbf{H}$$

where $\mathbf{H}$ is the matrix of all hidden states and $\mathbf{c}_t$ is the context vector. This mechanism is the precursor to the full self-attention of transformers (Lesson 5.4).

## The Kailash Engine: ModelVisualizer (training curves)

```python
from kailash_ml import ModelVisualizer

viz = ModelVisualizer()
fig = viz.line(training_df, x="epoch", y=["train_loss", "val_loss"],
               title="LSTM Training Curves")
```

## Worked Example: Singapore Stock Price Prediction with LSTM

```python
class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.2)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        # Temporal attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)
        return self.fc(context)

model = StockLSTM(input_dim=8, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

## Try It Yourself

**Drill 1.** Implement a character-level LSTM for text generation. Train on a corpus of Singapore Straits Times headlines. Generate 10 new headlines by sampling from the model's output distribution.

**Solution:**

```python
# Build character vocabulary, convert text to sequences of indices
# Train LSTM to predict next character given previous characters
# Generate by repeatedly sampling and feeding back
```

**Drill 2.** Compare LSTM and GRU on the stock price prediction task. Which has more parameters? Which converges faster? Which achieves lower test MSE?

**Solution:**

```python
lstm_params = sum(p.numel() for p in lstm_model.parameters())
gru_params = sum(p.numel() for p in gru_model.parameters())
print(f"LSTM: {lstm_params:,}, GRU: {gru_params:,}")
```

**Drill 3.** Add gradient clipping with max_norm = 1.0. Monitor the gradient norm during training. How often does clipping activate? Does it improve final performance?

**Solution:**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Drill 4.** Visualise attention weights for a specific prediction. Which time steps does the model attend to most? Do the attention patterns make sense for stock price prediction (e.g., attending to recent days more than distant ones)?

**Solution:**

```python
with torch.no_grad():
    lstm_out, _ = model.lstm(sample_input)
    weights = torch.softmax(model.attention(lstm_out), dim=1).squeeze()
    # Plot weights over time steps
```

**Drill 5.** Implement a multi-layer LSTM with residual connections between layers. Compare with a standard multi-layer LSTM. Does the residual connection improve training on longer sequences (length 100+)?

**Solution:**

```python
class ResidualLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dim = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.LSTM(dim, hidden_dim, batch_first=True))

    def forward(self, x):
        for i, lstm in enumerate(self.layers):
            out, _ = lstm(x if i == 0 else h)
            h = out + h if i > 0 and out.shape == h.shape else out
        return h
```

## Cross-References

- **Lesson 4.8** introduced backpropagation through layers. BPTT (Backpropagation Through Time) is the same algorithm unrolled through time steps.
- **Lesson 5.2** used CNNs for spatial data. RNNs handle temporal data. The two can be combined (ConvLSTM) for spatiotemporal data.
- **Lesson 5.4** will replace the RNN's sequential processing with parallel self-attention. The attention mechanism you just learned is the conceptual ancestor of the transformer.

## Reflection

You should now be able to:

- Write all six LSTM gate equations from memory and explain each gate's role.
- Explain why the cell state highway solves the vanishing gradient problem.
- Compare LSTM and GRU in terms of complexity and performance.
- Implement temporal attention and visualise attention weights.
- Train RNNs for time-series prediction and text generation.

---

# Lesson 5.4: Transformers

## Why This Matters

The transformer is the architecture behind GPT, BERT, Claude, and every major language model since 2017. It replaced RNNs for most sequence tasks because it processes all positions in parallel (instead of sequentially) and captures long-range dependencies through attention (instead of hoping information persists through gates).

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

Each head operates in a lower-dimensional subspace ($d_k/h$ per head), so the total computation is the same as a single head with full dimensionality.

```python
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

Transformers have no inherent sense of position — unlike RNNs, which process sequentially. Positional encodings are added to the input embeddings to provide position information:

$$\text{PE}(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$
$$\text{PE}(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

The sinusoidal encoding allows the model to attend to relative positions (the dot product between positions $p$ and $p+k$ is a function of $k$ only). Learned positional embeddings are an alternative and often perform similarly.

### FOUNDATIONS: Layer normalisation

Transformers use layer normalisation instead of batch normalisation because sequence lengths vary within a batch. Layer normalisation normalises across the feature dimension for each individual sample:

$$\hat{z}_i = \frac{z_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

where $\mu$ and $\sigma^2$ are computed over the feature dimension, not the batch dimension.

### FOUNDATIONS: Transformer variants

| Model    | Architecture    | Pre-training              | Strength                            |
| -------- | --------------- | ------------------------- | ----------------------------------- |
| **BERT** | Encoder only    | Masked language modelling | NLU tasks (classification, NER, QA) |
| **GPT**  | Decoder only    | Next token prediction     | Generation, few-shot learning       |
| **T5**   | Encoder-decoder | Text-to-text              | Unified framework for all NLP tasks |

## The Kailash Engine: ModelVisualizer (attention visualisation)

```python
from kailash_ml import ModelVisualizer
viz = ModelVisualizer()
fig = viz.heatmap(attention_weights, title="Self-Attention Weights")
```

## Worked Example: Self-Attention from Scratch and BERT Fine-Tuning

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

print(f"Scores shape: {scores.shape}")       # (1, 10, 10)
print(f"Attention shape: {attn_weights.shape}") # (1, 10, 10)
print(f"Output shape: {output.shape}")        # (1, 10, 64)

# Part B: BERT fine-tuning for text classification
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

# Freeze all layers except the classifier head
for param in model.bert.parameters():
    param.requires_grad = False

optimizer = optim.AdamW(model.classifier.parameters(), lr=2e-5)
```

## Try It Yourself

**Drill 1.** Implement scaled dot-product attention from scratch (no PyTorch modules, just matrix operations). Verify the output shape is correct for batch size 4, sequence length 20, and dimension 128.

**Solution:**

```python
B, L, D = 4, 20, 128
x = torch.randn(B, L, D)
Q = x @ torch.randn(D, D); K = x @ torch.randn(D, D); V = x @ torch.randn(D, D)
scores = Q @ K.transpose(-2, -1) / (D ** 0.5)
attn = torch.softmax(scores, dim=-1)
out = attn @ V
assert out.shape == (B, L, D)
```

**Drill 2.** Demonstrate the $\sqrt{d_k}$ effect empirically. Compute attention weights with and without scaling for $d_k = 512$. Show that without scaling, the attention distribution is peakier (higher max, lower entropy).

**Solution:**

```python
Q, K = torch.randn(1, 10, 512), torch.randn(1, 10, 512)
scores_unscaled = Q @ K.transpose(-2, -1)
scores_scaled = scores_unscaled / (512 ** 0.5)

attn_unscaled = torch.softmax(scores_unscaled, dim=-1)
attn_scaled = torch.softmax(scores_scaled, dim=-1)

print(f"Unscaled max attn: {attn_unscaled.max():.4f}")
print(f"Scaled max attn: {attn_scaled.max():.4f}")
```

**Drill 3.** Fine-tune BERT for text classification on a small dataset (e.g., 1000 labelled sentences). Compare accuracy when (a) only the classifier head is trained, (b) the last 2 transformer layers are also unfrozen.

**Solution:**

```python
# (a) Freeze all BERT layers, train classifier only
# (b) Unfreeze last 2 layers:
for param in model.bert.encoder.layer[-2:].parameters():
    param.requires_grad = True
```

**Drill 4.** Compare the BERT fine-tuned model with the LSTM baseline from Lesson 5.3 on the same text classification task. Report accuracy and training time.

**Solution:**

```python
# Run both models on same train/test split, measure accuracy and wall time
```

**Drill 5.** Implement sinusoidal positional encoding from scratch. Visualise the encoding matrix as a heatmap. Verify that the dot product between position $p$ and $p+k$ is a function of $k$ only (compute for several values of $p$ and $k$).

**Solution:**

```python
def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe
```

## Cross-References

- **Lesson 5.3** introduced attention as a mechanism on top of RNNs. Self-attention removes the RNN entirely.
- **Module 4, Lesson 4.6** used word embeddings as features. Transformers compute contextualised embeddings — the same word gets different representations depending on context.
- **Lesson 5.7** applies transfer learning with pre-trained transformers (BERT, GPT).
- **Module 6** builds extensively on transformers: LLM fundamentals (6.1), fine-tuning (6.2), and RAG (6.4).

## Reflection

You should now be able to:

- Derive scaled dot-product attention from first principles.
- Explain why dividing by $\sqrt{d_k}$ is necessary (prevent softmax saturation).
- Implement multi-head attention in PyTorch.
- Fine-tune BERT for a downstream classification task.
- Compare BERT, GPT, and T5 and know when each is appropriate.

---

# Lesson 5.5: Generative Models — GANs and Diffusion

## Why This Matters

Autoencoders (Lesson 5.1) generate data by sampling from a latent space, but the samples are often blurry. GANs produce sharper, more realistic outputs through adversarial training — a generator tries to fool a discriminator that tries to distinguish real from fake. The tension between these two networks drives the generator to produce increasingly realistic data.

## Core Concepts

### THEORY: GAN minimax objective

The GAN training objective is a minimax game:

$$\min_G \max_D \left[ \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))] \right]$$

The discriminator $D$ maximises the objective by correctly classifying real data as real ($D(\mathbf{x}) \to 1$) and generated data as fake ($D(G(\mathbf{z})) \to 0$). The generator $G$ minimises the objective by producing data that the discriminator classifies as real ($D(G(\mathbf{z})) \to 1$).

At the Nash equilibrium, the generator produces data indistinguishable from real data, and the discriminator outputs 0.5 for everything. In practice, training oscillates and rarely reaches the true equilibrium.

### FOUNDATIONS: Mode collapse

Mode collapse occurs when the generator learns to produce only a few types of outputs that fool the discriminator, ignoring the full diversity of the training data. For instance, a GAN trained on MNIST might generate only the digit 1 — the discriminator cannot tell these apart from real 1s, but the generator has stopped producing any other digit.

### THEORY: WGAN and gradient penalty

The Wasserstein GAN (WGAN) replaces the JS divergence (implicit in the original GAN) with the Wasserstein (Earth Mover's) distance:

$$\min_G \max_{D \in \text{1-Lip}} \left[ \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[D(\mathbf{x})] - \mathbb{E}_{\mathbf{z} \sim p_z}[D(G(\mathbf{z}))] \right]$$

where the discriminator (now called a critic) must be 1-Lipschitz. The Wasserstein distance provides a meaningful gradient even when the distributions do not overlap, which is why WGAN training is more stable.

**Gradient penalty** enforces the Lipschitz constraint by penalising the gradient norm of the critic:

$$\mathcal{L}_{\text{GP}} = \lambda \, \mathbb{E}_{\hat{\mathbf{x}}}[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2]$$

where $\hat{\mathbf{x}}$ is a random interpolation between a real and a generated sample.

```python
class WGAN_GP(nn.Module):
    def gradient_penalty(self, real, fake, critic):
        alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interp = critic(interp)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
        )[0]
        grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean()
```

### FOUNDATIONS: FID (Frechet Inception Distance)

FID measures the quality and diversity of generated images by comparing the distribution of real and generated image features (extracted by a pre-trained InceptionV3 network):

$$\text{FID} = \|\boldsymbol{\mu}_r - \boldsymbol{\mu}_g\|^2 + \text{Tr}(\boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g - 2(\boldsymbol{\Sigma}_r \boldsymbol{\Sigma}_g)^{1/2})$$

Lower FID means the generated distribution is closer to the real distribution. FID captures both quality (mean) and diversity (covariance).

### ADVANCED: Diffusion models

Diffusion models (DDPM — Denoising Diffusion Probabilistic Models) add noise to data progressively over $T$ steps, then learn to reverse the process:

- **Forward process:** gradually add Gaussian noise until the data is pure noise.
- **Reverse process:** a neural network learns to denoise step by step.

Diffusion models produce higher-quality and more diverse samples than GANs, at the cost of slower generation (requires many denoising steps). Stable Diffusion and DALL-E are based on diffusion models.

## Worked Example: DCGAN and WGAN on Fashion-MNIST

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 784), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
```

## Try It Yourself

**Drill 1.** Implement the full DCGAN training loop with alternating generator and discriminator updates. Train for 50 epochs and visualise generated images at epochs 1, 10, 25, and 50.

**Solution:**

```python
G = Generator(); D = Discriminator()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(50):
    for real, _ in train_loader:
        # Train D
        z = torch.randn(real.size(0), 100)
        fake = G(z).detach()
        loss_D = criterion(D(real), torch.ones(real.size(0), 1)) + \
                 criterion(D(fake), torch.zeros(real.size(0), 1))
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # Train G
        z = torch.randn(real.size(0), 100)
        fake = G(z)
        loss_G = criterion(D(fake), torch.ones(real.size(0), 1))
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()
```

**Drill 2.** Implement WGAN with gradient penalty. Compare training stability with the original DCGAN (plot discriminator and generator losses over epochs).

**Solution:**

```python
# WGAN critic loss: D(real).mean() - D(fake).mean() + gp
# WGAN generator loss: -D(fake).mean()
```

**Drill 3.** Compute FID between generated and real Fashion-MNIST images. How does FID change over training epochs?

**Solution:**

```python
from pytorch_fid import fid_score
# Save real and generated images to directories
# fid = fid_score.calculate_fid_given_paths([real_dir, gen_dir], batch_size=64, device="cpu", dims=2048)
```

**Drill 4.** Implement conditional generation: given a class label, generate an image of that class. Modify the generator to take both $z$ and a one-hot class label as input.

**Solution:**

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Tanh(),
        )

    def forward(self, z, label_onehot):
        return self.net(torch.cat([z, label_onehot], dim=1)).view(-1, 1, 28, 28)
```

**Drill 5.** Create a comparison table: VAE vs DCGAN vs WGAN. For each, report training stability, sample quality (FID), sample diversity, and training time. Which would you use for synthetic data generation in a production setting?

**Solution:** WGAN-GP offers the best balance of stability and quality. VAE produces more diverse but blurrier samples. DCGAN is fast but prone to mode collapse. For production synthetic data, WGAN-GP or diffusion models are preferred.

## Cross-References

- **Lesson 5.1** introduced VAEs for generation. GANs produce sharper samples; VAEs produce more diverse samples.
- **Module 6, Lesson 6.3** uses preference alignment (DPO) — a different approach to steering generative models.

## Reflection

You should now be able to:

- Write the GAN minimax objective and explain the generator-discriminator dynamic.
- Implement DCGAN and WGAN with gradient penalty.
- Explain mode collapse and how Wasserstein distance addresses it.
- Evaluate generative quality with FID.
- Compare VAE, GAN, and diffusion models for different generation tasks.

---

# Lesson 5.6: Graph Neural Networks

## Why This Matters

Social networks, molecular structures, supply chains, and knowledge graphs are naturally represented as graphs — nodes connected by edges. Standard neural networks cannot process graph-structured data directly. GNNs operate by message passing: each node aggregates information from its neighbours, then updates its representation. After several rounds of message passing, each node's representation captures information from its local neighbourhood.

## Core Concepts

### THEORY: GCN propagation rule

The Graph Convolutional Network (GCN) layer updates node features using the normalised adjacency matrix:

$$\mathbf{H}^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$

where $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ (adjacency matrix with self-loops), $\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{\mathbf{A}}_{ij}$ (degree matrix), $\mathbf{H}^{(l)}$ is the feature matrix at layer $l$, and $\mathbf{W}^{(l)}$ is the learnable weight matrix.

The normalisation $\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2}$ ensures that the aggregated features are averaged (not summed), preventing high-degree nodes from dominating.

### FOUNDATIONS: Message passing

The GCN propagation can be viewed as message passing:

1. **Message:** each node sends its current representation to all neighbours.
2. **Aggregate:** each node averages the messages from its neighbours (and itself, via the self-loop).
3. **Update:** the aggregated message is transformed by a linear layer and activation function.

After $L$ layers of message passing, each node's representation captures information from nodes up to $L$ hops away.

### THEORY: GAT attention weights

Graph Attention Networks (GAT) compute attention weights between neighbours:

$$e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j])$$
$$\alpha_{ij} = \text{softmax}_j(e_{ij})$$
$$\mathbf{h}_i' = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j\right)$$

This allows the model to learn which neighbours are more important for each node.

```python
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # graph-level readout
        return self.fc(x)
```

## Try It Yourself

**Drill 1.** Build a GCN for graph classification on TUDataset (MUTAG or PROTEINS). Report accuracy.

**Drill 2.** Compare GCN vs GAT on the same dataset. Does attention improve accuracy?

**Drill 3.** Visualise learned node embeddings after 2 layers of GCN. Do nodes of the same class cluster together?

**Drill 4.** Vary the number of GCN layers from 1 to 6. Does over-smoothing occur (all node embeddings become similar)?

**Drill 5.** Implement a simple message-passing network from scratch (without torch_geometric). Verify it produces the same output as GCNConv.

## Cross-References

- **Lesson 4.1** introduced spectral clustering, which uses the graph Laplacian — the same matrix that appears in GCN.
- **Lesson 5.4** introduced attention. GAT applies attention to graph neighbours.

## Reflection

You should now be able to build GCNs and GATs, explain message passing, and use torch_geometric for graph ML.

---

# Lesson 5.7: Transfer Learning

## Why This Matters

Training a model from scratch on a small dataset often leads to overfitting. Transfer learning solves this by starting from a model pre-trained on a large dataset (ImageNet for vision, Wikipedia/BookCorpus for NLP) and fine-tuning it on your small target dataset. The pre-trained model has already learned general features (edges, textures for vision; grammar, semantics for NLP) that transfer to new tasks.

## Core Concepts

### FOUNDATIONS: The transfer learning recipe

1. **Load a pre-trained model** (e.g., ResNet-50 from ImageNet, BERT from BookCorpus).
2. **Replace the classification head** with a new one matching your number of classes.
3. **Freeze early layers** — they contain general features that transfer well.
4. **Fine-tune later layers** and the new head on your target dataset.
5. **Optionally unfreeze more layers** if you have enough data.

### FOUNDATIONS: Architecture selection guide

| Data Type | Best Architecture  | When to Transfer              |
| --------- | ------------------ | ----------------------------- |
| Images    | CNN / ViT          | Always (ImageNet pre-trained) |
| Text      | Transformer        | Always (BERT/GPT pre-trained) |
| Sequences | LSTM / Transformer | Sometimes (domain-specific)   |
| Graphs    | GNN                | Rarely (task-specific)        |
| Tabular   | Gradient boosting  | Never (train from scratch)    |

### FOUNDATIONS: ONNX export and InferenceServer

```python
from kailash_ml import OnnxBridge, InferenceServer

bridge = OnnxBridge()
bridge.export(model, input_shape=(1, 3, 224, 224), output_path="model.onnx")

server = InferenceServer(model_path="model.onnx")
result = server.predict(sample_input)
batch_results = server.predict_batch(sample_batch)
```

## Worked Example: Fine-Tuning ResNet for Image Classification

```python
from torchvision import models

model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
model.fc = nn.Linear(512, 10)  # 10 classes

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# After a few epochs, optionally unfreeze layer4:
for param in model.layer4.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])
```

## Try It Yourself

**Drill 1.** Fine-tune ResNet-18 on Fashion-MNIST. Compare accuracy with the CNN from Lesson 5.2. How many epochs does transfer learning need to match the from-scratch accuracy?

**Drill 2.** Fine-tune BERT for sentiment classification. Compare with the LSTM from Lesson 5.3.

**Drill 3.** Export both fine-tuned models to ONNX. Measure inference latency.

**Drill 4.** Implement progressive unfreezing: start with only the head, then unfreeze one layer at a time every 5 epochs. Does this improve final accuracy?

**Drill 5.** Apply data augmentation (random crop, horizontal flip, colour jitter) to the image dataset. How much does augmentation improve transfer learning performance?

## Cross-References

- **Lesson 5.2** built CNNs from scratch. Transfer learning reuses pre-trained CNNs.
- **Lesson 5.4** introduced BERT. Transfer learning with BERT is the practical application.
- **Module 6, Lesson 6.2** extends transfer learning to LoRA and adapter-based fine-tuning.

## Reflection

You should now be able to fine-tune pre-trained models for new tasks, export to ONNX, and deploy with InferenceServer.

---

# Lesson 5.8: Reinforcement Learning

## Why This Matters

All deep learning so far learns from static datasets — images, text, sequences. Reinforcement learning (RL) learns from interaction with an environment. An agent takes actions, receives rewards, and learns a policy that maximises cumulative reward. RL powers game-playing AI (AlphaGo), robotics, and — crucially — the alignment of large language models (RLHF, which you will study in Module 6).

## Core Concepts

### THEORY: Bellman equations

The value of a state is the expected cumulative reward from that state:

$$V(s) = \mathbb{E}\left[R_{t+1} + \gamma V(S_{t+1}) \mid S_t = s\right]$$

The value of a state-action pair:

$$Q(s, a) = \mathbb{E}\left[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') \mid S_t = s, A_t = a\right]$$

where $\gamma \in [0, 1)$ is the discount factor — future rewards are worth less than immediate ones.

### THEORY: DQN (Deep Q-Network)

DQN approximates $Q(s, a)$ with a neural network $Q(s, a; \theta)$. The loss is:

$$\mathcal{L} = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

where $\theta^-$ is a target network (periodically copied from $\theta$) that stabilises training. DQN handles discrete action spaces.

### THEORY: PPO (Proximal Policy Optimization)

PPO is a policy gradient method with a clipped objective that prevents large policy updates:

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio, $\hat{A}_t$ is the advantage estimate, and $\epsilon$ (typically 0.2) controls how far the new policy can deviate from the old one.

PPO handles continuous action spaces and is the algorithm used in RLHF (Reinforcement Learning from Human Feedback) for LLM alignment.

### FOUNDATIONS: Five algorithms, five applications

| Algorithm | Action Space        | Application               |
| --------- | ------------------- | ------------------------- |
| DQN       | Discrete            | Customer churn prevention |
| DDPG      | Continuous          | Manufacturing control     |
| SAC       | Continuous          | Dynamic pricing           |
| A2C       | Discrete/Continuous | Resource allocation       |
| PPO       | Discrete/Continuous | Supply chain optimisation |

```python
import gymnasium as gym

class ChurnEnv(gym.Env):
    """Custom environment for customer churn prevention."""
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,))
        self.action_space = gym.spaces.Discrete(3)  # no action, discount, call

    def step(self, action):
        # Compute next state, reward based on action effectiveness
        reward = self._compute_reward(action)
        return next_state, reward, done, False, {}

    def reset(self, seed=None):
        return self._initial_state(), {}
```

## Worked Example: DQN for Customer Churn Prevention

```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)

# Training loop with experience replay
from collections import deque
import random

replay_buffer = deque(maxlen=10000)

def train_dqn(env, model, target_model, optimizer, episodes=500):
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < max(0.01, 1.0 - episode / 200):
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(state))
                    action = q_values.argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Sample mini-batch and update
            if len(replay_buffer) >= 64:
                batch = random.sample(replay_buffer, 64)
                # Compute DQN loss and update
```

## Try It Yourself

**Drill 1.** Implement DQN with experience replay on a CartPole environment. How many episodes until convergence?

**Drill 2.** Implement PPO for a continuous control task (e.g., Pendulum-v1). Verify the clipped objective prevents large updates.

**Drill 3.** Create a custom Gymnasium environment for Singapore taxi pricing. Define state (time, location, demand), actions (price multiplier), and reward (revenue minus customer loss).

**Drill 4.** Compare DQN and PPO on the same discrete-action environment. Which converges faster? Which achieves higher final reward?

**Drill 5.** Explain in a paragraph how PPO connects to RLHF for LLM alignment. What is the "environment"? What is the "reward"? What is the "policy"? (This connects to Module 6, Lesson 6.3.)

**Solution:** In RLHF, the LLM is the policy — it takes a prompt (state) and generates a response (action). The reward comes from a reward model trained on human preferences: responses preferred by humans get higher reward. PPO updates the LLM's weights to increase the probability of generating responses that score highly, while the clipping objective prevents the model from deviating too far from its pre-trained behaviour. Module 6, Lesson 6.3 will show how DPO achieves the same goal without the reward model.

## Cross-References

- **Lesson 4.8** introduced gradient descent and loss functions. RL uses the same optimisation but with rewards instead of labels.
- **Module 6, Lesson 6.3** uses DPO and GRPO as alternatives to RLHF, bypassing the reward model.

## Reflection

You should now be able to:

- Write the Bellman equations and explain what they represent.
- Implement DQN with experience replay.
- Explain PPO's clipped objective and why it stabilises training.
- Create custom Gymnasium environments.
- Articulate the PPO-to-RLHF connection.

---

# Chapter Summary

Module 5 covered every major deep learning architecture. You built eight types of neural networks, each exploiting a different structural assumption about the data:

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
- Export models for deployment with ONNX.
- Explain how RL connects to LLM alignment.

Module 6 will take you from trained models to production LLM applications: prompt engineering, fine-tuning with LoRA, preference alignment with DPO, RAG systems, AI agents with ReAct, multi-agent orchestration, AI governance with PACT, and full production deployment with Nexus.

---

# Glossary

**Attention.** A mechanism where each element in a sequence computes a weighted combination of all other elements, with weights based on relevance.

**Autoencoder.** A neural network that learns compressed representations by reconstructing its input through a bottleneck.

**Backpropagation Through Time (BPTT).** The application of backpropagation to recurrent networks by unrolling the computation graph through time steps.

**Batch normalisation.** Normalising layer inputs within each mini-batch to stabilise training.

**Bellman equation.** A recursive equation defining the value of a state as the immediate reward plus the discounted value of the next state.

**BERT.** Bidirectional Encoder Representations from Transformers. A pre-trained transformer encoder for NLU tasks.

**Cell state.** The memory component of an LSTM that flows through time with only additive modifications.

**CNN.** Convolutional Neural Network. Processes grid-structured data using learned filters.

**Convolution.** A template-matching operation that slides a filter across an input, producing a feature map.

**Cosine annealing.** A learning rate schedule that follows a cosine curve.

**DCGAN.** Deep Convolutional GAN. Uses convolutional layers in both generator and discriminator.

**Decoder.** The component of an autoencoder or transformer that maps from latent space to output space.

**Diffusion model.** A generative model that learns to reverse a gradual noising process.

**Discount factor.** The weight $\gamma$ applied to future rewards in RL, controlling how much the agent values long-term versus immediate rewards.

**DQN.** Deep Q-Network. Approximates the Q-function with a neural network for discrete action RL.

**ELBO.** Evidence Lower BOund. The objective function for VAE training, consisting of a reconstruction term and a KL divergence term.

**Encoder.** The component that maps from input space to latent space.

**Experience replay.** Storing past transitions in a buffer and sampling from them for training, decorrelating sequential samples.

**Feature map.** The output of a convolutional filter applied to an input.

**FID.** Frechet Inception Distance. A metric for evaluating generated image quality and diversity.

**Filter.** A small learnable matrix used in convolution to detect local patterns.

**Fine-tuning.** Adapting a pre-trained model to a new task by training on task-specific data.

**Forget gate.** The LSTM gate that controls what information to discard from the cell state.

**GAN.** Generative Adversarial Network. A generator and discriminator trained adversarially.

**GAT.** Graph Attention Network. A GNN that uses attention weights between neighbours.

**GCN.** Graph Convolutional Network. A GNN that aggregates neighbour features using the normalised adjacency matrix.

**GELU.** Gaussian Error Linear Unit. Activation function used in transformers.

**GPT.** Generative Pre-trained Transformer. An autoregressive decoder for text generation.

**Gradient penalty.** A regularisation term in WGAN that enforces the Lipschitz constraint on the critic.

**GRU.** Gated Recurrent Unit. A simplified RNN with update and reset gates.

**Hidden state.** The internal memory of an RNN at each time step.

**InferenceServer.** Kailash ML engine for serving model predictions.

**Input gate.** The LSTM gate that controls what new information to store.

**Key.** One of the three projections (Q, K, V) in attention, representing what each element contains.

**Latent space.** The low-dimensional space learned by an autoencoder or VAE.

**Layer normalisation.** Normalising across the feature dimension for each sample, used in transformers.

**LSTM.** Long Short-Term Memory. An RNN variant with gating mechanisms that prevent vanishing gradients.

**Message passing.** The GNN mechanism where nodes exchange information along edges.

**Mixed precision.** Using FP16 for computation and FP32 for weight updates.

**Mixup.** Data augmentation by linearly interpolating training examples.

**Mode collapse.** A GAN failure where the generator produces only a limited variety of outputs.

**Multi-head attention.** Running multiple attention operations in parallel with different projections.

**OnnxBridge.** Kailash ML engine for exporting models to ONNX format.

**Output gate.** The LSTM gate that controls what to expose from the cell state.

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

**Self-attention.** Attention where queries, keys, and values all come from the same sequence.

**Skip connection.** A shortcut that adds the input of a layer directly to its output.

**Stride.** The step size of a convolutional filter as it slides across the input.

**Transfer learning.** Reusing a pre-trained model's features for a new task.

**Transformer.** An architecture based entirely on attention, processing all positions in parallel.

**Transposed convolution.** An upsampling operation used in decoders to increase spatial dimensions.

**Value.** One of the three projections (Q, K, V) in attention, representing the information to contribute.

**Value function.** In RL, the expected cumulative reward from a state.

**VAE.** Variational Autoencoder. An autoencoder with a probabilistic latent space, enabling generation.

**ViT.** Vision Transformer. Applies transformer architecture to image patches.

**Wasserstein distance.** A distance metric between probability distributions used in WGAN.

**WGAN.** Wasserstein GAN. Uses Wasserstein distance for more stable training.

---

# Further Reading

**On autoencoders and VAEs**

- Kingma, D., and Welling, M. "Auto-Encoding Variational Bayes." _ICLR_, 2014. The original VAE paper.
- Doersch, C. "Tutorial on Variational Autoencoders." _arXiv:1606.05908_, 2016.

**On CNNs**

- He, K., et al. "Deep Residual Learning for Image Recognition." _CVPR_, 2016. The ResNet paper.
- Hu, J., Shen, L., and Sun, G. "Squeeze-and-Excitation Networks." _CVPR_, 2018. The SE block paper.
- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words." _ICLR_, 2021. The Vision Transformer paper.

**On RNNs and LSTMs**

- Hochreiter, S., and Schmidhuber, J. "Long Short-Term Memory." _Neural Computation_, 1997. The original LSTM paper.
- Cho, K., et al. "Learning Phrase Representations using RNN Encoder-Decoder." _EMNLP_, 2014. The GRU paper.

**On transformers**

- Vaswani, A., et al. "Attention Is All You Need." _NeurIPS_, 2017. The original transformer paper.
- Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers." _NAACL_, 2019.
- Radford, A., et al. "Language Models are Unsupervised Multitask Learners." OpenAI, 2019. The GPT-2 paper.

**On GANs and generative models**

- Goodfellow, I., et al. "Generative Adversarial Nets." _NeurIPS_, 2014. The original GAN paper.
- Arjovsky, M., Chintala, S., and Bottou, L. "Wasserstein GAN." _ICML_, 2017.
- Gulrajani, I., et al. "Improved Training of Wasserstein GANs." _NeurIPS_, 2017. WGAN-GP.
- Ho, J., Jain, A., and Abbeel, P. "Denoising Diffusion Probabilistic Models." _NeurIPS_, 2020.

**On GNNs**

- Kipf, T., and Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks." _ICLR_, 2017.
- Hamilton, W., Ying, R., and Leskovec, J. "Inductive Representation Learning on Large Graphs." _NeurIPS_, 2017. GraphSAGE.
- Velickovic, P., et al. "Graph Attention Networks." _ICLR_, 2018. GAT.

**On reinforcement learning**

- Sutton, R., and Barto, A. _Reinforcement Learning: An Introduction._ MIT Press, 2018. The definitive textbook. Free online at `incompleteideas.net/book/the-book.html`.
- Mnih, V., et al. "Human-level control through deep reinforcement learning." _Nature_, 2015. DQN.
- Schulman, J., et al. "Proximal Policy Optimization Algorithms." _arXiv:1707.06347_, 2017.

**On transfer learning**

- Zhuang, F., et al. "A Comprehensive Survey on Transfer Learning." _Proceedings of the IEEE_, 2020.

---
