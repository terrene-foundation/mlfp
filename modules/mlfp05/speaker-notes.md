# Module 5: Deep Learning and Machine Learning Mastery in Vision and Transfer Learning — Speaker Notes

Master speaker notes for the module deck (`deck.html`, 129 slides). One section per
slide, in deck order, numbered and titled exactly as the deck. Per-lesson pages with
deeper prose live in `lessons/NN/notes.html`; these master notes are the
slide-by-slide companion for the instructor presenting the full deck.

Total teaching time: ~310 minutes of presented material (about 5 hours 10 minutes),
plus hands-on exercise time (~3-4 hours across the 8 exercises) and breaks. Plan two
half-day sessions split after Slide 65 (end of Lesson 5.3), or a full intensive day
with generous breaks.

Audience: working professionals. Instructors must scaffold for both novices (first
deep network in M4) and practitioners (already train models at work). Every slide is
labelled FOUNDATIONS, THEORY, or ADVANCED — green for everyone, blue for stretch,
purple for bonus.

How to read each section:

- **Time** — budget for the slide. The timings sum to ~310 minutes.
- **Hook** — say this first, before advancing into the slide's detail.
- **Key question** — pose it to the room and wait for answers; do not answer it yourself.
- **Beginner cue** — what to do if the room looks lost.
- **Advanced cue** — what to offer experienced learners who look bored.
- **Transition** — the line that hands off to the next slide.

Kailash engines used in this module: ModelVisualizer (training curves, latent and
embedding scatter plots), OnnxBridge (ONNX export), InferenceServer (async serving
from the ModelRegistry), and RLTrainer (`kailash_ml.rl`, with `km.rl_train` as the
one-call entry point). Stable-Baselines3, RLTrainer's backend, is an optional extra
that is not installed in the course environment, so the RL exercises hand-write DQN
and PPO and present `km.rl_train` as the production path.

---

## Slide 1: Deep Learning and Machine Learning Mastery in Vision and Transfer Learning

**Time:** ~2 min

Welcome the room. This is the architecture module: 8 paradigms, 8 lessons, all
implemented end-to-end by the student. Nothing is left as "you will see this later".
By the end of Module 5, students can build anything from autoencoders to RL agents.

Make the promise concrete: every major deep-learning architecture, one paradigm per
lesson, all coded by hand in PyTorch and then connected to the Kailash engines for
visualisation, export, serving, and RL training.

**Key question:** "How many of you trained a neural network before Module 4?" A show
of hands gauges the room and calibrates how much time the FOUNDATIONS layer gets
versus the THEORY derivations.

**Beginner cue:** "Module 4 gave you the toolkit — forward pass, backprop,
optimisers. Today we use that toolkit to build every major architecture in modern
deep learning."

**Advanced cue:** "Even if you have used these architectures at work, very few
people have implemented all of them from scratch in one module. The progression is
what matters — you will see how every paradigm shares the same foundation."

**Transition:** "Here is exactly what you will be able to do by the end."

---

## Slide 2: What You Will Learn

**Time:** ~2 min

Walk through the three layers. This module is the most formula-heavy in the
programme, and the three-layer system governs how each person should spend their
attention. FOUNDATIONS students follow the intuition; THEORY students derive every
equation; ADVANCED students connect to current literature.

Reassure the mixed room explicitly: "Green slides are yours, blue is stretch, purple
is bonus. Nobody is expected to follow every slide. The exercises only assume
FOUNDATIONS."

**Beginner cue:** "If we hit a blue slide that feels too dense, just note the name
and move on — you can pass every exercise on the green track alone."

**Advanced cue:** "Stay for the derivations: the VAE ELBO, the sqrt(d_k) scaling
argument, the Bellman equations. These are the slides that come up in interviews."

**Transition:** "Eight lessons, one paradigm per lesson. Here is the journey."

---

## Slide 3: Your Journey: 8 Lessons

**Time:** ~1 min

Walk the table quickly — do not read every row. Highlight the progression:
autoencoders bridge from M4 neural networks. CNNs add spatial structure. RNNs add
temporal memory. Transformers replace both with attention. GANs generate. GNNs
handle graphs. Transfer learning applies everything. RL learns from interaction.

Key framing: each lesson pairs the theory with an exercise, and the exercises build
on each other — the conv layers students use in 5.1 become the backbone of 5.2.

**Beginner cue:** "Think of each lesson as one new vocabulary word for deep
learning. By the end of the module, you will speak all eight."

**Transition:** "You will use four Kailash engines in this module."

---

## Slide 4: Kailash Engines in Module 5

**Time:** ~2 min

Introduce the four engines and what each is for:

- **OnnxBridge** exports a trained model to ONNX. The flow is export, then register
  the artifact in the ModelRegistry.
- **InferenceServer** serves it: `await InferenceServer.from_registry(...)`, then
  `await start()`, then `await predict(...)` — the whole serving path is async.
- **ModelVisualizer** draws training curves and latent/embedding scatter plots from
  polars DataFrames — two lines instead of fifteen of matplotlib.
- **RLTrainer** lives in `kailash_ml.rl` (not top-level `kailash_ml`); `km.rl_train`
  is the one-call entry point. Its backend, Stable-Baselines3, is an optional extra
  (`kailash-ml[rl]`) that is not installed in the course environment — so the
  exercises hand-write DQN and PPO and show the library call as the production path.

Name the teaching pattern: "You learn the theory first, build the model by hand,
then see how the engine automates it. That way, when the engine gives you an
unexpected result, you know how to debug it."

**Beginner cue:** "Think of these as power tools. We show you the hand-tool version
first so you understand what the power tool is doing underneath."

**Advanced cue:** "OnnxBridge and InferenceServer are the production story —
train in PyTorch, export to ONNX, serve from the registry. That is the pipeline
every deployment in this module uses."

**Transition:** "Before we start building, let us make sure everyone has the M4
toolkit fresh."

---

## Slide 5: DL Toolkit Refresher (from M4)

**Time:** ~3 min

Recap the M4 toolkit: forward pass, backpropagation, gradient descent (SGD, Adam),
dropout, batch normalisation. Everything built today sits on these five ideas.

Run the 2-layer classifier check: "Build a 2-layer classifier in your head. Input →
Linear → ReLU → Linear → Softmax. Loss is cross-entropy. Optimiser is Adam. If that
feels familiar, you are ready." This refresher exists to confirm readiness — if
students are shaky on backprop, pause here. Two minutes now saves thirty minutes of
confusion in Lesson 5.1.

Then name the shift explicitly: M4 trained networks to predict labels (supervised).
M5 trains networks to learn representations — compressed (autoencoders), spatial
(CNNs), temporal (RNNs), attended (transformers), generated (GANs), connected
(GNNs), transferred (transfer learning), interactive (RL).

**Advanced cue:** "The shift from supervised labels to learned representations is
what modern deep learning is really about. Every architecture today is a different
answer to the question: what structure should the latent space have?"

**Transition:** "Before the first architecture, one more toolkit — the diagnostic
one. Here is why it exists."

---

## Slide 6: The Doctor's Bag

**Time:** 3 min (45 sec story + 2 min pair share)

**Hook:** "In 2023, a team at a major fintech spent 6 weeks debugging a model that
was 70% dead ReLU from epoch 2. Nobody checked. They blamed the architecture, the
data, the library. A 10-line diagnostic would have caught it the first afternoon.
This is what we're preventing today."

Run the pair share: two minutes, neighbours name symptoms they have hit. Collect
three or four on the board — you will map each to an instrument on the next slide.

**Key question:** "Why do most ML engineers never build a diagnostic protocol?"
Expected wrong answer: "because libraries hide it". Right answer: "because
re-running with different hyperparameters *feels* faster than instrumenting — until
the clock hits hour 6."

**Behind schedule:** cut the pair share; keep the 45-second story — it is the
emotional anchor for the whole section.

**Transition:** "You all just named symptoms. Now let's give you the five
instruments to diagnose them."

---

## Slide 7: The Five Instruments

**Time:** 3 min

**Hook:** "A doctor never prescribes without examining the patient. Same contract
here — four instruments of examination, one of treatment."

Walk the five icons: loss curves (the stethoscope), gradient flow (the blood test),
activation X-ray (the X-ray), the training dashboard (the patient chart), and the
prescription pad (the treatment). The headline is the divider: four instruments
*diagnose*, one *treats*. Diagnosis always comes first.

**Key question:** "If loss is oscillating wildly, which instrument am I READING, and
which am I about to USE?" Expected wrong answer: "the prescription pad". Right
answer: "I'm reading the stethoscope (loss); only then do I open the prescription
pad."

**Behind schedule:** skim the five icons and spend the time on the diagnose/treat
split — it is the headline.

**Transition:** "Before we use the first instrument, a 30-second primer on what it
is listening to: the loss."

---

## Slide 8: Meet DLDiagnostics — Your Toolkit in One Object

**Time:** 3 min

**Hook:** "Every subsequent slide in this section has a one-line call on this
object. You will never write a raw forward hook in M5."

Introduce `DLDiagnostics` as the single object that owns the five instruments.
Students construct it once per model and call the instruments from it — the hooks,
the recording order, and the cleanup are the library's problem.

**Key question:** "Why wrap the hooks in a class instead of pasting them into each
exercise?" Expected wrong answer: "it is cleaner". Right answer: "because hook
ordering matters — register after optimizer init, record after backward before
step. One line of mis-sequenced plumbing silently corrupts your diagnosis. The
library owns the sequence." Note for precision: the activation X-ray hooks
activation *modules* (ReLU, GELU outputs), not arbitrary layers.

**Behind schedule:** skip the `lr_range_test` classmethod mention; it recurs in
5.2.

**Transition:** "That is the tool. Now for the first instrument it surfaces — loss
curves — which means a 30-second primer on what loss actually is."

---

## Slide 9: Prerequisite — What is Loss?

**Time:** 1 min

**Hook:** "Before we read curves, one sentence: loss is how wrong the model is, on
average."

Use the two-bar picture: train loss and validation loss side by side. The gap
between them is the whole story of generalisation, and every loss-curve shape in
the next slide is a story about how those two bars evolve.

**Key question:** "If train loss is 0.02 and val loss is 2.0, what is going on?"
Expected wrong answer: "the model is great." Right answer: "it memorised the
homework; a 100x gap means overfitting."

**Behind schedule:** cut the callout; the two-bar SVG does the work.

**Transition:** "Now that loss and train/val are clear, let us read the SHAPES of
the loss curves."

---

## Slide 10: Instrument 1: Loss Curves — The Stethoscope

**Time:** 5 min (3 min drill + 2 min vote)

**Hook:** "Four shapes cover 95% of what you will see. Name the shape, then act —
never the other way round."

Drill the four canonical shapes: healthy (both drop, small gap), overfitting (val
bends up while train keeps falling), underfitting (both plateau high), unstable
(oscillation). For each, name the shape first, then the action.

**Key question:** after showing the four shapes, point to the 5th Mystery panel:
"Vote: healthy, overfitting, underfitting, or unstable?" Expected wrong answer:
"healthy — both dropped early". Right answer: "late-stage overfitting — val bent
up in the final third, which is the classic signature." Reveal after the vote.

**Behind schedule:** cut the mystery vote, leave the 5th panel visible as homework;
the four canonical shapes are the must-cover content.

**Transition:** "Now that you can name the shape, let me show you how to plot it so
the shape is actually visible."

---

## Slide 11: Reading Loss Curves — In Practice

**Time:** 3 min

**Hook:** "Most 'my loss is not decreasing' reports are plot problems, not model
problems."

Teach the three habits: plot log-y, smooth with an exponential moving average, and
re-plot before concluding anything. The re-plot rule: before touching a
hyperparameter, re-plot with log-y and smoothing — half of all "stuck" runs are
fine.

**Key question:** "Why log-y?" Expected wrong answer: "because it looks cleaner".
Right answer: "the first 100 steps compress 90% of the learning — linear-y hides it
behind the late-stage plateau."

**Behind schedule:** drop the code walkthrough, keep the three habits plus the
re-plot rule.

**Transition:** "Loss is the output; the next instrument looks at what is happening
INSIDE the model — gradient flow."

---

## Slide 12: Prerequisite — What is a Gradient?

**Time:** 1 min

**Hook:** "Before the blood test, one sentence: the gradient is how hard each knob
needs to turn to make the model less wrong."

One gradient per parameter, computed by backprop, consumed by the optimiser. The
four bullets on the slide — direction, magnitude, per-layer, per-step — are the
whole vocabulary the next instrument needs.

**Key question:** "If layer 1 has gradient near zero and layer 6 has a healthy
gradient, what is happening?" Expected wrong answer: "layer 1 is fine, it already
learned". Right answer: "layer 1 is not learning at all — the signal never reached
it. Vanishing gradient."

**Behind schedule:** cut the callout; the SVG plus the four bullets do the work.

**Transition:** "Now we can use the instrument that measures this."

---

## Slide 13: Instrument 2: Gradient Flow — The Blood Test

**Time:** 3 min

**Hook:** "Loss tells you the patient is sick. The blood test tells you which organ
is failing."

Walk the three patterns: healthy (per-layer gradient RMS within an order of
magnitude across layers), vanishing (early layers orders of magnitude below late
layers), exploding (late layers spiking, loss NaN shortly after). The healthy bar
pattern is the reference students match against all module long.

**Key question:** "If layer 1 gradient is 1e-9 and layer 6 gradient is 1e-2, what
single fix tends to help?" Expected wrong answer: "increase learning rate". Right
answer: "add skip connections or swap to GELU — the issue is signal reaching early
layers, not update size."

**Behind schedule:** collapse exploding and vanishing into one sentence each; spend
the saved time on the healthy bar pattern.

**Transition:** "Three shapes, one instrument — now the code that measures them."

---

## Slide 14: Gradient Flow — Code & Thresholds

**Time:** 4 min

**Hook:** "Two numbers per layer — RMS and update ratio. Everything else is noise."

Walk the thresholds: gradient RMS per layer should sit within roughly an order of
magnitude across layers; the update ratio (RMS of the update over RMS of the
weights) should hover near 1e-3. Far below means the layer is barely learning; far
above means the updates are tearing the weights apart.

**Key question:** "Why not just track `grad.norm()`?" Expected wrong answer: "no
good reason, norm is fine". Right answer: "norm scales with tensor size — a
1024x1024 weight dwarfs a 32x32 even if per-element signal is identical. RMS and
update ratio are scale-invariant."

**Behind schedule:** skip the ZClip mention; the callout stays on-screen for
self-study.

**Transition:** "Gradients tell us whether signal reaches each layer. The next
instrument looks at what each layer actually PRODUCES — activations."

---

## Slide 15: Instrument 3a: Statistical X-Ray (Activation X-Ray — part 1 of 2)

**Time:** 2.5 min

**Hook:** "Four numbers per layer. Runs in the training loop. Costs nothing."

The four numbers: mean, standard deviation, fraction dead (exact zeros after ReLU),
and saturation fraction (for tanh/sigmoid). Healthy: mean near 0, std in a
moderate range, dead fraction low and stable. These run inside the training loop on
activation modules, so the cost is negligible.

**Key question:** "If mean drifts toward +5 and std stays at 0.4, what is
happening?" Expected wrong answer: "exploding activations". Right answer:
"mean-drift, not variance blow-up — usually a missing normalisation
(BatchNorm/LayerNorm) or biased init."

**Behind schedule:** cut the saturation bullet; mean, std, and dead% are the
must-have three.

**Transition:** "Stats tell you the layer is misbehaving. The next slide tells you
WHY — by looking at what drove the output."

---

## Slide 16: Instrument 3b: Attribution X-Ray (Activation X-Ray — part 2 of 2)

**Time:** 3 min

**Hook:** "A model can be right for the wrong reason. Attribution is how you find
out."

Attribution asks which inputs drove the prediction. The three methods on the slide
— saliency, Grad-CAM, attention rollout — answer the same question at different
resolutions. Students meet Grad-CAM in Exercise 5.2 and attention visualisation in
5.6, so do not dwell on mechanics here.

**Key question:** "If a chest-X-ray classifier looks at a hospital marker instead
of the lung, what metric would catch it?" Expected wrong answer: "accuracy on the
test set". Right answer: "none — accuracy stays high on data from the same
hospital. Only transfer testing or attribution would catch it. Zech 2018 is the
canonical example."

**Behind schedule:** keep the Zech story, cut the three-method list — students will
meet Grad-CAM and attention rollout in exercises anyway.

**Transition:** "Stats tell you the layer is alive. Attribution tells you it is
looking at the right thing. Now the failure mode the stats catch best: dead
neurons."

---

## Slide 17: Dead Neurons — Detect & Fix

**Time:** 3 min

**Hook:** "A network can be 70% dead and still show a decreasing loss. That is the
scariest failure mode in this module."

Dead ReLU units output exactly zero for every input, so their incoming weights
receive zero gradient — they never recover. The X-ray's dead fraction is how you
see it; the loss curve will not tell you.

**Key question:** "If loss is still dropping and dead-fraction is 70%, what is the
fix priority order?" Expected wrong answer: "increase LR, the model is
under-trained". Right answer: "swap activation FIRST (ReLU → GELU), re-initialise
SECOND (Kaiming); LR change comes third if needed."

**Behind schedule:** cut the Kaiming callout; the activation swap is the one-minute
fix that handles most cases.

**Transition:** "Four instruments so far. Next we put them on ONE screen — the
training dashboard."

---

## Slide 18: Instrument 4: Training Dashboard — The Patient Chart

**Time:** 2 min

**Hook:** "Instruments 1 through 4 were examination. This is the patient chart —
every reading on one screen."

Walk the four panels: loss curves, gradient norms per layer, learning rate over
time, weight histograms. The point of the dashboard is the glance: one look per
epoch tells you whether to keep going or stop and diagnose.

**Key question:** "Which of the four panels do you glance at FIRST on a troubled
run?" Expected wrong answer: "the loss curves". Right answer: "gradient norms per
layer — loss can hide the failure, but a layer with flat gradients cannot."

**Behind schedule:** drop the weight-histograms panel explanation; the
loss + gradient + LR trio carries the point.

**Transition:** "Four instruments of examination — now the treatment side: the
prescription pad."

---

## Slide 19: Instrument 5: The Prescription Pad

**Time:** 4 min

**Hook:** "Seven symptoms cover more than 90% of DL failures you will hit this
module. Match symptom to prescription, THEN act."

Walk rows 1–5 slowly: loss not decreasing at all (overfit one batch first — if even
that fails, the pipeline is broken, not the model), overfitting (data, augmentation,
regularisation), underfitting (capacity, training time, LR), unstable loss (lower
LR, gradient clipping), dead neurons (activation swap, re-init). Rows 6 and 7 can
stay on screen for self-study.

**Key question:** "If loss will not decrease at ALL from step 0, which prescription
do you pick?" Expected wrong answer: "increase capacity". Right answer: "overfit to
one batch first — if even that fails, your pipeline is broken, not your model."

**Behind schedule:** cut rows 6 and 7 from discussion; keep them on-screen. Rows
1–5 cover the most common failures.

**Transition:** "That's the full toolkit. Now let me show you where each instrument
shows up across M5's eight lessons."

---

## Slide 20: How the Toolkit Threads Through M5

**Time:** 2 min

**Hook:** "The five instruments stay the same for the next eight lessons. What
changes is which symptom each architecture hands you."

Skim the mapping: autoencoders hand you posterior collapse and blurry
reconstructions; CNNs hand you dead channels; RNNs hand you exploding gradients;
transformers hand you attention pathologies; GANs hand you mode collapse; RL hands
you reward hacking. Same instruments, different failure patterns.

**Key question:** "Name one new failure mode a GAN hands you that an autoencoder
never will." Expected wrong answer: "vanishing gradients". Right answer: "mode
collapse — the generator ignores half the input distribution. Same instruments
(loss + gradients + prescription), different failure pattern."

**Behind schedule:** skip the 5.5–5.8 rows; the headline is that the TOOLKIT
carries forward, not the specific failure modes.

**Transition:** "Last slide of this section: the four-check protocol you will run
before every training run for the rest of M5."

---

## Slide 21: Your Diagnostic Workflow

**Time:** 5 min (3 min walk-through + 2 min sticky-note commitment)

**Hook:** "Four checks, one decision. Every model, every run, for the rest of M5."

Walk the protocol: check the loss shape, check gradient flow, check activation
stats, then change exactly one thing and re-run. Then run the sticky-note activity:
every student writes the four checks on a sticky note and puts it on their laptop.
Students leave with the protocol in their own handwriting — do not skip this.

**Key question:** "Why change only one thing at a time?" Expected wrong answer: "to
be safe". Right answer: "because if you change three things and the model works,
you learned nothing — you cannot attribute the fix. Information theory, not
caution."

**Behind schedule:** cut the walk-through, run ONLY the sticky-note activity.

**Transition:** "Put your sticky note next to your laptop. Three appendix slides of
modern practice next, then Lesson 5.1."

---

## Slide 22: Appendix: Modern Practice Notes

**Time:** ~30 sec

These three slides break the five-instruments narrative if delivered in-flow, so
they live here as reference. Point students at the appendix when they hit LR
decisions in 5.2, architecture decisions in 5.4, or data-budget questions in 5.5.

Say exactly that, then move on — do not teach the appendix in sequence.

**Transition:** "For reference whenever you need them. Now — the modern-defaults
table, sixty seconds."

---

## Slide 23: Modern Architecture Choices (2026)

**Time:** ~3 min

This is the "what would a 2026 foundation model do" slide. Every row is supported
by an ablation in a 2022–2025 paper: GELU/SwiGLU over ReLU, RMSNorm over LayerNorm
in transformers, AdamW over Adam+L2, cosine schedules over step decay.

The pedagogical point: students often implement textbook defaults (ReLU, BatchNorm,
Adam+L2) and then wonder why their models do not match modern benchmarks. The
answer is this table. Flag that SwiGLU and RMSNorm are transformer-specific; for
CNNs, GELU and GroupNorm/LayerNorm are the comparable upgrades.

**Transition:** "The cheapest hyperparameter decision in deep learning — the LR
range test."

---

## Slide 24: The LR Range Test (Leslie Smith, 2017)

**Time:** ~3 min

The LR range test is the single cheapest hyperparameter decision in DL — 100 steps
of training tells you the right LR for the real run. Walk the three zones: flat
(LR too low to move weights), sweet spot (loss drops fast), divergence (LR too
high, loss blows up). Pick 1/3 to 1/10 of the divergence point — this gives
headroom.

Flag the forward connection: this is why Lesson 5.2 starts every CNN training with
a 2-minute LR range test, and why `DLDiagnostics` carries it as a classmethod.

**Transition:** "The other budget question: how much training is enough?"

---

## Slide 25: Scaling Laws — How Much Training?

**Time:** ~3 min

This slide prevents a common student error: training for 5 epochs, seeing loss
still decreasing, and concluding "autoencoders/transformers/RL don't work for my
data."

Be precise about Chinchilla: ~20 tokens per parameter is the COMPUTE-OPTIMAL ratio
(the cheapest way to reach a given quality), not a minimum. Production models train
far past it — Llama 3 at roughly 1,875 tokens per parameter — because inference is
where the real cost lives, and a smaller, better-trained model is cheaper to serve.

For this course, the takeaway is humbler: give your model enough data AND enough
steps before declaring the architecture a failure.

**Transition:** "Toolkit complete. Now the first architecture — the autoencoder."

---

## Slide 26: Autoencoders

**Time:** ~1 min

Title the lesson. Autoencoders are the gentlest entry to unsupervised deep
learning. The metaphor: compress then reconstruct. If the network can rebuild the
input from a compressed version, it has learned the essential structure.

**Beginner cue:** "Think of the game where you describe a movie in three words and
a friend has to guess the movie. If they guess right, your three words captured the
essential plot."

**Transition:** "Three components, one idea — encode, bottleneck, decode."

---

## Slide 27: The Autoencoder Architecture

**Time:** ~3 min

Walk the three components: encoder compresses, latent space is the bottleneck,
decoder reconstructs. Emphasise: no labels. The target IS the input — this is
unsupervised learning.

The bottleneck is what makes the architecture useful. If the latent space were as
large as the input, the network would just copy. The compression forces learning.

Use the book-summary analogy: summarise a book in one sentence (encoder); someone
else rewrites the book from your sentence (decoder). The better the summary, the
closer the rewrite.

**Advanced cue:** "The auto in autoencoder is historically important — it
distinguished self-supervised reconstruction from supervised training, long before
'self-supervised learning' became a buzzword."

**Transition:** "So what does the loss function look like?"

---

## Slide 28: Reconstruction Loss

**Time:** ~2 min

Show the equation: L = ||x − x̂||². Mean squared error between input and
reconstruction. MSE is the default for continuous data; binary cross-entropy is
common for images normalised to [0,1].

The choice of loss affects what the autoencoder prioritises: MSE preserves
magnitude, BCE preserves binary structure.

**Advanced cue:** "The reconstruction loss is an implicit prior on the generative
distribution. MSE assumes Gaussian noise on pixels. BCE assumes Bernoulli. That is
why the loss changes what the model learns to preserve."

**Transition:** "The simplest autoencoder is fully connected."

---

## Slide 29: Variant 1: Vanilla Autoencoder

**Time:** ~3 min

Walk through the code. The encoder compresses 784 inputs (28×28 Fashion-MNIST
pixels) down to the latent dimension — 16 in Exercise 5.1. The decoder
reconstructs. Sigmoid output because the pixels are scaled to [0,1].

Key insight: if latent_dim equals input_dim, the network just copies. The
bottleneck is what forces learning. This is the simplest autoencoder — everything
else in the lesson changes ONE of its three boxes.

**Beginner cue:** "In the exercise you will feed Fashion-MNIST in with a 16-D
latent. The network has to squeeze 784 pixels down to 16 numbers and rebuild. That
forces it to learn what makes a shoe a shoe."

**Advanced cue:** "A vanilla AE with purely linear layers is exactly PCA.
Non-linearity is what buys you representation power."

**Transition:** "Now what if we corrupt the input to force robust features?"

---

## Slide 30: Variant 2: Denoising Autoencoder (DAE)

**Time:** ~3 min

The key difference from vanilla: the input is corrupted, the target is clean. The
loss compares the reconstruction against the ORIGINAL clean input, not the noisy
version.

Why it works: the encoder must throw away the noise and keep the signal. It cannot
memorise pixel patterns — those change every mini-batch — so it learns features
invariant to the noise. This is a form of self-supervised learning and a powerful
regulariser for free. Applications: image denoising, robust feature learning,
pre-training.

**Beginner cue:** "Imagine studying with blurry photocopies of the textbook. You
cannot memorise the exact pixels — you have to learn what the diagram MEANS."

**Advanced cue:** "Masking noise is the direct ancestor of BERT's masked language
modelling in Lesson 5.4."

**Transition:** "Now the architectural jump — what if the latent space were a
probability distribution?"

---

## Slide 31: Variant 3: Variational Autoencoder (VAE)

**Time:** ~3 min

This is the conceptual breakthrough slide. VAEs are generative models: they learn a
distribution, not a mapping. The encoder outputs mu and sigma, and you SAMPLE the
latent code from the Gaussian they parameterise.

Point at the left diagram: "A vanilla AE maps each digit to a precise point — if
you sample between two points, you get garbage." Point at the right: "A VAE maps
each digit to a fuzzy cloud — the overlap regions produce valid new digits." The
probabilistic latent space is continuous and smooth, which means you can
interpolate between data points and generate new ones.

Formally, the encoder approximates the posterior q_φ(z|x) ≈ p(z|x); the decoder
models the likelihood p_θ(x|z).

**Beginner cue:** "The practical punchline: the VAE is the first model we build
that can make NEW images. In the exercise you will generate new clothing items
from pure noise."

**Advanced cue:** "This is amortised variational inference. The encoder is the
inference network; the ELBO is the training signal."

**Transition:** "The VAE loss is called the ELBO. Here it is."

---

## Slide 32: VAE Loss: The ELBO

**Time:** ~4 min

This is the core equation. Walk both terms slowly.

Term 1 is the reconstruction likelihood — how well the decoder explains x given z.
For a Gaussian likelihood this becomes MSE plus a constant. Familiar from the
vanilla AE.

Term 2 is the KL divergence — how far the encoder's distribution is from the prior
N(0, I). It keeps the latent space smooth and continuous. Without it, the VAE
degenerates into a vanilla AE with isolated latent points and no generative
capacity.

The closed-form KL for Gaussians is a computational gift: no sampling needed for
that term, one line of Python.

**Beginner cue:** "Term 1 says 'reconstruct well.' Term 2 says 'keep the latent
space tidy.' You need both — reconstruction alone gives you a vanilla AE; tidiness
alone gives you random noise."

**Advanced cue:** "ELBO = Evidence Lower BOund, a lower bound on log p(x).
Maximising it maximises a lower bound on the true log-likelihood, which is
intractable because of the posterior."

**Transition:** "But we just said the latent code is sampled. How does gradient
flow through a random sample?"

---

## Slide 33: The Reparameterisation Trick

**Time:** ~4 min

This is the most elegant trick in the VAE paper. State the problem plainly:
sampling z ~ N(μ, σ²) is stochastic, and gradients cannot flow through random
sampling. Backprop breaks.

The solution: z = μ + σ·ε, where ε ~ N(0, I). Move the randomness out to an input.
Now gradients flow through μ and σ because they are deterministic paths — the
network only decides WHERE to sample (μ) and HOW WIDE (σ); the coin flip is a
fixed input.

Walk the three practical steps: encoder outputs μ and log σ², sample ε from a
standard normal, compute z = μ + exp(0.5·log σ²)·ε.

**Key question:** "Why log-variance instead of variance directly?" Answer:
numerical stability, and σ stays positive through the exponential.

**Advanced cue:** "The reparameterisation gradient has far lower variance than the
score-function gradient (REINFORCE), which is why VAEs train so much better than
early stochastic networks."

**Transition:** "Now let us swap fully connected layers for convolutions."

---

## Slide 34: Variant 4: Convolutional Autoencoder

**Time:** ~3 min

Motivation: fully connected layers ignore spatial structure and waste parameters on
images. Walk the encoder dimensions: 28×28×1 → 14×14×16 → 7×7×4. The decoder
reverses with transposed convolutions. Far fewer parameters than a fully connected
autoencoder on images.

Tell students this architecture is the foundation for CNNs in Lesson 5.2 — they do
not need Conv2d in depth yet, just the mirror structure: Conv2d downsamples in the
encoder, ConvTranspose2d upsamples in the decoder. Same loss, same training loop.

**Advanced cue:** "ConvTranspose has its own artefacts — checkerboard patterns —
that modern architectures address with nearest-neighbour upsampling plus a regular
conv."

**Transition:** "There are more variants in the exercise — here is the survey."

---

## Slide 35: Additional Autoencoder Variants (Survey)

**Time:** ~2 min

Survey in the lecture, but be clear that Exercise 5.1 builds these: file 04
(sparse), 05 (contractive — a real per-sample Jacobian penalty, computed with
vmap(jacrev), NOT weight decay), 07 (stacked), 08 (LSTM recurrent) and 10
(Contractive VAE). The point is that each variant adds one constraint to the same
three-box architecture. The four core ideas are vanilla, denoising, VAE and
convolutional.

Naming trap to state out loud: the exercise's "Contractive VAE" is not a CVAE —
CVAE conventionally means Conditional VAE. If students meet "CVAE" in the wild, it
almost always means the conditional one.

VQ-VAE is mentioned for advanced students interested in modern generative models:
the discrete latent space is what lets you use a transformer as the prior — the
DALL-E 1 / VQ-GAN design.

**Transition:** "Time to build. Here is the exercise."

---

## Slide 36: Exercise 5.1: Autoencoder Workshop

**Time:** ~1 min

Exercise 5.1 is 11 short files on Fashion-MNIST with a 16-D latent: overcomplete
then undercomplete AE, denoising, sparse, contractive (the real Jacobian penalty),
convolutional, stacked, LSTM-recurrent, VAE, Contractive VAE, and a grand
comparison of all ten.

The VAE generation task is the highlight — producing new clothing images from pure
noise. The 2-D latent projections should show garment classes separating. Stretch
goal: interpolate between two items in the VAE latent space and watch one morph
into the other.

Note: the exercise is PyTorch directly; ModelVisualizer provides the training
curves and latent scatter plots.

**Transition:** "Quick summary before we move to CNNs."

---

## Slide 37: Lesson 5.1 Summary

**Time:** ~1 min

One-line takeaway: autoencoders learn compressed representations by reconstructing
their own input through a bottleneck. Three equations to remember: reconstruction
loss, ELBO, reparameterisation.

Bridge to the next lesson: the convolutional autoencoder's encoder is essentially a
CNN feature extractor. In 5.2 we replace the decoder with a classifier head — and
we have a CNN. Natural progression.

**Transition:** "Where do autoencoders actually earn their keep? Four places."

---

## Slide 38: Lesson 5.1: Real-World Applications

**Time:** ~3 min

Walk through the fraud example first — it is the one everyone has an intuition for.
Emphasise: "This is the same model you just built on Fashion-MNIST. The only thing
that changes is the training data." That should land the message that the technique
generalises across domains.

**Key question:** "Why not just a supervised classifier?" Answer: imbalanced data —
labelled fraud is a tiny fraction of transactions, and the organisation cannot
afford to miss the first few instances of a new attack pattern. Train on normal
traffic; reconstruction error becomes the anomaly score.

State clearly that the organisations on this slide are generic illustrations, not
claims about any named institution's systems.

If someone raises synthetic patient data, give the honest answer: a generative
model can memorise and regenerate training records, so synthetic data is only
private with formal guarantees (DP-SGD) plus memorisation / membership-inference
tests. "Synthetic" does not mean "private" by default.

**Transition:** "From compressing images to classifying them — CNNs."

---

## Slide 39: CNNs and Computer Vision

**Time:** ~1 min

Title the lesson. CNNs are the workhorse of computer vision. Students already used
conv layers in the autoencoder; now we build a full classification pipeline, from
fundamentals through modern enhancements — LeNet to ResNet to SE blocks.

**Advanced cue:** "We cover 1998 to 2026 in one lesson. The arc is short but every
step matters."

**Transition:** "Start with the building block — the convolution."

---

## Slide 40: The Convolution Operation

**Time:** ~3 min

Walk through filter (kernel), stride, padding, feature map. A filter is a small
matrix (e.g. 3×3) that slides over the input. Stride is how far it moves each step.
Padding adds zeros around the border.

The output size formula — (W − F + 2P)/S + 1 — is essential for architecture
design. Students must be able to compute spatial dimensions at each layer. Worked
example: 28×28 input, 3×3 filter, stride 2, no padding = (28 − 3 + 0)/2 + 1 = 13×13.
Ask students to compute a few more on scratch paper.

**Beginner cue:** "Think of a magnifying glass scanning a photo, looking for one
specific pattern. At each position it outputs a number — how strongly the pattern
matches there. The feature map is that grid of numbers."

**Transition:** "After conv comes pooling — how we build hierarchy."

---

## Slide 41: Pooling and Feature Hierarchy

**Time:** ~2 min

Pooling reduces spatial dimensions: max pooling (dominant in practice), average
pooling, global average pooling (GAP). GAP is increasingly preferred over fully
connected layers at the end of networks — fewer parameters, less overfitting.

The feature hierarchy insight: early layers learn low-level features (edges,
textures); deep layers learn high-level features (objects, faces). This is the
empirical reason transfer learning works — hold that thought for Lesson 5.7.

**Advanced cue:** "GAP replacing FC layers is the architectural trick that made
transfer learning robust — you can attach a single linear classifier head to any
pre-trained GAP output."

**Transition:** "Now let us trace how CNNs evolved."

---

## Slide 42: CNN Architecture Evolution

**Time:** ~3 min

Walk the timeline: LeNet-5 (1998, handwritten digits), AlexNet (2012, deeper, ReLU,
dropout, the ImageNet breakthrough), VGGNet (2014, very small 3×3 filters, depth
matters), GoogLeNet/Inception (2015, multiple filter sizes in parallel).

The trend is clear: deeper networks learn better features, but training them is
hard. Each architecture solved the training problem of the previous generation.
ResNet is the breakthrough that enabled modern deep learning — the next slide.

**Beginner cue:** "For most of the last decade, the recipe was 'take the best
architecture, train it bigger on more data.' The architectures we talk about now
are the ones that survived."

**Transition:** "And then came ResNet — the single most important architectural
innovation in modern deep learning."

---

## Slide 43: ResNet: Skip Connections

**Time:** ~4 min

The equation: H(x) = F(x) + x. The network learns a residual F(x); the identity x
is added back through the skip connection.

Walk the gradient: even if F(x) contributes little gradient, the +1 from the
identity gives the signal a direct path. That is why ResNets reach 152 layers where
plain nets fail beyond ~20.

Be precise about the history (He et al., 2015): the 20+-layer failure of plain nets
was a degradation of TRAINING accuracy that the authors judged "unlikely to be
caused by vanishing gradients" — batch norm kept gradients healthy. It was an
optimisation difficulty: near-identity mappings are hard for stacked nonlinear
layers to learn, but trivial when the block only has to learn a residual.

**Beginner cue:** "Think of editing a Wikipedia article. It is much easier to
suggest 'add this sentence' than to rewrite the whole article. ResNet lets each
layer suggest small edits."

**Advanced cue:** "Residual connections also smooth the loss landscape — the famous
loss-surface visualisations show ResNet's surface nearly convex-looking where
VGG's is chaotic."

**Transition:** "Modern CNNs add a few more tricks on top. First — SE blocks."

---

## Slide 44: Modern Enhancement: SE Blocks

**Time:** ~2 min

Introduce the three-step concept: squeeze compresses spatial information (global
average pool), excitation learns channel importance (two small FC layers and a
sigmoid), recalibration applies the learned weights channel-wise.

Intuition: not all feature channels matter equally for every input. SE learns to
boost relevant channels and suppress irrelevant ones — a learned equaliser,
per-channel gating. The code on the right shows the full PyTorch implementation in
about 15 lines. It bolts onto any CNN — ResNet, VGG, MobileNet.

**Advanced cue:** "SE won the ImageNet 2017 challenge as SENet. It is the template
for all later attention-over-channels designs."

**Transition:** "Now watch the data actually flow through it."

---

## Slide 45: SE Block: Data Flow

**Time:** ~2 min

Walk the diagram left to right. Point out the skip connection (green dashed): the
original feature maps bypass the squeeze/excite pathway and are multiplied by the
learned weights. The channel importance bars show that not all channels matter
equally.

Do the parameter count out loud — SE is cheap: for a 64-channel layer with
reduction 16 it adds 64×4 + 4 + 4×64 + 64 = 580 parameters. The original paper
reports about 1.5 points of ImageNet top-1 for ResNet-50 — a modest but consistent
gain for almost zero cost.

**Transition:** "SE is one enhancement. Here are the training-time ones."

---

## Slide 46: Modern Training Enhancements

**Time:** ~3 min

These are the techniques that separate research-grade from production-grade CNN
training. Kaiming init is essential for deep networks — and watch the notation:
N(0, 2/n_in) gives the VARIANCE; the standard deviation is sqrt(2/n_in). Mixed
precision is free performance. Mixup and label smoothing are strong regularisers.

Be precise about what Exercise 5.2 actually exercises: SE blocks and automatic
mixed precision (the PRECISION constant picks 16-mixed on MPS/CUDA), plus a
flip-and-crop augmentation comparison in file 04. Kaiming init, Mixup and label
smoothing are NOT in the exercise files — set them as a one-change-at-a-time
extension: add one, rerun, compare the train/val gap.

**Advanced cue:** "Mixup is vicinal risk minimisation — training on the vicinity of
each data point — interpretable as Bayesian data augmentation."

**Transition:** "One more preview — Vision Transformers. They matter, and the full
derivation comes in Lesson 5.4."

---

## Slide 47: Vision Transformers (ViT) — Brief Introduction

**Time:** ~2 min

Brief introduction only. The key idea: split an image into patches (16×16 squares),
treat each patch as a token, feed the sequence through a transformer ENCODER —
ViT is encoder-only — with a classification head on top. The 2020 paper: "An Image
is Worth 16x16 Words."

ViT is covered here to plant the seed: transformers are not just for text. Image
patches are tokens. Position matters. Full attention derivation comes in 5.4.

**Advanced cue:** "ViT scales better than CNNs above roughly 300M images. Below
that, CNNs with strong inductive biases still win — which is why hybrids like Swin
are common in practice."

**Transition:** "Now the first Kailash engine of the lesson — OnnxBridge."

---

## Slide 48: Kailash Bridge: OnnxBridge

**Time:** ~2 min

OnnxBridge wraps the PyTorch ONNX exporter. Teach the three API facts students trip
on:

1. The framework argument is `"torch"` — `"pytorch"` is not supported and returns
   `success=False`.
2. Export does NOT validate and does not raise on failure — check `res.success`,
   then call `validate()` separately, which runs the model's `predict()` and ONNX
   Runtime on the same rows and compares them.
3. Trace with at least 2 rows in `sample_input` — a batch-of-1 trace freezes the
   batch size at 1. And pass `output_path` as a `pathlib.Path`, not a string.

Mention FlatImageAdapter (from the exercise helpers): it reshapes flat pixel rows
back to (3, 32, 32) because InferenceServer feeds one flat row per request record.

Connect forward: students export their CIFAR-10 ResNet-SE in Exercise 5.2 file 03,
register it, and serve it; Lesson 5.7 repeats the flow for the transfer model.

**Transition:** "Time to build. Here is the exercise."

---

## Slide 49: Exercise 5.2: CNN Classification Pipeline

**Time:** ~1 min

Exercise 5.2 is four files on CIFAR-10 (50K colour 32×32 images): 01 simple CNN,
02 ResNet-SE with Grad-CAM, 03 the production pipeline (OnnxBridge export +
validate, ModelRegistry, InferenceServer, latency benchmark), 04 a learning-rate
and augmentation study.

Students should see a clear accuracy jump from the simple CNN to ResNet-SE. Mixup,
label smoothing and Kaiming init are taught on the enhancements slide but are not
in the files — offer them as an extension. The ONNX export connects to the
deployment story in Lesson 5.7.

**Transition:** "Quick reference for the key formulas."

---

## Slide 50: Lesson 5.2 Key Formulas

**Time:** ~1 min

Quick reference for the four key formulas: conv output size (W − F + 2P)/S + 1,
ResNet skip H(x) = F(x) + x, SE gating s = sigmoid(W₂·ReLU(W₁·GAP(x))), Kaiming
init variance 2/n_in for ReLU.

Students should be able to compute conv output sizes by hand and explain in one
sentence why the +x in ResNet keeps gradients flowing.

**Transition:** "Lesson summary."

---

## Slide 51: Lesson 5.2 Summary

**Time:** ~1 min

CNNs capture spatial features through convolutions, pooling, and hierarchical
depth. ResNet skip connections let deep networks train. SE blocks add channel
recalibration for almost free. Modern training tricks compound for real accuracy
gains.

Bridge: CNNs handle spatial data (images). What about sequential data (text, time
series)? We need an architecture that remembers previous inputs. That is the RNN.

**Transition:** "Where do CNNs earn their keep?"

---

## Slide 52: Lesson 5.2: Real-World Applications

**Time:** ~3 min

The retail example lands best with business audiences: "most supermarkets you shop
in already have cameras; turning them into a shelf monitor is the opportunity."
For technical audiences, the manufacturing quality-control ROI story is stronger.

State clearly: the organisations and figures on this slide are illustrative
scenarios, not reported results from named companies.

Tie back to the exercise: "You just exported a CNN to ONNX. That is the exact
production format every one of these applications uses."

**Transition:** "From space to time — RNNs."

---

## Slide 53: RNNs and Sequence Models

**Time:** ~1 min

Title the lesson. Transition from spatial (CNN) to temporal (RNN) — it deserves a
pause. The key difference: order matters. In an image, pixel (3,4) is always next
to (3,5). In a sequence, the meaning of a word depends on what came before.
History matters.

**Advanced cue:** "RNNs feel dated in 2026, but they are still the clearest way to
teach sequence modelling, and LSTMs still ship on edge devices where transformers
are too heavy."

**Transition:** "What makes an RNN recurrent?"

---

## Slide 54: Recurrent Neural Networks

**Time:** ~3 min

Draw the unrolled RNN on the board. Show how the same weight matrix is applied at
every time step, and how the hidden state h_t carries memory from previous steps:
h_t = tanh(W_x·x_t + W_h·h_{t−1} + b).

Introduce the central problem: vanishing gradients. When you unroll the RNN over
many time steps, gradients either shrink to zero or explode. By time step 20, the
influence of time step 1 has been multiplied by W_h twenty times and effectively
erased. Long-range dependencies are unreachable. This motivates everything that
follows in the lesson.

**Beginner cue:** "Imagine whispering a message down a line of 20 people. By the
end, the message is unrecognisable. The vanilla RNN has exactly this problem."

**Transition:** "LSTM is the fix."

---

## Slide 55: LSTM: Long Short-Term Memory

**Time:** ~3 min

The LSTM is the most important sequence architecture before transformers. Four
components: the cell state (the memory highway), forget gate, input gate, output
gate.

The key insight is the ADDITIVE cell state. In C_t = f_t·C_{t−1} + i_t·candidate,
gradients flow through the addition: dC_t/dC_{t−1} = f_t. When f_t is near 1, the
gradient passes through nearly unchanged — so the forget gate literally decides how
much gradient survives from one time step to the next. That is the cell-state
highway: long-range gradients travel along C_t without being crushed by repeated
multiplication.

**Beginner cue:** "You do not have to memorise the equations. What matters is the
idea: a separate memory pathway that gradients can travel down without being
squashed."

**Advanced cue:** "Hochreiter & Schmidhuber, 1997 — a decade ahead of its time. The
cell-state pathway is structurally identical to the residual connections computer
vision rediscovered as ResNet in 2015."

**Transition:** "Let us derive the six equations."

---

## Slide 56: LSTM: The Six Equations (1–4)

**Time:** ~4 min

Walk through each equation slowly. Six in total — three gates, a candidate, the
cell update, and the hidden state; this slide covers the first four.

- Forget gate: f_t = σ(W_f·[h_{t−1}, x_t] + b_f) — what to throw away from the cell
  state.
- Input gate: i_t = σ(W_i·[h_{t−1}, x_t] + b_i) — what new information to store.
- Candidate: C̃_t = tanh(W_C·[h_{t−1}, x_t] + b_C) — the new candidate values.
- Cell update: C_t = f_t·C_{t−1} + i_t·C̃_t — forget some old, add some new.

The concatenation [h_{t−1}, x_t] means every gate decision depends on BOTH the
previous hidden state and the current input. All three gates use sigmoid (output in
[0,1]) because they are soft binary decisions (keep/discard). The candidate uses
tanh (output in [−1,1]) because it is new information.

**Beginner cue:** "Forget some of the past. Decide what new thing to remember. Add
them together. That is the cell update."

**Transition:** "Two equations left — how the hidden state comes out."

---

## Slide 57: LSTM: Output Gate and Hidden State (5–6)

**Time:** ~3 min

Complete the six equations:

- Output gate: o_t = σ(W_o·[h_{t−1}, x_t] + b_o) — which parts of the cell state to
  expose.
- Hidden state: h_t = o_t·tanh(C_t) — the cell state squashed through tanh, masked
  by the output gate.

The hidden state h_t is what goes to the next time step AND what is used for
prediction at the current step. The cell state C_t is internal memory only — it
never leaves the cell. Emphasise the tanh(C_t) squash: it keeps the hidden state
bounded, and o_t selects which dimensions to expose.

**Beginner cue:** "Think of C_t as private notes and h_t as what you say out loud.
The output gate decides which parts of your notes to share."

**Transition:** "GRU is the simpler cousin."

---

## Slide 58: GRU: Gated Recurrent Unit

**Time:** ~3 min

GRU (2014) merges the forget and input gates into a single update gate z_t, and
adds a reset gate r_t that controls how much of the previous hidden state to use
when computing the candidate. Fewer parameters, faster training, similar
performance on most tasks — the practical default for many sequence tasks.

In practice: try both LSTM and GRU on your task and keep whichever trains better.
Students do exactly this comparison in Exercise 5.3 file 05.

**Advanced cue:** "Empirically the GRU–LSTM gap is within noise on most benchmarks.
The cases where LSTM wins are extremely long sequences and certain RL settings."

**Transition:** "Now the idea that leads us to transformers — attention."

---

## Slide 59: Attention Mechanisms for Sequences

**Time:** ~3 min

The bottleneck problem with RNNs: the entire sequence context must fit in the final
hidden state. For long sequences, information is lost.

Attention's answer: instead of forcing everything through the last hidden state,
let the model decide which time steps matter for the current prediction — weight
ALL hidden states by relevance and combine them. This directly leads to
self-attention in Lesson 5.4.

**Beginner cue:** "Instead of making the student write a 500-word essay from
memory, give them permission to look back at the source text and highlight the
sentences they need. That is attention."

**Advanced cue:** "Bahdanau attention (2014) introduced additive attention for
neural machine translation — the prelude to 'Attention Is All You Need' three years
later."

**Transition:** "Two ways to push sequence models further."

---

## Slide 60: Going Deeper: Stacked LSTMs & Spatial Attention

**Time:** ~3 min

Two ways to get more out of a sequence model.

Depth: stacking LSTMs — `nn.LSTM(num_layers=...)` does plain stacking — learns
hierarchical temporal features. But deep stacks train badly for the same
optimisation reason plain CNNs did, so wrap each layer in a residual connection
plus LayerNorm.

Spatial attention: temporal attention (previous slide) weights time steps; spatial
attention weights relationships BETWEEN input features, using multi-head attention
with each feature as a token. The resulting (F, F) weight matrix is interpretable —
which features each feature looks at.

Both are previews: residual + LayerNorm + multi-head attention are exactly the
ingredients of the transformer block in Lesson 5.4.

**Transition:** "How do we measure sequence models?"

---

## Slide 61: Sequence Model Metrics

**Time:** ~2 min

Perplexity is the standard metric for language models: PP = exp(−(1/N)·Σ log
P(w_i)) — the exponential of mean cross-entropy. Lower is better; intuitively, "how
surprised is the model per token." A perplexity of 100 means the model is as
uncertain as if choosing uniformly among 100 options.

Students use MAE/RMSE for time-series prediction in the exercise. BLEU (n-gram
overlap with human references) is mentioned for completeness but is not assessed in
this module.

**Transition:** "Two practical applications bring RNNs to life."

---

## Slide 62: Applications: Finance and Text

**Time:** ~2 min

Two practical applications. Financial prediction is the relevant one for
professionals; text generation makes the technology tangible and connects to LLMs.

Be honest about scope: Exercise 5.3 forecasts prices from daily
Close/High/Low/Volume bars. Technical indicators (RSI, MACD, Bollinger) and
character-level text generation — with perplexity = exp of the mean cross-entropy —
are taught here on the slide but are not yet exercise files. Set them as
extensions: compute the indicators as features and re-run the comparison; or train
the char-LSTM and report its perplexity.

**Beginner cue:** "Stock prices are just a sequence of numbers over time. Text is a
sequence of characters. Same architecture, different data."

**Transition:** "Time for the sequence exercise."

---

## Slide 63: Exercise 5.3: Sequence Modelling

**Time:** ~1 min

Exercise 5.3 is five files on real daily prices for the STI and five APAC/US
tickers — Close, High, Low, Volume; 20-day lookback, 5-day horizon: vanilla RNN,
LSTM, GRU, LSTM + temporal attention, and a fair three-way comparison.

The spec's technical indicators (RSI, MACD, Bollinger) and character-level text
generation are not in the files yet — offer them as extensions.

One non-negotiable: gradient clipping. The shared training loop clips at
max_norm=1.0. Without it, exploding gradients can derail training on the first long
sequence. Students should include clipping in every RNN loop they ever write.

**Transition:** "Quick summary before transformers."

---

## Slide 64: Lesson 5.3 Summary

**Time:** ~1 min

RNNs process sequences one step at a time, carrying a hidden state. Vanilla RNNs
forget quickly (vanishing gradients). LSTMs and GRUs add gated cell states that
preserve long-range information. Attention lets the model look back across all time
steps.

Bridge: the attention mechanism in this lesson is the seed. Transformers ask: what
if attention was ALL you need? The answer changed everything.

**Transition:** "Where do sequence models earn their keep?"

---

## Slide 65: Lesson 5.3: Real-World Applications

**Time:** ~3 min

Focus on the "lead time, not prediction" framing — that is the line a non-technical
exec will remember. A forecast that a patient's vitals will deteriorate in six
hours is worth far more than a perfect diagnosis of the present; the clinical
deterioration example is the emotional hook. The rail-ridership demand-forecasting
example works well as the relatable case for Singaporean audiences.

State clearly: these are illustrative scenarios, not claims about any named
organisation's systems.

Note the GRU-vs-LSTM trade-off: students measured it themselves in Exercise 5.3, so
the retail distributor example closes the loop back to their own notebook.

**Transition:** "The paper that changed everything."

---

## Slide 66: Transformers

**Time:** ~1 min

Title the lesson. "Attention Is All You Need," Vaswani et al., 2017 — the paper
that made GPT, BERT, and every modern LLM possible. Transformers replaced RNNs for
almost every sequence task. This lesson derives self-attention from scratch; by the
end, students will have implemented it in PyTorch without looking at a reference.

**Beginner cue:** "Every AI product you have heard about in the last three years is
built on this architecture. The next 30 minutes are the foundation of the
industry."

**Transition:** "Start with the core idea — query, key, value."

---

## Slide 67: Self-Attention: The Core Idea

**Time:** ~3 min

This is the conceptual foundation — make sure the Q/K/V analogy lands before moving
to the math. Every input token is projected into three learned vectors: Query
(what am I looking for), Key (what do I contain), Value (what do I offer). The dot
product Q·K measures similarity; softmax turns similarities into weights; the
weighted sum of values is the output.

Use the library analogy: "If you are looking for information about cats, your query
is 'cats'. Each book has a key — its title. You compare. High similarity = high
attention weight. The weighted sum of the books' contents is what you read."

**Beginner cue:** "Three copies of the same input, each with a different purpose.
Query says what you want. Key says what I have. Value says what I contain. That is
the whole trick."

**Transition:** "Now the equation."

---

## Slide 68: Scaled Dot-Product Attention

**Time:** ~4 min

This is THE equation: Attention(Q, K, V) = softmax(QK^T / √d_k)·V. Walk each step:
(1) QK^T — raw similarity scores between every query and every key; (2) divide by
√d_k — the scaling; (3) softmax per row — scores become weights; (4) multiply by V
— weighted sum of values.

The scaling is crucial and often asked in interviews. For d_k = 512, the expected
magnitude of dot products is √512 ≈ 22.6. Without scaling, softmax would be nearly
one-hot, and gradients at the non-peak positions vanish. Scaling keeps the
distribution soft and gradients healthy.

**Beginner cue:** "This one equation is the entire transformer block. Everything
else is stacking and normalising this operation."

**Advanced cue:** "The dot-product form is efficient because it is a single batched
matmul. Additive attention (Bahdanau) needs a small MLP at every pair — much
slower."

**Transition:** "Why exactly do we divide by √d_k?"

---

## Slide 69: Why Divide by \sqrt{d_k}?

**Time:** ~3 min

This derivation is for understanding, not memorisation. Assume each component of Q
and K has mean 0 and variance 1. The dot product Q·K sums d_k such products, so its
variance is d_k and its standard deviation is √d_k.

Without scaling, dot products grow with dimension; softmax of a large value becomes
one-hot; gradients at non-peak positions die; training stalls. Dividing by √d_k
restores variance to O(1) regardless of dimension. Softmax stays soft. Gradients
flow.

And yes — companies ask this in interviews. It is the same unit-variance argument
behind layer norm, Kaiming init, and batch norm. It is everywhere once you see it.

**Transition:** "One head is not enough. Multi-head attention."

---

## Slide 70: Multi-Head Attention

**Time:** ~3 min

MultiHead(Q, K, V) = Concat(head₁, …, head_h)·W^O. Each head is scaled dot-product
attention with its own learned projections.

Multi-head attention seems expensive but is actually free: the per-head dimension
is reduced proportionally (d_model/h), so total compute stays the same. The benefit
is diverse attention patterns. Analogy: instead of one person reading a document, 8
people each read for a different purpose, then combine their notes.

**Advanced cue:** "Different heads specialise — famous visualisations show heads
that track subject-verb relations, phrase heads, punctuation boundaries. Many heads
are redundant, which is why head-pruning research exists."

**Transition:** "One problem — transformers have no idea about order."

---

## Slide 71: Positional Encoding

**Time:** ~3 min

Attention is permutation-invariant: shuffle the tokens and the output shuffles the
same way. The transformer has no inherent sense of order, so we add a position
vector to each token embedding. The original paper's sinusoidal form:

PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/d)).

Get the frequency direction right — many blogs state this backwards, so check it
from the formula: each dimension pair oscillates at 10000^(−2i/d) radians per
position. LOW dimensions (small i) change RAPIDLY — at i = 0 the frequency is 1
rad/position — so they capture fine, local position. HIGH dimensions change SLOWLY
(frequency approaching 1e-4) and capture coarse, global position. Point at the
heatmap: the top row flips colour every few positions; the bottom rows are almost
flat.

The linear-function property matters too: PE(pos + k) is a linear function of
PE(pos) for any fixed offset k, so the model can learn to attend to RELATIVE
positions.

**Advanced cue:** "Learned positional embeddings (BERT) also work; rotary position
embeddings (RoPE) are the current standard for long-context models."

**Transition:** "Now the full architecture — encoder and decoder."

---

## Slide 72: Transformer Architecture

**Time:** ~3 min

Walk both blocks. The encoder is simpler: multi-head self-attention → residual +
layer norm → feed-forward → residual + layer norm, stacked N times. No masking, no
cross-attention.

The decoder has two types of attention: masked self-attention (prevents looking at
future tokens during autoregressive generation) and cross-attention (queries from
the decoder, keys and values from the encoder). Then feed-forward, with residuals
and layer norm throughout.

The residual connections + layer norm are critical for training stability —
removing either causes training to diverge.

**Beginner cue:** "The encoder understands the input. The decoder generates the
output token by token, looking back at the encoder whenever it needs context. That
is translation in one paragraph."

**Transition:** "Let us see the whole machine at once."

---

## Slide 73: Encoder-Decoder: The Full Picture

**Time:** ~3 min

This is the full picture — walk it left to right. The encoder processes the input
sequence in parallel: no masking, all tokens see each other. The encoder output
flows as K and V into the decoder's cross-attention.

The decoder is autoregressive: masked self-attention prevents it from seeing future
tokens, and cross-attention lets it "look back" at the encoder output at every
step. This is how translation works: the encoder reads English, the decoder
generates French one token at a time, cross-attending to the English at each step.

**Beginner cue:** "Encoder reads the whole sentence at once. Decoder writes the
translation one word at a time, checking back with the encoder after every word."

**Transition:** "The transformer family tree."

---

## Slide 74: Transformer Variants

**Time:** ~2 min

Quick survey — students should know the landscape, not memorise every variant.
Three branches: encoder-only (BERT, RoBERTa — understanding, classification),
decoder-only (GPT, LLaMA — generation), encoder-decoder (T5, BART — translation,
summarisation). ViT belongs to the encoder-only branch, applied to image patches.

Efficiency variants (Transformer-XL, Reformer, Longformer) exist for long contexts
— name them only.

The distinction that matters is encoder vs decoder vs both: bidirectional
understanding vs autoregressive generation. BERT is the fine-tuning exercise in
this lesson; GPT is the generation paradigm covered in M6.

**Transition:** "Consolidation moment — we have seen four paradigms."

---

## Slide 75: Consolidation: Four Paradigms Compared

**Time:** ~3 min

This is the consolidation point. Students have now seen autoencoder, CNN, RNN, and
Transformer. Pause and connect them.

Quick matching game with the room: "A dataset of satellite images — which
architecture?" (CNN/ViT.) "Patient records over time?" (LSTM/Transformer.)
"Unlabelled images for pretraining?" (Autoencoder.) "A social network?" (GNN —
coming in 5.6.)

Keep the comparison table on screen while you work through examples.

**Beginner cue:** "You do not choose architectures in the exercises — we tell you
which one. But by the end of the module, you should look at a new problem and pick
the right family in 30 seconds."

**Advanced cue:** "Architectures are converging on transformers, but inductive
biases still matter: CNNs for small vision data, LSTMs for edge devices, gradient
boosting for tabular. No silver bullet."

**Transition:** "Time for the transformer exercise."

---

## Slide 76: Exercise 5.4: Transformers

**Time:** ~1 min

Exercise 5.4 is five files, all on AG News (120K news headlines, 4 topic classes —
World, Sports, Business, Sci/Tech): attention from scratch, a Transformer encoder,
a BiLSTM baseline, BERT fine-tuning (bert-base-uncased, HuggingFace Transformers),
and a three-way comparison.

The from-scratch implementation is crucial — students should be able to write the
attention function without a reference by the end.

Flag two reading traps for the results: (1) report test accuracy for the checkpoint
chosen on the VALIDATION split, not the best epoch on test; (2) a news-TOPIC
classifier cannot score sentiment — high softmax confidence on out-of-domain text
is not evidence of signal.

**Transition:** "Key formulas reference."

---

## Slide 77: Lesson 5.4 Key Formulas

**Time:** ~1 min

Quick reference. Four equations define the transformer: scaled dot-product
attention, multi-head attention, sinusoidal positional encoding, layer norm.
Students should know them by heart.

"These four lines are the core of every modern AI product. If you remember nothing
else from today, remember these."

**Transition:** "Summary and bridge."

---

## Slide 78: Lesson 5.4 Summary

**Time:** ~1 min

Self-attention lets every token attend to every other token. Scaling by √d_k keeps
softmax healthy. Multi-head runs attention in parallel with diverse patterns.
Positional encoding restores order. The full transformer stacks these blocks with
residuals and layer norm.

Bridge: the transformer lesson completes the core architecture sequence — four
paradigms for LEARNING from data. Now we shift to GENERATING new data.

**Transition:** "Where do transformers earn their keep?"

---

## Slide 79: Lesson 5.4: Real-World Applications

**Time:** ~3 min

The tax-query routing example is the cleanest "you could deliver this on Monday"
case: every organisation has a pile of customer tickets or emails that need
routing, and a fine-tuned encoder classifies them with a few thousand labelled
examples. State that all organisations on the slide are illustrative.

Emphasise the data-requirement drop: "A BERT fine-tune needs a few thousand
labelled examples, not a few million. That is the difference between a six-month
project and a six-week project."

**Transition:** "Two networks playing a game — GANs."

---

## Slide 80: Generative Models — GANs and Diffusion

**Time:** ~1 min

Title the lesson. GANs are one of the most creative ideas in machine learning: a
generator that creates fake data and a discriminator that tries to detect it. The
competition drives both to improve.

**Beginner cue:** "Imagine a forger and a detective. The forger gets better at
faking, the detective gets better at spotting fakes, and they keep training each
other up."

**Transition:** "Meet the two players."

---

## Slide 81: GAN: Generator vs Discriminator

**Time:** ~3 min

Generator G: takes random noise z from a prior, outputs a fake sample G(z) that
should look real. Discriminator D: takes a sample (real or fake), outputs the
probability it is real. Training alternates: update D to better distinguish, update
G to better fool. The loss is binary cross-entropy on D's outputs.

The adversarial framework is elegant but tricky to train. The alternating
optimisation means you must balance G and D carefully: if D is too strong, G gets
no useful gradients; if G is too strong, D cannot provide useful signal.

**Beginner cue:** "Two networks, two losses, alternating updates. That is the whole
framework. The tricky part is making the alternation stable."

**Transition:** "The mathematical objective."

---

## Slide 82: GAN Minimax Objective

**Time:** ~3 min

The objective: min_G max_D [E_x log D(x) + E_z log(1 − D(G(z)))]. D maximises it
(distinguish well); G minimises it (fool D). The inner max is solved at
D*(x) = p_data(x) / (p_data(x) + p_G(x)); substituting back gives a Jensen-Shannon
divergence between the real and generated distributions.

Walk the practical training trick: instead of minimising log(1 − D(G(z))), maximise
log D(G(z)). The original minimax G loss saturates when G is poor — D confidently
rejects the fakes, D(G(z)) is near 0, and log(1 − D(G(z))) is flat there. The
non-saturating version keeps a strong gradient in exactly that regime. This is what
Exercise 5.5 uses: BCE with "real" labels on the fakes, i.e. minimise −log D(G(z)).

Then the honest limitation: the non-saturating trick does not make GANs stable.
When the real and fake supports do not overlap, the JS divergence is stuck at its
maximum, log 2, and gives no useful direction. That motivates WGAN.

**Advanced cue:** "The D* derivation is worth doing once on paper — it is the
cleanest explanation of why the GAN objective estimates a divergence at all."

**Transition:** "Let us look at the architectural standard — DCGAN."

---

## Slide 83: DCGAN: Deep Convolutional GAN

**Time:** ~3 min

DCGAN established the standard architecture guidelines for convolutional GANs: no
pooling (use strided convolutions), batch norm in both networks, ReLU in G except
tanh on the output, LeakyReLU in D. The generator upsamples from a noise vector
with ConvTranspose2d; the discriminator is a strided CNN down to a single score.

Be precise with students: Exercise 5.5's generator and discriminator are MLPs
(fully connected), not DCGANs — the adversarial training loop is identical, only
the networks differ. Swapping in this convolutional generator plus a strided-conv
discriminator is the natural extension.

**Advanced cue:** "DCGAN was 2015, and the guidelines have held up — modern GANs
still use most of them. Progressive growing (StyleGAN) and attention (SAGAN) are
the main refinements since."

**Transition:** "WGAN is the stability upgrade."

---

## Slide 84: WGAN: Wasserstein GAN

**Time:** ~3 min

The key insight: replace Jensen-Shannon divergence with Wasserstein distance (Earth
Mover's Distance). Wasserstein varies smoothly with how far apart the distributions
are, so G always gets a useful gradient. JS divergence does not: when the supports
do not overlap it saturates at the constant log 2, and G receives no gradient at
all.

The objective is min_G max_D [E D(x) − E D(G(z))] with a Lipschitz constraint on D.
The original WGAN enforced the constraint with weight clipping; modern WGAN-GP uses
a gradient penalty, λ·(||∇D||₂ − 1)², added to the critic loss.

Teach the curve reading, because it surprises everyone: the critic loss (without
the GP term) is about minus the Wasserstein distance, so as G improves it RISES
toward 0. A more negative critic loss means the distributions are further apart —
the opposite of every other loss in the module. A rising-toward-zero critic loss is
the training working, not failing.

**Advanced cue:** "The Lipschitz constraint is what makes Wasserstein distance
computable via Kantorovich-Rubinstein duality. The gradient penalty is an elegant
soft enforcement."

**Transition:** "The GAN family in one table."

---

## Slide 85: GAN Variants (Survey)

**Time:** ~2 min

Survey only: Conditional GAN (class-conditioned generation), CycleGAN (unpaired
image-to-image — horses to zebras), StyleGAN (style-based, progressive growing, the
famous "this person does not exist" faces), Pix2Pix (paired image-to-image), BigGAN
(large-scale class-conditional).

Evaluation: FID (Frechet Inception Distance) is the standard — a distribution-level
score covering fidelity AND diversity, not a per-image quality score. Be precise
about the exercise: students compute FID with a small 64-d LeNet feature extractor
trained on MNIST, not Inception — so their numbers are comparable only with each
other, never with published Inception-FID thresholds. The exercise does not compute
Inception Score.

**Advanced cue:** "FID correlates with human quality judgments better than
Inception Score, but it still has failure modes — it can miss mode dropping and
reward texture artefacts."

**Transition:** "Diffusion is the current state of the art."

---

## Slide 86: Diffusion Models (Brief)

**Time:** ~3 min

Brief introduction — students do not implement diffusion models in this module (too
compute-intensive for exercises).

DDPM: gradually add Gaussian noise to an image over T steps until it becomes pure
noise; train a network to REVERSE the process — predict the noise at each step so
it can be subtracted. Generation starts from pure noise and applies the trained
denoiser iteratively until a clean sample appears.

Advantages over GANs: more stable training, better diversity, no mode collapse.
Disadvantages: slow generation (many forward passes), compute-intensive training.
Stable Diffusion is the practical application students have likely encountered.

**Beginner cue:** "Start from noise and clean it up step by step. Each step the
network says 'this pixel should be a little less noisy.' After enough steps, you
have an image."

**Advanced cue:** "Classifier-free guidance is the key trick that made conditional
diffusion work for text-to-image; the score-matching and SDE connections give the
field its mathematical depth."

**Transition:** "Training GANs is notoriously hard. Let us name the failure modes."

---

## Slide 87: GAN Training Challenges

**Time:** ~2 min

GAN training is notoriously finicky, and students will hit these in the exercise.
Mode collapse: the generator produces only a few output modes, ignoring most of the
real distribution — the classic symptom is every fake looking the same. Training
instability: oscillating losses, sudden quality drops, non-convergence. Evaluation
difficulty: vanilla GAN losses do not correspond to image quality (WGAN's critic
loss does — it tracks the Wasserstein distance).

Mitigations: WGAN-GP (the most stable default), spectral normalisation, one-sided
label smoothing, balanced update ratios.

**Beginner cue:** "GAN training is more like a weather system than a deterministic
optimisation. Expect oscillation. Monitor sample quality, not just loss numbers."

The key message: WGAN-GP is the default for new projects.

**Transition:** "Exercise time."

---

## Slide 88: Exercise 5.5: Generative Models

**Time:** ~1 min

Exercise 5.5 is three files on MNIST: a vanilla GAN (MLP networks, non-saturating
BCE loss), WGAN-GP, and an evaluation file.

The key comparison is vanilla GAN vs WGAN-GP. Students should see that the WGAN
critic loss tracks quality — it rises smoothly toward 0 as the generator improves —
while the vanilla GAN losses oscillate and say nothing. FID gives an objective
distribution-level comparison, and the nearest-neighbour novelty check gives a
first memorisation test.

The convolutional DCGAN is taught on the slides; building it is the extension.

**Transition:** "Lesson summary."

---

## Slide 89: Lesson 5.5 Summary

**Time:** ~1 min

GANs learn to generate by adversarial competition. WGAN stabilises training with
the Wasserstein distance. Diffusion models are the current SOTA, by noising and
denoising. FID is the standard evaluation metric.

Bridge: we have covered grids (CNN), sequences (RNN, Transformer), and generation
(VAE, GAN, diffusion). The next data structure is graphs — irregular, connected,
variable-size.

**Transition:** "Where does generative modelling earn its keep — and what must you
never claim about it?"

---

## Slide 90: Lesson 5.5: Real-World Applications

**Time:** ~3 min

Give decision-makers in regulated industries the honest version: synthetic data is
NOT private by default. Generative models can memorise; extraction attacks on
diffusion models are documented (Carlini et al., 2023). Sharing a model trained on
patient or customer data — or its samples — needs formal privacy guarantees
(DP-SGD), memorisation and membership-inference testing, and legal review.

Be equally clear about what FID does not say: FID answers "is the synthetic data
realistic and diverse?", never "is it private?" — a model that copies the training
set gets a great FID. Exercise 5.5's nearest-neighbour novelty check is a first
memorisation test, not a privacy guarantee.

State that the organisations on the slide are generic illustrations.

**Transition:** "Into the graph world."

---

## Slide 91: Graph Neural Networks

**Time:** ~1 min

Title the lesson. Graphs are everywhere: social networks, molecular structures,
knowledge graphs, citation networks, road networks. GNNs extend deep learning to
this non-Euclidean domain.

The key insight: a node's representation depends on its neighbours.

**Beginner cue:** "Whenever your data has connections — people who know people,
atoms bonded to atoms, papers citing papers — that is a graph. GNNs are how we
learn from it."

**Transition:** "Start with the vocabulary."

---

## Slide 92: Graph Data: Nodes, Edges, Adjacency

**Time:** ~2 min

Establish the vocabulary. Graph G = (V, E): V is the set of nodes, E the set of
edges. The adjacency matrix A is the mathematical representation of connections —
A_ij = 1 if an edge connects i and j — stored as an edge list for large sparse
graphs. Node features X are what we know about each entity: user demographics in a
social network, atom types in a molecule.

GNNs combine A and X to produce learned representations of each node.

**Beginner cue:** "A graph is just a table of connections plus a table of node
properties. The model learns to smear information along the connections."

**Transition:** "The GCN operation."

---

## Slide 93: GCN: Graph Convolutional Networks

**Time:** ~4 min

The GCN equation — H^(l+1) = σ(D^(−1/2)·A·D^(−1/2)·H^(l)·W^(l)) — looks complex,
but the intuition is simple: average your neighbours' features (normalised by
degree), transform with a learnable matrix, activate.

The D^(−1/2) A D^(−1/2) normalisation is the spectral theory connection — the
symmetric normalised Laplacian from graph signal processing. Students do not need
the math; they need the intuition: "degree-smoothed average of neighbours", which
keeps high-degree nodes from dominating.

After L layers, a node's representation reflects its L-hop neighbourhood. This is
why 2–3 layers is usually sufficient — more layers cause oversmoothing, where all
nodes become indistinguishable. The next slide visualises one round of message
passing.

**Beginner cue:** "Each layer, every node looks at its neighbours, averages their
features, transforms, and updates. Do that twice and each node has seen information
from 2 hops away."

**Transition:** "One picture of one round of message passing."

---

## Slide 94: GCN: Message Passing in One Picture

**Time:** ~2 min

Walk the diagram. Point at the centre node v and trace the green arrows from each
neighbour into it. Then say: "After this aggregation step, v's hidden state
reflects information from all four neighbours. Run another layer and it reaches
their neighbours too — that is how GNNs propagate information through a graph."

Connect back to the equation on the previous slide: the diagram IS
D^(−1/2) A D^(−1/2) H W for one node — sum the neighbours' transformed features,
normalise, activate.

**Transition:** "Two important variants."

---

## Slide 95: GraphSAGE and GAT

**Time:** ~3 min

GCN treats all neighbours equally. The two variants relax that in different
directions.

GraphSAGE (Sample and Aggregate): SAMPLE a fixed number of neighbours per layer
instead of using the full neighbourhood — this makes GNNs scalable to very large
graphs, and inductive: it can handle unseen nodes at inference time, unlike
transductive GCN.

GAT (Graph Attention Network): learn attention weights over neighbours, e_ij =
LeakyReLU(a^T [Wh_i || Wh_j]) then softmax over j. Different neighbours get
different importance — self-attention restricted to the graph's edge structure,
connecting directly back to Lesson 5.4.

These three cover the core design space: spectral (GCN), sampling (GraphSAGE),
attention (GAT).

**Beginner cue:** "GCN is 'average your friends.' GraphSAGE is 'average a random
subset of your friends to handle big networks.' GAT is 'weight your friends by how
relevant they are to you.'"

**Transition:** "One more architecture completes the set — GIN."

---

## Slide 96: GIN: Graph Isomorphism Network

**Time:** ~3 min

GIN completes the four architectures in the spec: GCN (normalised averaging),
GraphSAGE (sampling), GAT (attention), GIN (sum aggregation).

The key idea is expressiveness. Mean and max aggregators lose information about how
MANY neighbours share a feature, so two different neighbourhoods can map to the
same vector. Summing and passing through an MLP keeps them apart, which makes GIN
as discriminative as the Weisfeiler-Lehman test — the theoretical ceiling for
message-passing GNNs.

The code on the slide is the standard torch_geometric graph-classification pattern:
TUDataset for a benchmark of many small graphs, the PyG DataLoader that batches
graphs into one big disconnected graph plus a batch vector, conv layers, then a
readout (global_add_pool) giving one row per graph.

Position it honestly: Exercise 5.6 itself does node classification on Cora with
layers written from scratch; this graph-classification pattern is the extension.

**Transition:** "What can GNNs actually predict?"

---

## Slide 97: GNN Task Types

**Time:** ~2 min

The three task types correspond to different levels of the graph hierarchy.

Node classification: a label per node (fraud detection, topic classification of
papers) — the most common, and the one Exercise 5.6 does. Graph classification: a
label per whole graph (molecule toxicity) — needs a readout function (sum, mean,
max pooling) to aggregate node embeddings. Link prediction: whether an edge should
exist (recommender systems) — score pairs of node embeddings; Exercise 5.6 scores
link prediction on held-out edges.

**Beginner cue:** "Same GNN body, different heads, different tasks. Like the
classification vs regression split in M3."

**Transition:** "Exercise time."

---

## Slide 98: Exercise 5.6: Graph Neural Networks

**Time:** ~1 min

Exercise 5.6 is five files on the Cora citation network — torch_geometric only
loads the data: GCN, GAT and GraphSAGE layers written from scratch for node
classification, link prediction scored on held-out edges, and a three-way
comparison.

Students focus on building the layers and interpreting results. The attention
weight visualisation is the highlight — it shows which citations the model
considers important. Interpretability for free.

Selection discipline to enforce: pick the epoch by VALIDATION accuracy and report
test accuracy at that epoch — never the best test epoch. Graph classification on
TUDataset with GINConv/GCNConv + global pooling (the previous slides) is the
extension.

**Transition:** "Lesson summary."

---

## Slide 99: Lesson 5.6 Summary

**Time:** ~1 min

GNNs extend deep learning to graph-structured data through message passing. GCN
averages neighbours. GraphSAGE samples them. GAT attends over them. GIN sums them
through an MLP for maximum expressiveness. Three task types: node, graph, link.

Bridge: after six lessons of building architectures from scratch, students are
ready for the practical shortcut — start with pre-trained models and fine-tune.

**Transition:** "Where do GNNs earn their keep?"

---

## Slide 100: Lesson 5.6: Real-World Applications

**Time:** ~3 min

The anti-money-laundering example is the strongest business case for a finance
audience — rules engines are expensive and brittle, and GNNs generalise across new
laundering typologies because they learn from the network structure, not just
account features. The drug discovery example is the strongest ROI case generally:
shrinking the wet-lab shortlist by orders of magnitude is where the money is. The
port example grounds the lesson in Singapore's role as a transhipment hub.

State clearly: all organisations here are generic illustrations; the figures are
not reported results.

One precision worth volunteering: GraphSAGE is not the only inductive option — GAT
is inductive too.

**Transition:** "The most practical technique in modern DL."

---

## Slide 101: Transfer Learning

**Time:** ~1 min

Title the lesson. Transfer learning is the most practical technique in modern deep
learning. Almost no one trains from scratch anymore. Pre-trained models on ImageNet
(vision) and large text corpora (language) provide universal features that transfer
to almost any downstream task.

**Beginner cue:** "For the rest of your career, this is how you will build models
for new problems. Start with someone else's pre-trained model. Swap the head.
Fine-tune. Done."

**Transition:** "The paradigm."

---

## Slide 102: The Transfer Learning Paradigm

**Time:** ~2 min

Transfer learning is the practical payoff for understanding architectures. Because
early CNN layers learn edges and early transformer layers learn grammar, these
features transfer across tasks. Later layers learn task-specific features — those
you replace.

Pre-training: a large model on a massive dataset learns general features.
Fine-tuning: attach a new head, train on your small target dataset — much faster,
much less data needed.

The smaller your dataset, the more you benefit. Below ~10k examples, transfer
learning is essentially required.

**Advanced cue:** "The scaling-laws literature makes this explicit: pre-trained
compute transfers logarithmically to downstream tasks."

**Transition:** "Let us fine-tune ResNet for computer vision."

---

## Slide 103: CV Transfer Learning: ResNet Fine-Tuning

**Time:** ~3 min

Walk through the code — it is the ResNet-18 recipe from Exercise 5.7. Note the API:
`weights=` replaces the deprecated `pretrained=True`; `DEFAULT` is the ImageNet-1k
checkpoint (~1.28M images).

The key decisions: which layers to freeze, what learning rate to use, how much
augmentation. Progressive unfreezing is the safest strategy — start with only the
head, then unfreeze one block at a time. Low learning rate is critical — too high
and you destroy the pre-trained features. Data augmentation matters more the
smaller the target dataset.

**Beginner cue:** "Load a model, freeze most of it, swap the last layer, train at a
very low learning rate. Three lines of PyTorch, massive accuracy improvement over
training from scratch."

**Advanced cue:** "Which layers to freeze is task-dependent. Very dissimilar
domains (satellite imagery vs natural photos) → unfreeze more. Similar domains →
keep most frozen."

**Transition:** "Now the NLP side."

---

## Slide 104: NLP Transfer Learning: BERT Fine-Tuning

**Time:** ~3 min

Same pattern as CV but with BERT. HuggingFace makes it straightforward: the Trainer
handles training, and the `pipeline()` API handles inference — tokenise, forward
pass and softmax in one call, returning a label and a score.

Key hyperparameters: learning rate (2e-5 to 5e-5 is standard), epochs (3–5
typical), batch size (16–32).

One API warning worth stating: if students write their own TrainingArguments, the
evaluation argument is `eval_strategy` — `evaluation_strategy` was removed in
recent transformers versions.

Adapters are mentioned as a preview of M6: instead of fine-tuning all weights,
insert small trainable modules inside the frozen backbone — the idea behind LoRA
and parameter-efficient fine-tuning. Full fine-tuning changes all weights; adapters
change only a few.

**Advanced cue:** "BERT fine-tuning is being displaced by zero/few-shot prompting
with large decoder-only LLMs. But for specialised domains with labelled data,
fine-tuned BERT-sized encoders are still the highest accuracy-per-dollar option."

**Transition:** "One big table to internalise."

---

## Slide 105: Architecture Selection Guide

**Time:** ~2 min

This table is a key reference for professional practice. Walk every row: Images →
CNN or ViT, always with transfer learning. Text → Transformer, always with transfer
learning. Sequences → LSTM or Transformer, sometimes transfer. Graphs → GNN, rarely
transfer. Tabular → gradient boosting (XGBoost, LightGBM), never transfer, train
from scratch.

The "tabular data resists transfer" point is important — many professionals work
with tabular data, and the advice is gradient boosting, not deep learning. This
connects back to M3 supervised learning.

**Beginner cue:** "Memorise this table. It will save you from months of
wrong-architecture projects."

**Transition:** "Two Kailash engines together — OnnxBridge and InferenceServer."

---

## Slide 106: Kailash Bridge: OnnxBridge + InferenceServer

**Time:** ~3 min

This is the deployment story, and it is exactly what Exercise 5.7 file 05 runs.
Three steps:

1. `OnnxBridge.export(model, "torch", output_path=..., sample_input=...)` — check
   `res.success`. Trace with 2 rows so the batch size is not frozen, and pass
   `output_path` as a `pathlib.Path`.
2. Register the model in the ModelRegistry and attach the `.onnx` file as
   `model.onnx` for that version.
3. `InferenceServer.from_registry(...)` is a coroutine: await it, `await start()`
   to load the ONNX file, then `await predict()` with `{"records": [...]}` — one
   record per image, one key per pixel, which is why the CNN is exported behind
   FlatImageAdapter.

`predict` returns a plain mapping — `{"predictions": [...]}`. Be explicit: there is
no `predict_batch`, no `warm_cache`, no `PredictionResult` in kailash-ml 2.x.

The served predictions come from ONNX Runtime, not PyTorch — file 05 compares the
two for parity.

**Beginner cue:** "Three calls. Export. Register. Serve. You do not need the
internals — you need to know when to call each."

**Transition:** "The exercise."

---

## Slide 107: Exercise 5.7: Transfer Learning Pipeline

**Time:** ~1 min

This is the most practical exercise in the module: five files, all on CIFAR-10
resized to 96×96 — scratch baseline, ResNet-18 transfer, a data-efficiency curve,
adapters, and production deployment.

Set expectations precisely: there is no mask-detection dataset and no BERT here —
BERT fine-tuning is Exercise 5.4.

Adapters start as an exact identity — the adapter adds only its residual
(`adapted − pooled`), so the pre-trained features are untouched at step 0 — and the
trainable share is about 1% of parameters.

The transfer vs from-scratch comparison is compelling: transfer learning should win
convincingly, especially on small data fractions. Students then read their own
data-efficiency curve on the applications slide.

**Transition:** "Lesson summary."

---

## Slide 108: Lesson 5.7 Summary

**Time:** ~1 min

Transfer learning takes pre-trained models and fine-tunes them on target tasks:
low learning rate, partial freezing, data augmentation. ResNet for CV, BERT for
NLP. OnnxBridge exports, InferenceServer serves.

Bridge: all seven lessons so far learn from STATIC data — images, text, sequences,
graphs. The last lesson is different. RL learns from INTERACTION with an
environment. No fixed dataset — the agent generates its own training data through
experience.

**Transition:** "Where does transfer learning earn its keep?"

---

## Slide 109: Lesson 5.7: Real-World Applications

**Time:** ~3 min

This is the "business case" slide for the module — transfer learning is often the
single technique that lets an ML project ship at all. The rare-disease example is
the most emotionally resonant (hundreds of labelled images, not millions); the
manufacturing visual-inspection one is the most commercially grounded. State that
the organisations here are generic illustrations.

Tie back directly to the data-efficiency experiment from Exercise 5.7: "You
measured this yourself — read your curve: where does transfer at 10% of CIFAR-10
sit against scratch at 100%?"

**Transition:** "The final paradigm — reinforcement learning."

---

## Slide 110: Reinforcement Learning

**Time:** ~1 min

Title the lesson. RL is the final paradigm in this module. It completes the DL
picture: supervised (labelled data), unsupervised (structure discovery), generative
(creating data), and now RL (learning from interaction).

This lesson connects directly to RLHF for LLM alignment in M6 — PPO, covered today,
is the classic optimiser behind RLHF.

**Beginner cue:** "All the models so far learned from a fixed dataset. An RL agent
doesn't have a dataset — it has an environment it can interact with. It takes
actions, sees rewards, and learns a strategy."

**Transition:** "The paradigm shift."

---

## Slide 111: The RL Paradigm Shift

**Time:** ~2 min

Make the distinction clear. Supervised learning: input X, output y, fixed dataset,
minimise loss. Static. Independent samples. RL: no dataset. The agent interacts
with an environment — takes an action, receives a reward, transitions to a new
state. Dynamic. Sequential. Correlated samples. Delayed rewards.

Four hard problems RL agents face that supervised models do not: delayed rewards
(credit assignment), sparse rewards (most steps give no feedback), exploration vs
exploitation (try something new or exploit what works?), and non-stationarity (the
data distribution changes as the agent learns).

**Beginner cue:** "Supervised learning is a student with a textbook. RL is a baby
learning to walk — falling, getting up, trying again, with no manual."

**Advanced cue:** "Non-stationarity is what makes RL hard. The policy changes,
which changes the state distribution, which changes what should be learned. No
supervised-style convergence guarantees."

**Transition:** "The vocabulary of RL."

---

## Slide 112: RL Fundamentals

**Time:** ~3 min

Establish the vocabulary — every RL algorithm uses these components. Agent: the
learner. Environment: everything outside the agent; provides observations and
rewards. State: the agent's knowledge of the environment at time t. Action: what
the agent does. Reward: a scalar signal indicating how well the agent is doing.
Episode: a sequence of steps from start to termination.

The policy π(a|s) is what we are trying to learn — deterministic (state → action)
or stochastic (state → distribution over actions). The value function V(s) is the
expected cumulative reward from state s; Q(s, a) the same for state-action pairs.
The discount factor γ (next slide) controls how much the agent cares about future
vs immediate rewards.

**Beginner cue:** "Think of a game. The agent is the player, the environment is the
game, the state is the screen, the action is the controller input, the reward is
the score. The policy is your strategy. The value function is how good you think a
given screen is."

**Transition:** "The equations that make RL possible."

---

## Slide 113: Bellman Equations

**Time:** ~4 min

The Bellman equations are the foundation of all RL. Walk them slowly, and keep the
two labelled apart — the slide does.

Bellman EXPECTATION: V(s) = E[R_{t+1} + γ·V(S_{t+1}) | S_t = s]. The value of a
state is the immediate reward plus the discounted value of the next state —
recursive, and defined for a given policy.

Bellman OPTIMALITY: Q*(s, a) = E[R_{t+1} + γ·max_{a'} Q*(S_{t+1}, a')]. The max
over next actions is what makes it the OPTIMAL value function — the agent picks the
best future action.

γ is the discount factor (typically 0.9–0.99): higher is more forward-looking. The
recursive definition is what makes dynamic programming and Q-learning possible.
Estimate Q* well, and the optimal policy is just argmax_a Q*(s, a).

**Beginner cue:** "The value of RIGHT NOW is the reward now plus a shrunken version
of the value of the next moment. Apply the same rule at the next moment and the
recursion covers the whole future."

**Advanced cue:** "The max in the optimality equation is what makes Q-learning
off-policy — you can learn about the optimal policy while following a different
behaviour policy."

**Transition:** "The first deep RL algorithm — DQN."

---

## Slide 114: DQN: Deep Q-Network

**Time:** ~4 min

DQN is the entry point for deep RL: a neural network approximates Q(s, a), trained
to minimise the Bellman loss L = E[(r + γ·max_{a'} Q(s', a'; θ⁻) − Q(s, a; θ))²].

Three innovations made it work:

- Experience replay: store transitions in a buffer, sample random minibatches —
  breaks correlation in sequential data.
- Target network: a separate, slowly updated copy of Q for computing the target —
  stabilises training.
- Epsilon-greedy: with probability ε take a random action, else argmax Q — balances
  exploration and exploitation.

The churn-prevention example makes RL tangible for professionals: the action is
"intervene or not", the state is customer history, the reward is retention.

**Advanced cue:** "The 2015 DeepMind Atari paper is the historical landmark. Every
subsequent deep RL algorithm addresses a limitation of DQN."

**Transition:** "DQN handles discrete actions. What about continuous?"

---

## Slide 115: Policy Gradient Methods

**Time:** ~3 min

DQN works for DISCRETE actions (choose from a menu). Continuous actions — how much
to adjust temperature, how much discount to offer — need a different approach:
learn the policy π(a|s) directly, estimating the gradient of expected return from
rollouts.

The actor-critic architecture combines the best of both: the actor learns the
policy (what to do), the critic learns the value function (how good it is), and the
critic's estimate reduces variance in the policy gradient update.

**Beginner cue:** "Instead of learning 'how good is each action,' learn 'what
action should I take' directly. A second network helps reduce noise in the
updates."

**Advanced cue:** "REINFORCE has high variance. Actor-critic subtracts a baseline —
the critic's V(s) — which reduces variance without bias. The advantage
A(s, a) = Q(s, a) − V(s) is the key quantity."

**Transition:** "Five algorithms in one overview slide."

---

## Slide 116: Five Algorithms, Five Applications

**Time:** ~2 min

Quick overview of all five. DQN — customer churn prevention (discrete actions).
DDPG — manufacturing control (continuous actions). SAC — dynamic pricing
(continuous, entropy-regularised). A2C — resource allocation (variance reduction
via baseline). PPO — supply chain optimisation (clipped objective prevents
destructively large updates).

Students implement DQN and PPO in the exercise; the other three are covered
conceptually and via `km.rl_train`.

Selection guidance: pick by action space and stability requirements. Discrete: DQN
or PPO. Continuous: SAC or PPO. Very unstable environment: PPO.

**Transition:** "Two continuous-action algorithms."

---

## Slide 117: DDPG and SAC

**Time:** ~3 min

DDPG (Deep Deterministic Policy Gradient) extends DQN to continuous actions: a
deterministic policy μ(s), actor-critic, off-policy — it can reuse old experience
via the replay buffer.

SAC (Soft Actor-Critic) adds entropy regularisation to the objective: prefer
stochastic policies unless the environment rewards determinism. Better exploration
by construction.

Both are off-policy and scale to high-dimensional continuous action spaces. SAC is
generally preferred in modern practice — more stable than DDPG and less
hyperparameter-sensitive; its entropy coefficient is auto-tuned in modern
implementations, which removes the main pain point.

**Beginner cue:** "DDPG is continuous-action DQN. SAC is DDPG plus 'stay
exploratory' as part of the loss."

**Transition:** "A simpler actor-critic."

---

## Slide 118: A2C: Advantage Actor-Critic

**Time:** ~2 min

A2C is the simplest actor-critic — the synchronous version of A3C. The advantage
function A(s, a) = Q(s, a) − V(s) tells the actor how much better or worse an
action was compared to the baseline (the critic's estimate). This reduces variance
in the policy gradient, making training more stable.

A2C is on-policy — it uses current rollouts only — which makes it simpler but less
sample-efficient than off-policy methods.

**Beginner cue:** "The advantage tells the actor 'this action was better than
average, do more of it' or 'worse, do less.' The baseline makes training much
smoother."

**Transition:** "And the flagship algorithm — PPO."

---

## Slide 119: PPO: Proximal Policy Optimization

**Time:** ~4 min

PPO is the most important RL algorithm to know. The core idea: when updating the
policy, do not move too far from the old policy in a single step — too large a step
can destroy everything the agent has learned.

The clipped objective: L^CLIP = E[min(r_t·A_t, clip(r_t, 1−ε, 1+ε)·A_t)], where
r_t = π_new(a|s) / π_old(a|s) is the probability ratio. Clipping achieves the
trust-region constraint of TRPO ("don't change the policy too much") with a simple
min + clip — much simpler to implement, similar empirical performance.

The RLHF connection is critical for M6 — PPO is the classic optimiser in RLHF. Keep
two mechanisms apart, because they are constantly conflated: PPO's clip bounds each
update relative to the PREVIOUS policy (π_old); staying close to the ORIGINAL
supervised model is enforced by a separate KL penalty to the reference model in the
reward. Two different references, two different mechanisms. And in RLHF, the LLM's
action is a DISCRETE token from the vocabulary — not a continuous distribution.

**Beginner cue:** "Take small steps. Do not let the new policy stray too far from
the old one. That is literally the whole algorithm."

**Advanced cue:** "PPO is a first-order approximation to TRPO — it trades
theoretical optimality for simplicity and gets ~95% of the performance."

**Transition:** "To use RL for business problems, you need a custom environment."

---

## Slide 120: Custom Gymnasium Environments

**Time:** ~3 min

Custom environments are where RL becomes practical for business. The Gymnasium API
is the standard (successor to OpenAI Gym). Required: `reset()` starts a new episode
and returns the initial observation; `step(action)` applies the action and returns
the next observation, reward, flags, and info.

One API precision: Gymnasium's `step` returns `(obs, reward, terminated, truncated,
info)` — end the episode loop on terminated OR truncated. Terminated means the
environment reached a natural end; truncated means the time limit cut it off. They
are not the same thing, and bootstrapping differs between them.

Define the observation space, action space, and reward function. The reward
function is the most important design decision — it defines what "success" means.
Bad reward design = reward hacking; good reward design = aligned behaviour.

Students build seven environments in the exercise (inventory, ride-hailing pricing,
churn, portfolio, queue, energy, traffic). The helpers on this slide
(`_get_customer`, `_compute_reward`, …) are a sketch — the exercise's
ChurnPreventionEnv implements them in full.

**Advanced cue:** "Reward shaping is where most business RL projects live or die.
Specification gaming is a real failure mode — the agent finds a way to maximise
reward that violates the spirit of the task."

**Transition:** "Meet the last Kailash engine — RLTrainer."

---

## Slide 121: Kailash Bridge: RLTrainer

**Time:** ~3 min

`km.rl_train` is the one-call entry point: pass a Gymnasium id, a zero-argument env
factory or an env instance, an algorithm string, a timestep budget and SB3
hyperparameters; it returns an RLTrainingResult (mean_reward, std_reward,
reward_curve, artifact path, lineage). The class form lives in `kailash_ml.rl`
(RLTrainer + RLTrainingConfig, used with environment and policy registries) — there
is no top-level `kailash_ml.RLTrainer`. DDPG, SAC and TD3 need a continuous (Box)
action space.

Be candid about status: the backend, Stable-Baselines3, is an optional extra
(`kailash-ml[rl]`) that is not installed in the course environment, so `rl_train`
raises ImportError there. Students therefore hand-write DQN and PPO in the exercise
— every moving part visible — and Exercise 5.8 file 04 closes by showing this call
as the production path.

Note the shape: same define → train → evaluate pattern as M3's TrainingPipeline.

**Beginner cue:** "You write the environment. The engine handles everything else —
replay buffer, target network updates, logging, checkpointing. In production, two
lines go from 'I have an environment' to 'I have a trained agent.'"

**Transition:** "The exercise."

---

## Slide 122: Exercise 5.8: Reinforcement Learning

**Time:** ~1 min

Exercise 5.8 is four files: DQN and PPO written from scratch and trained on
CartPole-v1, then applied to custom environments — retail inventory, ride-hailing
pricing, and five more in file 03 (churn, portfolio, queue, energy, traffic).

All exercise environments use discrete actions. PPO on a continuous action space,
and DDPG/SAC/A2C, are covered on the slides and via `km.rl_train` — not in the
files.

The RLHF bridge to M6 is critical: students who understand PPO here have a head
start on LLM alignment. The custom environments are the creative part — designing
reward functions that capture the business objective without gameable loopholes.

**Transition:** "Key formulas."

---

## Slide 123: Lesson 5.8 Key Formulas

**Time:** ~1 min

Quick reference for the four key RL equations. Bellman expectation and Bellman
optimality are the foundation. The DQN loss turns Bellman into a supervised
learning problem. The PPO clipped objective bounds each policy update for
stability.

"These four lines are the foundation of deep RL. Internalise them."

**Transition:** "Lesson summary."

---

## Slide 124: Lesson 5.8 Summary

**Time:** ~1 min

RL learns from interaction, not fixed datasets. Bellman equations define optimal
value functions. DQN uses neural Q-networks with replay and a target network. PPO
clips the policy update. Custom Gymnasium environments enable business
applications. `km.rl_train` handles the loop in production.

Bridge to M6: RL is the last building block. M6 combines everything — transformers
(Lesson 5.4), transfer learning (Lesson 5.7), and RL (Lesson 5.8) — into RLHF for
LLM alignment. The progression is complete.

**Transition:** "Where does RL earn its keep?"

---

## Slide 125: Lesson 5.8: Real-World Applications

**Time:** ~4 min

Give this slide an extra minute — it is the module finale for applications.

Five of the six cards map to a Gymnasium env the student has already coded
(ChurnPrevention and PortfolioRebalancing in file 03, RetailInventory in file 01,
RideHailingPricing in file 02); ResourceAllocation is conceptual. Call it out: "You
built these. They are simplified, but they have the same state / action / reward
shape a production system would."

State that the organisations are generic illustrations.

The bridge to M6 is the most important takeaway: PPO is not a CartPole trick — it
is the classic optimiser in RLHF for LLMs, with two separate controls: the clip
bounds each update against the previous policy, and the KL-to-reference penalty
keeps the model close to where it started. Two mechanisms, two references. This is
the hook into Module 6.

**Transition:** "Module-level consolidation — the formula reference."

---

## Slide 126: Module 5: Complete Formula Reference

**Time:** ~2 min

This is the cheat sheet. Walk through quickly, connecting each formula to its
lesson context: autoencoder reconstruction loss, VAE ELBO, reparameterisation, conv
output size, ResNet skip, SE gating, the six LSTM equations, GRU update, scaled
dot-product attention, positional encoding, GAN minimax, Wasserstein objective, GCN
propagation, Bellman equations, DQN loss, PPO clip.

Students should have all of these memorised or readily derivable by the end of the
module. "If any feel unfamiliar, that is a signal to revisit the corresponding
lesson."

**Transition:** "And the decision tree for when to use what."

---

## Slide 127: Architecture Decision Tree

**Time:** ~2 min

This is the practical summary. Students should be able to look at a new problem and
immediately answer three questions:

1. What data type? Images → CNN/ViT. Text → Transformer. Sequences →
   LSTM/Transformer. Graphs → GNN. Tabular → gradient boosting, not DL.
2. What goal? Classify → supervised. Generate → VAE/GAN/diffusion/transformer.
   Control → RL.
3. Is there a pre-trained model to fine-tune? Transfer learning is the default
   unless you have a good reason not to.

**Beginner cue:** "Three questions, done. You can solve 90% of new architecture
problems by answering them."

**Transition:** "How the module is assessed."

---

## Slide 128: End of Module Assessment

**Time:** ~2 min

Describe what exists in the module's `quiz/` and `assessment/` folders.

The quiz is a 90-minute coding notebook: four build questions and one prescribe
question that uses the DL Diagnostics toolkit from the start of this deck. Each is
pass/fail against a threshold.

The assessment is four auto-graded tasks, one per pillar — autoencoders, CNNs,
sequences, transformers — each worth 25 marks and passed only when every grader
check is true. They are deliberately CPU-sized: synthetic or bundled data, small
models, no pre-trained backbones.

The export-and-serve pipeline (OnnxBridge, ModelRegistry, InferenceServer) is
exercised in Exercises 5.2 and 5.7 rather than graded here.

**Transition:** "Module complete."

---

## Slide 129: Deep Learning and Machine Learning Mastery in Vision and Transfer Learning

**Time:** ~1 min

Thank the class. Module 5 is the most technically dense in the programme.

Recap the progression: autoencoders compress, CNNs see space, RNNs remember time,
transformers attend, GANs generate, GNNs connect, transfer learning scales, RL
interacts.

"Students who have completed all 8 lessons and exercises have a solid foundation in
every major deep learning paradigm. Module 6 builds on this: LLMs, alignment,
governance, and production AI."

**Beginner cue:** "You did it. You now know more deep learning than most people who
call themselves data scientists. Take a moment."

**Advanced cue:** "You have seen the lineage. Every modern AI system is a
composition of these building blocks. Module 6 is where the compositions get
interesting."

**Transition:** "See you in Module 6 for LLMs and alignment."

---

**End of speaker notes for Module 5 — Deep Learning and Machine Learning Mastery in Vision and Transfer Learning.**

Timing summary (from the deck's per-slide budgets):

- Intro + engines + refresher (Slides 1–5): ~10 min
- DL Diagnostics toolkit (Slides 6–21): ~47 min
- Appendix: modern practice (Slides 22–25): ~9 min
- Lesson 5.1 Autoencoders (Slides 26–38): ~33 min
- Lesson 5.2 CNNs (Slides 39–52): ~30 min
- Lesson 5.3 RNNs (Slides 53–65): ~32 min
- Lesson 5.4 Transformers (Slides 66–79): ~34 min
- Lesson 5.5 GANs & diffusion (Slides 80–90): ~25 min
- Lesson 5.6 GNNs (Slides 91–100): ~22 min
- Lesson 5.7 Transfer learning (Slides 101–109): ~19 min
- Lesson 5.8 RL (Slides 110–125): ~41 min
- Module close (Slides 126–129): ~7 min

Presented total: ~310 minutes, plus exercise time. Split after Slide 65 (end of
Lesson 5.3) for two balanced half-day sessions (~162 min / ~148 min). Instructors
running short on time should compress the GAN variants survey (Slide 85), the
DDPG/SAC slide (Slide 117), and the appendix (Slides 23–25, which are reference
material). The slides that must not be cut: 33 (reparameterisation), 43 (ResNet
skip connections), 68–69 (scaled dot-product attention and the √d_k derivation),
and 119 (PPO). These are the conceptual anchors of the module.
