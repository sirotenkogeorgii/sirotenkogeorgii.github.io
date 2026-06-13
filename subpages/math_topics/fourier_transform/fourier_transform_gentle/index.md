---
layout: default
title: "The Fourier Transform: A Prism for Functions"
tags:
  - operator-theory
  - fourier-analysis
  - fourier-transform
  - change-of-basis
---

# The Fourier Transform: A Prism for Functions

## The one idea you should take away

If you remember nothing else from this post, remember this:

> **Complicated things are often sums of simple oscillations, and the Fourier transform is the device that tells you exactly which oscillations, and how much of each.**

A prism takes white light — a hopeless-looking jumble — and fans it out into a rainbow, revealing that "white" was secretly a blend of every color. The Fourier transform does the same thing to *functions*. Hand it a messy signal, and it hands back a clean list: *here is how much of each pure frequency was hiding inside.*

<figure>
  <img src="{{ '/assets/images/notes/fourier_transform/prism.png' | relative_url }}" alt="Left: a complicated wiggly signal in time, built from three sine waves. Right: its magnitude spectrum — three clean coloured lines at frequencies 3, 7 and 12, like a rainbow split out of white light." loading="lazy">
  <figcaption>The whole subject in one picture. The messy curve on the left is secretly a sum of just three pure sinusoids; the Fourier transform on the right reads off exactly which frequencies are present ($\xi = 3, 7, 12$) and how much of each. "White light" on the left, its hidden "rainbow" of pure tones on the right.</figcaption>
</figure>

That single sentence is the whole subject. Everything below is just making it precise, and then discovering that this innocent-looking idea quietly powers your phone, your music, your medical scans, and a large fraction of modern physics.

Let me set out the conventions first, the way one always should, and then we'll build the thing from scratch.

## A note on conventions (read this once, then forget it)

There is no single "correct" Fourier transform; there is a family of them that differ only in where you sprinkle factors of $2\pi$. This causes endless confusion when comparing textbooks, so I'll pin down my choice now and use it consistently. I'll measure frequency in **cycles per unit length** (e.g. cycles per second, not radians per second), which makes all the constants vanish and the formulas symmetric:

$$
\widehat{f}(\xi) \;=\; \int_{-\infty}^{\infty} f(x)\, e^{-2\pi i x \xi}\, dx,
\qquad
f(x) \;=\; \int_{-\infty}^{\infty} \widehat{f}(\xi)\, e^{+2\pi i x \xi}\, d\xi.
$$

The first formula is the **analysis** (decompose $f$ into frequencies); the second is the **synthesis** (rebuild $f$ from them). The variable $x$ lives in the *time* (or *space*) domain; the variable $\xi$ lives in the *frequency* domain. With this convention the two formulas are mirror images of each other — a hint that the time and frequency pictures are on completely equal footing. We'll see this is no accident.

If that integral looks intimidating, don't worry. We're going to *derive* it, not memorize it.

## Warm-up: the music your ear already knows

Strike three keys on a piano at once and the air pressure at your eardrum traces out a single wiggly curve in time — one number per instant, a genuinely complicated function. Yet you don't hear "one complicated wiggle." You hear *three notes*. Somehow your ear takes the tangled curve apart into its constituent pure tones.

<figure>
  <img src="{{ '/assets/images/notes/fourier_transform/piano_chord.png' | relative_url }}" alt="Four stacked plots sharing a time axis: three pure sine waves labelled C, E and G, and beneath them their sum — a single complicated curve labelled 'ear'." loading="lazy">
  <figcaption>Three pure tones in the frequency ratio $4:5:6$ (a major triad) add, instant by instant, into the one wiggly pressure curve your eardrum actually feels (bottom). Hearing the chord as three separate notes is your cochlea running this addition <em>backwards</em> — a biological Fourier transform you've been computing your whole life.</figcaption>
</figure>

That act of taking-apart is, essentially, a Fourier transform. The cochlea in your inner ear is lined with hair cells tuned to different pitches; each responds to *its own* frequency and ignores the rest. The brain reads off which cells fired and how strongly, and reports: a C, an E, a G. Your auditory system has been computing Fourier transforms your whole life without telling you.

So the mathematical question is simply: **given the wiggly curve, how do we recover the recipe of pure tones?** A pure tone is a sine wave, so let's start there.

---

## Step 1: Periodic functions and the orthogonality trick

We'll first solve an easier problem — *periodic* signals, the ones that repeat — because the answer there is a discrete list (this is the **Fourier series**), and discreteness is friendlier. The leap to the full transform comes afterward.

Say $f$ repeats every $2\pi$. The candidate building blocks are the pure oscillations that *also* fit neatly into a window of length $2\pi$: namely $\cos(nx)$ and $\sin(nx)$ for whole numbers $n$, the harmonics. It's cleaner to bundle each sine–cosine pair into a single **complex exponential** using Euler's identity,

$$
e^{i\theta} = \cos\theta + i\sin\theta,
$$

which you can picture as a point marching counterclockwise around the unit circle in the complex plane at unit speed. Then $e^{inx}$ is a point racing around the circle $n$ times as $x$ runs over a period — a pure oscillation of frequency $n$. The hope is that we can write

$$
f(x) = \sum_{n=-\infty}^{\infty} c_n\, e^{inx},
$$

and the entire game is to find the coefficients $c_n$ — the "amount of frequency $n$."

<figure>
  <img src="{{ '/assets/images/notes/fourier_transform/partial_sums.png' | relative_url }}" alt="A dashed square wave overlaid with four smooth partial-sum approximations using 1, 3, 7 and 21 sine harmonics; adding more harmonics tracks the square wave more closely, leaving a persistent overshoot at the jump." loading="lazy">
  <figcaption>"Complicated things are sums of simple oscillations" made literal. Stacking up more and more harmonics steadily reshapes a single smooth wave into a sharp-cornered square wave. The stubborn little overshoot that refuses to die at the jump (the Gibbs phenomenon) is the price of building a discontinuity out of perfectly smooth pieces.</figcaption>
</figure>

**The key idea is that the building blocks are mutually orthogonal.** This is the crux of the whole subject, so let me say what it means. Think of functions as vectors, with the role of the dot product played by

$$
\langle f, g\rangle \;=\; \frac{1}{2\pi}\int_0^{2\pi} f(x)\,\overline{g(x)}\, dx
$$

(the bar is complex conjugation; it's there so that $\langle f,f\rangle$ comes out as a genuine, positive "length-squared"). A direct computation — just integrating an exponential — gives the magic relation

$$
\langle e^{inx}, e^{imx}\rangle \;=\; \frac{1}{2\pi}\int_0^{2\pi} e^{i(n-m)x}\, dx \;=\;
\begin{cases} 1 & n = m,\\[2pt] 0 & n \neq m.\end{cases}
$$

In words: **distinct harmonics are perpendicular, and each has unit length.** They form an orthonormal basis, exactly like the perpendicular axes $\hat{x}, \hat{y}, \hat{z}$ of ordinary space — only now there are infinitely many of them.

And here is the payoff. In ordinary space, to find the $\hat{x}$-component of a vector, you take its dot product with $\hat{x}$; the other axes contribute nothing precisely *because* they're perpendicular. The same move works here. Take the inner product of $f = \sum_m c_m e^{imx}$ with $e^{inx}$: every term dies except the one with $m=n$, leaving

$$
\boxed{\,c_n \;=\; \langle f, e^{inx}\rangle \;=\; \frac{1}{2\pi}\int_0^{2\pi} f(x)\, e^{-inx}\, dx.\,}
$$

**That integral is not a magic incantation — it is a projection.** It asks: *how much does $f$ overlap with the pure tone $e^{inx}$?* Multiplying by $e^{-inx}$ and averaging is a way of "tuning in" to frequency $n$ — like rotating a radio dial until one station comes through and the others average out to nothing. The minus sign in the exponent is just the conjugate that the inner product demands.

<figure>
  <img src="{{ '/assets/images/notes/fourier_transform/orthogonality.png' | relative_url }}" alt="Two panels over one period. Left: a cosine wave whose green positive area and red negative area are equal and cancel to zero. Right: a constant function at height one whose shaded area is entirely positive." loading="lazy">
  <figcaption>Why the projection actually works. For any <em>mismatched</em> harmonic (left) the integrand wobbles, and its positive and negative areas cancel exactly — that frequency contributes nothing to the average. Only the matching harmonic (right) collapses to a constant that reinforces everywhere, so it alone survives. That is the orthogonality $\langle e^{inx}, e^{imx}\rangle = \delta_{nm}$ seen as cancellation.</figcaption>
</figure>

This is the conceptual seed of everything. The forbidding integral defining the Fourier transform is, at heart, *the formula for a coordinate in an orthonormal basis of oscillations.*

---

## Step 2: From a discrete list to a continuous spectrum

Periodic functions gave us a discrete spectrum: amounts $c_n$ at the integer frequencies $n = 0, \pm 1, \pm 2, \dots$. But most signals of interest — a single hand-clap, a localized pulse, a function that decays to zero far away — aren't periodic. What happens to them?

Here's the heuristic that bridges the gap. Pretend your non-periodic signal *is* periodic, with some enormous period $L$. The allowed frequencies are then spaced $1/L$ apart. As you let $L \to \infty$ to capture the genuinely non-periodic function, that spacing shrinks to zero: **the discrete comb of allowed frequencies fills in to a continuum.** The sum over $n$ becomes an integral over a continuous frequency variable $\xi$, and the discrete coefficients $c_n$ blur into a continuous function $\widehat{f}(\xi)$.

<figure>
  <img src="{{ '/assets/images/notes/fourier_transform/comb_to_continuum.png' | relative_url }}" alt="Three panels for periods L = 2, 5 and 20. Each shows the same dashed continuous transform; the discrete spectral samples (orange stems spaced 1/L apart) grow denser from left to right until they trace out the continuous curve." loading="lazy">
  <figcaption>The bridge from Fourier <em>series</em> to Fourier <em>transform</em>. Pretending a signal is periodic with period $L$ permits only the frequencies spaced $1/L$ apart (the orange comb). As $L \to \infty$ that spacing shrinks toward zero and the comb fills in the underlying continuous spectrum (dashed) — the discrete list of amounts $c_n$ blurs into a continuous function $\widehat{f}(\xi)$.</figcaption>
</figure>

Carrying the bookkeeping through (which I'll spare you, since it's all in the limit and not in any new idea) turns the projection formula into

$$
\widehat{f}(\xi) = \int_{-\infty}^{\infty} f(x)\, e^{-2\pi i x \xi}\, dx
$$

and the reconstruction sum into

$$
f(x) = \int_{-\infty}^{\infty} \widehat{f}(\xi)\, e^{+2\pi i x \xi}\, d\xi.
$$

These are the formulas I promised at the start, now arrived at honestly. The interpretation carries over verbatim: $\widehat{f}(\xi)$ is *how much of the pure frequency $\xi$ is present in $f$*, and the second formula reassembles $f$ by stacking up all those frequencies, each weighted by its amount. The transform is a **change of coordinates** — the same function, viewed in the "frequency" coordinate system instead of the "time" one. No information is created or destroyed; it's merely re-described.

---

## Step 3: Why *complex exponentials*, of all things?

A fair objection: oscillations are sines and cosines, things you can draw. Why phrase everything in terms of the more abstract $e^{2\pi i x\xi}$? There are two answers, and the second is the deep one.

**The cheap answer: convenience.** A complex exponential bundles a cosine and a sine into one object and tracks both their amplitude and their *phase* (their timing) in a single complex number. Real formulas with two pieces become complex formulas with one.

<figure>
  <img src="{{ '/assets/images/notes/fourier_transform/helix.png' | relative_url }}" alt="A 3-D purple helix winding along the x-axis, with its projection onto the floor tracing a cosine and its projection onto the back wall tracing a sine." loading="lazy">
  <figcaption>A complex exponential $e^{2\pi i \xi x}$ is a single corkscrew spiralling through the complex plane as $x$ advances. Its shadow on one wall is a cosine (the real part); its shadow on the other is a sine (the imaginary part). One object carries both at once, together with their relative timing — the phase — which is exactly why it, and not the sine or cosine alone, is the natural building block.</figcaption>
</figure>

**The real answer: they are the eigenfunctions of the operations we care about.** An "eigenfunction" of an operation is an input the operation merely *scales* rather than reshaping — the operation's preferred, simplest input. Now notice two facts.

First, differentiate a complex exponential:

$$
\frac{d}{dx}\, e^{2\pi i \xi x} = (2\pi i \xi)\, e^{2\pi i \xi x}.
$$

It comes back unchanged except for a number out front. **The exponential is an eigenfunction of differentiation.** Sines and cosines don't have this property individually — differentiating turns one into the other — which is exactly why the complex version is the *right* primitive.

Second, *shift* a complex exponential in time by some amount $a$:

$$
e^{2\pi i \xi (x - a)} = \big(e^{-2\pi i \xi a}\big)\, e^{2\pi i \xi x}.
$$

Again it returns unchanged up to a constant. **The exponential is an eigenfunction of translation.** This is the property that secretly runs the whole show, because so many of the operations we perform on signals — delaying, blurring, smoothing, averaging, the physics of any medium that doesn't care *where* you are — are *translation-invariant*. The Fourier basis is the one basis in which every translation-invariant operation becomes trivial: plain multiplication by a number, frequency by frequency. We're about to cash this in.

---

## Step 4: The properties that make it useful

The Fourier transform would be a pretty curiosity if it merely re-described functions. What makes it indispensable is a short list of "dictionary" entries translating operations in the time domain into *simpler* operations in the frequency domain. Here are the ones worth internalizing.

**Linearity.** $\widehat{af + bg} = a\widehat{f} + b\widehat{g}$. Superposition in, superposition out. Unremarkable but constantly used.

**A delay becomes a phase twist.** Shifting $f$ in time by $a$ multiplies its transform by $e^{-2\pi i a \xi}$. The *amounts* of each frequency don't change — of course they don't, you only delayed the signal — but their *timing* (phase) does. This is the translation-eigenfunction fact, dressed for use.

**Differentiation becomes multiplication.** This is the headline:

$$
\widehat{f'}(\xi) = (2\pi i \xi)\, \widehat{f}(\xi).
$$

Taking a derivative — a genuine operation of calculus — turns into *multiplying by $2\pi i \xi$*, a trivial operation of algebra. Differential equations, which are hard, become algebraic equations, which are easy. This single line is why the transform is the central tool for partial differential equations.

**Blurring becomes multiplying (the convolution theorem).** "Convolution" $f * g$ is the operation of smearing one function by another — blurring an image, applying an echo, taking a moving average. It's defined by an awkward integral and is a pain to compute directly. Under the Fourier transform it becomes the most innocent operation imaginable:

$$
\widehat{f * g} = \widehat{f}\cdot \widehat{g}.
$$

Tangled smearing in one domain is plain pointwise multiplication in the other. (This is the eigenfunction story again: every translation-invariant operation, and blurring is one, acts on each frequency independently, simply by scaling it.)

**Squeezing in time stretches in frequency.** Compress a signal horizontally by a factor $a$ and its transform spreads out by the same factor:

$$
\widehat{f(ax)}(\xi) = \frac{1}{|a|}\,\widehat{f}\!\left(\frac{\xi}{a}\right).
$$

A short, sharp spike is built from a *broad* band of frequencies; a long, lazy wave needs only a *narrow* band. Brevity in time demands richness in frequency. Hold onto this — it's about to become a law of nature.

<figure>
  <img src="{{ '/assets/images/notes/fourier_transform/reciprocal_spreads.png' | relative_url }}" alt="A two-by-two grid of Gaussians. Top row: a narrow pulse in time beside its broad transform in frequency. Bottom row: a broad pulse in time beside its narrow transform." loading="lazy">
  <figcaption>The scaling law in pictures. Squeeze a pulse to be sharp in time (top left) and its transform spreads across a wide band of frequencies (top right); let it relax to be broad in time (bottom left) and only a narrow band is needed (bottom right). Sharpness in one domain is always bought with spread in the other — a fact that is about to become the uncertainty principle.</figcaption>
</figure>

**Energy is conserved (Plancherel's theorem).**

$$
\int_{-\infty}^{\infty} |f(x)|^2\, dx = \int_{-\infty}^{\infty} |\widehat{f}(\xi)|^2\, d\xi.
$$

The total energy of a signal is the same whether you measure it tone by tone or instant by instant. Mathematically, the transform is an **isometry** — a rigid rotation of the (infinite-dimensional) space of functions. It doesn't distort lengths or angles; it just spins you from the time axes to the frequency axes. This is the precise sense in which "no information is lost."

---

## Step 5: The uncertainty principle, for free

That squeeze-and-stretch rule has a famous consequence. A function and its transform **cannot both be sharply concentrated.** Pin a signal down to a tiny interval in time and its frequency content necessarily smears across a wide band; conversely, a signal made of a narrow band of frequencies must be spread out in time. You can trade localization in one domain for the other, but you can never win in both at once. Making this quantitative gives an honest inequality,

$$
(\text{spread in time}) \times (\text{spread in frequency}) \;\geq\; \frac{1}{4\pi},
$$

with the spreads measured as standard deviations. In quantum mechanics, a particle's position and momentum are related by *exactly* this Fourier pairing, and the inequality becomes **Heisenberg's uncertainty principle** — not a statement about clumsy measuring devices, but a hard mathematical fact about any object that has a position description and a frequency (momentum) description simultaneously.

**A remark on sharpness.** The inequality is not idle: it is achieved, with equality, precisely by the **Gaussian** bell curve $e^{-\pi x^2}$. The Gaussian is the most efficient possible compromise between the two domains — and, delightfully, with our normalization it is *its own Fourier transform*: $\widehat{e^{-\pi x^2}} = e^{-\pi \xi^2}$. It's the one shape that looks identical from both sides of the prism. (This is also why the Gaussian turns up everywhere from probability to the heat equation: it is the natural fixed point of this whole circle of ideas.)

<figure>
  <img src="{{ '/assets/images/notes/fourier_transform/gaussian_uncertainty.png' | relative_url }}" alt="Left: the Gaussian e^{-pi x squared} and its Fourier transform plotted on top of each other, perfectly identical. Right: a curve of time-spread versus frequency-spread forming a hyperbola, with the symmetric Gaussian marked at its balance point." loading="lazy">
  <figcaption>Left: with this normalisation the Gaussian $e^{-\pi x^2}$ is <em>its own</em> Fourier transform — the one shape that looks identical from both sides of the prism. Right: sweep through the family of Gaussians and the product of time-spread and frequency-spread stays pinned to the floor $\sigma_t\,\sigma_\xi \geq \tfrac{1}{4\pi}$, with the symmetric Gaussian sitting exactly at the optimal trade-off. That floor <em>is</em> Heisenberg's uncertainty principle.</figcaption>
</figure>

## Step 6: A worked showcase — heat, melting away high frequencies

Let me make the "differential equations become algebra" slogan concrete, because it's the most persuasive thing the transform does. Consider how heat spreads along a metal rod, governed by the heat equation

$$
\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2},
$$

where $u(x,t)$ is the temperature at position $x$ and time $t$. As written, this couples every point on the rod to its neighbors — hard.

Now Fourier transform *in the space variable* $x$. Two derivatives in $x$ become multiplication by $(2\pi i \xi)^2 = -4\pi^2 \xi^2$, and the equation collapses to

$$
\frac{\partial \widehat{u}}{\partial t} = -4\pi^2 \xi^2\, \widehat{u}.
$$

The frequencies have *decoupled* completely. For each fixed $\xi$ this is the simplest differential equation there is — a quantity whose rate of change is proportional to itself — and its solution is plain exponential decay:

$$
\widehat{u}(\xi, t) = \widehat{u}(\xi, 0)\, e^{-4\pi^2 \xi^2 t}.
$$

Read off the physics directly: each frequency just fades on its own, and the decay rate $4\pi^2\xi^2$ grows with $\xi$, so **high frequencies die fastest.** High frequency means rapid spatial wiggling — sharp bumps and fine detail — so the equation is telling us that heat flow smooths out the fine structure first, leaving the broad trends to settle slowly. That is exactly what we mean physically by "things even out." We turned an intractable PDE into a one-line exponential, *and* we extracted intuition we didn't have before. That is the Fourier transform earning its keep.

<figure>
  <img src="{{ '/assets/images/notes/fourier_transform/heat_decay.png' | relative_url }}" alt="Left: a jagged temperature profile along a rod that smooths into a gentle curve over successive times. Right: a bar chart of the frequency amplitudes at those same times, with the tallest high-frequency bars collapsing first." loading="lazy">
  <figcaption>The heat equation watched in both domains at once. In space (left) a bumpy initial temperature relaxes toward a smooth one. In frequency (right) the mechanism is laid bare: each mode decays on its own at rate $4\pi^2 k^2$, so the high frequencies — the fine wiggles — vanish first while the broad trends linger. "Things even out" is precisely <em>high frequencies dying fastest.</em></figcaption>
</figure>

---

## Where this shows up (a quick tour)

- **Audio (MP3, AAC).** Transform the sound, discard the frequencies the human ear can't perceive, store only the rest. Lossy compression is Fourier analysis plus a model of your ear.
- **Images (JPEG).** The same trick in two dimensions: most images are dominated by low spatial frequencies, so the high-frequency fine print can be heavily compressed with little visible loss.
- **The Fast Fourier Transform (FFT).** A staggeringly clever algorithm that computes the discrete transform in $O(N\log N)$ steps instead of $O(N^2)$. It is, by some accounting, the single most-used numerical algorithm in the world, running invisibly in essentially every digital communication you make.
- **Solving PDEs** — heat, waves, diffusion, and more — exactly as above.
- **Quantum mechanics**, where position and momentum are Fourier partners and uncertainty is built into the formalism.
- **Spectroscopy, MRI, radio, optics** — anywhere a physical apparatus naturally separates a signal into frequencies, it is doing Fourier analysis in hardware.

---

## What's really going on, and where it goes next

Let me close with the higher vantage point, now that the machinery is in hand.

**The Fourier transform is the act of diagonalizing translation.** Strip away the analysis and the deepest fact is the one from Step 3: the exponentials $e^{2\pi i x\xi}$ are the eigenfunctions shared by *every* translation-invariant operation. Choosing the Fourier basis is choosing the coordinate system in which all such operations — blurring, delaying, differentiating, the response of any homogeneous physical medium — become simultaneously diagonal, i.e. mere multiplication. Hard operations look hard only because we were using the wrong coordinates. Fourier hands us the right ones.

**This reframing is what lets the subject generalize.** Once you see the transform as "expand in the characters of the translation group," you can swap out the group:

- The integers (periodic, sampled data) give **Fourier series** and the **discrete Fourier transform** computed by the FFT.
- Finite and non-commutative groups give the Fourier analysis underpinning much of number theory and **representation theory** — the same orthogonality-of-characters idea, one floor up in abstraction.
- Refusing to commit to *one* global frequency at a time, and instead localizing in time and frequency together, leads to **wavelets** and the modern time-frequency analysis behind much of signal processing.

And it all traces back to the one sentence at the top: complicated things are sums of simple oscillations, and the transform reads off the recipe. Everything else — the projection formula, the eigenfunctions, the conservation of energy, the uncertainty principle, the melting of heat — is that single idea, followed honestly to its conclusions.

---

*A small reading suggestion: the genuinely deepest entry point is Step 1 — the orthogonality of the harmonics and the reading of the defining integral as a projection. If that clicks, the rest of the subject is commentary.*