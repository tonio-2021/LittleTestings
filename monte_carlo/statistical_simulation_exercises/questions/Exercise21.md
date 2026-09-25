# Exercise 2(i): Rejection sampling with a Cauchy proposal

The target density is

$$
f(x)=\frac{1+\sin^2(\pi x)}{C(x^8+8)},\qquad x\in\mathbb{R},
$$

where $C$ is the normalizing constant. Only Uniform(0, 1) draws may be used
as random input. Use the standard Cauchy proposal
$g(x)=1/[\pi(x^2+1)]$.

Describe a valid rejection sampling algorithm, including an envelope
constant. The inequalities $|\sin(\pi x)|\leq 1$ and
$x^8+8\geq4(x^2+1)$ may help. Generate $10^5$ target samples, plot a
histogram, and overlay the normalized target density. Numerical integration
may be used to compute $C$ for the overlay.
