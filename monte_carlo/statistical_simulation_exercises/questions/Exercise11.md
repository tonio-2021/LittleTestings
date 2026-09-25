# Exercise 1(i): Inverse transform sampling

Let $X$ have a normal distribution with mean $\mu$ and variance
$\sigma^2$, conditional on $X \geq a$. Its density is proportional to
$\exp[-(x-\mu)^2/(2\sigma^2)]$ on $[a,\infty)$.

1. Derive the cumulative distribution function for general $\mu$,
   $\sigma^2$, and $a$, using the standard normal CDF $\Phi$.
2. Describe an inverse transform algorithm for the case
   $\mu=0,\sigma^2=2,a=1$.
3. Generate $10^4$ samples, plot their histogram, and overlay the true
   density.
