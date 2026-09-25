# Exercise 2(iii): Self-normalized importance sampling

Estimate $\theta=\mathbb{E}_f[X^2]$, where
$f(x)=[1+\sin^2(\pi x)]/[C(x^8+8)]$, using $10^5$ draws from the
standard Cauchy importance distribution
$g(x)=1/[\pi(x^2+1)]$.

Compute the self-normalized importance sampling estimate and its 95%
asymptotic confidence interval. Compare the result with the simple Monte
Carlo estimate from Exercise 2(ii), and decide which method performs better.
