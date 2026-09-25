# Exercise 3(ii): Importance sampling for bankruptcy probability

Keep the company value model from Exercise 3(i), with
$F(x_1,\ldots,x_{10})=\min_{1\leq k\leq10}[1+\sum_{i=1}^k x_i]$.
The target annual changes are independent
$N(0.05,0.15^2)$ variables.

For self-normalized importance sampling, draw ten independent annual changes
from $N(-0.02,0.15^2)$ per simulated path. Generate $10^5$ paths, compute
$Z=F(W_1,\ldots,W_{10})$, and plot its histogram. Use the ratio of the
target and proposal joint densities to estimate
$p^*=\Pr(F(X_1,\ldots,X_{10})<0)$. Report a 95% asymptotic confidence
interval for the estimate.
