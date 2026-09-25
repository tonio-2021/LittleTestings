# Exercise 3(i): Simulating a rare bankruptcy event

A company's initial value is 1 (measured in millions of dollars), and its value after year $k$ is
$V_k=1+\sum_{i=1}^k X_i$, for $k=1,\ldots,10$. The annual changes are
independent normal variables with mean 0.05 and standard deviation 0.15.
Define $Y=\min_{1\leq k\leq10}V_k$ and
$p^*=\Pr(Y<0)$.

Simulate 20 independent datasets, each containing $10^5$ values of $Y$.
Plot a histogram for the first dataset. For each dataset, estimate $p^*$
using the proportion of negative values. Decide whether the 20 estimates
agree when compared to their first two significant figures.
