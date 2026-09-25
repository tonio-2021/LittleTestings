# Exercise 1(iii): Rejection sampling with a shifted exponential proposal

Sample from a normal distribution with mean 0 and variance 2, conditional on
being at least 1. Use a shifted exponential proposal with density
$g(x)=\lambda e^{-\lambda(x-a)}$ for $x\geq a$.

Find the optimal rate $\lambda$ when $a=1$, and describe the resulting
rejection sampling algorithm. Using only Uniform(0, 1) random draws, generate
$10^4$ accepted values. Plot a histogram with the true density overlaid.
Compare the empirical mean number of proposals per acceptance with its
theoretical value.
