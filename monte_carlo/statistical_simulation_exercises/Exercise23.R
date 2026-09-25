set.seed(1)

source("Exercise21.R")
source("Exercise22.R")

weight_func <- function(x) {
  f_func(x) / g_density(x)
}

# Draw Cauchy samples from g.

G_inv <- function(y) {
  tan(pi * y - pi / 2)
}

c_samples <- G_inv(runif(n))

weights <- weight_func(c_samples)

enumerator <- sum(c_samples^2 * weights)

denominator <- sum(weights)

SIS_estimator <- enumerator / denominator
print(SIS_estimator) # 0.6948207

# Estimated variance for importance sampling.
var_enumerator <- sum(weights^2 * (c_samples^2 - SIS_estimator)^2)

var_denominator <- denominator^2

SISsigmasq_hat_n <- var_enumerator / var_denominator

SISmargin95 <- 1.96 * sqrt(SISsigmasq_hat_n)

SISconfidence95 <- c(
  SIS_estimator - SISmargin95,
  SIS_estimator + SISmargin95
)

print(SIS_estimator) # 0.6948207
print(SISconfidence95) # 0.6899583 0.6996831

# Comparison.
margins95 - SISmargin95 # 0.0004683745
