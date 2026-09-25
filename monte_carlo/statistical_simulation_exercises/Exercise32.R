set.seed(1)

source("Exercise31.R")

n <- 10^5

f_single <- function(x) {
  dnorm(x, mean = 0.05, sd = 0.15)
}

g_single <- function(x) {
  dnorm(x, mean = -0.02, sd = 0.15)
}

prod(g_single(c(1, 0.8, 0.6, 0.4)))

# Estimation of Z.
w_data <- matrix(
  rnorm(n * 10, mean = -0.02, sd = 0.15),
  nrow = n,
  ncol = 10
)
z_data <- apply(w_data, 1, F)
hist(z_data, probability = TRUE, breaks = 40)

# Estimation with SNIS.

snis_weight <- function(x) {
  prod(f_single(x)) / prod(g_single(x))
}

snis_summand <- function(w) {
  if (F(w) >= 0) {
    return(0)
  }
  return(snis_weight(w))
}

weights_array <- apply(w_data, 1, snis_weight)
snis_summands <- apply(w_data, 1, snis_summand)

snis_estimator <- sum(snis_summands) / sum(weights_array)
# 0.001309455

# Confidence interval.

snis_var_summand <- function(w) {
  snis_weight(w)^2 * ((F(w) < 0) - snis_estimator)^2
}
snis_var_summands <- apply(w_data, 1, snis_var_summand)

snis_var_estimator <- sum(snis_var_summands) / sum(weights_array)^2
snis_var_estimator # 6.132923e-10

margins <- c(
  snis_estimator - 1.96 * sqrt(snis_var_estimator),
  snis_estimator + 1.96 * sqrt(snis_var_estimator)
)
snis_estimator # 0.001309455
margins # 0.001260916 0.001357994
