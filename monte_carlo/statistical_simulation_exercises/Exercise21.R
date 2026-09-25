set.seed(1)

n <- 10^5
M <- pi/2

f_func <- function(x) {
  (1 + sin(pi * x)^2) / (x^8 + 8)
}
g_density <- function(x) {
  1 / (pi * (x^2 + 1))
}

counter <- 0
results <- c()
while (length(results) < n) {
  u1 <- runif(1)
  Y <- tan(pi * u1 - pi * 0.5)

  counter <- counter + 1
  u2 <- runif(1)
  if (u2 < f_func(Y) / (M * g_density(Y))) {
    results <- c(results, Y)
  }
}

hist(results, probability = TRUE, xlim = c(-5, 5), ylim = c(0, 0.7), breaks = 100)

# True density function.
fnormalizing_constant <- integrate(f_func, -Inf, Inf)$value

f_density <- function(x) {
  f_func(x) / fnormalizing_constant
}

x_eval <- seq(min(results), max(results), length.out = 500)
lines(x_eval, f_density(x_eval))
