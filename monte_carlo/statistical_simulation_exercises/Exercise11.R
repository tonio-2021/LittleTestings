set.seed(1)

n <- 10^4

# Parameters for the distribution.
mu <- 0
sigmasq <- 2
a <- 1
# Normalizing constant.
C <- 1 / (sqrt(4 * pi) * pnorm(-1 / sqrt(2)))

inverser <- function(y) {
  sqrt(2) * qnorm(y / (C * sqrt(4 * pi)) + pnorm(1 / sqrt(2)))
}

y_unif <- runif(n)
results <- inverser(y_unif)
hist(results, probability = TRUE, breaks = 50)

real_density <- function(x) {
  C * exp(-x^2 / 4)
}
x_seq <- seq(1, max(results), length.out = 500)

lines(x_seq, real_density(x_seq))
