set.seed(5)

n <- 10^4

results <- c()
counter <- 0

while (length(results) < n) {
  x <- rnorm(n = 1, mean = 0, sd = sqrt(2))
  counter <- counter + 1
  if (x >= 1) {
    results <- c(results, x)
  }
}
print(counter)
print(length(results))
hist(results, probability = TRUE, breaks = 40)

# True density.
C <- 1 / (sqrt(4 * pi) * pnorm(-1 / sqrt(2)))
real_density <- function(x) {
  C * exp(-x^2 / 4)
}
x_eval <- seq(1, max(results), length.out = 500)

lines(x_eval, real_density(x_eval))

# Empirical average number of proposals per accepted sample.
emp_average <- counter / n
print(emp_average)

real_average <- 1 / pnorm(-1 / sqrt(2))
real_average - emp_average
