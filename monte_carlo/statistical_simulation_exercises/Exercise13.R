set.seed(1)
n <- 10^4

C <- 1 / (sqrt(4 * pi) * pnorm(-1 / sqrt(2)))
M_opt <- C

f_density <- function(x) {
  # if (x < 1) {
  #   return(0)
  # }
  return(C * exp(-x^2 / 4))
}

g_density <- function(x) {
  if (x < 1) {
    return(0)
  }
  return(exp(-(x - 1)))
}

results <- c()
counter <- 0
while (length(results) < n) {
  # Draw a proposal from g.
  u <- runif(1)
  Y <- -log(u) + 1

  counter <- counter + 1
  u2 <- runif(1)
  if (u2 < f_density(Y) / (M_opt * g_density(Y))) {
    results <- c(results, Y)
  }
}
hist(results, probability = TRUE, breaks = 40)

x_eval <- seq(1, max(results), length.out = 500)
lines(x_eval, f_density(x_eval))

# Average proposals per accepted sample.
emp_ave <- counter / n
emp_ave # 1.1785
theo_ave <- C # 1.17662
theo_ave - emp_ave # -0.001879688
