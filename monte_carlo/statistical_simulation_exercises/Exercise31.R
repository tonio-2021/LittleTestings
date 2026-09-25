set.seed(1)

n <- 10^5
set_amoutn <- 20

F <- function(x_data) {
  min(1 + cumsum(x_data))
}

x_data <- array(
  rnorm(n * 10 * 20, mean = 0.05, sd = 0.15),
  dim = c(10, n, 20)
)

results <- apply(x_data, c(2, 3), F)
dim(results) # 100000 rows, 20 columns.

hist(results[, 1], probability = TRUE, breaks = 40)

p_est <- c()
for (i in seq(1:20)) {
  p_est <- c(p_est, length(which(results[, i] < 0)) / n)
}
print(p_est)
# 0.00146 0.00139 0.00132 0.00144 0.00134 0.00144
# 0.00134 0.00122 0.00117 0.00132 0.00147 0.00141
# 0.00149 0.00143 0.00137 0.00115 0.00121 0.00161
# 0.00115 0.00143

# We see that there is some variance around the first 2 significant digits. We have
# values from 0.00115 to 0.00161. They do NOT agree!
print(max(p_est))
print(min(p_est))
