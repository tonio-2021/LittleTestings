set.seed(1)

source("Exercise21.R")

mc_est <- mean(results^2)

# Estimated variance of the simple sampling estimator.
sigmasq_hat <- mean(results^4) - mc_est^2

# 95% confidence interval.
margins95 <- 1.96 * sqrt(sigmasq_hat/n)

conf_interval95 <- c(mc_est - margins95, mc_est + margins95)
print(conf_interval95) # 0.6908624 0.7015239
print(mc_est) # 0.6961931
