# Weibull PH custom response distribution for brms
# Andrea Discacciati
# 2022-10-27

weibullPH <- brms::custom_family(
  "weibullPH",
  dpars = c("mu", "gamma"), # log(mu) = xb
  links = c("log", "identity"), # identity link for gamma as in rstanarm
  lb = c(0, 0),
  ub = c(NA, NA),
  type = "real"
)

stan_funs_weibullPH <- "
  real weibullPH_lpdf(real y, real mu, real gamma) {
    real sigma;
    sigma = mu^(-1/gamma);
    return weibull_lpdf(y | gamma, sigma);
  }
  real weibullPH_lcdf(real y, real mu, real gamma) {
    real sigma;
    sigma = mu^(-1/gamma);
    return weibull_lcdf(y | gamma, sigma);
  }
  real weibullPH_lccdf(real y, real mu, real gamma) {
    real sigma;
    sigma = mu^(-1/gamma);
    return weibull_lccdf(y | gamma, sigma);
  }"

stanvars_weibullPH <- brms::stanvar(scode = stan_funs_weibullPH, 
                              block = "functions")

posterior_epred_weibullPH <- function(prep) { # expected survival
  mu <- brms::get_dpar(prep, "mu")
  gamma <- brms::get_dpar(prep, "gamma")
  mu^(-1/gamma) * gamma(1 + 1/gamma)
}

log_lik_weibullPH <- function(i, prep) { # log-likelihood
  mu <- brms::get_dpar(prep, "mu", i = i)
  gamma <- brms::get_dpar(prep, "gamma", i = i)
  sigma <- mu^(-1/gamma) # PH -> AFT
  
  args <- list(shape = gamma, scale = sigma)
  out <- brms:::log_lik_censor(
    dist = "weibull", args = args, i = i, prep = prep 
  )
  out <- brms:::log_lik_truncate(
    out, cdf = pweibull, args = args, i = i, prep = prep
  )
  return(out)
}

posterior_predict_weibullPH <- function(i, prep, ntrys = 5, ...) { # posterior predictive distribution
  mu <- brms::get_dpar(prep, "mu", i = i)
  gamma <- brms::get_dpar(prep, "gamma", i = i)
  sigma <- mu^(-1/gamma) # PH -> AFT
  
  brms:::rcontinuous(
    n = prep$ndraws, dist = "weibull",
    shape = gamma, scale = sigma,
    lb = prep$data$lb[i], ub = prep$data$ub[i],
    ntrys = ntrys
  )
}
