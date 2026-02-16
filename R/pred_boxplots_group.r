#' Predictive boxplots by group (Year), conditional or marginal, for glmmTMB or brms
#'
#' @param model glmmTMB or brmsfit
#' @param data  original data used to fit
#' @param group_var name of grouping column on x-axis (default "Year_C")
#' @param class_var name of class column (default "Class")
#' @param classes character/integer vector of classes to facet (NULL = all)
#' @param K number of draws per observation (default 100)
#' @param conditional logical; TRUE = include REs, FALSE = marginal (exclude REs)
#' @param as_rate logical; if TRUE divide by `exposure_var` (e.g., Payroll_M)
#' @param exposure_var name of exposure column used for rates (default "Payroll_M")
#'
library(dplyr)
library(tidyr)
library(ggplot2)

predictive_boxplot <- function(model, data,
                               group_var   = "Year_C",   # x-axis (e.g., centered year)
                               class_var   = "Class",    # facet variable
                               classes     = NULL,       # optional subset of classes
                               K           = 100,        # draws per observation
                               conditional = TRUE,       # include REs? (TRUE = conditional)
                               as_rate     = FALSE,      # divide by exposure?
                               exposure_var= "Payroll_M" # exposure col if as_rate = TRUE
) {
  stopifnot(group_var %in% names(data), class_var %in% names(data))
  if (as_rate) stopifnot(exposure_var %in% names(data))

  N          <- nrow(data)
  group_vec  <- data[[group_var]]
  class_vec  <- data[[class_var]]
  expo_vec   <- if (as_rate) data[[exposure_var]] else rep(1, N)

  if (inherits(model, "brmsfit")) {
    re_arg <- if (conditional) NULL else NA
    yrep <- brms::posterior_predict(model, ndraws = K, re_formula = re_arg) # K x N
    mu   <- brms::posterior_epred(model,   ndraws = K, re_formula = re_arg) # K x N
  } else if (inherits(model, "glmmTMB")) {
    mu_mean <- predict(model, type = "response",
                       re.form = if (conditional) NULL else NA)             # length N
    phi   <- sigma(model)^2
    shape <- 1 / phi
    scale <- phi * mu_mean
    yrep  <- matrix(stats::rgamma(N * K, shape = shape, scale = rep(scale, each = K)),
                    nrow = K, ncol = N, byrow = TRUE)
    mu    <- matrix(rep(mu_mean, each = K), nrow = K, ncol = N, byrow = FALSE)
  } else {
    stop("model must be a brmsfit or glmmTMB object")
  }

  # convert to rates if requested
  yrep <- sweep(yrep, 2, expo_vec, "/")
  mu   <- sweep(mu,   2, expo_vec, "/")

  # long frames (key: map obs index -> group/class via vector indexing)
  sim_long <- as_tibble(yrep) %>%
    mutate(.draw = row_number()) %>%
    pivot_longer(-.draw, names_to = "obs", values_to = "y") %>%
    mutate(obs = as.integer(sub("^V","", obs)),
           .group = group_vec[obs],
           .class = class_vec[obs])

  mu_long <- as_tibble(mu) %>%
    mutate(.draw = row_number()) %>%
    pivot_longer(-.draw, names_to = "obs", values_to = "mu") %>%
    mutate(obs = as.integer(sub("^V","", obs)),
           .group = group_vec[obs],
           .class = class_vec[obs])

  if (!is.null(classes)) {
    sim_long <- filter(sim_long, .class %in% classes)
    mu_long  <- filter(mu_long,  .class %in% classes)
  }

  # median fitted line per draw and group (clean overlay)
  mu_by_group <- mu_long %>%
    group_by(.class, .group, .draw) %>%
    summarise(mu = median(mu), .groups = "drop")

  ggplot(sim_long, aes(x = factor(.group), y = y)) +   # <- force discrete x
    geom_boxplot(fill = "grey85", width = 0.6, outlier.alpha = 0.15) +
    geom_line(data = mu_by_group,
              aes(y = mu, group = .draw),
              color = "red", linewidth = 0.5, alpha = 0.4) +
    facet_wrap(~ .class, scales = "free_y") +
    scale_y_log10() +
    labs(
      x = group_var,
      y = if (as_rate) "Rate (log scale)" else "Loss (log scale)",
      title = sprintf("Predictive simulations (%s): boxplots by %s",
                      if (conditional) "conditional" else "marginal", group_var),
      subtitle = sprintf("K = %d draws per observation; red = fitted mean", K)
    ) +
    theme_minimal(base_size = 12)
}

