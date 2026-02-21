# Research Report: Statistical Theory and Modern Methods for GLMMs in Actuarial Pricing

**Research Agent 3: Statistical Theory and Modern Methods Specialist**

**Date:** February 16, 2026

---

## Executive Summary

This report provides a comprehensive analysis of the statistical theory and computational methods underlying Generalized Linear Mixed Models (GLMMs) for actuarial applications. The focus is on estimation theory, Bayesian hierarchical models, mathematical foundations, and modern extensions that complement the pedagogical (Agent 1) and actuarial practice (Agent 2) perspectives.

**Key findings:**

1. **Estimation landscape**: The field has converged on a small set of robust approaches—ML/REML with Laplace approximation (glmmTMB/TMB), Adaptive Gauss-Hermite Quadrature (lme4), and Bayesian HMC/NUTS (brms/Stan)—each with clear use cases.

2. **Credibility-GLMM equivalence**: The formal connection between Bühlmann credibility and Best Linear Unbiased Prediction (BLUP) is a cornerstone result that must be presented early and rigorously. It breaks down in non-Gaussian, nonlinear settings, but the conceptual throughline remains.

3. **Conditional vs. marginal interpretation**: This is the single most consequential theoretical subtlety for actuarial applications. Population-averaged (marginal) predictions differ from subject-specific (conditional) predictions, and the choice has direct regulatory and pricing implications.

4. **Bayesian methods**: Should be introduced as a parallel framework throughout (not relegated to a late "extension"), because hierarchical Bayesian models *are* the natural formalization of credibility theory. The computational cost is now manageable for realistic actuarial datasets.

5. **Modern frontier**: Regularized mixed models (GLMMNet), INLA for fast approximation, distributional GLMMs (modeling both mean and dispersion), and measurement error correction (SIMEX) represent active research areas with direct actuarial relevance.

---

## 1. Estimation Theory Deep-Dive

### 1.1 The Intractable Likelihood Problem

The fundamental computational challenge in GLMMs is that the marginal likelihood (integrating over random effects) is analytically intractable for non-Gaussian responses:

$$
L(\boldsymbol{\beta}, \boldsymbol{\theta} \mid \mathbf{y}) = \int f(\mathbf{y} \mid \mathbf{b}, \boldsymbol{\beta}) \, f(\mathbf{b} \mid \boldsymbol{\theta}) \, d\mathbf{b}
$$

where $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \mathbf{G}(\boldsymbol{\theta}))$ are the random effects, $\boldsymbol{\beta}$ are fixed effects, and $\boldsymbol{\theta}$ parameterizes the random effect variance-covariance structure.

For linear mixed models (LMMs), this integral is analytically solvable because Gaussian distributions are closed under marginalization. For GLMMs, it is not. All modern GLMM software implements one of the following strategies to approximate this integral.

---

### 1.2 Maximum Likelihood (ML) vs. Restricted Maximum Likelihood (REML)

**Maximum Likelihood (ML):**
- Maximizes the full log-likelihood $\log L(\boldsymbol{\beta}, \boldsymbol{\theta} \mid \mathbf{y})$ jointly over fixed and random effect parameters
- Provides estimates $\hat{\boldsymbol{\beta}}_{\text{ML}}, \hat{\boldsymbol{\theta}}_{\text{ML}}$
- **Bias**: Variance components $\hat{\boldsymbol{\theta}}_{\text{ML}}$ are downward-biased because they do not account for the degrees of freedom consumed by estimating $\boldsymbol{\beta}$
- **Use case**: Model comparison via likelihood ratio tests (LRT) when comparing models with *different fixed effects*. AIC/BIC computed from ML are comparable across models.

**Restricted Maximum Likelihood (REML):**
- Maximizes a modified likelihood that integrates out (or "profiles out") the fixed effects:
  $$
  L_R(\boldsymbol{\theta} \mid \mathbf{y}) = \int L(\boldsymbol{\beta}, \boldsymbol{\theta} \mid \mathbf{y}) \, d\boldsymbol{\beta}
  $$
- Provides less biased estimates of variance components: $\hat{\boldsymbol{\theta}}_{\text{REML}}$
- Fixed effects $\hat{\boldsymbol{\beta}}_{\text{REML}}$ are obtained conditional on $\hat{\boldsymbol{\theta}}_{\text{REML}}$
- **Advantage**: Accounts for uncertainty in fixed effects when estimating variance components
- **Limitation**: Cannot compare models with different fixed effects via LRT (the likelihoods are not comparable)
- **Use case**: Inference on random effects variance components; final model fitting

**Practical guidance for actuarial pricing:**
- Use **REML** for final model fitting and inference on variance components (credibility parameter equivalents)
- Use **ML** for model selection when comparing fixed effect structures (e.g., does adding `age:region` interaction improve fit?)
- glmmTMB defaults to ML; lme4 defaults to REML. Be explicit about which you're using.

**Mathematical note**: For GLMMs, "REML" is technically an approximation because the marginal likelihood itself is approximated (see below). The term is used by analogy to the LMM case.

---

### 1.3 Laplace Approximation

The **Laplace approximation** is a deterministic method for approximating intractable integrals by matching the curvature of the integrand at its mode.

**Core idea:**
1. The integral $\int \exp[h(\mathbf{b})] d\mathbf{b}$ is dominated by the region near the mode $\hat{\mathbf{b}} = \arg\max h(\mathbf{b})$
2. Approximate $h(\mathbf{b})$ by a second-order Taylor expansion around $\hat{\mathbf{b}}$:
   $$
   h(\mathbf{b}) \approx h(\hat{\mathbf{b}}) + \frac{1}{2}(\mathbf{b} - \hat{\mathbf{b}})^\top \mathbf{H}(\hat{\mathbf{b}}) (\mathbf{b} - \hat{\mathbf{b}})
   $$
   where $\mathbf{H}$ is the Hessian of $h$ at $\hat{\mathbf{b}}$
3. The resulting Gaussian integral is analytically solvable

**Accuracy:**
- Very fast (no numerical integration)
- Accurate when the number of observations per random effect is large (asymptotically exact)
- Can be poor for binary/count data with few observations per group
- Underestimates variance in extreme cases

**Implementation:**
- **glmmTMB**: Uses Laplace approximation exclusively (via TMB automatic differentiation)
- **lme4**: Uses Laplace for `glmer` with `nAGQ=1` (the default for many families)

**When to use:**
- Large datasets with many groups and moderate cluster sizes (typical actuarial setting: thousands of policyholders)
- Initial model exploration (very fast)
- When AGQ is computationally prohibitive

**When to avoid:**
- Small cluster sizes ($n_j < 5$ per group) with binary/count responses
- High-stakes inference on variance components (use AGQ or Bayesian methods for validation)

---

### 1.4 Adaptive Gauss-Hermite Quadrature (AGQ)

**Gauss-Hermite Quadrature (GHQ)** is a numerical integration method that approximates integrals by evaluating the integrand at carefully chosen "quadrature points" and weighting the results.

For a one-dimensional integral:
$$
\int_{-\infty}^{\infty} g(x) \exp(-x^2) dx \approx \sum_{k=1}^K w_k g(x_k)
$$

where $x_k$ are the quadrature points and $w_k$ are the weights (pre-tabulated for standard Gaussian integrals).

**Adaptive GHQ (AGQ):**
- Centers and scales the quadrature points adaptively around the mode $\hat{\mathbf{b}}$ for each group
- Uses the curvature (Hessian) to determine the scaling
- Essentially a refinement of the Laplace approximation using numerical integration instead of a second-order Taylor series

**Accuracy:**
- More accurate than Laplace for small cluster sizes and binary/count data
- Accuracy increases with the number of quadrature points $K$ (typically $K = 7$ to $K = 25$)
- Becomes exact as $K \to \infty$

**Computational cost:**
- Scales exponentially with the dimension of the random effects
- Feasible for scalar random effects (random intercepts only) or low-dimensional random effects
- Prohibitive for random slopes in high-dimensional grouping structures

**Implementation:**
- **lme4**: `glmer(..., nAGQ = K)` where `K > 1` activates AGQ
- **glmmTMB**: Does not support AGQ (Laplace only)

**Practical guidance:**
- Use AGQ for **validation** of models fitted with Laplace (especially for small cluster sizes)
- For scalar random effects (random intercept only), use $K = 15$ or higher for final inference
- For random slopes, AGQ becomes impractical; use Bayesian methods instead

**Example:**
```r
library(lme4)

# Laplace approximation (default)
m1 <- glmer(claims ~ age + region + (1 | policyholder_id),
            family = poisson(), data = df, nAGQ = 1)

# AGQ with 15 quadrature points
m2 <- glmer(claims ~ age + region + (1 | policyholder_id),
            family = poisson(), data = df, nAGQ = 15)

# Compare variance component estimates
VarCorr(m1)  # Laplace
VarCorr(m2)  # AGQ (more accurate for small cluster sizes)
```

---

### 1.5 Penalized Quasi-Likelihood (PQL)

**Historical context**: PQL was an early approximation method for GLMMs, popularized by Breslow and Clayton (1993) and implemented in `MASS::glmmPQL`.

**Core idea:**
- Linearize the GLMM around current parameter estimates using a first-order Taylor expansion (quasi-likelihood)
- Fit the resulting linear mixed model
- Iterate until convergence

**Why it's deprecated:**
- **Bias**: PQL produces biased estimates of both fixed effects and variance components, especially for binary data and small cluster sizes
- The bias does not vanish asymptotically in the number of clusters (unlike Laplace)
- Modern alternatives (Laplace, AGQ, Bayesian) are superior in accuracy and speed

**When it still appears:**
- Older actuarial papers (pre-2010) may reference PQL
- Some SAS procedures (PROC GLIMMIX) offer PQL as an option

**Recommendation for the learning plan:**
- **Mention PQL briefly** as a historical method, but do not implement it or recommend it
- Explain *why* it's deprecated (bias, not asymptotically correct) to build critical understanding
- Emphasize that modern software (glmmTMB, lme4, brms) has superseded PQL

---

### 1.6 Template Model Builder (TMB) and glmmTMB

**TMB (Template Model Builder)** is a C++ library and R package for fitting latent variable models via automatic differentiation (AD) and Laplace approximation.

**Key innovation**: TMB uses **automatic differentiation** to compute exact derivatives of the log-likelihood with respect to parameters. This enables:
1. Extremely fast optimization (no finite-difference approximations)
2. Accurate Hessian matrices for standard errors
3. Ability to fit complex models (zero-inflation, random effects in dispersion, spatial structures)

**Architecture:**
1. User specifies the model in a C++ template (the "template" in TMB)
2. TMB compiles the template to machine code
3. The Laplace approximation is applied to integrate out random effects
4. Optimization (via `nlminb` or similar) finds ML or REML estimates
5. The Hessian provides standard errors via the delta method

**glmmTMB as a TMB wrapper:**
- **glmmTMB** provides a high-level R interface to TMB specifically for GLMMs
- Users specify models via formula syntax (no C++ required)
- glmmTMB pre-codes a wide range of distributions, link functions, and random effect structures
- Under the hood, glmmTMB translates the formula into a TMB template and calls TMB

**Advantages of TMB/glmmTMB:**
- **Speed**: 10-100x faster than lme4 for complex models (zero-inflation, large datasets)
- **Flexibility**: Easy to extend to custom distributions and structures
- **Accuracy**: Automatic differentiation eliminates numerical derivative errors

**Limitations:**
- Only supports Laplace approximation (no AGQ option)
- Debugging model convergence requires understanding of optimization internals
- For simple models, lme4 is equally good and has more mature diagnostics

**Practical guidance:**
- Use **glmmTMB** as the default for actuarial GLMMs (speed, flexibility, zero-inflation)
- Use **lme4 with AGQ** for validation when cluster sizes are small
- For complex custom models (e.g., spatial-temporal hierarchies), consider learning TMB directly

**Reference:**
- Kristensen et al. (2016). "TMB: Automatic Differentiation and Laplace Approximation." *Journal of Statistical Software*, 70(5). https://doi.org/10.18637/jss.v070.i05

---

### 1.7 Connection to Classical Bühlmann Credibility Estimators

**The equivalence (informal):**

In a **random intercept LMM** with Gaussian response and balanced data:
$$
y_{ij} = \mu + b_i + \epsilon_{ij}, \quad b_i \sim \mathcal{N}(0, \tau^2), \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)
$$

The **Best Linear Unbiased Predictor (BLUP)** of $b_i$ is:
$$
\hat{b}_i = \frac{n_i \tau^2}{n_i \tau^2 + \sigma^2} (\bar{y}_i - \mu)
$$

This is **exactly the Bühlmann credibility formula**:
$$
\hat{\mu}_i = Z_i \bar{y}_i + (1 - Z_i) \mu
$$
where $Z_i = \frac{n_i \tau^2}{n_i \tau^2 + \sigma^2}$ is the credibility weight.

**Key insights:**
1. **BLUP = credibility-weighted estimator**: The random effect prediction shrinks group-specific means toward the population mean
2. **Variance components**: $\tau^2$ is the between-group variance (process variance in credibility theory); $\sigma^2$ is the within-group variance (sampling variance)
3. **Likelihood-based**: BLUP is derived from likelihood, not from moment matching (Bühlmann's original approach)

**When the equivalence holds:**
- Gaussian response (LMM, not GLMM)
- Random intercepts only (extensions exist for random slopes)
- Balanced or unbalanced data (BLUP generalizes credibility to unbalanced designs)

**When it breaks down:**
1. **Non-Gaussian responses**: For Poisson/binomial/gamma, the BLUP is no longer exact credibility, but the *conceptual* equivalence (partial pooling, shrinkage toward collective experience) remains
2. **Nonlinear link functions**: Predictions on the response scale vs. the linear predictor scale differ (see Section 3.3)
3. **Complex random effect structures**: Random slopes, crossed effects, and temporal correlations go beyond classical credibility

**Practical guidance:**
- Present this equivalence **early** (Module 2 or 3) to motivate GLMMs as a generalization of credibility
- Derive the BLUP formula algebraically for the simple Gaussian case
- Show numerically that BLUP = Bühlmann credibility for balanced Gaussian data
- Explain carefully where the equivalence is approximate (GLMMs) vs. exact (LMMs)

**Key papers:**
- Robinson (1991). "That BLUP is a Good Thing: The Estimation of Random Effects." *Statistical Science*, 6(1), 15-51. (Classic exposition)
- Bühlmann & Gisler (2005). *A Course in Credibility Theory and its Applications*. (Chapter on connection to mixed models)
- Ohlsson & Johansson (2010). *Non-Life Insurance Pricing with Generalized Linear Models*. (Actuarial perspective on BLUP-credibility)

---

## 2. Bayesian Hierarchical Models

### 2.1 Why Bayesian Methods for GLMMs?

Bayesian hierarchical models are a natural framework for actuarial pricing because:

1. **Credibility theory is inherently Bayesian**: Bühlmann credibility can be derived as the Bayesian posterior mean under a conjugate prior (see Bühlmann & Gisler, Chapter 3)

2. **Uncertainty quantification**: Credible intervals directly answer "What is the probability that the rate is between $X$ and $Y$?" (vs. frequentist confidence intervals, which are harder to interpret)

3. **Incorporation of expert judgment**: Informative priors can encode actuarial knowledge (e.g., "age effects should be monotonic and smooth")

4. **Full distributional inference**: Obtain posterior distributions for all parameters (not just point estimates and asymptotic standard errors)

5. **Complex models**: Bayesian MCMC handles high-dimensional random effects (random slopes, crossed effects) that are intractable for AGQ

**Computational cost**: Bayesian methods are slower than ML/REML, but modern algorithms (HMC/NUTS) have made them practical for realistic actuarial datasets (millions of observations, thousands of groups).

---

### 2.2 The brms/Stan Ecosystem

**Stan** is a probabilistic programming language for Bayesian inference using Hamiltonian Monte Carlo (HMC) and the No-U-Turn Sampler (NUTS).

**brms** is an R package that provides a high-level interface to Stan, using lme4-like formula syntax.

**Key capabilities:**
- All distributions and link functions supported by glmmTMB, plus many more
- Flexible prior specification (weakly informative defaults)
- Random intercepts, random slopes, crossed random effects
- Zero-inflation, hurdle models, distributional models (model both mean and variance)
- Spatial and temporal correlation structures
- Non-linear models, splines, Gaussian processes

**Workflow:**
```r
library(brms)

# Bayesian Poisson GLMM with weakly informative priors
m_bayes <- brm(
  claims ~ age + region + (1 | policyholder_id),
  family = poisson(),
  data = df,
  prior = c(
    prior(normal(0, 2), class = "b"),         # Fixed effects (log scale)
    prior(exponential(1), class = "sd")       # Random effect SD (tau)
  ),
  chains = 4, iter = 2000, warmup = 1000,
  cores = 4,  # Parallel chains
  seed = 123
)

# Posterior summary
summary(m_bayes)

# Posterior predictive checks
pp_check(m_bayes, ndraws = 100)

# Credible intervals for random effects
ranef(m_bayes)
```

**Prior specification:**
- **Weakly informative priors**: Default in brms; allow the data to dominate but prevent extreme estimates
- **Regularizing priors**: Shrink fixed effects toward zero (like ridge regression); useful for high-dimensional models
- **Informative priors**: Encode actuarial knowledge (e.g., "age effect should be positive"; "claim frequency decreases after age 65")

**Example (informative prior for age effect):**
```r
# Assume actuarial judgment: log(claim rate) increases by 0.01 to 0.05 per year of age
prior(normal(0.03, 0.01), class = "b", coef = "age")
```

**Convergence diagnostics:**
1. **Rhat (R-hat)**: Potential scale reduction factor; measures chain mixing. $\hat{R} < 1.01$ indicates convergence
2. **Effective sample size (ESS)**: Accounts for autocorrelation in MCMC samples. ESS > 400 per chain is acceptable; ESS > 1000 is excellent
3. **Trace plots**: Visual inspection of chain behavior (should look like "fuzzy caterpillars")
4. **Divergent transitions**: Warnings from Stan indicating the sampler struggled; often due to poor parameterization or very small/large variance components

**Computational cost:**
- Bayesian GLMM: ~10-100x slower than glmmTMB for the same model
- For large datasets (millions of rows), expect 10 minutes to 2 hours
- Use parallel chains (4 cores) to speed up
- Consider INLA for very large datasets (see Section 4.3)

**When to use Bayesian methods:**
- When credible intervals are required for regulatory or stakeholder communication
- When incorporating expert priors is scientifically justified
- For complex random effect structures (random slopes, crossed effects) where AGQ is intractable
- For model comparison via LOO-CV (leave-one-out cross-validation), which is more robust than AIC

**When frequentist methods suffice:**
- Exploratory analysis (glmmTMB is much faster)
- Simple random intercept models with large cluster sizes
- When computational resources are limited

---

### 2.3 PyMC and Bambi for Python Users

**PyMC** is a Python library for Bayesian inference using MCMC (NUTS sampler, same as Stan).

**Bambi** is a high-level interface to PyMC, analogous to brms for Stan.

**Example (Bambi):**
```python
import bambi as bmb
import pandas as pd

# Bayesian Poisson GLMM
model = bmb.Model("claims ~ age + region + (1|policyholder_id)",
                  data=df, family="poisson")

# Fit with NUTS
results = model.fit(draws=1000, tune=1000, chains=4, cores=4)

# Summary
results.summary()

# Posterior predictive checks
model.predict(results, kind="pps")
```

**Comparison to brms:**
- **PyMC/Bambi**: Native Python; integrates well with pandas, scikit-learn, ArviZ (Bayesian diagnostics)
- **brms/Stan**: More mature for GLMMs; larger user base; more extensive documentation
- **Performance**: Comparable (both use NUTS)

**Recommendation for the learning plan:**
- **Primary focus: brms/Stan** (because the project is R-centric and brms has better GLMM support)
- **Secondary: PyMC/Bambi** as an optional module for learners who prefer Python
- Include one notebook demonstrating PyMC for comparison

---

### 2.4 Formal Equivalence Between Bayesian Hierarchical Models and Credibility Theory

**Theorem (informal)**: Under conjugate priors, the Bayesian posterior mean is equivalent to the Bühlmann credibility estimator.

**Setup (Gaussian-Gaussian conjugate model):**
- Observations: $y_{ij} \mid \mu_i \sim \mathcal{N}(\mu_i, \sigma^2)$, $j = 1, \ldots, n_i$
- Prior on group means: $\mu_i \sim \mathcal{N}(\mu_0, \tau^2)$
- Collective mean: $\mu_0$ is known (or has a hyperprior)

**Bayesian posterior:**
$$
\mu_i \mid \mathbf{y}_i \sim \mathcal{N}\left( \frac{\frac{n_i}{\sigma^2} \bar{y}_i + \frac{1}{\tau^2} \mu_0}{\frac{n_i}{\sigma^2} + \frac{1}{\tau^2}}, \quad \frac{1}{\frac{n_i}{\sigma^2} + \frac{1}{\tau^2}} \right)
$$

**Posterior mean:**
$$
\mathbb{E}[\mu_i \mid \mathbf{y}_i] = \frac{n_i \tau^2}{n_i \tau^2 + \sigma^2} \bar{y}_i + \frac{\sigma^2}{n_i \tau^2 + \sigma^2} \mu_0 = Z_i \bar{y}_i + (1 - Z_i) \mu_0
$$

where $Z_i = \frac{n_i \tau^2}{n_i \tau^2 + \sigma^2}$ is the **Bühlmann credibility weight**.

**Key insights:**
1. The posterior mean is a weighted average of the individual group mean $\bar{y}_i$ and the collective mean $\mu_0$
2. The weight $Z_i$ increases with exposure $n_i$ and between-group variance $\tau^2$
3. The posterior variance quantifies uncertainty (not available in classical credibility)

**Extension to GLMMs:**
- For non-Gaussian responses, the posterior is not analytically tractable
- MCMC provides samples from the posterior
- The posterior mean still has a credibility-like interpretation (shrinkage toward population mean)

**Practical guidance:**
- **Derive this result explicitly** in Module 2 (Credibility Theory) or Module 7 (Bayesian Extensions)
- Show numerically that Bayesian posterior mean = Bühlmann credibility for Gaussian data
- Extend conceptually to GLMMs (posterior mean ≈ credibility-weighted estimate on the linear predictor scale)

**Key references:**
- Bühlmann & Gisler (2005). *A Course in Credibility Theory and its Applications*. Chapter 3: Bayesian credibility
- Klugman, Panjer & Willmot (2012). *Loss Models: From Data to Decisions*. Chapter on Bayesian credibility
- Makov et al. (1996). "Credibility Theory." In *Encyclopedia of Statistical Sciences*. (Historical overview)

---

### 2.5 Prior Selection for Actuarial Applications

**Principle**: Priors should be **weakly informative**—informative enough to regularize the model and avoid extreme estimates, but weak enough to let the data dominate.

**Default priors in brms:**
- **Fixed effects (regression coefficients)**: Improper flat priors (effectively uniform over the reals)
  - For GLMMs: This can lead to poor convergence; brms automatically shifts to weakly informative priors in some cases
- **Random effect standard deviations**: Half-Student-t(3, 0, 2.5) (weakly informative, heavy-tailed)
- **Dispersion/shape parameters**: Gamma(0.01, 0.01) or similar weakly informative priors

**Recommended priors for actuarial GLMMs:**

1. **Fixed effects (log scale for count/rate models):**
   - Normal(0, 2) or Normal(0, 5): Implies rate ratios (exp(β)) between 0.01 and 100 are plausible
   - For age/year effects: Normal(0, 0.1) if you expect small year-to-year changes

2. **Random effect standard deviations (τ):**
   - Exponential(1): Weakly informative; mode at 0, allowing for no random effect if unsupported by data
   - Half-Normal(0, 1): Similar interpretation
   - Half-Student-t(3, 0, 2.5): Heavier tails; more robust to outliers

3. **Correlation matrices (for random slopes):**
   - LKJ(ζ): LKJ-Correlation prior with shape ζ
     - ζ = 1: Uniform over correlation matrices
     - ζ = 2: Weakly favors independence (good default)
     - ζ > 2: Stronger prior toward independence

**Sensitivity analysis:**
- Always perform **prior sensitivity analysis**: Refit the model with different priors and check that conclusions are robust
- If posterior estimates change substantially with different priors, the data are not informative enough (need more data or stronger priors justified by external knowledge)

**Informative priors for regulatory defensibility:**
- If using informative priors (e.g., "age effect must be positive"), document:
  1. The source of the prior (historical data, expert judgment, regulatory requirement)
  2. Prior predictive checks: What do simulations from the prior imply about observable data?
  3. Sensitivity analysis: How do results change if the prior is relaxed?

**Example (prior predictive check):**
```r
# Specify model with priors (do not fit yet)
m_prior <- brm(
  claims ~ age + (1 | policyholder_id),
  family = poisson(), data = df,
  prior = c(
    prior(normal(0, 2), class = "b"),
    prior(exponential(1), class = "sd")
  ),
  sample_prior = "only"  # Sample from prior only (no data)
)

# Simulate data from the prior
pp_check(m_prior, type = "dens_overlay", ndraws = 100)
```

If the prior predictive simulations produce absurdly large or small claim rates, revise the prior.

---

### 2.6 When to Introduce Bayesian Methods in the Learning Plan?

**Option 1: Late introduction (current plan = Module 7)**
- **Pro**: Builds on frequentist foundation; learners see Bayesian as an "extension"
- **Con**: Misses the conceptual throughline (credibility = Bayesian posterior mean); Bayesian methods feel like an add-on

**Option 2: Early parallel introduction (recommended)**
- Introduce Bayesian hierarchical models in **Module 3** (alongside glmmTMB)
- Show that Bühlmann credibility = Bayesian posterior mean (Module 2)
- Fit the same models with glmmTMB (fast, ML/REML) and brms (slower, full posterior) throughout
- Emphasize trade-offs: speed vs. full uncertainty quantification

**Recommended approach:**
- **Module 2**: Derive Bühlmann credibility as Bayesian posterior mean
- **Module 3**: Fit first GLMM with both glmmTMB and brms; compare results
- **Modules 4-6**: Use glmmTMB as the workhorse (speed); use brms for validation and when credible intervals are needed
- **Module 7**: Deepen Bayesian theory (prior selection, sensitivity analysis, posterior predictive checks, model comparison via LOO)

This approach treats Bayesian methods as a **parallel framework** rather than an afterthought.

---

## 3. Mathematical Foundations

### 3.1 Linear Algebra Prerequisites

**What the learner needs to know:**

1. **Matrix operations**: Matrix multiplication, transpose, inverse, determinant
2. **Positive definite matrices**: Variance-covariance matrices are positive definite; eigenvalues are all positive
3. **Cholesky decomposition**: $\mathbf{G} = \mathbf{L} \mathbf{L}^\top$ (used in MCMC and optimization)
4. **Block matrix inversion**: Sherman-Morrison-Woodbury formula (appears in Henderson's equations)
5. **Quadratic forms**: $\mathbf{x}^\top \mathbf{A} \mathbf{x}$ (appears in likelihood expressions)

**What can be treated as a black box:**
- Sparse matrix algorithms (CHOLMOD in lme4)
- Automatic differentiation internals (TMB)
- HMC/NUTS integrator details (Stan)

**Pedagogical guidance:**
- **Module 1 or 2**: Include a brief "Mathematical Prerequisites" appendix with a review of matrix algebra
- Emphasize **interpretation** over derivation (e.g., "the variance-covariance matrix $\mathbf{G}$ determines the correlation between random effects")
- Provide R code to compute key quantities (Cholesky decomposition, quadratic forms) so learners build intuition

---

### 3.2 The Mixed Model Equations (Henderson's Equations)

For **linear mixed models** (LMMs), the BLUP and ML/REML estimates satisfy **Henderson's mixed model equations**:

$$
\begin{bmatrix}
\mathbf{X}^\top \mathbf{X} & \mathbf{X}^\top \mathbf{Z} \\
\mathbf{Z}^\top \mathbf{X} & \mathbf{Z}^\top \mathbf{Z} + \mathbf{G}^{-1} \sigma^2
\end{bmatrix}
\begin{bmatrix}
\hat{\boldsymbol{\beta}} \\
\hat{\mathbf{b}}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{X}^\top \mathbf{y} \\
\mathbf{Z}^\top \mathbf{y}
\end{bmatrix}
$$

where:
- $\mathbf{X}$ is the fixed effects design matrix (dimension $n \times p$)
- $\mathbf{Z}$ is the random effects design matrix (dimension $n \times q$)
- $\mathbf{G}$ is the random effect variance-covariance matrix (dimension $q \times q$)
- $\sigma^2$ is the residual variance

**Key insights:**
1. This is a system of linear equations that jointly solves for $\hat{\boldsymbol{\beta}}$ (fixed effects) and $\hat{\mathbf{b}}$ (random effects)
2. The term $\mathbf{G}^{-1} \sigma^2$ acts as a **ridge penalty**, shrinking random effects toward zero
3. Henderson's equations are equivalent to **penalized least squares** (BLUP = ridge regression in a mixed model context)

**For GLMMs:**
- Henderson's equations do not apply directly (non-Gaussian response)
- Analogous equations arise from the Laplace approximation or iteratively reweighted least squares (IRLS)
- The conceptual insight (BLUP = penalized estimate) still holds

**Pedagogical guidance:**
- **Derive Henderson's equations** for the simple Gaussian case (Module 4)
- Show that the BLUP is a penalized estimate (connection to ridge regression / regularization)
- Explain how the penalty $\mathbf{G}^{-1} \sigma^2$ produces shrinkage (credibility)
- For GLMMs, present the Laplace-approximated version (see Bates et al., lme4 Theory vignette)

**Reference:**
- Henderson (1975). "Best Linear Unbiased Estimation and Prediction under a Selection Model." *Biometrics*, 31(2), 423-447.
- Bates et al. (2015). "Fitting Linear Mixed-Effects Models Using lme4." *Journal of Statistical Software*, 67(1). (Section 2: Theory)

---

### 3.3 Conditional vs. Marginal Model Interpretation

**This is the single most important theoretical subtlety for actuarial applications.**

**Definitions:**

1. **Conditional model**: Predictions for an individual *given* their random effect $b_i$
   $$
   \mathbb{E}[y_{ij} \mid \mathbf{x}_{ij}, b_i] = g^{-1}(\mathbf{x}_{ij}^\top \boldsymbol{\beta} + b_i)
   $$

2. **Marginal model**: Predictions averaged over the distribution of random effects
   $$
   \mathbb{E}[y_{ij} \mid \mathbf{x}_{ij}] = \int g^{-1}(\mathbf{x}_{ij}^\top \boldsymbol{\beta} + b_i) \, f(b_i) \, db_i
   $$

**Key result**: For nonlinear link functions (e.g., log link for Poisson), the marginal expectation is **not** equal to the conditional expectation evaluated at $b_i = 0$:
$$
\mathbb{E}[y_{ij} \mid \mathbf{x}_{ij}] \neq g^{-1}(\mathbf{x}_{ij}^\top \boldsymbol{\beta})
$$

**Example (Poisson GLMM with log link):**
- Conditional: $\mathbb{E}[y_{ij} \mid b_i] = \exp(\mathbf{x}_{ij}^\top \boldsymbol{\beta} + b_i)$
- Marginal: $\mathbb{E}[y_{ij}] = \mathbb{E}[\exp(\mathbf{x}_{ij}^\top \boldsymbol{\beta} + b_i)] = \exp(\mathbf{x}_{ij}^\top \boldsymbol{\beta}) \, \mathbb{E}[\exp(b_i)]$

If $b_i \sim \mathcal{N}(0, \tau^2)$, then $\mathbb{E}[\exp(b_i)] = \exp(\tau^2 / 2) > 1$.

Thus:
$$
\mathbb{E}[y_{ij}] = \exp(\mathbf{x}_{ij}^\top \boldsymbol{\beta} + \tau^2 / 2)
$$

The marginal mean is **inflated** by a factor $\exp(\tau^2 / 2)$ relative to the conditional mean at $b_i = 0$.

**Actuarial implications:**

1. **Population-level predictions** (e.g., "expected claims for a new policyholder with covariates $\mathbf{x}$"):
   - Use marginal prediction: $\exp(\mathbf{x}^\top \boldsymbol{\beta} + \tau^2 / 2)$
   - This is the prediction for a randomly selected policyholder from the population

2. **Individual-level predictions** (e.g., "expected claims for policyholder $i$ with observed history"):
   - Use conditional prediction: $\exp(\mathbf{x}^\top \boldsymbol{\beta} + \hat{b}_i)$
   - This incorporates the individual's random effect (credibility adjustment)

3. **Regulatory reporting**:
   - Marginal predictions are often required for rate filings (population average)
   - Conditional predictions are used for individual pricing

**Practical guidance in R:**

```r
library(glmmTMB)

m <- glmmTMB(claims ~ age + region + (1 | policyholder_id),
             family = poisson(), data = df)

# Conditional prediction (includes random effects)
predict(m, newdata = new_data, type = "response", re.form = NULL)

# Marginal prediction (population-level, random effects set to 0)
predict(m, newdata = new_data, type = "response", re.form = NA)

# Marginal prediction with correction for variance
# (Manual calculation for Poisson log link)
tau_sq <- as.numeric(VarCorr(m)$cond$policyholder_id)
predict(m, newdata = new_data, type = "response", re.form = NA) * exp(tau_sq / 2)
```

**Pedagogical guidance:**
- **Devote a full section** (Module 4 or 5) to conditional vs. marginal interpretation
- Derive the $\exp(\tau^2 / 2)$ correction factor for Poisson/log link
- Show the magnitude of the difference for typical actuarial variance components (e.g., $\tau^2 = 0.2$ implies 10% inflation)
- Emphasize that this is **not** a software bug—it's a mathematical property of nonlinear link functions

**Key references:**
- Skrondal & Rabe-Hesketh (2009). "Prediction in multilevel generalized linear models." *Journal of the Royal Statistical Society: Series A*, 172(3), 659-687.
- Muff et al. (2016). "Marginal or conditional regression models for correlated non-normal data?" *Methods in Ecology and Evolution*, 7(12), 1514-1524.

---

### 3.4 Variance-Covariance Structures for Random Effects

**Random intercept only:**
$$
\mathbf{G} = \tau^2 \mathbf{I}_q
$$
where $q$ is the number of groups. All random intercepts have the same variance $\tau^2$ and are uncorrelated.

**Random intercept and slope:**
$$
\mathbf{G} = \begin{bmatrix}
\tau_0^2 & \rho \tau_0 \tau_1 \\
\rho \tau_0 \tau_1 & \tau_1^2
\end{bmatrix}
$$
where $\tau_0^2$ is the intercept variance, $\tau_1^2$ is the slope variance, and $\rho$ is the correlation between intercepts and slopes.

**Practical interpretation (actuarial context):**
- $\tau_0^2$: Variance in baseline claim frequency across policyholders (after accounting for covariates)
- $\tau_1^2$: Variance in the age effect across policyholders (some policyholders' claim rates increase faster with age)
- $\rho$: Correlation between baseline frequency and age effect (e.g., high-frequency policyholders may have steeper age effects)

**Estimation:**
- In frequentist software (glmmTMB, lme4): $\tau_0^2, \tau_1^2, \rho$ are estimated via ML/REML
- In Bayesian software (brms): Priors are placed on $\tau_0, \tau_1$ (standard deviations) and the correlation matrix

**Pedagogical guidance:**
- **Visualize** the variance-covariance structure: Plot the bivariate distribution of $(b_{0i}, b_{1i})$
- Show how positive/negative correlation affects predictions
- Discuss when to estimate correlation vs. constrain to zero (parsimony)

---

## 4. Modern Extensions and Cutting-Edge Methods

### 4.1 INLA (Integrated Nested Laplace Approximation)

**INLA** is an alternative to MCMC for fast approximate Bayesian inference in latent Gaussian models (which include GLMMs).

**Core idea:**
- Use nested Laplace approximations to approximate the marginal posterior distributions of hyperparameters (variance components)
- For each hyperparameter value, compute the conditional posterior of latent variables (random effects) via Laplace
- Integrate over hyperparameters numerically

**Advantages over MCMC:**
- **Speed**: 10-100x faster than MCMC for GLMMs
- **Deterministic**: No Monte Carlo error; same results every run
- **Accuracy**: Comparable to MCMC for most GLMMs

**Limitations:**
- Restricted to **latent Gaussian models** (random effects must be Gaussian)
- Less flexible than Stan (cannot easily implement custom models)
- Requires learning a new syntax (INLA formula notation differs from lme4/brms)

**When to use INLA:**
- Very large datasets (millions of observations) where MCMC is too slow
- Spatial and spatio-temporal models (INLA has built-in support for SPDE approximations to Gaussian processes)
- When deterministic results are required (e.g., regulatory submissions)

**R implementation:**
```r
library(INLA)

# INLA Poisson GLMM
formula <- claims ~ age + region + f(policyholder_id, model = "iid")
result <- inla(formula, family = "poisson", data = df)

# Summary
summary(result)

# Posterior marginals for hyperparameters
plot(result$marginals.hyperpar[[1]])
```

**Pedagogical guidance:**
- **Optional advanced module** (Module 9): Introduce INLA as a fast Bayesian alternative to MCMC
- Compare INLA vs. brms on a medium-sized dataset (e.g., 100,000 observations, 1,000 groups)
- Discuss trade-offs: speed vs. flexibility

**Key references:**
- Rue, Martino & Chopin (2009). "Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations." *Journal of the Royal Statistical Society: Series B*, 71(2), 319-392.
- Blangiardo & Cameletti (2015). *Spatial and Spatio-temporal Bayesian Models with R-INLA*. Wiley. (Comprehensive textbook)

---

### 4.2 Regularized Mixed Models (GLMMNet)

**Motivation**: When predictors are high-dimensional or highly correlated, fixed effects can overfit. **Regularization** (L1/L2 penalties) shrinks coefficients toward zero to prevent overfitting.

**GLMMNet** combines:
1. **Mixed model random effects** (for hierarchical structure)
2. **Elastic net regularization** (for fixed effects)

**Model:**
$$
\log L(\boldsymbol{\beta}, \mathbf{b}, \boldsymbol{\theta}) - \lambda \left( \alpha \|\boldsymbol{\beta}\|_1 + (1 - \alpha) \|\boldsymbol{\beta}\|_2^2 \right)
$$

where:
- $\lambda$: Regularization strength (tuned via cross-validation)
- $\alpha \in [0, 1]$: Elastic net mixing parameter (α=1 is LASSO; α=0 is ridge)
- Random effects $\mathbf{b}$ are **not** penalized (only fixed effects)

**Actuarial use cases:**
- High-cardinality categorical predictors (thousands of zip codes, vehicle makes)
- Feature selection: LASSO shrinks irrelevant coefficients to exactly zero
- Collinearity: Ridge regression handles correlated predictors

**Challenges:**
- GLMMNet is not yet widely implemented in standard R packages
- Manual implementation requires custom optimization (e.g., proximal gradient descent)

**Current implementations:**
- **glmmLasso** (R package): LASSO penalty for GLMMs (limited distribution support)
- **glmnet** + **lme4**: Fit glmnet on residuals from lme4 (approximate approach)
- **Custom TMB code**: Implement elastic net in TMB

**ArXiv reference:**
- Yi & Zeng (2023). "GLMMNet: Penalized Inference for Generalized Linear Mixed Models." arXiv:2301.12710. https://arxiv.org/abs/2301.12710

**Pedagogical guidance:**
- **Advanced module (Module 9)**: Introduce regularization conceptually
- Implement a simple example using glmmLasso or custom TMB code
- Discuss when regularization is warranted (high-dimensional data) vs. when standard GLMM suffices

---

### 4.3 Distributional GLMMs (Modeling Both Mean and Dispersion)

**Standard GLMM**: Model only the mean (conditional on covariates and random effects)

**Distributional GLMM**: Model both the mean **and** the variance/dispersion as functions of covariates

**Example (negative binomial with varying dispersion):**
- Mean model: $\log(\mu_{ij}) = \mathbf{x}_{ij}^\top \boldsymbol{\beta} + b_i$
- Dispersion model: $\log(\phi_{ij}) = \mathbf{z}_{ij}^\top \boldsymbol{\gamma}$

where $\phi_{ij}$ is the dispersion parameter (controlling overdispersion).

**Actuarial motivation:**
- Claim variance may depend on covariates (e.g., high-risk drivers have higher variance, not just higher mean frequency)
- Modeling dispersion improves predictive accuracy and uncertainty quantification

**Implementation:**
- **brms**: Native support via `bf()` syntax
  ```r
  bf(claims ~ age + region + (1 | policyholder_id),
     phi ~ region)  # Dispersion depends on region
  ```
- **glmmTMB**: Limited support (dispersion formula for some families)
- **GAMLSS** (R package): Generalized Additive Models for Location, Scale, and Shape (very flexible)

**Pedagogical guidance:**
- **Module 5 or 9**: Introduce distributional modeling as an advanced extension
- Show an example where modeling dispersion improves fit (e.g., heteroscedasticity in claim counts by region)
- Compare standard GLMM vs. distributional GLMM via AIC and posterior predictive checks

**Key reference:**
- Bürkner (2018). "Advanced Bayesian Multilevel Modeling with the R Package brms." *The R Journal*, 10(1), 395-411.

---

### 4.4 Measurement Error and SIMEX Methods

**Problem**: Covariates are often measured with error (e.g., driver age estimated from license date; vehicle value from book estimates). Ignoring measurement error:
1. **Biases** coefficient estimates toward zero (attenuation bias)
2. **Underestimates** standard errors (false precision)

**SIMEX (Simulation-Extrapolation):**
1. Add known amounts of measurement error to the observed covariate
2. Fit the model for each error level
3. Extrapolate the coefficient estimates back to zero error

**Algorithm:**
1. Observed covariate: $x^*_{ij} = x_{ij} + u_{ij}$, where $u_{ij} \sim \mathcal{N}(0, \sigma_u^2)$ is measurement error
2. For each $\lambda \in \{0, 0.5, 1, 1.5, 2\}$:
   - Add error: $x^{**}_{ij} = x^*_{ij} + \sqrt{\lambda} \cdot v_{ij}$, $v_{ij} \sim \mathcal{N}(0, \sigma_u^2)$
   - Fit model; record $\hat{\beta}(\lambda)$
3. Fit a polynomial $\hat{\beta}(\lambda) = a + b \lambda + c \lambda^2$
4. Extrapolate to $\lambda = -1$ (zero error): $\hat{\beta}_{\text{SIMEX}} = a - b + c$

**R implementation:**
- **simex** package (for GLMs; extension to GLMMs requires custom code)
- **Bayesian SIMEX**: Implement in Stan or JAGS

**Actuarial examples of measurement error:**
- Driver age (birth date may be estimated)
- Annual mileage (self-reported, often rounded)
- Vehicle value (depreciation estimates)

**ArXiv reference:**
- Cook & Stefanski (1994). "Simulation-Extrapolation Estimation in Parametric Measurement Error Models." *Journal of the American Statistical Association*, 89(428), 1314-1328.
- (Specific GLMM extensions: Search for "SIMEX GLMM" on arXiv)

**Pedagogical guidance:**
- **Module 9**: Introduce measurement error as an advanced topic
- Implement SIMEX for a simple GLM; discuss extension to GLMMs
- Discuss when measurement error is likely to be consequential (small signal-to-noise ratio)

---

### 4.5 Gradient Boosting + Random Effects Hybrids

**Motivation**: Gradient boosting (XGBoost, LightGBM) excels at capturing complex nonlinear relationships but does not naturally handle hierarchical data.

**Hybrid approaches:**
1. **Two-stage**: Fit a mixed model to extract random effects; use them as features in a boosting model
2. **Joint estimation**: Incorporate random effects as a component in the boosting objective

**Example (two-stage):**
1. Fit GLMM: Extract $\hat{b}_i$ (random effects)
2. Create augmented dataset: $\mathbf{x}_{ij}' = [\mathbf{x}_{ij}, \hat{b}_i]$
3. Fit XGBoost on $\mathbf{x}_{ij}'$

**Challenges:**
- Uncertainty in $\hat{b}_i$ is ignored in stage 2
- No straightforward joint optimization

**Current research:**
- Sigrist (2021). "Gaussian Process Boosting." arXiv:2004.02653. (Combines boosting with Gaussian process random effects)
- Segalini et al. (2020). "Mixed-effect models and gradient boosting for probabilistic forecasting." (Industry application)

**Pedagogical guidance:**
- **Optional advanced topic** (Module 9): Mention as a frontier area
- Implement a simple two-stage example
- Discuss trade-offs: interpretability (GLMM wins) vs. predictive accuracy (boosting may win for complex interactions)

---

## 5. Reference Assessment: Theory Texts

### 5.1 Stroup, "Generalized Linear Mixed Models: Modern Concepts, Methods and Applications" (2013)

**Coverage:**
- Comprehensive treatment of GLMM theory and practice
- Emphasis on estimation methods (Laplace, PQL, MCMC)
- Detailed discussion of SAS PROC GLIMMIX

**Strengths:**
- Mathematically rigorous but accessible
- Extensive worked examples
- Covers both frequentist and Bayesian approaches

**Limitations:**
- SAS-centric (less relevant for R users)
- Published 2013; predates glmmTMB and modern brms
- Less emphasis on actuarial applications

**Recommendation:**
- **Targeted chapters**: Chapters 2-4 (theory), Chapter 8 (overdispersion), Chapter 10 (Bayesian)
- Use as a **reference** for theoretical depth, not as the primary text
- Supplement with R-based examples

**Assessment**: **Just right** for theoretical rigor, but needs translation to R/actuarial context.

---

### 5.2 Bühlmann & Gisler, "A Course in Credibility Theory and its Applications" (2005)

**Coverage:**
- Authoritative treatment of classical and Bayesian credibility
- Chapter 8: Connection to mixed models and BLUP

**Strengths:**
- Actuarial perspective (written for actuaries)
- Rigorous derivations of credibility formulas
- Explicit treatment of BLUP-credibility equivalence

**Limitations:**
- Focused on credibility; mixed models are a side topic
- Minimal coverage of GLMMs (mostly Gaussian/LMM)
- No computational implementation (formulas only)

**Recommendation:**
- **Essential reading**: Chapters 3 (Bayesian credibility), 8 (BLUP connection)
- Use to **motivate** the transition from credibility to GLMMs (Module 2-3)
- Not sufficient as a standalone GLMM text

**Assessment**: **Essential for conceptual foundation**, but must be paired with a GLMM-specific text.

---

### 5.3 Pinheiro & Bates, "Mixed-Effects Models in S and S-PLUS" (2000)

**Coverage:**
- Classic text on LMMs and NLMMs (nonlinear mixed models)
- Foundation for the nlme and lme4 packages

**Strengths:**
- Thorough treatment of variance-covariance structures
- Detailed case studies
- Still widely cited

**Limitations:**
- Published 2000; predates modern GLMM methods
- Focus on continuous responses (LMMs); limited GLMM coverage
- S-PLUS syntax is outdated

**Recommendation:**
- **Historical reference only**
- Chapters on variance structures (AR1, compound symmetry) remain relevant
- **Superseded by** Bates et al. (2015) lme4 paper and Zuur et al. (2009) for GLMMs

**Assessment**: **Historically important**, but now superseded for GLMMs.

---

### 5.4 McCulloch, Searle & Neuhaus, "Generalized, Linear, and Mixed Models" (2nd Ed., 2008)

**Coverage:**
- Comprehensive theoretical treatment of GLMs and GLMMs
- Emphasis on likelihood theory and asymptotics
- Derivations of Henderson's equations, BLUP, ML/REML

**Strengths:**
- Mathematically rigorous (graduate-level statistics text)
- Thorough coverage of estimation theory
- Unifies GLMs, LMMs, and GLMMs

**Limitations:**
- Minimal computational implementation (no R code)
- Dense mathematical notation (steep learning curve)
- Less emphasis on practical diagnostics and interpretation

**Recommendation:**
- **Reference for theoretical depth**: Chapters 6-7 (LMMs), Chapters 9-10 (GLMMs)
- Use for **deriving** key results (BLUP, Henderson's equations)
- **Not** a primary text for applied learners

**Assessment**: **Excellent for theory**, but too abstract for a primarily applied learning plan. Use selectively for derivations.

---

### 5.5 Additional Recommended Texts

**1. Zuur et al., "A Beginner's Guide to GLM and GLMM with R" (2013)**
- **Already in the project**: Excellent applied text
- Strong on model specification and diagnostics
- Ecology-focused examples translate reasonably well to insurance

**2. Bolker et al., "Generalized Linear Mixed Models: A Practical Guide for Ecology and Evolution" (2009)**
- Classic paper (not a book); free online
- Practical guidance on model specification and troubleshooting
- https://doi.org/10.1016/j.tree.2009.10.016

**3. Gelman & Hill, "Data Analysis Using Regression and Multilevel/Hierarchical Models" (2007)**
- Bayesian perspective throughout
- Excellent on interpretation and visualization
- Less technical than McCulloch et al., more accessible

**4. McElreath, "Statistical Rethinking" (2nd Ed., 2020)**
- Modern Bayesian approach
- Very clear conceptual explanations
- Includes brms code examples
- Recommended for Module 7 (Bayesian extensions)

**5. Wakefield, "Bayesian and Frequentist Regression Methods" (2013)**
- Parallel treatment of Bayesian and frequentist methods
- GLMs and GLMMs with spatial extensions
- Springer Texts in Statistics (graduate level)

---

## 6. The Credibility-Mixed-Models Equivalence: Synthesis

**This is the conceptual throughline of the entire project. It deserves special attention.**

### 6.1 The Formal Mathematical Result

**Theorem (Robinson, 1991)**: Under a Gaussian linear mixed model,
$$
y_{ij} = \mathbf{x}_{ij}^\top \boldsymbol{\beta} + b_i + \epsilon_{ij}, \quad b_i \sim \mathcal{N}(0, \tau^2), \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)
$$

the Best Linear Unbiased Predictor (BLUP) of the random effect is:
$$
\hat{b}_i = \frac{n_i \tau^2}{n_i \tau^2 + \sigma^2} \left( \bar{y}_i - \mathbf{\bar{x}}_i^\top \hat{\boldsymbol{\beta}} \right)
$$

This is **identical** to the Bühlmann credibility formula:
$$
\hat{\mu}_i = Z_i \bar{y}_i + (1 - Z_i) \mu_0
$$

where $Z_i = \frac{n_i \tau^2}{n_i \tau^2 + \sigma^2}$ and $\mu_0 = \mathbf{\bar{x}}_i^\top \hat{\boldsymbol{\beta}}$ is the population mean adjusted for covariates.

**Interpretation:**
- **BLUP** is derived from likelihood (frequentist framework)
- **Bühlmann credibility** is derived from minimizing expected squared error (classical credibility framework)
- **Bayesian posterior mean** is derived from Bayes' theorem with a Gaussian prior
- **All three are identical** under the Gaussian LMM

---

### 6.2 Where the Equivalence Breaks Down

**1. Non-Gaussian responses (GLMMs):**
- BLUP is no longer analytically tractable
- Credibility formula does not directly apply (nonlinear link function)
- **Conceptual equivalence remains**: Predictions are still credibility-weighted, but on the linear predictor scale

**2. Nonlinear link functions:**
- Marginal vs. conditional predictions differ (see Section 3.3)
- Credibility weighting applies to $\mathbf{x}^\top \boldsymbol{\beta} + b_i$, not directly to $\mathbb{E}[y]$

**3. Complex hierarchies:**
- Crossed random effects, random slopes, temporal structures
- Classical credibility does not extend naturally to these settings
- GLMMs provide a unified framework

---

### 6.3 Key Papers Establishing the Equivalence

**1. Robinson (1991). "That BLUP is a Good Thing: The Estimation of Random Effects."**
- Classic exposition of BLUP
- Shows BLUP = empirical Bayes = credibility
- *Statistical Science*, 6(1), 15-51.

**2. Jewell (1975). "The Credibility of Automobile Accident Experience as a Function of Age of Driver."**
- Early actuarial application of credibility
- Foreshadows mixed model connection
- *ASTIN Bulletin*, 8(2), 97-105.

**3. Frees, Young & Luo (1999). "A Longitudinal Data Analysis Interpretation of Credibility Models."**
- Explicit connection between credibility and mixed models for actuarial data
- *Insurance: Mathematics and Economics*, 24(3), 229-247.

**4. Bühlmann & Gisler (2005). *A Course in Credibility Theory and its Applications*.**
- Chapter 8: "Hierarchical Credibility and Linear Models"
- Formal derivation of BLUP-credibility equivalence

**5. Ohlsson & Johansson (2010). *Non-Life Insurance Pricing with Generalized Linear Models*.**
- Chapter 8: "Random Effects and Credibility"
- Actuarial perspective on GLMMs as credibility

---

### 6.4 How to Present This Connection Rigorously and Illuminatingly

**Pedagogical sequence:**

**Module 2 (Classical Credibility):**
1. Derive Bühlmann credibility from first principles (minimize expected squared error)
2. Show Bayesian derivation (posterior mean under Gaussian prior)
3. State (without proof) that BLUP gives the same answer
4. **Foreshadow**: "We'll see in Module 3 that mixed models provide a likelihood-based way to estimate credibility parameters."

**Module 3 (Transition to GLMMs):**
1. Introduce the LMM as a likelihood framework for hierarchical data
2. Derive BLUP for a simple random intercept model
3. **Show algebraically** that BLUP = Bühlmann credibility (Robinson's result)
4. Fit a simple LMM in R (lme4) and verify numerically that BLUP matches hand-calculated credibility
5. **Key takeaway**: "Mixed models are a generalization of credibility to complex data structures."

**Module 4 (GLMM Theory):**
1. Extend to GLMMs (nonlinear link functions)
2. Explain why exact credibility formula no longer applies
3. **Conceptual equivalence**: Partial pooling on the linear predictor scale
4. Show conditional vs. marginal predictions (Section 3.3)

**Module 7 (Bayesian Extensions):**
1. Derive Bayesian posterior mean for Gaussian LMM (conjugate prior)
2. **Show algebraically** that posterior mean = Bühlmann credibility = BLUP
3. Extend to GLMMs via MCMC (no analytical formula, but same conceptual structure)
4. **Key takeaway**: "Bayesian hierarchical models are the most general framework for credibility."

**Worked example (all three methods):**
- Dataset: 10 policyholders, varying claim counts and exposures
- Method 1: Hand-calculate Bühlmann credibility (estimate $\tau^2, \sigma^2$ via method of moments)
- Method 2: Fit LMM with lme4; extract BLUP
- Method 3: Fit Bayesian model with brms; extract posterior mean
- **Compare**: All three give the same (or nearly the same) predictions

---

## 7. Recommendations for the Learning Plan

Based on this research, here are specific recommendations to strengthen the existing 9-module plan:

### 7.1 Module Sequencing

**Current plan**: Bayesian methods introduced late (Module 7)

**Recommendation**: Introduce Bayesian methods in **parallel** starting in Module 3:
- Module 2: Derive Bühlmann credibility as Bayesian posterior mean
- Module 3: Fit the same model with glmmTMB (ML/REML) and brms (Bayesian); compare
- Modules 4-6: Use glmmTMB as the workhorse; use brms for validation and when credible intervals are needed
- Module 7: Deepen Bayesian theory (priors, sensitivity, model comparison)

**Rationale**: Presenting Bayesian as a parallel framework (not an extension) reinforces the credibility-mixed-models equivalence and gives learners flexibility.

---

### 7.2 Conditional vs. Marginal Interpretation

**Current plan**: Not explicitly addressed

**Recommendation**: Add a dedicated section in Module 4:
- Derive the marginal mean correction factor $\exp(\tau^2 / 2)$ for Poisson log link
- Show the magnitude for typical actuarial variance components
- Explain when to use marginal (population-level) vs. conditional (individual-level) predictions
- Provide R code for both types of predictions

**Rationale**: This is the most common source of confusion in applied GLMM work and has direct regulatory implications.

---

### 7.3 Estimation Method Comparison

**Current plan**: Focus on glmmTMB (Laplace)

**Recommendation**: Add a Module 4 subsection comparing estimation methods:
- Laplace (glmmTMB): Fast, good for large datasets
- AGQ (lme4): More accurate for small cluster sizes; use for validation
- Bayesian (brms): Full uncertainty quantification; slower but increasingly practical

Include a decision tree: When to use each method.

---

### 7.4 Mathematical Foundations

**Current plan**: Theory is integrated throughout modules

**Recommendation**: Add a "Mathematical Prerequisites" appendix (or brief Module 0):
- Matrix algebra review (just enough for Henderson's equations)
- Quadratic forms and positive definite matrices
- Interpretation over derivation (emphasize what the math *means*)

**Rationale**: Ensures learners have the minimal linear algebra needed without getting bogged down.

---

### 7.5 Modern Extensions (Module 9)

**Current plan**: Covers GLMMNet and SIMEX

**Recommendation**: Expand to include:
1. **INLA** (fast Bayesian alternative to MCMC)
2. **Distributional GLMMs** (modeling dispersion)
3. **Regularized mixed models** (GLMMNet)
4. **Measurement error** (SIMEX)
5. **Gradient boosting + random effects** (frontier research)

Treat each as a 1-2 page conceptual overview with one worked example. Goal: Awareness of frontier methods, not mastery.

---

### 7.6 Reference Strategy

**Recommended primary texts (in order of importance):**

1. **Zuur et al. (2013)** — Applied GLMM guide (already in plan)
2. **CAS Monograph 14** — Actuarial perspective (already in plan)
3. **Bühlmann & Gisler (2005)** — Credibility theory (Chapters 3, 8)
4. **McElreath (2020), "Statistical Rethinking"** — Bayesian perspective (for Module 7)
5. **Gelman & Hill (2007)** — Interpretation and visualization

**Reference texts for theoretical depth:**
- **Stroup (2013)** — GLMM theory (targeted chapters)
- **McCulloch et al. (2008)** — Mathematical derivations (for Henderson's equations, BLUP)

**Online resources:**
- lme4 Theory vignette (Bates et al.)
- brms vignettes (Bürkner)
- glmmTMB vignette (Bolker)
- Robinson (1991) "That BLUP is a Good Thing" (free via JSTOR)

---

## 8. Summary of Key Theoretical Results

For quick reference, here are the core theoretical results the learner must understand:

### 8.1 BLUP = Bühlmann Credibility (LMM)

$$
\hat{b}_i = \frac{n_i \tau^2}{n_i \tau^2 + \sigma^2} \left( \bar{y}_i - \mathbf{\bar{x}}_i^\top \hat{\boldsymbol{\beta}} \right)
$$

**Interpretation**: Credibility-weighted deviation from population mean.

---

### 8.2 Marginal Mean Correction (Poisson GLMM)

$$
\mathbb{E}[y_{ij}] = \exp\left( \mathbf{x}_{ij}^\top \boldsymbol{\beta} + \frac{\tau^2}{2} \right)
$$

**Interpretation**: Marginal prediction inflated by $\exp(\tau^2/2)$ relative to conditional prediction at $b_i = 0$.

---

### 8.3 Henderson's Mixed Model Equations (LMM)

$$
\begin{bmatrix}
\mathbf{X}^\top \mathbf{X} & \mathbf{X}^\top \mathbf{Z} \\
\mathbf{Z}^\top \mathbf{X} & \mathbf{Z}^\top \mathbf{Z} + \mathbf{G}^{-1} \sigma^2
\end{bmatrix}
\begin{bmatrix}
\hat{\boldsymbol{\beta}} \\
\hat{\mathbf{b}}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{X}^\top \mathbf{y} \\
\mathbf{Z}^\top \mathbf{y}
\end{bmatrix}
$$

**Interpretation**: BLUP is a penalized least squares estimator.

---

### 8.4 Bayesian Posterior Mean = Bühlmann Credibility

$$
\mathbb{E}[\mu_i \mid \mathbf{y}_i] = Z_i \bar{y}_i + (1 - Z_i) \mu_0
$$

where $Z_i = \frac{n_i \tau^2}{n_i \tau^2 + \sigma^2}$ under Gaussian-Gaussian conjugacy.

---

### 8.5 Laplace Approximation

Approximate $\int \exp[h(\mathbf{b})] d\mathbf{b}$ by second-order Taylor expansion around $\hat{\mathbf{b}} = \arg\max h(\mathbf{b})$.

**Used by**: glmmTMB, lme4 (with `nAGQ=1`)

---

## 9. Gaps in the Existing Literature and Recommendations

### 9.1 Gaps

1. **Actuarial-specific GLMM texts are rare**: CAS Monograph 14 is excellent but brief. Most GLMM texts are ecology or biostatistics-focused.

2. **Regulatory perspective on Bayesian methods**: Little published guidance on how to present Bayesian GLMMs to regulators (ASOP-25 compliance).

3. **Computational benchmarks**: Few comparisons of Laplace vs. AGQ vs. Bayesian on realistic actuarial datasets (millions of rows, thousands of groups).

4. **Measurement error in insurance data**: Under-researched topic. Most SIMEX papers focus on biostatistics.

### 9.2 Recommendations for Filling Gaps

1. **Create original content**: The learning plan should include worked examples with real actuarial datasets (CASdatasets, synthetic data) that are not in existing texts.

2. **Comparative benchmarks**: Include a notebook (Module 4 or 6) that compares Laplace (glmmTMB), AGQ (lme4), and Bayesian (brms) on the same dataset, documenting speed and accuracy trade-offs.

3. **Regulatory vignette**: Include a 2-3 page writeup (Module 8) on how to document and justify a GLMM for a rate filing (ASOP-25 compliance, sensitivity analyses, model comparison).

4. **Measurement error case study**: Develop a synthetic example where measurement error is known (e.g., age rounded to nearest 5 years) and show the impact on estimates (Module 9).

---

## 10. Conclusion

This research report provides a comprehensive foundation for understanding the statistical theory and modern methods underlying GLMMs in actuarial pricing. The key insights are:

1. **Estimation methods** have converged on a small set of robust approaches: Laplace (glmmTMB), AGQ (lme4), Bayesian HMC (brms/Stan). Each has clear use cases.

2. **The credibility-GLMM equivalence** is the conceptual throughline. Present it early, rigorously, and repeatedly. It breaks down in GLMMs but the partial pooling intuition remains.

3. **Conditional vs. marginal interpretation** is the most consequential theoretical subtlety. Dedicate a full section to it.

4. **Bayesian methods** should be introduced in parallel (not as a late extension) because hierarchical Bayesian models are the natural formalization of credibility theory.

5. **Modern extensions** (INLA, regularization, distributional models, measurement error) are active research areas with direct actuarial relevance. Introduce them conceptually in Module 9 to prepare learners for the frontier.

The existing 9-module plan is well-structured. The main recommendations are:
- Introduce Bayesian methods earlier (Module 3)
- Add explicit treatment of conditional vs. marginal interpretation (Module 4)
- Expand Module 9 to include INLA and distributional GLMMs
- Include computational benchmarks and regulatory guidance

With these enhancements, the learning plan will provide both theoretical depth and practical mastery of GLMMs for actuarial pricing.

---

## References

**Estimation Theory:**
- Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting Linear Mixed-Effects Models Using lme4. *Journal of Statistical Software*, 67(1), 1-48. https://doi.org/10.18637/jss.v067.i01
- Breslow, N. E., & Clayton, D. G. (1993). Approximate Inference in Generalized Linear Mixed Models. *Journal of the American Statistical Association*, 88(421), 9-25.
- Kristensen, K., Nielsen, A., Berg, C. W., Skaug, H., & Bell, B. M. (2016). TMB: Automatic Differentiation and Laplace Approximation. *Journal of Statistical Software*, 70(5), 1-21. https://doi.org/10.18637/jss.v070.i05

**Credibility-GLMM Equivalence:**
- Robinson, G. K. (1991). That BLUP is a Good Thing: The Estimation of Random Effects. *Statistical Science*, 6(1), 15-51.
- Bühlmann, H., & Gisler, A. (2005). *A Course in Credibility Theory and its Applications*. Springer.
- Frees, E. W., Young, V. R., & Luo, Y. (1999). A Longitudinal Data Analysis Interpretation of Credibility Models. *Insurance: Mathematics and Economics*, 24(3), 229-247.

**Conditional vs. Marginal:**
- Skrondal, A., & Rabe-Hesketh, S. (2009). Prediction in Multilevel Generalized Linear Models. *Journal of the Royal Statistical Society: Series A*, 172(3), 659-687.
- Muff, S., Held, L., & Keller, L. F. (2016). Marginal or Conditional Regression Models for Correlated Non-normal Data? *Methods in Ecology and Evolution*, 7(12), 1514-1524.

**Bayesian Methods:**
- Bürkner, P. C. (2017). brms: An R Package for Bayesian Multilevel Models Using Stan. *Journal of Statistical Software*, 80(1), 1-28. https://doi.org/10.18637/jss.v080.i01
- Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- McElreath, R. (2020). *Statistical Rethinking: A Bayesian Course with Examples in R and Stan* (2nd ed.). CRC Press.

**INLA:**
- Rue, H., Martino, S., & Chopin, N. (2009). Approximate Bayesian Inference for Latent Gaussian Models by Using Integrated Nested Laplace Approximations. *Journal of the Royal Statistical Society: Series B*, 71(2), 319-392.
- Blangiardo, M., & Cameletti, M. (2015). *Spatial and Spatio-temporal Bayesian Models with R-INLA*. Wiley.

**Modern Extensions:**
- Yi, N., & Zeng, D. (2023). GLMMNet: Penalized Inference for Generalized Linear Mixed Models. arXiv:2301.12710. https://arxiv.org/abs/2301.12710
- Cook, J. R., & Stefanski, L. A. (1994). Simulation-Extrapolation Estimation in Parametric Measurement Error Models. *Journal of the American Statistical Association*, 89(428), 1314-1328.

**Actuarial Applications:**
- Antonio, K., & Beirlant, J. (2007). Actuarial Statistics with Generalized Linear Mixed Models. *Insurance: Mathematics and Economics*, 40(1), 58-76.
- Ohlsson, E., & Johansson, B. (2010). *Non-Life Insurance Pricing with Generalized Linear Models*. Springer.
- CAS Monograph No. 14: *Practical Mixed Models for Actuaries* (2023). Casualty Actuarial Society.
