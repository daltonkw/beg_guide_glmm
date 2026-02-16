# Project Prompt: Generalized Linear Mixed Models for Actuarial Pricing

## Executive Summary

This self-directed project develops mastery in Generalized Linear Mixed Models (GLMMs) for non-life insurance pricing. You will progress from classical GLM and credibility foundations through advanced GLMM applications, with emphasis on practical R implementation. The project bridges actuarial theory, statistical rigor, and regulatory considerations, enabling you to apply these models confidently to real-world casualty pricing problems.

---

## Project Vision & Outcomes

By completion, you will:

1. **Master foundational theory:** Deeply understand GLMs, credibility theory, and their interplay in insurance rating—with ability to explain *why* these approaches work and *when* to use each.

2. **Implement production-ready GLMMs:** Write clean, reproducible R code using `glmmTMB` (and supplementary tools) that properly specifies random effects structures, variance components, and distributional assumptions for actuarial contexts.

3. **Bridge statistical methods to actuarial practice:** Understand how credibility emerges naturally from partial pooling in mixed models, how to validate assumptions, and how modeling choices align with ASOP-25 and professional standards.

4. **Handle complexity:** Implement zero-inflated models, random slopes, temporal structures, and Bayesian extensions to address real-world data challenges in casualty pricing.

5. **Evaluate models rigorously:** Perform diagnostics, comparisons to simpler alternatives, and validation exercises that demonstrate your ability to recommend appropriate models to stakeholders.

---

## Project Structure: Nine Integrated Modules

### Module 1: Foundations of GLMs & Credibility in Insurance Pricing

**Duration:** Foundational; revisit throughout project  
**Core Objective:** Establish the theoretical and practical bedrock for all subsequent modules.

#### Deliverables

**Conceptual Document (3–4 pages):**
- Define the fundamental distributions used in casualty insurance (Poisson, negative binomial, gamma, lognormal) and explain when each is appropriate for frequency, severity, and pure premium modeling.
- Explain the exponential family framework and canonical links (log for frequency, identity/log for severity).
- Motivate GLMs: Why are they preferable to classical rating techniques? How do they improve upon multiplicative models?
- Introduce the concept of insufficient data and explain why credibility theory is essential.

**R Workflow Example – Poisson Frequency GLM:**

```r
# Load and explore casualty data (synthetic or real)
# Fit a Poisson GLM for claim frequency
gl_freq <- glm(claims ~ age + region + vehicle_type, 
               family = poisson(), data = pricing_data)
summary(gl_freq)

# Extract and interpret coefficients, deviance, AIC
# Compare to baseline/null models
# Visualize exposure and fitted rates by rating variables
```

**Credibility Theory Primer (2–3 pages):**
- Bühlmann credibility formula: $Z = \frac{n}{n + k}$ where $k = \frac{\tau^2}{\sigma^2}$.
- Manual calculation of credibility-weighted estimates.
- Illustration: Show how a small insured's claim history is credibility-weighted toward the collective experience.
- Contrast credibility weighting with simple averaging.

**Comparative Analysis:**
- Fit a simple rate (collective average) vs. GLM-only model vs. (preview) credibility-adjusted GLM.
- Create a comparison table showing predictions for example risk profiles.
- Highlight how GLMs outperform rate-averaging, especially with sparse data.

#### Key Concepts to Embed

- Maximum likelihood estimation for GLMs; deviance and goodness-of-fit metrics.
- How rating factors (age, region, vehicle_type) affect claim probability and pure premiums.
- Distinction between frequency and severity modeling.

---

### Module 2: Classical Credibility Theory & Manual Implementation

**Duration:** Intermediate  
**Core Objective:** Develop intuition for credibility; implement it by hand before embedding in models.

#### Deliverables

**Credibility Theory Deep-Dive (3–4 pages):**
- Formal derivation of Bühlmann credibility (Bayesian framework).
- Explain the variance of the process mean ($\tau^2$) and the expected value of the variance ($\sigma^2$).
- Parameter estimation: How to estimate $\tau^2$ and $\sigma^2$ from data?
- Limited fluctuation credibility vs. Bühlmann credibility.

**R Implementation – Manual Credibility Calculations:**

```r
# Example: Rate a portfolio of policyholders with limited claim history
data <- data.frame(
  policyholder_id = 1:10,
  claims = c(0, 1, 2, 1, 0, 3, 1, 2, 0, 1),
  exposure = c(1, 0.5, 1, 1, 1, 1, 0.8, 1, 1, 1)
)

# Estimate collective experience
collective_freq <- sum(data$claims) / sum(data$exposure)

# Calculate credibility parameters
# (Estimate tau_sq and sigma_sq from observed data)

# Apply Bühlmann credibility
Z <- credibility_factor_function(data$exposure, tau_sq, sigma_sq)
data$credible_rate <- Z * (data$claims / data$exposure) + 
                      (1 - Z) * collective_freq
```

**Case Study Walkthrough:**
- Take a small hypothetical dataset of policyholders with varying claim counts.
- Compute credibility-weighted premiums manually and explain each step.
- Compare to uncredibility-weighted rates and raw empirical rates.
- Document insights on how credibility stabilizes sparse-data estimates.

#### Key Concepts to Embed

- The credibility coefficient $Z$ as a balance between individual and collective experience.
- Why credibility becomes unnecessary as data volume increases ($Z \to 1$).
- Connection to Bayesian posterior estimation and shrinkage.

---

### Module 3: Transition to GLMMs – Why & How

**Duration:** Intermediate  
**Core Objective:** Motivate random effects modeling as a modern, flexible extension of classical credibility.

#### Deliverables

**Conceptual Bridge Document (2–3 pages):**
- Explain why classical credibility has limitations (e.g., assumes exchangeable risks, requires known variance components).
- Introduce random effects as a generalization: Instead of estimating a single credibility coefficient, use likelihood-based methods to estimate the variance structure of random intercepts.
- Show how a random intercept model naturally produces partial pooling—a generalization of credibility.
- Highlight advantages: Flexible within-subject structure, simultaneous estimation of fixed and random effects, ability to handle unbalanced data and multiple levels of hierarchy.

**Conceptual Diagram:**
- Visualize: Fixed-effects-only GLM → Random intercept GLMM → Random slopes GLMM.
- Show how predictions shrink toward the population mean as a function of group-level variance.

**R Comparison: GLM vs. GLMM**

```r
library(glmmTMB)

# Fixed-effects-only GLM (like classical credibility baseline)
glm_fixed <- glm(claims ~ age + region + policyholder_id, 
                 family = poisson(), data = df)

# Random intercept GLMM (partial pooling of policyholder effects)
glmm_ri <- glmmTMB(claims ~ age + region + (1 | policyholder_id), 
                   family = poisson(), data = df)

# Compare predictions for a typical policyholder
# Show how GLMM predictions are "shrunk" relative to fixed-effects GLM
```

**Visualization Exercise:**
- Plot fitted rates by policyholder for both models.
- Highlight the shrinkage effect: smaller insureds' predictions move toward collective estimates.
- Annotate with data volume per policyholder to show relationship between data and shrinkage.

#### Key Concepts to Embed

- Partial pooling and its relationship to credibility.
- Random effects variance as an estimate of group-level heterogeneity.
- Likelihood-based estimation vs. moment-based credibility.

---

### Module 4: GLMM Theory, Estimation & Assumptions

**Duration:** Advanced; theoretical foundation  
**Core Objective:** Understand the mechanics of GLMM inference and know the assumptions that underpin model validity.

#### Deliverables

**GLMM Theory Document (4–5 pages):**

- **Model formulation:** Define the conditional and marginal models.
  - Conditional on random effects: $\mathbf{y}_i | \mathbf{b}_i \sim f(\mu_i, \phi)$ where $\mu_i = g^{-1}(\mathbf{X}_i \boldsymbol{\beta} + \mathbf{Z}_i \mathbf{b}_i)$.
  - Marginal distribution: Integrate out random effects (often intractable; motivates approximations).

- **Estimation methods:**
  - **REML (Restricted Maximum Likelihood):** Accounts for loss of degrees of freedom in estimating fixed effects; recommended for random effects inference.
  - **ML (Maximum Likelihood):** Used for model comparison (likelihood ratio tests).
  - **Laplace and AGQ (Adaptive Gaussian Quadrature):** Approximations for the marginal likelihood when exact integration is infeasible.

- **Assumptions:**
  - Random intercepts/slopes are normally distributed.
  - Predictors are measured without error.
  - No unmeasured confounding.
  - Conditional independence given random effects.
  - Correctly specified variance structure.

- **Consequences of violations:** How do they affect inference? When can we be flexible?

**Variance Component Interpretation (2–3 pages):**
- Explain random intercept variance $\tau_0^2$: Heterogeneity of baseline risk across groups.
- Relate to credibility: Groups with higher variance are shrunk less toward the collective estimate.
- Discuss ICC (Intra-Class Correlation): Proportion of total variance due to grouping; a measure of within-group clustering.
- Practical guidance: When is within-group variance meaningful in pricing?

**R Implementation – Model Fitting & Diagnostics:**

```r
library(glmmTMB)
library(lme4)

# Fit GLMM with REML
m1 <- glmmTMB(claims ~ age + region + (1 | policyholder_id), 
              family = poisson(), data = df, REML = TRUE)

# Extract random effects and variance components
ranef(m1)
VarCorr(m1)

# Compute ICC
tau_sq <- attr(VarCorr(m1)$cond$policyholder_id, "stddev")^2
sig_sq <- sigma(m1)^2
icc <- tau_sq / (tau_sq + sig_sq)

# Compare REML vs. ML for LRT
m1_ml <- update(m1, REML = FALSE)
m0_ml <- glmmTMB(claims ~ age + region + (1 | policyholder_id), 
                 family = poisson(), data = df, REML = FALSE)
anova(m0_ml, m1_ml)  # LRT for variance component
```

**Assumptions Check Document:**
- List key assumptions and provide R code to assess each (covered more formally in Module 6).
- Explain trade-offs: When is it acceptable to relax assumptions, and what are the consequences?

#### Key Concepts to Embed

- Marginal vs. conditional models and why marginal predictions can differ from conditional ones.
- The role of Laplace approximation in `glmmTMB`.
- REML as the principled choice for random effects inference.

---

### Module 5: Advanced GLMMs – Real-World Actuarial Complexity

**Duration:** Advanced; practical application  
**Core Objective:** Extend GLMMs to handle non-standard data features common in casualty insurance.

#### 5.1: Zero-Inflation & Overdispersion

**Motivation:**
- Many casualty datasets have excess zeros (many policyholders with zero claims in a period).
- Poisson/negative binomial models may not capture this excess; zero-inflation components address it.
- Overdispersion: Variance exceeds the mean; address via negative binomial or zero-inflation.

**Deliverables:**

**Conceptual Document (2 pages):**
- Zero-Inflated Poisson (ZIP) and Zero-Inflated Negative Binomial (ZINB) models.
- Interpretation: Component 1 (inflation process) determines zero probability; Component 2 (count process) models claim counts conditional on claims occurring.
- When to use: Claims data with structural zeros vs. sampling zeros.
- Connection to credibility: How does zero-inflation interact with partial pooling?

**R Implementation:**

```r
library(glmmTMB)

# Standard Poisson GLMM
m_pois <- glmmTMB(claims ~ age + region + (1 | policyholder_id), 
                  family = poisson(), data = df)

# Zero-inflated negative binomial GLMM
m_zinb <- glmmTMB(claims ~ age + region + (1 | policyholder_id),
                  ziformula = ~1,  # Zero-inflation intercept only
                  family = nbinom2(), data = df)

# Extract zero-inflation parameter
summary(m_zinb)

# Compare AIC
AIC(m_pois, m_zinb)

# Predictions under each model
pred_pois <- predict(m_pois, newdata = new_data, type = "link")
pred_zinb <- predict(m_zinb, newdata = new_data, type = "link")
```

**Case Study:**
- Simulate or use data with structural zeros.
- Fit Poisson, NB, ZIP, ZINB models.
- Compare via AIC, residual diagnostics, and prediction accuracy.
- Document when zero-inflation is warranted.

#### 5.2: Random Slopes

**Motivation:**
- Not all rating factors affect all groups equally. E.g., age may have different effects across regions.
- Random slopes allow the effect of a predictor to vary by group.
- Improves model fit and allows richer, group-specific pricing rules.

**Deliverables:**

**Conceptual Document (2 pages):**
- Model formulation: $\mu_{ij} = g^{-1}(\alpha + \beta_{0j} + (\gamma + \beta_{1j}) \times x_{ij})$ where $\beta_{0j}, \beta_{1j}$ vary by group.
- Interpretation: Both intercepts and slopes are random; they are typically correlated.
- Practical use: Different age-rating curves for different regions or policy types.
- Trade-off: More flexibility but more parameters to estimate; requires more data per group.

**R Implementation:**

```r
library(glmmTMB)

# Random intercept only
m_ri <- glmmTMB(claims ~ age + region + (1 | state), 
                family = poisson(), data = df)

# Random intercept and random slope for age
m_rs <- glmmTMB(claims ~ age + region + (age | state), 
                family = poisson(), data = df)

# Random slope for age, no random intercept
m_rs_no_int <- glmmTMB(claims ~ age + region + (0 + age | state), 
                       family = poisson(), data = df)

# Extract and visualize random slopes
re_slopes <- ranef(m_rs)$cond$state
plot(re_slopes$age, main = "Random slopes for age by state")
```

**Visualization:**
- Plot fitted age-rating curves by state under both RI and RS models.
- Show how random slopes capture regional heterogeneity in age effects.

#### 5.3: Temporal & Correlation Structures

**Motivation:**
- Insurance data often has temporal structure: Claims for the same policyholder over multiple years are correlated.
- Ignoring temporal dependence underestimates standard errors and biases variance components.
- Temporal structures (e.g., AR1) model how correlation decays with time.

**Deliverables:**

**Conceptual Document (2 pages):**
- Common structures: Compound symmetry (exchangeable), AR(1) (autoregressive), unstructured.
- When to use: Longitudinal data with multiple observations per subject.
- Implementation challenges: `glmmTMB` supports selected structures; more complex structures may require custom code or alternative software.

**R Implementation (Conceptual):**

```r
library(glmmTMB)

# Longitudinal data: Multiple years per policyholder
# Simple random intercept (no temporal structure)
m_no_time <- glmmTMB(claims ~ age + year + (1 | policyholder_id), 
                     family = poisson(), data = df_long)

# AR(1) structure via explicit covariance (simplified; glmmTMB has limited native temporal support)
# For full temporal covariance structures, consider lme4 with additional packages 
# or software like PROC GENMOD (SAS)

# Workaround in glmmTMB: Add (1 | policyholder_id:year) for compound symmetry
m_compound <- glmmTMB(claims ~ age + (1 | policyholder_id) + (1 | policyholder_id:year), 
                      family = poisson(), data = df_long)
```

**Documentation:**
- Acknowledge `glmmTMB`'s limitations with complex temporal structures.
- Suggest alternative tools (e.g., `nlme`, `mgcv`, or specialized Bayesian software) for advanced temporal modeling.
- Provide guidance on when to use simpler structures vs. when to escalate to other software.

#### Overall Module 5 Deliverables

- Comprehensive guide document (8–10 pages total) synthesizing zero-inflation, random slopes, and temporal structures.
- R workflow demonstrating fitting, comparison, and interpretation of each.
- Case study comparing multiple competing models and justifying the final choice.
- Practical guidance on when each extension is warranted.

#### Key Concepts to Embed

- Model selection via AIC, BIC, and likelihood ratio tests.
- Parsimony vs. fit: Overfitting risk with too many random effects.
- Interpretation of complex models for actuarial audiences.

---

### Module 6: Model Diagnostics, Validation & Comparison

**Duration:** Intermediate–Advanced; practical  
**Core Objective:** Learn rigorous practices for assessing model adequacy, comparing alternatives, and validating pricing models before deployment.

#### Deliverables

**Comprehensive Diagnostics Guide (4–5 pages):**

- **Residual analysis:**
  - Pearson and deviance residuals; their interpretation.
  - Quantile–Quantile plots for residuals.
  - Residual vs. fitted plots; assessment of heteroscedasticity and systematic patterns.

- **Overdispersion testing:**
  - Pearson chi-square goodness-of-fit test.
  - Dispersion parameter estimation and interpretation.
  - Residual-based overdispersion measures.

- **Variance component assessment:**
  - Confidence intervals for random intercepts via profile likelihood or bootstrap.
  - Likelihood ratio tests for variance components (Model A: $\tau^2 = 0$ vs. Model B: $\tau^2 > 0$).
  - ICC interpretation in context of pricing (how much within-group clustering is meaningful?).

- **Fixed effects diagnostics:**
  - Collinearity checks (VIF, correlation matrix).
  - Influential points and outliers (leverage, cook's distance adapted for mixed models).

- **Prediction validation:**
  - Hold-out test set predictions: Compare fitted vs. held-out claim counts.
  - Calibration plots: Observed vs. predicted claim frequencies by decile.
  - Lift analysis: How well does the model rank risks?

**R Implementation – Diagnostic Workflow:**

```r
library(glmmTMB)
library(ggplot2)
library(DHARMa)  # Residual diagnostics for GLMMs

# Fit GLMM
m <- glmmTMB(claims ~ age + region + (1 | policyholder_id), 
             family = poisson(), data = df)

# Extract residuals
resid_pearson <- residuals(m, type = "pearson")
resid_deviance <- residuals(m, type = "deviance")

# Diagnostic plots
par(mfrow = c(2, 2))
plot(fitted(m), resid_pearson, main = "Pearson Residuals vs. Fitted")
qqnorm(resid_pearson, main = "Q-Q Plot")
hist(resid_pearson, main = "Histogram of Residuals")

# DHARMa diagnostics (specialized for GLMMs)
sim_resid <- simulateResiduals(m)
plot(sim_resid)
testDispersion(sim_resid)

# Overdispersion ratio
sum(resid_pearson^2) / df.residual(m)

# LRT for variance component
m0 <- glm(claims ~ age + region, family = poisson(), data = df)
anova(m, m0)  # Comparison (note: different structures; use glmmTMB versions for proper LRT)

# Prediction validation
pred_freq <- predict(m, newdata = test_data, type = "link", re.form = NA)  # Population-level
obs_freq <- test_data$claims / test_data$exposure

# Calibration plot
ggplot(data.frame(observed = obs_freq, predicted = exp(pred_freq)), 
       aes(x = predicted, y = observed)) +
  geom_point() + geom_abline(slope = 1, intercept = 0) +
  labs(title = "Calibration: Predicted vs. Observed Frequency")
```

**Model Comparison Framework (2–3 pages):**
- Define competing hypotheses: Simple fixed-effects GLM vs. random intercept GLMM vs. random slopes vs. zero-inflation variants.
- Comparison criteria:
  - **AIC/BIC:** Relative model fit, penalizing complexity.
  - **Likelihood ratio test:** For nested models.
  - **Out-of-sample prediction error:** Practical measure of model performance.
  - **Interpretability & regulatory acceptance:** Can you explain and defend the model to regulators?
- Decision tree: Guide the user through model selection based on data and business context.

**Case Study – Comprehensive Model Comparison:**
- Motivating scenario: You've been given casualty insurance claim frequency data with policyholders clustered by state and agent.
- Fit 5–6 competing models: GLM, GLMM RI, GLMM RS, GLMM with zero-inflation, etc.
- Conduct diagnostics for each.
- Compare via AIC, LRT, and out-of-sample accuracy.
- Document decision process and final recommendation with justification.
- Deliverable: 2-page technical summary suitable for actuarial stakeholders.

#### Key Concepts to Embed

- The difference between inference goals (estimating variance components) and prediction goals (accurate claim forecasts).
- Trade-offs between model complexity and interpretability.
- Regulatory and professional standards for model validation (ASOP-25 alignment).

---

### Module 7: Bayesian Extensions – Full Credibility & Flexibility

**Duration:** Advanced; conceptual & practical  
**Core Objective:** Leverage Bayesian inference to obtain credible intervals, incorporate expert judgment, and adapt to complex data structures.

#### Deliverables

**Bayesian Credibility & GLMM Bridge (3–4 pages):**
- Classical credibility as an implicit Bayesian prior: Bühlmann credibility naturally emerges from a hierarchical model with normal priors.
- Full Bayesian GLMMs: Specify informative or non-informative priors; obtain posterior distributions for all parameters.
- Advantages: Credible intervals, direct probability statements, incorporation of expert priors, flexibility in model structure.
- Computational methods: MCMC (Markov Chain Monte Carlo), Hamiltonian Monte Carlo (HMC).

**Bayesian Modeling Using `brms`:**

```r
library(brms)

# Specify a Bayesian Poisson GLMM
m_bayes <- brm(
  claims ~ age + region + (1 | policyholder_id),
  family = poisson(),
  data = df,
  prior = c(
    prior(normal(0, 1), class = "b"),  # Priors on fixed effects
    prior(exponential(1), class = "sd")  # Prior on random effect SD
  ),
  chains = 4, iter = 2000, warmup = 1000
)

# Extract posterior samples
posterior_samples(m_bayes)

# Posterior predictions with credible intervals
posterior_epred(m_bayes, newdata = new_data)

# Posterior predictive check
pp_check(m_bayes, ndraws = 100)
```

**Bayesian Zero-Inflation Example:**

```r
m_bayes_zinb <- brm(
  claims ~ age + region + (1 | policyholder_id),
  family = zero_inflated_negbinomial(),
  data = df,
  prior = c(...),
  chains = 4, iter = 2000
)
```

**Incorporation of Expert Prior Information (2 pages):**
- Example: Actuaries have historical experience that age effects should be smooth and monotonic.
- Specify regularizing priors to encode this knowledge.
- Comparison: Prior without expert information vs. with; impact on posterior and predictions.
- Discussion: When is informative priors appropriate vs. when should we let data dominate?

**Convergence Diagnostics (1–2 pages):**
- $\hat{R}$ (Rhat): Potential scale reduction factor; assess chain mixing.
- Effective sample size (ESS): Accounting for autocorrelation in posterior samples.
- Trace plots: Visual inspection of chain behavior.
- R workflow:
  ```r
  # Check convergence
  plot(m_bayes)  # Trace plots
  summary(m_bayes)  # Rhat and ESS
  
  # All Rhat values should be < 1.01 for convergence
  ```

**Case Study – Bayesian Hierarchical Pricing Model:**
- Motivating scenario: You want to price policies with limited individual claims history; use agent-level hierarchies and expert priors on agent effects.
- Fit Bayesian hierarchical model with priors on agent random effects and fixed effects.
- Demonstrate posterior distributions, credible intervals for agent-specific parameters.
- Show how posteriors "borrow strength" across agents, improving stability for small agents.
- Compare to frequentist GLMM: Posterior means vs. fixed estimates; width of credible intervals vs. confidence intervals.

#### Key Concepts to Embed

- Bayesian inference as a natural extension of frequentist GLMMs with explicit probability modeling.
- MCMC as a computational tool; understanding when convergence is achieved.
- Advantages for communication: Credible intervals directly answer "What is the probability that the true rate is in this interval?"

---

### Module 8: Industry Standards & Regulatory Alignment

**Duration:** Intermediate; conceptual & applied  
**Core Objective:** Understand professional expectations, standards, and regulatory requirements for actuarial models; ensure your GLMM pricing models meet these standards.

#### Deliverables

**ASOP-25 Alignment Document (3–4 pages):**
- **ASOP-25 (Credibility)** requirements and how GLMMs satisfy them:
  - Appropriateness of credibility method (partial pooling via GLMMs is credible).
  - Data quality and sufficiency checks.
  - Documentation of assumptions and sensitivity analyses.
  - Comparison to alternatives.
- **Risk-Based Capital (RBC) and reserving standards:** How do GLMM assumptions affect capital requirements?
- **SOA/CAS learning objectives:** Coverage of GLM and credibility topics; how this project aligns.
- **Practical mapping:** Link each module objective to actuarial professional standards.

**Documentation & Reproducibility Guide (2–3 pages):**
- Model documentation checklist: Objectives, data sources, variable definitions, transformations, model structure, assumptions, diagnostics, results, limitations.
- Code reproducibility: Version control (Git), R session info, package versions, random seed management.
- Regulatory readiness: How to present models to regulators and auditors; transparency and auditability.

**Professional Communication (1–2 pages):**
- How to explain GLMMs to non-technical audiences: Avoid jargon; use analogies to classical credibility.
- Structuring technical summaries for actuarial peers vs. senior management vs. regulators.
- Visualizations: Effective plots for communicating model insights and uncertainty.

**Governance & Model Lifecycle (1–2 pages):**
- Model governance frameworks: Approval, monitoring, update triggers.
- When to refit GLMMs: Annual? Quarterly? After significant data changes?
- Validation in production: Ongoing monitoring of actual vs. predicted claim experience.

#### Key Concepts to Embed

- Regulatory humility: GLMMs are flexible and powerful but must be applied with rigor and transparency.
- Professional credibility: Use of advanced statistical methods must be coupled with clear communication and robust governance.

---

### Module 9: Advanced Topics & Frontier Extensions

**Duration:** Advanced; applied research  
**Core Objective:** Explore cutting-edge techniques and adaptations of GLMMs to emerging actuarial challenges.

#### 9.1: High-Cardinality Predictors & GLMMNet

**Motivation:**
- Modern datasets often have many categorical predictors (e.g., zip codes, agent IDs, vehicle models).
- Classical GLMM can lead to overfitting (too many fixed effects) or sparse groups with few observations.
- **GLMMNet** combines machine learning (elastic net regularization) with mixed models to handle high-cardinality data.

**Deliverables:**

**Conceptual Document (2–3 pages):**
- Problem: When do classical GLMMs struggle? What is overfitting in this context?
- Solution: GLMMNet uses regularization (L1/L2 penalties) on fixed effects to encourage shrinkage and handle many predictors.
- Relationship to actuarial pricing: Automatic feature selection; reduces model complexity without manual variable selection.
- Reference: ArXiv paper on GLMMNet (https://arxiv.org/pdf/2301.12710).

**R Implementation (Conceptual & Example Code):**

```r
# Note: GLMMNet not yet widely available in standard R; 
# this is a conceptual/research module.
# Alternative: Use standard glmmTMB with careful feature engineering or cross-validation.

# Example: Fit GLMM with many zip codes (high-cardinality predictor)
# Use cross-validation to select regularization parameter

library(glmmTMB)

# Without regularization: all zip codes estimated
m_all_zips <- glmmTMB(claims ~ age + vehicle_type + zip_code + (1 | policyholder_id),
                      family = poisson(), data = df)

# With feature selection: pre-filter zip codes with low exposure
df_filtered <- df %>%
  group_by(zip_code) %>%
  filter(n() >= 30) %>%  # Arbitrary threshold
  ungroup()

m_filtered <- glmmTMB(claims ~ age + vehicle_type + zip_code + (1 | policyholder_id),
                      family = poisson(), data = df_filtered)
```

**Case Study:**
- Motivating scenario: Pricing data with 500+ zip codes.
- Fit standard GLMM (all zips as fixed effects) vs. filtered GLMM vs. zip as random effect.
- Compare model complexity, interpretability, and cross-validation error.
- Document trade-offs and practical guidance.

#### 9.2: Mis-measured Covariates & SIMEX Methods

**Motivation:**
- Real-world data often includes measurement error in covariates (e.g., driver age estimated from license, vehicle values from book estimates).
- Ignoring measurement error biases coefficient estimates and inflates or deflates standard errors.
- **SIMEX (SIMulation-EXtrapolation):** A Bayesian-inspired method to correct for measurement error.

**Deliverables:**

**Conceptual Document (2–3 pages):**
- Sources of measurement error in insurance data: Data entry, third-party errors, aggregation.
- Impact: Bias toward zero for error-prone predictors; underestimated standard errors.
- SIMEX intuition: Simulate additional measurement error, fit models, and extrapolate back to zero error.
- Reference: ArXiv paper on Bayesian + SIMEX methods (https://arxiv.org/pdf/2310.0745).

**R Implementation (Conceptual):**

```r
# Simplified illustration: Assume age is measured with error (e.g., SD = 2 years)

library(simex)

# Fit standard model (ignoring error)
m_naive <- glm(claims ~ age_observed + region, family = poisson(), data = df)

# SIMEX approach: 
# 1. Simulate measurement error for observed predictor
# 2. Fit models with increasing error levels
# 3. Extrapolate back to true (error-free) scenario

# (Full SIMEX implementation for GLMMs is complex; this is a conceptual illustration)

# Rough workflow:
set.seed(123)
lambda_values <- seq(0, 2, by = 0.5)  # Levels of added error
coefficients <- matrix(NA, nrow = length(lambda_values), ncol = 2)

for (i in seq_along(lambda_values)) {
  noise <- rnorm(nrow(df), 0, lambda_values[i])
  df$age_noisy <- df$age_observed + noise
  m_temp <- glm(claims ~ age_noisy + region, family = poisson(), data = df)
  coefficients[i, ] <- coef(m_temp)[c("age_noisy", "region1")]
}

# Extrapolate back to lambda = 0 (error-free)
# (Simplified; typically use polynomial regression and extrapolate)
```

**Case Study:**
- Motivating scenario: You suspect policyholder age is recorded with error.
- Fit naive GLM vs. SIMEX-corrected model.
- Compare coefficients: Are they substantially different?
- Practical guidance: When is measurement error likely to be consequential in pricing?

#### 9.3: Emerging Directions & Open Problems

**Conceptual Exploration (2–3 pages):**

- **High-dimensional data:** How do GLMMs scale to thousands of predictors? What are computational and statistical challenges?

- **Causal inference:** Can GLMMs help identify causal effects of rating factors? Discussion of confounding, colliders, and instrumental variables.

- **Fair & ethical pricing:** Do GLMM rating factors inadvertently embed bias? How to audit for fairness?

- **Unstructured data integration:** Can we incorporate image, text, or behavioral data into GLMMs?

- **Federated learning:** Privacy-preserving model fitting across distributed insurer databases.

**Suggested Further Reading:**
- Academic papers on each topic.
- Industry discussions (CAS forums, actuarial blogs).
- Emerging software and tools.

#### Key Concepts to Embed

- GLMMs are not the end of the story; they are a stepping stone to more sophisticated approaches.
- Actuarial practice will evolve; staying abreast of research ensures professional relevance.

---

## Project Execution & Deliverables

### Overall Deliverables

1. **Module Workbooks (9 documents, 40–50 pages total):**
   - Each module is a self-contained document combining theory, R code, and exercises.
   - Includes examples, walkthroughs, and case studies.

2. **Comprehensive R Code Repository:**
   - Reproducible workflows for each module.
   - Well-commented; suitable for reuse on real data.
   - Example datasets (synthetic or public) to run code against.

3. **Glossary & Reference (2–3 pages):**
   - Definitions of key statistical and actuarial terms.
   - Quick lookup for formulas and R functions.

4. **Capstone Project (Optional but Recommended):**
   - Apply all modules to a realistic casualty pricing scenario.
   - Deliverable: Technical report (8–10 pages) + R code.
   - Covers data exploration → model development → diagnostics → comparison → recommendations.

---

## Learning Outcomes

Upon completion, you will:

- **Theoretically:** Understand GLM, credibility, GLMM foundations, estimation, assumptions, and extensions.
- **Practically:** Write production-grade R code using `glmmTMB`, `brms`, and related packages.
- **Professionally:** Align models with ASOP-25 and actuarial standards; communicate findings to stakeholders.
- **Critically:** Evaluate model fit, compare alternatives, and diagnose problems.
- **Flexibly:** Adapt GLMMs to complex data structures (zero-inflation, random slopes, hierarchies, measurement error).
- **Creatively:** Explore emerging extensions (GLMMNet, SIMEX, fairness, causal inference).


## Success Criteria

You have successfully completed the project when you can:

1. Explain the relationship between classical credibility and modern GLMMs to a peer.
2. Fit, diagnose, and compare competing GLMM specifications in R for a realistic pricing problem.
3. Interpret random effects, variance components, and predictions in actuarial terms.
4. Defend a chosen model based on statistical rigor and actuarial appropriateness.
5. Extend GLMMs to handle non-standard data (zero-inflation, hierarchies, measurement error).
6. Communicate findings clearly to actuarial and regulatory audiences.

---