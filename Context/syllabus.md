# Syllabus

1. Use classical credibility techniques with actuarial rigor.
2. Implement and assess GLMMs (via `glmmTMB`) tailored to casualty risks.
3. Understand partial pooling and variance components in modern pricing.
4. Explore advanced modeling like zero‑inflation, random slopes, and Bayesian inference.
5. Relate modeling choices to regulatory frameworks and actuarial standards (e.g., ASOP‑25).
## Module 1: Foundations of GLMs & Credibility

**Objective:** Reinforce GLM theory and its role in insurance rating.  In cases with insufficient data, combine GLMs with credibility theory to obtain reliable estimates.
- thoroughly describe fundamental concepts and statistical distributions used in pricing of non-life insurance contracts
- thoroughly describe and motivate the use of generalized linear models and credibility theory in non-life insurance pricing
- determine maximum likelihood estimates of the parameters of a GLM
- apply GLM for tariff analysis
- apply credibility theory, combined with GLM, for tariff analysis
**Reading:**
- _“Generalized Linear Models for Insurance Rating”_ (CAS Monograph No. 5, 2nd Ed.)
- https://www.casact.org/sites/default/files/2021-01/05-Goldburd-Khare-Tevet.pdf
**Practice:**
- Fit a Poisson GLM for claim frequency:
```{r}
gl <- glm(claims ~ age + region, family=poisson(), data=df)
summary(gl)
```

## Module 2: Classical Credibility Theory

- **Objective:** Learn Bühlmann credibility for rating.
- **Reading:**
	- _“An Introduction to Credibility”_ (G. Dean, CAS paper)


## Module 3: Transition to GLMMs

- **Objective:** Move from credibility to random effects modeling.
- **Reading:**
    - _“Generalized Linear Mixed Models for Ratemaking”_ (CAS forum paper)
    
```{r}
library(glmmTMB)
m1 <- glmmTMB(claims ~ age + region + (1 | policyholder), family=poisson(), data=df)
summary(m1)
```

## Module 4: GLMM Theory and Estimation

- **Objective:** Understand marginal likelihood, REML, and mixed model assumptions.
- **Textbook:**
    - standard GLMM theory texts
- **Actuarial Context:** How mixed models build credibility via partial pooling.

## Module 5: Advanced GLMMs and Data Features

- **Objective:** Extend GLMMs to use-case-specific features.
- **Topics & Code Examples:**
    - **Zero inflation / overdispersion:**
```{r}
m2 <- glmmTMB(claims ~ age + (1 | policyholder),
              ziformula=~1, family=nbinom2(), data=df)
```
- **Random slopes**:
```{r}
m3 <- glmmTMB(claims ~ age + (age | policyholder), family=poisson(), data=df)
```
- **Temporal/correlation structures** (e.g., AR1) via glmmTMB documentation.

## Module 6: Model Diagnostics and Validation

- **Objective:** Learn best practices for model checking.
- **Tasks:**
    - Residuals, overdispersion checks.
    - Compare fixed vs. random intercept models.
    - Use AIC, likelihood ratio tests to assess variance components.

## Module 7: Bayesian Extensions

- **Objective:** Leverage full Bayesian credible intervals and flexibility.
- **Reading:**
    - Stuart Klugman on Bayesian credibility.
    - _brms_ & _bamlss_ package tutorials.
```{r}
library(brms)
m_bayes <- brm(claims ~ age + (1 | policyholder), family=poisson(), data=df)
```

## Module 8: Industry & Regulatory Integration

- **Objective:** Explore actuarial standards and professional practice.
- **Resources:**
	- ASOP‑25
	- SOA/CAS learning objectives covering GLMs and credibility

## Module 9: Advanced Topics & Extensions

- **Topics:**
    - **High-cardinality predictors:** GLMMNet (ML + random effects)
    - https://arxiv.org/pdf/2301.12710
	- **Mis-measured covariates and overdispersion:** Bayesian + SIMEX methods
	- https://arxiv.org/pdf/2310.0745
- **Exercise:** Explore these extensions in R using synthetic insurance datasets.