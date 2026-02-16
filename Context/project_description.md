# Generalized Linear Mixed Models for Non-Life Insurance Pricing

## Executive Summary

This self-directed project develops mastery in Generalized Linear Mixed Models (GLMMs) for non-life (casualty/property) insurance pricing. The conceptual throughline is that **credibility theory, mixed models, and Bayesian hierarchical models are the same idea** --- partial pooling of sparse group-level data toward a collective estimate --- implemented with increasing statistical rigor. The project takes a credentialed actuary (FCAS-level) with graduate-level statistical training from GLM and credibility foundations through full competence with GLMMs, including both frequentist and Bayesian approaches.

### Guiding Principle

> "Credibility theory is partial pooling. GLMMs automate partial pooling via likelihood-based estimation. Bayesian GLMMs make the credibility prior explicit. They are all the same idea --- just implemented differently."

This equivalence (BLUP = Buhlmann credibility = Bayesian posterior mean, under appropriate conditions) is introduced early and reinforced throughout every module.

---

## Project Vision & Outcomes

By completion, you will:

1. **Explain the credibility-mixed-model equivalence** rigorously to a peer, including where it holds exactly (Gaussian LMM) and where it is approximate (GLMMs with nonlinear link functions).

2. **Fit, diagnose, and compare GLMMs** using both frequentist (`glmmTMB`, `lme4`) and Bayesian (`brms`) tools, choosing the right approach for the problem at hand.

3. **Interpret variance components and random effects** in actuarial terms --- as credibility parameters, heterogeneity measures, and shrinkage predictors.

4. **Handle real-world complexity** --- zero-inflation, overdispersion, random slopes, model selection --- on realistic insurance datasets.

5. **Defend modeling choices** to regulators and stakeholders, with proper documentation aligned to ASOP-25 and CAS professional standards.

6. **Know the frontier** --- INLA, distributional GLMMs, regularized mixed models --- well enough to evaluate when these tools are warranted.

---

## Pedagogical Design

### Core Principles

1. **Concrete before abstract.** Fit models early (Module 3), understand estimation theory later (Module 4). Actuarial learners are action-oriented; they need to see the tool working before investing in the mechanics.

2. **Spiral learning.** The credibility-partial-pooling-shrinkage concept is revisited in every module from a different angle: manual calculation (Module 2), GLMM random effects (Module 3), variance components (Module 4), Bayesian posteriors (Module 5).

3. **Single running example.** A core insurance dataset is used across Modules 2--5 to build deep familiarity. The learner sees manual credibility, frequentist GLMM, and Bayesian GLMM applied to the same data, making the equivalences concrete.

4. **Just-in-time theory.** Mathematical foundations (Henderson's equations, Laplace approximation, conditional vs. marginal interpretation) are introduced when they are needed to solve a specific problem, not before.

5. **Bayesian as parallel framework, not extension.** Bayesian methods are introduced alongside frequentist methods from Module 3 onward, not deferred to a late module. This reinforces the credibility connection and gives learners flexibility.

6. **Show failures.** Convergence warnings, poor diagnostics, overparameterized models --- the learner encounters and resolves these deliberately. Troubleshooting skill is as important as model-fitting skill.

### What Makes This Plan Different from the Standard GLMM Curriculum

Most GLMM curricula (ecology-focused, biostatistics-focused) start from linear models and build up to mixed models without any credibility context. For actuaries, this misses the crucial insight: **you already understand partial pooling intuitively from credibility theory**. This plan exploits that prior knowledge ruthlessly, using it as the bridge that makes every new concept click faster.

---

## Project Structure: Eight Modules

### Module 1: GLM Refresher & The Credibility Problem

**Duration:** ~1 week
**Objective:** Refresh GLM mechanics and surface the sparse-data problem that motivates everything that follows.

**Core Content:**
- Quick review of exponential family distributions for insurance (Poisson, negative binomial, gamma, Tweedie) and canonical links
- Fit a Poisson GLM for claim frequency on insurance data; interpret coefficients and deviance
- Identify where GLM predictions are unstable: sparse cells, low-exposure groups, extreme rate estimates
- Motivate credibility: "We need a principled way to stabilize sparse-data estimates by borrowing strength from the collective."

**Key Connection:** GLMs are the *starting point* for insurance pricing, but they treat every group independently. When groups have little data, this produces unreliable estimates. Credibility theory and mixed models solve this.

**References:**
- CAS Monograph No. 5, "Generalized Linear Models for Insurance Rating" (Goldburd, Khare, Tevet, 2nd Ed.) --- prerequisite refresher
- CAS Monograph No. 14, Ch. 1--2 --- actuarial historical context and GLM review

**Datasets:** `freMTPL2freq` from `CASdatasets` (678k French motor policies; rich covariates, clear regional hierarchy)

**Deliverable:** Quarto notebook showing a fitted GLM with identified problem areas (unstable predictions for sparse groups).

---

### Module 2: Classical Credibility Theory

**Duration:** ~1--2 weeks
**Objective:** Develop deep intuition for credibility as manual partial pooling. Derive the Buhlmann formula, implement it by hand, and foreshadow the connection to mixed models and Bayesian inference.

**Core Content:**
- Buhlmann credibility formula: $\hat{\mu}_j = Z_j \bar{y}_j + (1 - Z_j) \hat{\mu}$, where $Z_j = \frac{n_j}{n_j + k}$ and $k = \sigma^2 / \tau^2$
- Variance decomposition: between-group variance ($\tau^2$, the "process variance" or VHM) and within-group variance ($\sigma^2$, the "sampling variance" or EVPV)
- Parameter estimation via method of moments (ANOVA-based)
- Buhlmann-Straub extension (heterogeneous weights/exposures)
- Bayesian derivation: under Gaussian-Gaussian conjugacy, the Bayesian posterior mean IS the Buhlmann credibility estimate. This is the first appearance of the central equivalence.
- Limited fluctuation credibility (briefly, for contrast)

**Key Insight:** The credibility weight $Z$ balances individual experience against collective experience. Large $\tau^2$ (groups really differ) $\Rightarrow$ trust individual data more. Large $\sigma^2$ (noisy observations) $\Rightarrow$ trust the collective more. This is **shrinkage**.

**References:**
- CAS Monograph No. 14, Ch. 3 --- outstanding Buhlmann/Buhlmann-Straub treatment with R code
- Buhlmann & Gisler, *A Course in Credibility Theory and its Applications* --- Ch. 3 (Bayesian credibility), Ch. 8 (BLUP connection)
- `actuar` R package --- `cm()` function for credibility models

**Datasets:** Small synthetic portfolio (3--5 classes, 4--5 periods) following CAS Monograph 14 Ch. 3 example. Same data reused in Module 3.

**Deliverable:** Quarto notebook with complete hand calculations of variance components, credibility weights, and credibility-weighted estimates. Include the Bayesian derivation showing posterior mean = Buhlmann formula.

---

### Module 3: From Credibility to Mixed Models

**Duration:** ~2 weeks
**Objective:** Fit the first random intercept GLMM and demonstrate that it produces the same credibility-weighted estimates automatically, while handling covariates that classical credibility cannot.

**This is the pivotal module.** If the learner doesn't internalize the credibility-GLMM equivalence here, nothing else will stick.

**Core Content:**
- The random intercept LMM as a likelihood framework for hierarchical data
- BLUP (Best Linear Unbiased Predictor) derivation for the simple Gaussian case
- **The central result:** BLUP = Buhlmann credibility weight $\times$ (group mean $-$ population mean). Derive algebraically and verify numerically on the Module 2 dataset.
- First GLMM with `glmmTMB`: `claims ~ age + region + (1 | territory), family = poisson()`
- First Bayesian GLMM with `brms`: same model, flat priors. Show results are nearly identical to frequentist.
- Extract variance components and random effects; compare to manual credibility calculations
- **Shrinkage visualization:** Plot no-pooling (raw group means), complete-pooling (grand mean), and partial-pooling (GLMM estimates) on the same axis. Show that GLMM predictions are always between the extremes, with more shrinkage for smaller groups.

**Key Insight:** The GLMM does everything credibility does --- and more. It estimates variance components from data (no manual calculation), handles covariates naturally, and extends to non-Gaussian responses. Credibility is a special case of the mixed model.

**References:**
- Frees, Young & Luo (1999), "A Longitudinal Data Analysis Interpretation of Credibility Models" --- **the landmark paper** connecting credibility to mixed models
- Robinson (1991), "That BLUP is a Good Thing" --- classic exposition of BLUP
- CAS Monograph No. 14, Ch. 4--5 (presumed: mixed models content)
- Existing notebook: `pooling_explanations.qmd` --- excellent shrinkage visualizations to build on

**Datasets:** Same synthetic portfolio from Module 2 (to verify credibility-GLMM equivalence numerically), then `usworkcomp` from `CASdatasets` (workers' comp with state hierarchy) for a realistic example.

**Deliverable:** Quarto notebook demonstrating the equivalence on toy data, then applying GLMM to real insurance data. Both frequentist and Bayesian fits.

---

### Module 4: GLMM Theory, Estimation & Diagnostics

**Duration:** ~2--3 weeks
**Objective:** Understand the mechanics of GLMM inference, know the assumptions that underpin model validity, and diagnose problems.

Now that the learner has seen GLMMs work, they're ready to peek under the hood.

**Core Content:**

*Estimation Theory:*
- The intractable marginal likelihood and why it matters
- ML vs. REML: use REML for variance component inference, ML for model comparison
- Laplace approximation (how `glmmTMB` works): fast, accurate for large clusters, can be poor for small clusters with binary/count data
- Adaptive Gauss-Hermite Quadrature (how `lme4` with `nAGQ > 1` works): more accurate, but scales poorly with random effect dimension
- PQL: mention as deprecated; explain why (biased, not asymptotically correct)
- TMB (Template Model Builder): what it does under the hood (automatic differentiation + Laplace)
- Decision tree: when to use `glmmTMB` (speed, flexibility) vs. `lme4` with AGQ (validation) vs. `brms` (full posterior)

*Variance Components & Interpretation:*
- Random intercept variance $\tau^2$: heterogeneity of baseline risk across groups
- ICC (Intra-Class Correlation): proportion of total variance due to grouping
- Relationship to credibility: $k = \sigma^2 / \tau^2$ determines shrinkage strength

*Conditional vs. Marginal Interpretation (critical section):*
- For nonlinear link functions (log, logit), the marginal mean $\neq$ the conditional mean at $b = 0$
- Poisson log-link: $E[y] = \exp(X\beta + \tau^2/2)$, inflated by $\exp(\tau^2/2)$
- When to use marginal predictions (new business pricing, rate filings) vs. conditional predictions (individual policyholder pricing)
- R code: `predict(..., re.form = NA)` vs. `predict(..., re.form = NULL)`

*Random Effects as Predictions, Not Parameters:*
- Fixed effects are estimated; random effects are predicted (conditional modes/means)
- You can test whether $\tau^2 = 0$ (LRT), but you cannot hypothesis-test individual random effects
- The `re.form` argument in `predict()`: `NA` for population-average, `NULL` for subject-specific

*Diagnostics:*
- DHARMa simulation-based residual diagnostics (the best tool for GLMM residuals)
- Overdispersion testing (Pearson chi-square, DHARMa `testDispersion`)
- Q-Q plots of random effects (checking normality assumption)
- Variance component confidence intervals (profile likelihood, bootstrap)
- Convergence checking: what warnings mean, how to fix them (simplify random effects, try different optimizer, scale predictors)

**Key Insight:** You don't need to fully understand the math to use GLMMs, but you do need to know when the tool is lying to you. Diagnostics are not optional.

**References:**
- Zuur et al., *A Beginner's Guide to GLM and GLMM with R* --- Ch. 6 (model validation), Ch. 11 (zero-inflation diagnostics)
- Stroup, *Generalized Linear Mixed Models* --- Ch. 2--4 (estimation theory, targeted reading)
- Antonio & Beirlant (2007), "Actuarial Statistics with GLMMs" --- insurance-specific GLMM application
- Skrondal & Rabe-Hesketh (2009), "Prediction in multilevel GLMs" --- conditional vs. marginal
- Ben Bolker's GLMM FAQ (bbolker.github.io/mixedmodels-misc/glmmFAQ.html) --- troubleshooting
- DHARMa package documentation (Hartig)

**Datasets:** Synthetic data with known $\tau^2$ (for calibration exercises: "did the model recover the true variance component?"), then `freMTPL2freq` for realistic diagnostics.

**Deliverable:** Quarto notebook covering estimation method comparison on one dataset, diagnostics workflow, and a deliberately overparameterized model that fails to converge (with recovery steps).

---

### Module 5: Bayesian GLMMs & The Credibility Connection

**Duration:** ~2--3 weeks
**Objective:** Deepen the Bayesian approach. Show that Bayesian hierarchical models are the most general framework for credibility, with explicit priors, full uncertainty quantification, and natural handling of complex structures.

**Core Content:**

*Bayesian Credibility, Formally:*
- Derive the Bayesian posterior mean under Gaussian-Gaussian conjugacy --- it IS the Buhlmann credibility formula
- Show algebraically: posterior mean = BLUP = Buhlmann credibility (all three are identical for LMMs)
- Extend conceptually to GLMMs: the posterior still produces credibility-weighted predictions, just not analytically tractable

*brms/Stan in Depth:*
- Prior specification: weakly informative (default), regularizing, informative (from actuarial judgment)
- Prior predictive checks: "What do my priors imply about observable data?"
- Posterior predictive checks: "Does the fitted model generate data that looks like the real data?"
- Convergence diagnostics: R-hat, ESS, trace plots, divergent transitions
- Model comparison: LOO-CV (leave-one-out cross-validation via `loo` package) as a robust alternative to AIC

*Practical Comparison:*
- Refit Module 3 models in `brms`; compare posterior means to REML estimates
- Show that with flat priors, Bayesian and frequentist results converge
- Show a case where informative priors improve estimates (small-sample groups)
- Show a case where `glmmTMB` gives convergence warnings but `brms` with regularizing priors converges cleanly

*Prior Selection for Actuarial Applications:*
- Fixed effects (log scale): Normal(0, 2) --- implies rate ratios between 0.01 and 100
- Random effect SD: Exponential(1) or Half-Normal(0, 1) --- weakly informative, allows $\tau \to 0$
- Correlation matrices (for random slopes): LKJ($\zeta = 2$) --- weakly favors independence
- Sensitivity analysis: always refit with different priors; if conclusions change, data is not informative enough

*Optional: PyMC/Bambi Introduction:*
- Same model in Python for learners who want a second implementation language
- Brief comparison to brms/Stan

**Key Insight:** Bayesian hierarchical models make the credibility assumption *explicit*: the prior on $\tau$ is literally a belief about between-group variability. The posterior for each group is a credibility-weighted blend of individual data and the population prior.

**References:**
- McElreath, *Statistical Rethinking* (2nd Ed.) --- best conceptual explanations of hierarchical models
- Gelman & Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models* --- interpretation and visualization
- Burkner (2017, 2018), brms papers in *Journal of Statistical Software* and *The R Journal*
- Buhlmann & Gisler, Ch. 3 (Bayesian credibility derivation)

**Datasets:** Same running example from Modules 2--3 (to complete the three-method comparison), then `usworkcomp` for a realistic Bayesian analysis.

**Deliverable:** Quarto notebook showing the three-method comparison (manual credibility, frequentist GLMM, Bayesian GLMM) on the same data, plus a full Bayesian analysis with prior/posterior diagnostics. Optional: Jupyter notebook with PyMC/Bambi.

---

### Module 6: Extensions & Practical Complexity

**Duration:** ~2--3 weeks
**Objective:** Extend GLMMs to handle the non-standard features common in real casualty insurance data.

**Core Content:**

*Zero-Inflation:*
- Structural zeros (never-claim policyholders) vs. sampling zeros (claim-eligible but no claim this period)
- Zero-Inflated Poisson (ZIP) and Zero-Inflated Negative Binomial (ZINB) with random effects
- Hurdle models as an alternative
- `glmmTMB(..., ziformula = ~1)` and `brms(..., family = zero_inflated_negbinomial())`

*Overdispersion:*
- Negative binomial as an alternative to Poisson when variance > mean
- Observation-level random effects as a flexible overdispersion solution
- Diagnosis: DHARMa `testDispersion`, Pearson chi-square ratio

*Random Slopes:*
- When the effect of a covariate varies by group (e.g., age effect differs by territory)
- `(age | territory)` syntax: random intercept + random slope + their correlation
- Interpretation of the variance-covariance matrix for random effects
- When random slopes are warranted (domain knowledge + LRT) vs. overparameterized

*Model Selection:*
- AIC/BIC for nested and non-nested model comparison
- Likelihood ratio tests for variance components (boundary issues: $\tau^2 = 0$ is on the boundary)
- LOO-CV for Bayesian models
- Domain knowledge as the tiebreaker: "Can you explain and defend this model?"

*Comprehensive Case Study:*
- Fit 5--6 competing models to a realistic insurance dataset: GLM, random intercept GLMM, random slopes GLMM, ZIP GLMM, ZINB GLMM, Bayesian version of the best
- Compare via AIC, diagnostics, holdout prediction
- Document the decision process and final recommendation

**Key Insight:** The random intercept model is just the starting point. Real actuarial data has more structure --- overdispersion, excess zeros, group-varying effects. But complexity must be justified: start simple, add only what's needed.

**References:**
- Zuur et al. --- Ch. 7--8 (random slopes), Ch. 11 (zero-inflation)
- glmmTMB vignettes (zero-inflation, covariance structures)
- Bolker et al. (2009), "GLMMs: A practical guide for ecology and evolution" --- model specification guidance

**Datasets:** `freMTPL2freq` (high zero proportion, regional hierarchy --- ideal for zero-inflation and random slopes), synthetic data with known random slope structure (for calibration).

**Deliverable:** Case study Quarto notebook (5--7 pages equivalent) with full model comparison, diagnostics, and justified recommendation.

---

### Module 7: Professional Practice & Regulatory Standards

**Duration:** ~1--2 weeks
**Objective:** Ensure the learner can document, communicate, and defend GLMM-based pricing models in a professional actuarial context.

**Core Content:**

*ASOP-25 Alignment:*
- How GLMMs satisfy ASOP-25 credibility requirements (explicit credibility weights, objective variance component estimation, covariate adjustment)
- Documentation requirements: data sources, variable definitions, model structure, assumptions, diagnostics, sensitivity analyses, limitations
- Where GLMMs may face regulatory scrutiny: complexity, software dependency, boundary estimates, rate stability when random effects shift

*Regulatory Communication:*
- Frame GLMMs as "modern credibility methods" --- not "advanced statistics"
- Provide GLM comparison alongside GLMM results (show improvement, not just complexity)
- Visualize credibility weights by group (high-volume groups get $Z \approx 1$; low-volume get $Z \approx 0$)
- Anticipate regulator questions: "Why did Territory X's rate change when its own data didn't change much?"

*Model Documentation Package:*
- Checklist: objectives, data, transformations, model structure, assumptions, diagnostics, results, limitations
- Code reproducibility: Git, R session info, package versions, random seeds
- Governance: when to refit (annually? quarterly?), monitoring actual vs. expected

*Professional Communication:*
- Writing technical memos for actuarial peers vs. senior management vs. regulators
- Effective visualizations for non-technical audiences
- The distinction between inference goals (estimating variance components) and prediction goals (accurate claim forecasts)

**Key Insight:** Technical mastery is necessary but not sufficient. You must be able to explain and defend your work to people who don't know what a random effect is.

**References:**
- ASOP-25, "Credibility Procedures" (Actuarial Standards Board)
- CAS Statement of Principles Regarding Property and Casualty Insurance Ratemaking

**Deliverable:** A model documentation package (3--5 pages) for a GLMM-based rate filing, including model rationale, diagnostics summary, sensitivity analysis, and a 1-page executive summary for a non-technical audience. Mock rate filing exercise.

---

### Module 8: Advanced Topics & Frontiers

**Duration:** ~1--2 weeks
**Objective:** Survey cutting-edge methods with enough depth for the learner to evaluate when they are warranted, but not full mastery. Awareness, not implementation.

**Core Content (brief overviews with one worked example each):**

*INLA (Integrated Nested Laplace Approximation):*
- Fast approximate Bayesian inference (10--100x faster than MCMC)
- When to use: very large datasets, spatial/temporal models, deterministic results needed
- R-INLA package; comparison to brms on a medium-sized dataset

*Distributional GLMMs:*
- Model both mean and variance/dispersion as functions of covariates
- `brms::bf(claims ~ ..., phi ~ region)` syntax
- Actuarial motivation: claim variance may depend on risk factors

*Regularized Mixed Models (GLMMNet):*
- Elastic net penalty on fixed effects + random effects for hierarchical structure
- For high-cardinality predictors (thousands of zip codes, agents)
- `glmmLasso` R package; GLMMNet ArXiv paper (Yi & Zeng, 2023)

*Measurement Error & SIMEX:*
- Sources of measurement error in insurance data (self-reported mileage, estimated vehicle values)
- SIMEX algorithm: add error, refit, extrapolate back to zero error
- When measurement error is consequential (small signal-to-noise ratio)

*Other Frontiers (1-paragraph overviews):*
- Gradient boosting + random effects hybrids
- Spatial random effects (CAR/SAR models for territorial pricing)
- Causal inference with GLMMs
- Fair and ethical pricing: auditing random effects for bias

**Key Insight:** GLMMs are not the end of the story. They are a platform from which more specialized methods are built.

**References:**
- Rue, Martino & Chopin (2009) --- INLA foundational paper
- Burkner (2018) --- distributional brms models
- Yi & Zeng (2023), ArXiv:2301.12710 --- GLMMNet
- Cook & Stefanski (1994) --- SIMEX

**Deliverable:** Quarto notebook with brief worked examples for INLA and distributional GLMMs. 2-page survey document covering the remaining topics.

---

## Capstone Project (Optional but Recommended)

**Duration:** ~2--3 weeks

Apply all skills from Modules 1--7 to a realistic casualty pricing problem.

**Requirements:**
1. Select a dataset with clear hierarchical structure (`usworkcomp` or `freMTPL2freq` recommended)
2. Fit at least 5 competing models: GLM, random intercept GLMM, random slopes GLMM, zero-inflated GLMM, Bayesian GLMM
3. Perform diagnostics on each (DHARMa, posterior predictive checks)
4. Compare via AIC, LOO-CV, and holdout prediction
5. Choose a final model and justify the choice
6. Document assumptions, limitations, and recommendations
7. Write a 5--8 page technical report with visualizations suitable for an actuarial audience

---

## Assessed Reference List

### Tier 1: Essential (Read These)

| Reference | Scope | Use In |
|-----------|-------|--------|
| **CAS Monograph No. 14**, "Practical Mixed Models for Actuaries" | Actuarial GLM/credibility/mixed models bridge | Modules 1--3 |
| **Zuur et al. (2013)**, *A Beginner's Guide to GLM and GLMM with R* | Applied GLMM guide (ecology examples, translate to insurance) | Modules 4, 6 |
| **Frees, Young & Luo (1999)**, "A Longitudinal Data Analysis Interpretation of Credibility Models" | The landmark credibility-mixed-models paper | Module 3 |
| **CAS Monograph No. 5** (Goldburd, Khare, Tevet, 2nd Ed.) | GLM foundations for insurance | Module 1 (prerequisite) |
| **brms documentation** (Burkner) | Bayesian GLMMs in R | Modules 3, 5 |

### Tier 2: Strongly Recommended (Targeted Chapters)

| Reference | Scope | Use In |
|-----------|-------|--------|
| **Buhlmann & Gisler (2005)**, *A Course in Credibility Theory and its Applications* | Credibility theory, Ch. 3 (Bayesian), Ch. 8 (BLUP connection) | Modules 2, 5 |
| **Antonio & Beirlant (2007)**, "Actuarial Statistics with GLMMs" | Insurance-specific GLMM application (Belgian motor data) | Module 4 |
| **Frees (2014)**, *Regression Modeling with Actuarial and Financial Applications* | Ch. 8: LMMs for panel data; insurance examples | Modules 3--4 |
| **Robinson (1991)**, "That BLUP is a Good Thing" | Classic exposition of BLUP = credibility | Module 3 |
| **DHARMa documentation** (Hartig) | Simulation-based GLMM diagnostics | Module 4, 6 |
| **McElreath (2020)**, *Statistical Rethinking* (2nd Ed.) | Best Bayesian hierarchical model explanations | Module 5 |

### Tier 3: Reference for Depth

| Reference | Scope | Use In |
|-----------|-------|--------|
| **Stroup (2013)**, *Generalized Linear Mixed Models* | GLMM theory (Ch. 2--4, 8, 10) | Module 4 |
| **McCulloch, Searle & Neuhaus (2008)**, *Generalized, Linear, and Mixed Models* | Henderson's equations, mathematical derivations | Module 4 |
| **Gelman & Hill (2007)**, *Data Analysis Using Regression and Multilevel/Hierarchical Models* | Interpretation and visualization | Module 5 |
| **Ohlsson & Johansson (2010)**, *Non-Life Insurance Pricing with GLMs* | Ch. 7--8: GLMMs, credibility in insurance context | Modules 3--4 |
| **De Jong & Heller (2008)**, *GLMs for Insurance Data* | Tweedie theory (Ch. 7) | Module 1 (optional) |

### Online Resources

- Michael Clark, "Mixed Models with R" (m-clark.github.io) --- best free tutorial, excellent shrinkage visualizations
- Ben Bolker's GLMM FAQ (bbolker.github.io/mixedmodels-misc/glmmFAQ.html) --- essential troubleshooting reference
- glmmTMB vignettes (CRAN) --- zero-inflation, covariance structures, extensions
- Richard McElreath's "Statistical Rethinking" lectures (YouTube) --- best conceptual explanations

---

## Datasets

### Primary Datasets

| Dataset | Source | Modules | Why |
|---------|--------|---------|-----|
| `freMTPL2freq` / `freMTPL2sev` | `CASdatasets` | 1, 4, 6 | 678k policies, rich covariates, regional hierarchy, high zero proportion |
| `usworkcomp` | `CASdatasets` | 3, 5, Capstone | Workers' comp with state hierarchy; natural for credibility exercises |
| Synthetic GLMM data | Generated in-notebook | 2, 3, 4 | Known $\tau^2$ for calibration; controlled experiments |

### Secondary Datasets

| Dataset | Source | Modules | Why |
|---------|--------|---------|-----|
| `ausautoBI8999` | `CASdatasets` | 6 | Severity modeling, temporal structure |
| `ausprivauto0405` | `CASdatasets` | 6 | Multi-line; Tweedie GLMM potential |
| Frees' Singapore Auto | SOA / Frees (2014) | 3--4 | Panel data with textbook solutions |

### Synthetic Data Strategy

- **Module 2:** Small portfolio (3--5 classes, 4--5 periods) for manual credibility calculations
- **Module 3:** Reuse Module 2 data for GLMM comparison; add a Gaussian LMM version for exact BLUP = credibility verification
- **Module 4:** Poisson GLMM with known $\tau^2$ (50 groups, 20 obs/group) for estimation calibration
- **Module 6:** Zero-inflated Poisson with structural zeros (known zero-inflation probability) and random slopes with known variance

### Synthetic Data Generation Tools

- Base R: `rnorm()`, `rpois()`, `rgamma()` with hierarchical structure
- `simstudy` or `fabricatr` packages for declarative multilevel data simulation
- `brms::simulate()` for generating data from a fitted Bayesian model

---

## Key R Packages

| Package | Purpose | Modules |
|---------|---------|---------|
| `glmmTMB` | Frequentist GLMMs (primary) | 3--6 |
| `lme4` | LMMs; AGQ for validation | 3--4 |
| `brms` | Bayesian GLMMs via Stan | 3, 5--6 |
| `DHARMa` | Simulation-based GLMM diagnostics | 4, 6 |
| `tidybayes` | Tidy posterior extraction | 5 |
| `performance` | ICC, R-squared, model quality checks | 4 |
| `actuar` | Classical credibility (`cm()` function) | 2 |
| `CASdatasets` | Actuarial datasets | 1, 3, 6, Capstone |
| `loo` | LOO-CV for Bayesian model comparison | 5--6 |

---

## Conventions

- **Notebooks** (Quarto `.qmd`, Jupyter `.ipynb`) are the primary deliverable format --- theory (LaTeX math), code, and narrative together.
- **Model caching:** Fitted models are saved as `.rds` (frequentist) or via `brms::brm(file = ...)` (Bayesian) in `Models/` to avoid refitting during rendering. Always check for cached models before refitting.
- **Code style:** R follows the [Tidyverse style guide](https://style.tidyverse.org/); Python follows the [Google style guide](https://google.github.io/styleguide/pyguide.html).
- **Parallel MCMC:** `brms` models use `future::plan(multicore, workers = 4)` for parallel chains.
- **Notebook length:** Keep each notebook under ~50 lines of substantive code; split longer analyses into multiple notebooks.

---

## Success Criteria

You have successfully completed the project when you can:

1. Derive the BLUP-credibility equivalence for the Gaussian LMM and explain where it extends to (and breaks down for) GLMMs.
2. Fit, diagnose, and compare competing GLMM specifications in R for a realistic pricing problem using both frequentist and Bayesian tools.
3. Explain the difference between conditional and marginal predictions and know when to use each.
4. Interpret variance components, random effects, and credibility weights in actuarial terms.
5. Defend a chosen model to regulators with proper documentation, diagnostics, and sensitivity analyses.
6. Handle zero-inflation, overdispersion, and random slopes when the data warrants it.
7. Evaluate when Bayesian methods, INLA, or regularized mixed models would add value over standard frequentist GLMMs.
