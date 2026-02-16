# Syllabus: GLMMs for Non-Life Insurance Pricing

**Conceptual throughline:** Credibility theory = partial pooling = random effects = Bayesian shrinkage. Same idea, increasing rigor.

## Learning Objectives

1. Derive the BLUP-credibility-Bayesian posterior equivalence and explain where it holds and breaks down.
2. Fit and diagnose GLMMs using `glmmTMB` (frequentist) and `brms` (Bayesian) for actuarial pricing.
3. Interpret variance components, random effects, and credibility weights in insurance terms.
4. Handle zero-inflation, overdispersion, random slopes, and model selection on real insurance data.
5. Document and defend GLMM-based models to regulators (ASOP-25 compliance).

---

## Module 1: GLM Refresher & The Credibility Problem (~1 week)

**Objective:** Review GLMs for insurance; surface the sparse-data problem.

- Exponential family distributions (Poisson, NB, gamma, Tweedie) and canonical links
- Fit Poisson GLM for claim frequency; identify unstable predictions in sparse cells
- Motivate: why GLMs alone fail with limited group-level data

**Reading:** CAS Monograph 5 (Goldburd et al.); CAS Monograph 14 Ch. 1--2
**Data:** `freMTPL2freq` (CASdatasets)
**Deliverable:** Quarto notebook --- fitted GLM with problem diagnosis

---

## Module 2: Classical Credibility Theory (~1--2 weeks)

**Objective:** Build deep intuition for credibility as manual partial pooling; foreshadow mixed models.

- Buhlmann credibility: $Z = n/(n + k)$, $k = \sigma^2/\tau^2$
- Variance decomposition: VHM ($\tau^2$) and EVPV ($\sigma^2$)
- Buhlmann-Straub (heterogeneous exposures)
- Bayesian derivation: posterior mean = Buhlmann formula under Gaussian conjugacy
- `actuar::cm()` for computational verification

**Reading:** CAS Monograph 14 Ch. 3; Buhlmann & Gisler Ch. 3 & 8
**Data:** Synthetic portfolio (3--5 classes, 4--5 periods)
**Deliverable:** Quarto notebook --- hand calculations + Bayesian derivation

---

## Module 3: From Credibility to Mixed Models (~2 weeks)

**Objective:** Fit first GLMM; demonstrate BLUP = credibility algebraically and numerically.

- Random intercept LMM; BLUP derivation (simple Gaussian case)
- **Central result:** BLUP = Buhlmann credibility weight $\times$ (group mean $-$ population mean)
- First `glmmTMB` fit: `claims ~ age + region + (1 | territory), family = poisson()`
- First `brms` fit: same model, flat priors; compare to frequentist
- Shrinkage visualization: no pooling vs. complete pooling vs. partial pooling

**Reading:** Frees, Young & Luo (1999); Robinson (1991); CAS Monograph 14 Ch. 4--5
**Data:** Module 2 synthetic data (equivalence check) + `usworkcomp` (realistic)
**Deliverable:** Quarto notebook --- credibility-GLMM equivalence + real insurance application

---

## Module 4: GLMM Theory, Estimation & Diagnostics (~2--3 weeks)

**Objective:** Understand estimation mechanics; diagnose problems; know when the model is wrong.

**Estimation:**
- ML vs. REML (when to use each)
- Laplace approximation (glmmTMB) vs. AGQ (lme4, `nAGQ > 1`) vs. Bayesian (brms)
- PQL: why deprecated
- TMB architecture (automatic differentiation + Laplace)

**Interpretation:**
- Variance components ($\tau^2$), ICC
- **Conditional vs. marginal predictions** (critical): $E[y] = \exp(X\beta + \tau^2/2)$ for Poisson/log
- `re.form = NA` (population-average) vs. `re.form = NULL` (subject-specific)
- Random effects as predictions, not parameters

**Diagnostics:**
- DHARMa simulation-based residuals
- Overdispersion testing
- Random effects normality (Q-Q plots)
- Convergence issues: diagnosis and recovery

**Reading:** Zuur et al. Ch. 6; Stroup Ch. 2--4; Antonio & Beirlant (2007); Bolker GLMM FAQ
**Data:** Synthetic (known $\tau^2$) + `freMTPL2freq`
**Deliverable:** Quarto notebook --- estimation comparison, diagnostics workflow, convergence failure/recovery

---

## Module 5: Bayesian GLMMs & The Credibility Connection (~2--3 weeks)

**Objective:** Show Bayesian hierarchical models as the general credibility framework.

- Formal proof: posterior mean = BLUP = Buhlmann (Gaussian LMM)
- brms/Stan: prior specification, prior/posterior predictive checks, convergence diagnostics
- LOO-CV for model comparison
- Prior selection for actuarial applications (weakly informative, regularizing, informative)
- Sensitivity analysis: do conclusions change with different priors?
- Three-method comparison on running example (manual credibility, glmmTMB, brms)
- Optional: PyMC/Bambi introduction

**Reading:** McElreath *Statistical Rethinking*; Gelman & Hill; Burkner brms papers
**Data:** Running example from Modules 2--3 + `usworkcomp`
**Deliverable:** Quarto notebook --- three-method comparison + full Bayesian analysis

---

## Module 6: Extensions & Practical Complexity (~2--3 weeks)

**Objective:** Handle real-world data features: zero-inflation, overdispersion, random slopes.

- Zero-inflation: ZIP, ZINB, hurdle models with random effects (`ziformula = ~1`)
- Overdispersion: negative binomial, observation-level random effects
- Random slopes: `(age | territory)`; variance-covariance interpretation
- Model selection: AIC/BIC, LRT (boundary issues), LOO-CV, domain knowledge
- Comprehensive case study: 5--6 competing models, full comparison

**Reading:** Zuur et al. Ch. 7--8, 11; glmmTMB vignettes; Bolker et al. (2009)
**Data:** `freMTPL2freq` + synthetic (known random slopes)
**Deliverable:** Case study notebook --- model comparison with justified recommendation

---

## Module 7: Professional Practice & Regulatory Standards (~1--2 weeks)

**Objective:** Document, communicate, and defend GLMM pricing models professionally.

- ASOP-25 compliance: how GLMMs satisfy credibility procedure requirements
- Regulatory communication: frame as "modern credibility," show GLM comparison
- Model documentation checklist
- Mock rate filing exercise
- Writing for actuarial peers vs. management vs. regulators

**Reading:** ASOP-25; CAS Statement of Principles
**Deliverable:** Model documentation package (3--5 pages) + executive summary

---

## Module 8: Advanced Topics & Frontiers (~1--2 weeks)

**Objective:** Survey cutting-edge methods (awareness, not mastery).

- INLA: fast approximate Bayesian inference (R-INLA)
- Distributional GLMMs: model mean + dispersion (`brms::bf()`)
- Regularized mixed models / GLMMNet: elastic net + random effects
- Measurement error / SIMEX
- Brief: gradient boosting hybrids, spatial random effects, causal inference, fairness

**Reading:** Rue et al. (2009); Burkner (2018); Yi & Zeng (2023)
**Deliverable:** Quarto notebook with INLA/distributional examples + 2-page survey

---

## Capstone (Optional, ~2--3 weeks)

Full pricing analysis on `usworkcomp` or `freMTPL2freq`: 5+ competing models, diagnostics, comparison, 5--8 page technical report.

---

## Key References

**Tier 1 (Essential):**
- CAS Monograph 14, *Practical Mixed Models for Actuaries*
- Zuur et al. (2013), *A Beginner's Guide to GLM and GLMM with R*
- Frees, Young & Luo (1999), "A Longitudinal Data Analysis Interpretation of Credibility Models"
- CAS Monograph 5, *GLMs for Insurance Rating* (prerequisite)
- brms package documentation (Burkner)

**Tier 2 (Targeted chapters):**
- Buhlmann & Gisler (2005), *A Course in Credibility Theory* --- Ch. 3, 8
- Antonio & Beirlant (2007), "Actuarial Statistics with GLMMs"
- Frees (2014), *Regression Modeling with Actuarial and Financial Applications* --- Ch. 8
- Robinson (1991), "That BLUP is a Good Thing"
- McElreath (2020), *Statistical Rethinking* (2nd Ed.)

**Online:**
- Michael Clark, "Mixed Models with R" (m-clark.github.io)
- Ben Bolker's GLMM FAQ (bbolker.github.io)
- glmmTMB vignettes (CRAN)

## Estimated Timeline

| Module | Duration | Cumulative |
|--------|----------|------------|
| 1. GLM Refresher | 1 week | 1 week |
| 2. Credibility Theory | 1--2 weeks | 2--3 weeks |
| 3. Credibility → Mixed Models | 2 weeks | 4--5 weeks |
| 4. GLMM Theory & Diagnostics | 2--3 weeks | 6--8 weeks |
| 5. Bayesian GLMMs | 2--3 weeks | 8--11 weeks |
| 6. Extensions | 2--3 weeks | 10--14 weeks |
| 7. Professional Practice | 1--2 weeks | 11--16 weeks |
| 8. Frontiers | 1--2 weeks | 12--18 weeks |
| Capstone (optional) | 2--3 weeks | 14--21 weeks |

**Target: ~15 weeks for core content (Modules 1--7), ~18 weeks with frontiers and capstone.**
