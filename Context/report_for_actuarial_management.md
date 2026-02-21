# Tweedie Lasso GLM: Python Migration Risk Assessment

**Prepared for:** Actuarial Management / Model Governance
**Date:** 2026-02-19
**Subject:** Can we replicate our HDTweedie pricing model in Python without compromising model integrity?

---

## Executive Summary

Our Tweedie lasso GLM, fitted using the R package HDTweedie, is a specialized statistical model that no single Python package fully replicates. After exhaustive research, we identified one credible Python alternative for model training (`glum` by QuantCo) and one zero-risk approach for Python-only scoring (exporting the R-fitted coefficients to a simple Python scoring function). The critical finding for actuarial governance is: **any approach that re-fits the model in Python will produce different coefficients than HDTweedie**, because the implementations differ in subtle but real ways — and this triggers re-validation under SR 11-7, OCC 2011-12, and PRA SS1/23 model risk management guidance. The coefficient export approach avoids this entirely by preserving the exact R model output.

This document provides the technical evidence to assess whether a Python migration preserves model integrity, and what the regulatory and actuarial costs would be.

---

## 1. What HDTweedie Does and Why It's Special

### 1.1 The Algorithm

HDTweedie (Qian, Yang, Yang & Zou, 2016, *Journal of Computational and Graphical Statistics*) solves a problem that sits at the intersection of two capabilities:

1. **Tweedie compound Poisson distribution** — the natural model for insurance aggregate claims (point mass at zero from no-claims policies + continuous positive values from claims)
2. **Lasso (L1) penalization** — automatic variable selection that shrinks irrelevant predictor coefficients to exactly zero

Combining these requires a purpose-built algorithm called **IRLS-BMD** (Iteratively Reweighted Least Squares with Blockwise Majorization Descent). The standard tools — `glmnet` for lasso, `glm()` for Tweedie — each handle one piece but not both. HDTweedie was specifically designed to handle both simultaneously.

### 1.2 Why the Tweedie Distribution Matters for This Discussion

The Tweedie distribution for power parameter 1 < p < 2 has an unusual property: its probability density function involves an **infinite series with no closed form**. HDTweedie sidesteps this by optimizing the **Tweedie deviance**, which *does* have a closed form:

```
d(y, mu) = 2 * [ y^(2-p)/((1-p)(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p) ]
```

This formula is mathematically exact but **numerically sensitive** — particularly when p approaches 1 (Poisson boundary) or 2 (Gamma boundary), where terms cancel and floating-point precision erodes. The HDTweedie Fortran implementation handles these edge cases through careful log-space computation. Any reimplementation must handle these same numerical subtleties, or risk producing different optimization trajectories and therefore different final coefficients.

### 1.3 The Package

- **Authors:** Wei Qian (Rochester Institute of Technology), Yi Yang (McGill University), Hui Zou (University of Minnesota)
- **Core:** 508 lines of Fortran 90 (the numerical engine) + ~450 lines of R (API, cross-validation, plotting)
- **License:** GPL-2
- **Status:** Stable on CRAN, last updated May 2022. The algorithm is mature and the package is not under active feature development — which, for a validated production model, is a good thing.
- **Academic foundation:** Peer-reviewed methodology published in a top-tier computational statistics journal

---

## 2. The Central Question: Will a Python Model Give the Same Results?

### 2.1 Defining "Same Results"

Three levels of equivalence, from strongest to weakest:

| Level | Definition | Achievable? |
|---|---|---|
| **Bit-identical** | Same coefficients to 15+ decimal places | Only with coefficient export |
| **Numerically equivalent** | Same coefficients to 4-5 significant digits, same variable selection | Possible with `glum`, requires validation |
| **Statistically comparable** | Similar predictive performance on holdout data, possibly different variable selection | Likely with any reasonable approach |

For actuarial governance, the relevant question is: **which level does your model change management process require?**

### 2.2 The Coefficient Export Approach: Exact Equivalence

The strongest guarantee comes from not re-fitting the model at all. A Tweedie GLM with log link makes predictions using a simple formula:

```
predicted_loss = exp(beta_0 + beta_1 * x_1 + beta_2 * x_2 + ... + beta_p * x_p)
```

Or equivalently with an exposure offset:

```
predicted_loss = exposure * exp(beta_0 + beta_1 * x_1 + ... + beta_p * x_p)
```

This is a dot product followed by exponentiation. The prediction formula does not require the Tweedie distribution, the lasso penalty, or any optimization — those are only needed during training.

**The approach:**
1. Train the model in R using HDTweedie (as we already do)
2. Export the fitted coefficient vector (including zeros from lasso selection) to a JSON file
3. In Python, load the JSON and compute predictions using the formula above

**Why this gives exact equivalence:**
- The coefficients are *identical* — they are the same numbers, exported from R
- Matrix multiplication (`X @ beta`) and `exp()` are deterministic operations
- IEEE 754 floating-point arithmetic produces the same results in R and Python for these operations
- The difference between R and Python predictions will be at the level of floating-point epsilon (~10^-15) — well below any actuarial or regulatory threshold

**What this does NOT cover:**
- Model retraining — when the model needs to be re-estimated on new data, that still happens in R
- Prediction intervals or distributional predictions — these require the Tweedie power parameter p and dispersion phi in addition to the coefficients, plus a Tweedie distribution implementation in Python. This adds moderate complexity beyond the simple scoring formula. For point prediction (expected loss cost), which is our current production use case, this is not needed.
- Any feature preprocessing — if features are transformed before entering the model (standardization, encoding, etc.), those transformations must be replicated identically in the Python scoring path. In our case, no preprocessing is applied by HDTweedie beyond what we control, so this is not a concern.

**Actuarial governance implication:** The coefficient export approach does **not** constitute a model change under SR 11-7 or OCC 2011-12. The model — defined as the algorithm, the fitted parameters, and the mathematical specification — is unchanged. Only the computational environment executing the arithmetic is different, and the arithmetic is deterministic. This is analogous to running the same Excel formula on a different computer.

### 2.3 The glum Re-fitting Approach: Close but Not Identical

If the engineering team requires Python for model *training* (not just scoring), the best option is `glum` (by QuantCo). `glum` implements the same class of algorithm as HDTweedie: IRLS with coordinate descent, Tweedie deviance loss, L1 penalization.

However, the implementations differ:

| Aspect | HDTweedie | glum |
|---|---|---|
| Inner solver | BMD (blockwise majorization descent) | Coordinate descent |
| Hessian computation | Fisher information (expected) | True Hessian (observed) |
| Convergence tolerance | `eps = 1e-8` on relative coefficient change | Different default tolerance |
| Lambda path construction | glmnet-style automatic grid | Different grid logic |
| Strong rules / screening | glmnet-style | Different active set strategy |
| Numerical precision | Fortran 90 double precision | Cython/NumPy double precision |
| Warm-start strategy | Coefficients from previous lambda | Similar but different initialization |

These differences mean that **even with the same data, the same features, the same power parameter, and the same lambda value, HDTweedie and glum will generally produce different coefficient vectors.** The differences are typically small (matching to 2-4 significant digits for most coefficients), but they are real and systematic — not random noise.

**Why this matters for actuarial governance:**

1. **Different variable selection is possible.** At the lasso boundary, where a coefficient is being driven toward zero, the two implementations may disagree on whether the coefficient is exactly zero or slightly nonzero. This means the two models may include/exclude different predictors.

2. **Different relativities.** Even when both models select the same variables, the coefficient magnitudes will differ. For a rating factor like territory, a coefficient of 0.123 in R vs. 0.119 in Python means a different relativity applied to every policy in that territory.

3. **Cascading differences.** Because lasso coefficients are jointly determined (changing one coefficient affects the optimal values of all others through the IRLS reweighting), a small difference in one coefficient propagates through the entire model.

### 2.4 Evidence: Port Drift Is Real, Not Hypothetical

The risk of coefficient differences between implementations is documented:

**statsmodels Issue #8300:** Python's `statsmodels` NegativeBinomial GLM produces structurally different standard errors from R's `glm.nb()`. Root cause: the two implementations parameterize the dispersion parameter differently (`alpha` vs. `theta = 1/alpha`) and use different IRLS weighting. This is not a rounding error — it is an algorithmic divergence that affects predictions on every row.

**Knoyd SAS-to-Python banking migration:** A credit scoring model migrated from SAS to Python in a regulated banking environment found that regression coefficients differed "after the second decimal place." The consultants acknowledged that "it's impossible for regression to converge to the exact same coefficients twice" across implementations. The migration cost was EUR 60,000 and required documented tolerance sign-off from model risk governance.

**Finalyse analysis of language migration in credit risk:** "Calculations performed using SAS packages versus Python packages can result in differences which, while generally minuscule, cannot always be ignored, and **getting an exact match between values calculated in SAS and values calculated in Python may be difficult**."

For a Tweedie lasso model specifically, the risk surface is larger than for standard GLMs because:
- The Tweedie deviance computation is numerically sensitive (see Section 1.2)
- Lasso variable selection is a discontinuous function of the coefficients — small numerical differences can flip a variable in or out
- The regularization path depends on warm starts, so differences compound along the path

---

## 3. Regulatory Framework

### 3.1 SR 11-7: "Processing" Is a Validatable Model Component

Federal Reserve Supervisory Letter SR 11-7 (April 2011) — the foundational US model risk management guidance — explicitly identifies the code and computational implementation ("processing") as a model component subject to validation:

> "All model components — inputs, **processing**, outputs, and reports — should be subject to validation."

SR 11-7 further identifies **implementation errors** as a distinct source of model risk:

> "A model is a simplification of reality, and all models are subject to model risk... [including] incorrectly implemented models."

Changing the implementation language changes the "processing" component. Under SR 11-7, this is a model change that requires, at minimum, implementation validation (code review + output comparison) and potentially full revalidation for high-risk models. A pricing model directly affecting policyholder premiums is typically classified as high-risk.

**If the model stays in R (including the coefficient export approach), no revalidation is triggered.** The processing component is unchanged.

### 3.2 OCC 2011-12 and the 2021 Model Risk Management Handbook

The OCC's 2021 Model Risk Management Handbook extends SR 11-7 explicitly:

> "Algorithms, formulae, code/script, software, and IT systems that implement models should be examined thoroughly. These supporting tools should have rigorous controls for quality, accuracy, **change management**, and user access."
>
> "Banks should have a change management process to validate updates to existing models before implementation."

A language migration is unambiguously an "update to the existing model implementation." The change management process must be invoked, documented, and approved. This has direct financial cost: actuarial time, validation team time, and documentation.

### 3.3 UK PRA SS1/23 (Effective May 2024)

For UK-regulated entities, PRA Supervisory Statement SS1/23 requires:
- Model change governance with "clear roles and responsibilities of dedicated model approval authorities" (Principle 2)
- Validation of model implementation as part of the model lifecycle (Principle 3)
- During any transition period, additional compensating controls must be documented and operated (Principle 5)

A Python re-implementation would need to pass through the model approval authority as a material model change.

### 3.4 Rate Filing Implications

In insurance rate filings, regulators can and do ask about model implementation. The audit trail for the current model is clean:

- The R code in version control IS the production code
- The model that was backtested IS the model that scores policies
- The coefficients in the `.rds` file are the coefficients the actuary validated

If the model is rewritten in Python, the filing record changes:
- "We rewrote the model in a different language" must be documented
- The regulator asks: "How do you know the Python model produces the same results?"
- The answer requires: parallel run comparison, numerical tolerance documentation, edge case analysis, and independent validation sign-off
- If the Python model produces *different* results (even slightly), the rate filing may need to be amended

---

## 4. Cost of Migration

### 4.1 Coefficient Export (Option B): Minimal Cost

| Cost Category | Estimate | Notes |
|---|---|---|
| Python scoring implementation | 0.5 days | NumPy dot product + exp() |
| Validation testing | 1-2 days | Confirm R and Python predictions match to floating-point precision |
| Documentation | 0.5 days | Document the export/deploy workflow |
| Regulatory impact | None | Not a model change; same coefficients, deterministic arithmetic |
| **Total** | **~2-3 person-days** | |

### 4.2 glum Re-fitting (Option C): Substantial Cost

| Cost Category | Estimate | Notes |
|---|---|---|
| Python training pipeline | 1-3 days | Translate HDTweedie API calls to glum |
| Side-by-side coefficient comparison | 2-4 weeks | Run both models on full portfolio, document differences |
| Tolerance analysis | 1-2 weeks | Define acceptable coefficient/prediction deviations |
| Model validation review | 4-8 weeks | Actuarial validation team reviews the Python implementation |
| Regulatory documentation | 1-2 weeks | Model change documentation, possibly rate filing amendment |
| Dual maintenance period | Ongoing | Both R and Python models in parallel until migration certified |
| **Total** | **~8-16 person-weeks of actuarial/data science time** | |

### 4.3 Full Python Port of HDTweedie (Option D): Not Recommended

| Cost Category | Estimate | Notes |
|---|---|---|
| Python implementation | 15-25 weeks | Rewrite 508 lines of Fortran + 450 lines of R in Python/Cython |
| Numerical validation | 4-6 weeks | Ensure convergence to same coefficients as R across test datasets |
| Model validation review | 4-8 weeks | Full revalidation of the new implementation |
| Long-term maintenance | Ongoing | Your team owns the codebase permanently |
| GPL-2 licensing review | 1-2 weeks legal | HDTweedie is GPL-2; derivative works inherit the license |
| **Total** | **~25-40 person-weeks** | |

### 4.4 The Opportunity Cost

Every week of actuarial and data science time spent proving that a Python model matches the R model is a week not spent:
- Improving the model (new features, updated power parameter, interaction effects)
- Expanding coverage to new lines of business
- Refining the regularization path or cross-validation strategy
- Building monitoring and A/B testing for the existing model

---

## 5. What Happens to "The Same Model" Under Each Option

### 5.1 Option A: Keep R + Plumber + Docker

| Property | Status |
|---|---|
| Same coefficients | Yes — identical, it IS the same model |
| Same variable selection | Yes |
| Same predictions | Yes, bit-for-bit |
| Regulatory status | No change triggered |
| Audit trail | Perfect — production code = validated code |

### 5.2 Option B: Export Coefficients to Python

| Property | Status |
|---|---|
| Same coefficients | Yes — exported from R, identical values |
| Same variable selection | Yes — same coefficients, same zeros |
| Same predictions | Yes, to floating-point precision (~10^-15) |
| Regulatory status | No model change — same mathematical model, different execution environment |
| Audit trail | JSON coefficient file is versioned and auditable |
| Caveat | Training still in R; if prediction intervals or distributional output ever needed, additional complexity arises |

### 5.3 Option C: Re-fit with glum

| Property | Status |
|---|---|
| Same coefficients | **No** — close (2-4 significant digits) but not identical |
| Same variable selection | **Possibly different** — lasso boundary effects may include/exclude different predictors |
| Same predictions | Close but not identical; differences accumulate across the portfolio |
| Regulatory status | **Model change** — triggers validation under SR 11-7 / OCC 2011-12 |
| Audit trail | New codebase; must document equivalence or justify differences |

### 5.4 Option D: Full Python Port

| Property | Status |
|---|---|
| Same coefficients | **Unknown** — depends on implementation fidelity; risk of subtle divergence |
| Same variable selection | **Unknown** |
| Same predictions | **Unknown** — requires extensive validation |
| Regulatory status | **Model change + new implementation risk** — highest regulatory burden |
| Audit trail | Entirely new codebase with no R provenance |

---

## 6. The Python Ecosystem: What Actually Exists

For context, here is what we found when we exhaustively searched for Python packages that could replace HDTweedie:

### 6.1 Packages That Cannot Do Tweedie Lasso

| Package | Why Not |
|---|---|
| scikit-learn `TweedieRegressor` | L2 (ridge) penalty only — no L1/lasso, no variable selection |
| statsmodels `GLM.fit_regularized` | Known Tweedie + L1 instability bug (GitHub #7476) — "Tweedie still has the additional problem that the log-likelihood function is just an approximation" |
| glmnet-python (all variants) | No Tweedie distribution support |
| pyglmnet | No Tweedie distribution support |
| PySpark GLM | Requires Spark cluster; no regularization path |

### 6.2 The One Credible Alternative

**`glum`** (QuantCo, BSD-3 license, `pip install glum`):
- Implements IRLS with coordinate descent for Tweedie + L1
- Built specifically for insurance pricing use cases
- Actively maintained (latest release January 2026)
- sklearn-compatible API
- Does NOT support grouped lasso (not needed for our model)
- Coefficients will differ from HDTweedie (different solver implementation)

### 6.3 What Does Not Exist in Python

- A package that uses the same IRLS-BMD algorithm as HDTweedie
- A package that will produce identical coefficients to HDTweedie
- A package that supports grouped lasso with Tweedie deviance
- An ONNX or serialization format that captures lasso-penalized Tweedie training

---

## 7. Recommendation

### Primary Recommendation: Keep the R Model, Offer Python Scoring via Coefficient Export

This is the lowest-risk path that addresses the engineering team's request for Python involvement while preserving model integrity:

1. **Training stays in R** using HDTweedie — the validated, peer-reviewed, production-proven tool
2. **Coefficients are exported** to a versioned JSON file after each training cycle
3. **Scoring moves to Python** (if engineering prefers) — a trivial NumPy implementation of `exp(X @ beta + intercept)`
4. **No model change** is triggered under any regulatory framework
5. **Exact numerical equivalence** is guaranteed by construction

The actuarial workflow is:
```
Actuarial team retrains model in R (quarterly/annually)
  -> Exports coefficient JSON
    -> Engineering deploys updated JSON to Python scoring service
      -> Production predictions use identical coefficients
```

### If Full Python Training Is Required

Use `glum` and budget for:
- 8-16 person-weeks of actuarial/validation time
- Side-by-side comparison on the full portfolio
- Documented tolerance analysis
- Formal model change submission through governance
- Ongoing dual maintenance until migration is certified

### What We Do Not Recommend

- A full Python port of HDTweedie (15-25 weeks implementation + 25-40 weeks total with validation — not justified for one model)
- Accepting coefficient differences without formal validation ("it's close enough" is not a governance-compliant position)
- Abandoning R training without a validated Python alternative in place

---

## 8. References

### Regulatory Guidance
- [SR 11-7 — Federal Reserve Supervisory Letter (April 2011)](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)
- [SR 11-7 Full Attachment (PDF)](https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107a1.pdf)
- [OCC Bulletin 2011-12](https://www.occ.gov/news-issuances/bulletins/2011/bulletin-2011-12.html)
- [OCC Model Risk Management Handbook 2021 (PDF)](https://www.occ.treas.gov/publications-and-resources/publications/comptrollers-handbook/files/model-risk-management/pub-ch-model-risk.pdf)
- [UK PRA SS1/23 — Model Risk Management Principles (May 2023)](https://www.bankofengland.co.uk/prudential-regulation/publication/2023/may/model-risk-management-principles-for-banks-ss)

### Port Drift Evidence
- [statsmodels Issue #8300 — NegativeBinomial Does Not Match R's glm.nb](https://github.com/statsmodels/statsmodels/issues/8300)
- [statsmodels Issue #7476 — fit_regularized Unstable with Tweedie](https://github.com/statsmodels/statsmodels/issues/7476)
- [Knoyd — SAS to Python Migration in Banking (Numerical Equivalence Findings)](https://www.knoyd.com/blog/migration-from-sas-to-python)
- [Finalyse — The Language War in Credit Risk Modelling](https://www.finalyse.com/blog/the-language-war-in-credit-risk-modelling-sas-python-or-r)

### HDTweedie
- Qian, W., Yang, Y., Yang, Y. and Zou, H. (2016). "Tweedie's Compound Poisson Model With Grouped Elastic Net." *Journal of Computational and Graphical Statistics*, 25(2), 606-625.
- [HDTweedie CRAN page](https://cran.r-project.org/web/packages/HDtweedie/)
- [Paper preprint (free)](https://repository.rit.edu/cgi/viewcontent.cgi?article=2822&context=article)

### Python Alternatives
- [glum documentation (QuantCo)](https://glum.readthedocs.io/)
- [glum GitHub](https://github.com/Quantco/glum)
- [scikit-learn TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html)

### Production R Deployments
- [Swiss Re — R/Insurance Series, R Consortium (January 2024)](https://r-consortium.org/webinars/r-insurance-series.html)
- [R Consortium R Submissions Working Group — FDA Pilots 1-4](https://r-consortium.org/all-projects/isc-working-groups.html)
- [Joel Spolsky — Things You Should Never Do, Part I (2000)](https://www.joelonsoftware.com/2000/04/06/things-you-should-never-do-part-i/)

---

*This report synthesizes findings from five parallel research workstreams. Full technical details available in the supporting reports in the `Context/` directory: `research_hdtweedie_deep_dive.md`, `research_python_1to1_alternatives.md`, `research_python_workarounds.md`, `research_transcoding_feasibility.md`, and `research_r_plumber_production_case.md`.*
