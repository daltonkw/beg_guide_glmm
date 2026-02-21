# HDTweedie Python Transcoding Feasibility Report

**Date:** 2026-02-19
**Author:** Research Agent 4 (Tweedie Research Team)
**Context:** Insurance actuarial model (~20-50 predictors), fixed Tweedie power parameter, regulatory environment, R version currently deployed and working.

---

## Executive Summary

**Recommendation: Do not build a full Python port of HDTweedie.** Instead, use **glum** for the unpenalized and standard elastic-net cases, and — if grouped elastic net is genuinely required — write a thin **f2py wrapper** around the existing Fortran core (4–8 person-weeks). A from-scratch Cython/Python rewrite carries 3–6× the effort and substantial numerical-validation risk for a production regulatory environment.

---

## 1. HDTweedie Package Anatomy

### 1.1 Language Breakdown

| Language | Share | File(s) | Lines |
|----------|-------|---------|-------|
| Fortran 90 | ~48.8% | `src/tweediegrpnet.f90` | ~508 |
| R | ~49.4% | `R/*.R` (7 files) | ~450 total |
| C | ~1.8% | `src/init.c` | ~25 |

The package is nearly a 50/50 split between R glue code and compiled Fortran — there is **no pure-R implementation of the core algorithm**.

### 1.2 Source File Inventory

**Compiled code (`src/`):**
- `tweediegrpnet.f90` — 508 lines of Fortran 90; the entire numerical engine
- `init.c` — 25 lines of C boilerplate; registers the Fortran symbol `tweediegrpnet` (29 arguments) with R's dynamic loader. Contains zero algorithm logic.
- `Makevars` — build configuration

**R layer (`R/`):**
- `HDtweedie.R` (~95 lines) — main user-facing function; validates inputs, sets up lambda grid, calls `tweediegrppath()`
- `tweediegrppath.R` — thin bridge; calls `.Fortran("tweediegrpnet", ...)` with 29 arguments; zero algorithm logic in R
- `cv.R` (~67 lines) — k-fold cross-validation and fold-wise prediction-error computation
- `utilities.R` — Tweedie deviance (`devi()`), lambda interpolation, CV aggregation; mostly adapted from **glmnet**
- `tools.R`, `plot.HDtweedie.R`, `plot.cv.HDtweedie.R` — diagnostics and plotting

### 1.3 External Dependencies

- **R `methods` package only** — no other R package dependencies
- **LAPACK `dgesvd`** — called directly from Fortran for SVD; linked via R's built-in BLAS/LAPACK
- No BLAS, Rcpp, or other compiled dependencies beyond standard LAPACK

This extremely lean dependency tree is both an asset (easy to isolate) and a liability (the algorithm is entirely self-contained — there is nothing to reuse from an existing Python library without reimplementing the math).

### 1.4 Package Metadata

| Field | Value |
|-------|-------|
| Version | 1.2 (released 2022-05-09) |
| License | **GPL-2** |
| Authors | Wei Qian, Yi Yang, Hui Zou (University of Minnesota) |
| Maintainer | Wei Qian `<weiqian@stat.umn.edu>` |
| CRAN | Active (as of report date) |
| GitHub activity | 2 commits total (read-only CRAN mirror) |

---

## 2. Core Algorithm: IRLS-BMD

HDTweedie implements a **two-loop optimization** for the Tweedie compound Poisson log-likelihood with grouped elastic net regularization, as described in:

> Qian, W., Yang, Y., Yang, Y. and Zou, H. (2016). *Tweedie's Compound Poisson Model With Grouped Elastic Net.* **Journal of Computational and Graphical Statistics**, 25, 606–625.

### 2.1 Outer Loop: IRLS (Iteratively Reweighted Least Squares)

At each IRLS iteration, the Fortran code computes working response `yt` and working weights `vtt` from the current linear predictor `r = X*beta`:

```fortran
r1  = vt * y * exp(-(rho - 1.0D0) * r)   ! y * mu^(1-p)
r2  = vt * exp((2.0D0 - rho) * r)         ! mu^(2-p)
vtt = (rho - 1.0D0) * r1 + (2.0D0 - rho) * r2  ! working weights (2nd deriv of log-lik)
yt  = r + (r1 - r2) / vtt                  ! working response (1st deriv / 2nd deriv)
```

This is numerically sensitive: `r1` and `r2` can have catastrophic cancellation when `rho` is near 1 (Poisson) or 2 (Gamma). The log-space exponential formulation (`exp(...)`) somewhat mitigates this compared to a naive `mu^p` approach.

### 2.2 Inner Loop: BMD (Blockwise Majorization Descent)

The penalized WLS subproblem is solved by cycling over groups:

1. **Majorization**: for each group `g`, compute `γ_g` = largest singular value of `X_g' diag(vtt) X_g` via LAPACK `dgesvd`. This is the tight Lipschitz upper bound on the group's curvature — the key step that makes BMD faster than naive coordinate descent.

2. **Group elastic net proximal update** (from Fortran, simplified):
```fortran
u = matmul(vtt * (yt - r), x(:, start:end))   ! gradient w.r.t. group
u = gam(g) * b(start:end) + u                  ! gradient step using majorization
unorm = sqrt(dot_product(u, u))
t = unorm - pf(g) * lam * alpha                ! group lasso threshold
IF (t > 0.0D0) THEN
    b(start:end) = u * t / ((gam(g) + lam * (1.0D0 - alpha)) * unorm)
ELSE
    b(start:end) = 0.0D0
END IF
```
This is the group elastic net proximal operator: L1 term shrinks the group ℓ₂-norm, L2 term (`1-alpha`) shrinks the scale.

3. **Strong rules / screening**: the Fortran maintains a `jxx(g)` flag array using the KKT-based strong rules (Tibshirani et al., adapted for groups). Groups unlikely to be active at the current lambda are skipped entirely, making the algorithm scale to high dimensions despite the inner SVD cost.

### 2.3 Numerically Sensitive Components

| Component | Sensitivity | Notes |
|-----------|-------------|-------|
| Tweedie deviance: `2[y^(2-p)/((1-p)(2-p)) - y*μ^(1-p)/(1-p) + μ^(2-p)/(2-p)]` | **HIGH** — catastrophic cancellation near p≈1 or p≈2 | Requires careful branch handling |
| IRLS weight computation: `w_i = μ_i^(2-p)` | **MEDIUM** — can produce near-zero weights for large μ | |
| Convergence check on solution path | MEDIUM | Active set tracking across lambda |
| Lambda grid generation | LOW | Log-uniform grid, standard practice |
| SVD for group blocks | LOW | Delegated to LAPACK; numerically stable |

The deviance formula is the single most dangerous porting target: it is numerically unstable at the Poisson boundary (p=1) and the Gamma boundary (p=2) and requires limit expressions (L'Hôpital) in those regimes. The Fortran code computes `exp(-(rho-1.0D0)*r)` and related exponential transforms, suggesting a log-space implementation that avoids some cancellation — but replicating this exactly in Python requires close reading of 508 lines of Fortran.

### 2.4 Solution Path and Cross-Validation

- **Solution path**: the Fortran subroutine computes the entire lambda path in a single call using warm starts (the active set and coefficient estimates from lambda_k seed the computation for lambda_{k+1}).
- **CV**: implemented entirely in R (~67 lines). Fits the full path on each training fold, computes fold-wise deviance/MAE/MSE, aggregates with weights. This is the easiest piece to port.

---

## 3. Engineering Effort Estimates

All estimates assume a **competent Python developer who is not a Tweedie or coordinate descent expert**. A senior numerical methods developer could cut estimates by ~40%.

### Option A: f2py Wrapper (Reuse Fortran Core)

Extract the Fortran file from the package, compile it into a Python extension via NumPy's `f2py`, and write a thin Python API layer around it.

| Phase | Tasks | Effort |
|-------|-------|--------|
| Environment setup | Extract Fortran from CRAN tarball; set up f2py build with LAPACK linkage | 1–2 days |
| f2py interface | Write `.pyf` signature file for 29-argument `tweediegrpnet`; handle array ordering (Fortran-contiguous) | 3–5 days |
| Python wrapper | sklearn-compatible `HDTweedie` class; input validation; lambda grid; `fit()`, `predict()`, `coef_path_` | 3–5 days |
| CV wrapper | Port the 67-line `cv.R` to Python (straightforward) | 2–3 days |
| Numerical validation | Run identical inputs through R and Python; assert results match to 1e-8 tolerance | 5–10 days |
| Packaging | `pyproject.toml`, `setup.py` with Fortran compilation, GitHub Actions CI, PyPI | 3–5 days |
| Documentation | API docs, usage examples, README | 3–5 days |
| **Total** | | **~4–7 person-weeks** |

**Key risks**: Fortran compilation is notoriously platform-sensitive (gfortran version, LAPACK linkage on macOS ARM, Windows). The 29-argument Fortran interface has no type safety — a single argument-order mistake silently produces garbage coefficients.

### Option B: Cython Port (Rewrite Core in Cython)

Translate `tweediegrpnet.f90` to Cython, calling `scipy.linalg.dgesvd` (or `scipy.linalg.svd`) for the SVD blocks.

| Phase | Tasks | Effort |
|-------|-------|--------|
| Algorithm study | Carefully read and annotate all 508 lines of Fortran; map to algorithm in the 2016 paper | 2–4 weeks |
| Cython implementation | Implement IRLS outer loop + BMD inner loop + active set management + solution path | 6–10 weeks |
| Numerical validation | Extremely important; subtle differences in floating-point evaluation order can produce different solution paths | 4–6 weeks |
| Python wrapper + CV + packaging + docs | Same as Option A | 3–5 weeks |
| **Total** | | **~15–25 person-weeks** |

### Option C: Pure Python / NumPy Port

Similar to Option B but using NumPy array operations. Roughly equivalent effort to Cython but ~10–50× slower in the inner loop (unacceptable for a solution-path solver with 100 lambda values × many IRLS iterations). Only appropriate if performance is not a concern.

### Option D: Contribute Tweedie + Group Lasso to glum

Add grouped elastic net support to the **glum** library (Quantco), which already has Tweedie support and a production-grade coordinate descent engine.

| Phase | Tasks | Effort |
|-------|-------|--------|
| Study glum internals | Understand glum's coordinate descent, Cython extension, extension points | 2–3 weeks |
| Implement grouped penalty | Add group soft-thresholding operator to glum's inner loop | 3–6 weeks |
| PR + review cycle | Open-source contribution; Quantco review process; iterate on design | 4–8 weeks |
| **Total** | | **~9–17 person-weeks** |

This produces a maintained, community-owned artifact. However, it is uncertain whether Quantco will accept the contribution, and the timeline is driven partly by their review process.

---

## 4. Benefits of a Native Python Port

| Benefit | Weight | Notes |
|---------|--------|-------|
| No R runtime in production | HIGH | Eliminates R + renv in Docker; reduces image size, startup time |
| pip-installable | HIGH | `pip install hdtweedie-py`; integrates into standard Python MLOps |
| sklearn-compatible API | HIGH | Works with `Pipeline`, `GridSearchCV`, `SHAP`, etc. |
| CI/CD friendly | HIGH | Standard Python tooling; no R CMD check |
| Team familiarity | MEDIUM | Only valuable if the team is Python-primary |
| Actuarial Python ecosystem | LOW-MEDIUM | Contributes to nascent ecosystem (chainladder-python, skglm, etc.) |

The production engineering argument is strong: R-in-Docker is a solved but annoying problem. An R Plumber microservice works, but it adds operational complexity (R runtime management, health checks, renv reproducibility) that a pip-installable Python package avoids entirely.

---

## 5. Drawbacks and Risks

### 5.1 Risk Matrix

| Risk | Probability | Impact | Severity |
|------|-------------|--------|----------|
| Numerical divergence from R version | HIGH | HIGH | **CRITICAL** |
| Regulatory re-validation required | HIGH (if results differ) | HIGH | **CRITICAL** |
| Port becomes abandonware | HIGH | MEDIUM | HIGH |
| Fortran compilation fails on target platform | MEDIUM | HIGH | HIGH |
| Subtle Tweedie deviance bug | MEDIUM | HIGH | HIGH |
| No community → bugs unfixed for months | HIGH | MEDIUM | HIGH |
| f2py interface breakage on NumPy major version | MEDIUM | MEDIUM | MEDIUM |

### 5.2 Numerical Accuracy

This is the central risk. The Tweedie deviance formula is numerically unstable near p=1 and p=2. The Fortran implementation may use specific floating-point evaluation strategies that differ from a naive Python implementation. In a regulatory insurance environment, you cannot accept "close but not identical" — auditors will ask why your Python model gives a different answer than the validated R model.

**Acceptable tolerance**: In practice, coefficients matching to 4–5 significant digits on standard test cases is achievable and defensible. Matching to 8+ digits requires matching the Fortran's evaluation order exactly.

### 5.3 Maintenance Burden

The R package has had 2 commits total on GitHub (CRAN mirror). The academic authors (University of Minnesota) are unlikely to port it themselves or provide support for a Python derivative. Once written, **your organization owns the maintenance**:
- Bug fixes
- Compatibility updates (NumPy, Python, gfortran API changes)
- Numerical edge cases from new datasets
- LAPACK API changes

For a package used in a regulatory model, this is a significant long-term commitment.

### 5.4 License Implications (GPL-2)

HDTweedie is licensed under **GPL-2**. Key implications:

| Scenario | GPL-2 Requirement |
|----------|------------------|
| Use R package internally (no distribution) | No restrictions |
| Create Python derivative and use internally | No restrictions (private use) |
| Create Python derivative and distribute (open-source) | Must distribute under GPL-2 or compatible |
| Create Python derivative and distribute (commercial/proprietary) | **Not permitted** — GPL-2 prohibits proprietary distribution of derivative works |
| f2py wrapper distributing the Fortran binary | The Fortran is GPL-2; the wrapper is a derivative work |

**Practical implication**: If your organization wants to distribute or sell the Python port, or if you use it in a SaaS product, you must open-source it under GPL-2. Internal use only is fine. Consult legal counsel if commercial distribution is intended.

A **fresh reimplementation** based solely on the published algorithm description (the 2016 JCGS paper) — without copying Fortran code — is NOT a derivative work of HDTweedie and can be licensed freely. However, it requires more effort and more careful numerical validation.

---

## 6. Alternative: Use Existing Python Packages

### 6.1 glum (Quantco)

**[glum](https://github.com/Quantco/glum)** is the most mature Python Tweedie elastic-net implementation.

| Feature | glum | HDTweedie |
|---------|------|-----------|
| Tweedie distribution | ✅ configurable power | ✅ configurable power |
| L1 / elastic net | ✅ | ✅ |
| **Grouped elastic net** | ❌ not supported | ✅ |
| sklearn API | ✅ | ❌ |
| CV built-in | ✅ | ✅ (separate function) |
| Offsets | ✅ | ❌ |
| Sample weights | ✅ | ✅ |
| Active maintainer | ✅ Quantco | ⚠️ Academic (low activity) |
| License | BSD-3 | GPL-2 |

**If grouped elastic net is not required**, glum is a drop-in replacement. The effort is essentially zero: `pip install glum`, change the API call. For an insurance model with ~20–50 predictors where you control the feature engineering, grouped penalties are often not necessary.

**If grouped elastic net IS required** (e.g., one-hot encoded categoricals that should be penalized as a group, or rating factors with multiple levels), glum cannot help today.

### 6.2 scikit-learn TweedieRegressor

[`sklearn.linear_model.TweedieRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html) supports L2 (ridge) regularization only. No L1/lasso. Not a substitute for HDTweedie.

### 6.3 scikit-learn Issue #16637

A GitHub issue ([sklearn #16637](https://github.com/scikit-learn/scikit-learn/issues/16637)) explicitly requested adding L1/coordinate-descent support to `TweedieRegressor`. It was not merged as of 2025. Contributing this to sklearn would be the highest-impact path (massive user base, production-grade testing infrastructure) but the review bar is high and the Tweedie-specific numerical handling may not meet sklearn's generality requirements.

### 6.4 pyglmnet

[pyglmnet](https://github.com/glm-tools/pyglmnet) supports grouped lasso and elastic net for Gaussian, Binomial, Poisson, Gamma, and Softplus GLM families — but **does not include Tweedie** as a named family. Adding Tweedie would require implementing the variance function and deviance. Research-grade, not production-validated.

### 6.5 skglm

[skglm](https://github.com/scikit-learn-contrib/skglm) is a fast sklearn-compatible library for sparse GLMs with various penalties. Has strong support for L1-type penalties on GLMs. Tweedie support exists in principle via custom datafit classes. This is the most promising path for a fresh Python implementation that avoids GPL-2 entanglement.

---

## 7. Decision Framework

Answer these questions in order:

**Q1: Is grouped elastic net actually required?**

If no: **use glum**. Zero porting effort. BSD-3 license. Actively maintained. Far superior sklearn integration. Tweedie with configurable power and elastic net — exactly what HDTweedie provides, minus grouping.

If yes → proceed to Q2.

**Q2: Is commercial distribution of the Python artifact required?**

If no: **f2py wrapper** is the pragmatic path. 4–7 person-weeks, reuses the validated Fortran core, numerical parity is guaranteed by construction, and internal use is GPL-2 compliant.

If yes → proceed to Q3.

**Q3: Is there budget and appetite for a 15–25 week engineering project?**

If yes: Fresh reimplementation based on the 2016 paper (BSD or MIT license), either standalone or as a contribution to skglm. Requires deep algorithm knowledge and extensive numerical validation.

If no: Stay on R for this model, wrap it in a Plumber microservice, and invest Python effort where R has no good story.

---

## 8. Recommendation

For the stated context (insurance actuarial model, ~20–50 predictors, fixed power parameter, regulatory environment, R version working):

### Primary Recommendation: Stay on R (short-term)

The R version is **working, validated, and deployed**. The cost of Python parity — even via f2py — is 4–7 person-weeks minimum, plus the regulatory cost of re-validating a "new" implementation. That is a high bar for a model already in production.

**Invest instead in making R production-friendly**: `plumber` microservice + Docker + `renv` + CI/CD. This pattern is well-documented, widely used in insurance, and eliminates the operational objections to R in production without requiring a port.

### Secondary Recommendation: Migrate to glum (medium-term, if grouped penalties not required)

Audit whether the grouped elastic net feature is actually used and material. If HDTweedie is being used with `group = NULL` (i.e., no grouping), then **glum is a direct substitute** with no algorithmic cost, better tooling, and active maintenance. The migration effort is a few hours of API translation.

### Tertiary Recommendation: f2py wrapper (if grouped penalties required and internal-only distribution)

If grouped penalties are required AND the Python artifact stays internal (not commercially distributed), the f2py wrapper in 4–7 weeks is the most defensible path in a regulatory environment: numerical parity with the R version is guaranteed because it literally runs the same Fortran code.

### Do Not Recommend: Full Cython/Python rewrite

For a single organization's production model, a 15–25 week rewrite of a validated statistical package is hard to justify. The maintenance burden is high, the regulatory re-validation is expensive, and the numerical risk is real. This only makes sense as a community-wide contribution (to skglm or similar) where the maintenance burden is shared.

---

## 9. Sources

- [HDTweedie CRAN page](https://cran.r-project.org/web/packages/HDtweedie/index.html)
- [HDTweedie GitHub mirror](https://github.com/cran/HDtweedie)
- [HDTweedie source: R/utilities.R](https://rdrr.io/cran/HDtweedie/src/R/utilities.R)
- [glum documentation](https://glum.readthedocs.io/en/latest/)
- [glum GitHub (Quantco)](https://github.com/Quantco/glum)
- [scikit-learn TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html)
- [pyglmnet](https://github.com/glm-tools/pyglmnet)
- [scikit-learn issue #16637: Add L1 to TweedieRegressor](https://github.com/scikit-learn/scikit-learn/issues/16637)
- [f2py documentation (NumPy)](https://numpy.org/doc/stable/f2py/)
- [conda-forge r-hdtweedie feedstock](https://github.com/conda-forge/r-hdtweedie-feedstock)
- Qian, W., Yang, Y., Yang, Y. and Zou, H. (2016). *Tweedie's Compound Poisson Model With Grouped Elastic Net.* Journal of Computational and Graphical Statistics, 25, 606–625.
