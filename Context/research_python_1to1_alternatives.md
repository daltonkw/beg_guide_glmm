# Python Alternatives to R HDTweedie: Exhaustive Assessment

**Prepared by:** Agent 2 (Python Package Research)
**Date:** 2026-02-19
**Purpose:** Identify direct 1:1 Python replacements for the R HDTweedie package

---

## What HDTweedie Actually Does

HDTweedie is an R package implementing the **Lasso (and grouped elastic net) for Tweedie's Compound Poisson model** via an **IRLS-BMD** (Iteratively Reweighted Least Squares with Blockwise Majorization Descent) algorithm. Its key characteristics:

| Capability | HDTweedie Behavior |
|---|---|
| **Loss function** | Tweedie deviance (not MSE, not Gaussian likelihood) |
| **Penalization** | L1 (lasso) and grouped elastic net |
| **Algorithm** | Coordinate descent embedded in IRLS |
| **Output** | Full solution path across a lambda grid |
| **Cross-validation** | Built-in CV with deviance, MAE, or MSE loss |
| **Power parameter** | Fixed by user (1 < p < 2) |
| **Scale** | Designed for high-dimensional insurance data |

The algorithm reference is: Qian, Yang, Yang, Zou (2016), "Tweedie's Compound Poisson Model with Grouped Elastic Net," *JCGS*. The CRAN package is `HDtweedie`.

---

## Comparison Table

| Package | Tweedie Deviance Loss | L1 Penalization | Coordinate Descent | Solution Path | Built-in CV | Fixed p | Verdict |
|---|---|---|---|---|---|---|---|
| **glum** | ✅ Yes | ✅ Yes | ✅ IRLS-CD | ⚠️ Implicit (CV-only) | ✅ Yes | ✅ Yes | **Best alternative** |
| **h2o (Python API)** | ✅ Yes | ✅ Yes | ⚠️ IRLSM (not pure CD) | ✅ lambda_search | ✅ nfolds | ✅ Yes | Strong, but heavyweight |
| **scikit-learn TweedieRegressor** | ✅ Yes | ❌ L2 only | ❌ LBFGS/Newton | ❌ No | ❌ External | ✅ Yes | Not a replacement |
| **statsmodels GLM** | ⚠️ Approx. | ✅ Technically | ❌ Not CD | ❌ No | ❌ No | ✅ Yes | Unstable — avoid |
| **glmnet-python / python-glmnet** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | N/A | No Tweedie support |
| **pyglmnet** | ❌ No | ✅ Yes | ✅ Yes | ❌ No | ❌ No | N/A | No Tweedie support |
| **PySpark GLM** | ✅ Yes | ✅ Yes | ❌ IRLS | ❌ No native | ❌ External | ✅ Yes | Overkill, no path |

---

## Detailed Candidate Assessments

### 1. scikit-learn TweedieRegressor

**Does it optimize Tweedie deviance?** Yes — uses the proper Tweedie log-likelihood.
**L1 penalization?** **No.** The `alpha` parameter multiplies an **L2 penalty only** (Ridge). There is no `l1_ratio` or Lasso option in `TweedieRegressor`.
**Coordinate descent?** No — uses LBFGS or Newton-Cholesky solvers.
**Solution paths?** No — fits a single model per `alpha`. Would require manual looping.
**Cross-validation?** Only externally via `GridSearchCV`.
**Fixed power p?** Yes, via the `power` parameter.

**Evidence:** The [TweedieRegressor docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html) state: *"alpha: Constant that multiplies the L2 penalty term."* The Lasso and ElasticNet classes in scikit-learn do not support non-Gaussian loss functions.

**Verdict:** ❌ **Not a replacement.** Cannot do L1 penalization on Tweedie deviance. The two capabilities (Tweedie family + L1) live in separate classes that cannot be combined.

---

### 2. statsmodels GLM with Tweedie family + `fit_regularized()`

**Does it optimize Tweedie deviance?** ⚠️ **Approximately.** statsmodels uses an approximation to the Tweedie log-likelihood, not the true density (which is itself a series expansion). This matters for optimization accuracy.
**L1 penalization?** Technically yes — `fit_regularized(L1_wt=1.0)` gives pure L1.
**Coordinate descent?** Not the primary solver — uses a proximal gradient / ADMM approach.
**Solution paths?** No — fits a single model per `alpha`.
**Cross-validation?** No built-in CV.
**Fixed power p?** Yes, via `sm.families.Tweedie(var_power=p)`.

**Critical known issues:** GitHub Issue [#7476](https://github.com/statsmodels/statsmodels/issues/7476) ("fit_regularized method is unstable/too sensitive to alpha when using Tweedie distribution") documents that:
> *"Tweedie still has the additional problem that the log-likelihood function is just an approximation and not the true loglike. Additionally, fit_regularized doesn't do anything explicitly with scale, so scale could possibly differ between loglike and score."*

Small changes in alpha can cause all coefficients to collapse to zero or behave erratically.

**Verdict:** ❌ **Not a replacement.** Technically the pieces exist, but known bugs make it unreliable for Tweedie + L1. No solution path, no built-in CV, and unstable behavior documented in open GitHub issues. Not production-quality.

---

### 3. glum (Quantco)

**Repository:** [Quantco/glum](https://github.com/Quantco/glum)
**PyPI:** `pip install glum`
**Stars:** ~356 (active as of Feb 2026, v3.1.2 released Jan 30, 2026)

**Does it optimize Tweedie deviance?** ✅ **Yes.** glum minimizes the negative log-likelihood (unit deviance) as the loss function. For Tweedie, this is the Tweedie deviance. The documentation states: *"Switching to using the true Hessian for special cases (including Tweedie regression for 1 < p < 2) gave huge reductions in the number of IRLS iterations."* Code and formulas for Tweedie are hand-optimized.
**L1 penalization?** ✅ **Yes.** `l1_ratio=1.0` gives pure lasso. Elastic net with any mix is supported.
**Coordinate descent?** ✅ **Yes.** The `irls-cd` solver uses IRLS with a coordinate descent inner loop for L1-penalized problems, directly analogous to HDTweedie's IRLS-BMD approach.
**Solution paths?** ⚠️ **Partial.** `GeneralizedLinearRegressorCV` searches across a path of `alphas` values (automatically generated or user-specified via `min_alpha_ratio`). However, it does **not** expose the full coefficient matrix across the path as a single output (unlike HDTweedie's matrix output at each lambda). You can recover the path by iterating manually.
**Cross-validation?** ✅ **Yes.** `GeneralizedLinearRegressorCV` selects optimal alpha via CV, stores results in `.alpha_` and `.l1_ratio_` attributes.
**Fixed power p?** ✅ **Yes.** Pass `family=glum.TweedieDistribution(1.5)` or `family='tweedie (1.5)'`.

**Example usage:**
```python
import glum

# Single model
model = glum.GeneralizedLinearRegressor(
    family=glum.TweedieDistribution(power=1.5),
    alpha=0.01,
    l1_ratio=1.0,  # pure lasso
    solver='irls-cd'
)
model.fit(X, y)

# CV over regularization path
model_cv = glum.GeneralizedLinearRegressorCV(
    family=glum.TweedieDistribution(power=1.5),
    l1_ratio=1.0,
    min_alpha_ratio=1e-4,
    cv=5
)
model_cv.fit(X, y)
print(model_cv.alpha_)  # optimal lambda
```

**Production maturity:** QuantCo built glum specifically for **insurance pricing** (used in e-commerce pricing, insurance claims prediction). Actively maintained, latest release Jan 2026. Not as widely used as scikit-learn (~356 stars vs. scikit-learn's 60k+), but purpose-built for the exact use case.

**Key gaps vs. HDTweedie:**
- Does not natively output a full coefficient path matrix (all betas at all lambdas in one call)
- No named `deviance`, `mae`, `mse` CV criterion options (defaults to deviance-like loss)
- Slightly heavier API than HDTweedie's minimal `hdtweedie(X, y, p=p)` call

**Verdict:** ✅ **Best Python alternative.** Matches HDTweedie in loss function, penalization type, and optimization algorithm. The CV class covers lambda selection. The main gap is the all-lambdas coefficient path as a first-class output — but this is recoverable with a short loop.

---

### 4. glmnet-python / python-glmnet / glmnet (PyPI)

There are multiple Python glmnet packages:
- **`glmnet`** (PyPI): Fortran wrapper, implements linear and logistic only.
- **`python-glmnet`** (Civis Analytics): Python port, supports linear, logistic, Cox, Poisson — **not Tweedie.**
- **`glmnet-py`** (PyPI): Similar scope, again no Tweedie.

**Evidence:** From [python-glmnet GitHub](https://github.com/civisanalytics/python-glmnet): *"only linear and logistic are implemented in this package."* R glmnet itself added Tweedie family support in v4.0 via custom family objects, but none of the Python ports have implemented this.

**Verdict:** ❌ **Not a replacement.** Zero Tweedie support in any Python glmnet port.

---

### 5. h2o Python API

**Package:** `h2o` (Python client for H2O cluster)
**Installation:** `pip install h2o` + requires Java/H2O server

**Does it optimize Tweedie deviance?** ✅ **Yes.** H2O GLM uses the Tweedie deviance as its loss function. Supports full Tweedie distribution with fixed or estimated variance power.
**L1 penalization?** ✅ **Yes.** `alpha=1.0` gives pure LASSO.
**Coordinate descent?** ⚠️ **Not pure CD.** Uses IRLSM (Iteratively Reweighted Least Squares Method), not a BMD/coordinate descent inner loop. For `lambda_search=True` with L1, IRLSM is recommended for small-to-moderate p (which fits the 20-50 predictor use case).
**Solution paths?** ✅ **Yes.** `lambda_search=True` computes the full regularization path from lambda_max down to lambda_min, directly analogous to glmnet's path.
**Cross-validation?** ✅ **Yes.** `nfolds=5` enables k-fold CV across the lambda path.
**Fixed power p?** ✅ **Yes.** `tweedie_variance_power=1.5`.

**Example usage:**
```python
import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator

h2o.init()
model = H2OGeneralizedLinearEstimator(
    family='tweedie',
    tweedie_variance_power=1.5,
    alpha=1.0,              # pure lasso
    lambda_search=True,     # compute full path
    nlambdas=100,
    nfolds=5               # cross-validation
)
model.train(x=predictors, y=target, training_frame=train_h2o)
```

**Production maturity:** H2O is enterprise-grade, widely deployed, with a large user base. Very mature (10+ years). Actively maintained by H2O.ai.

**Key limitations:**
- **Heavy dependency.** Requires Java JVM + H2O cluster startup. Not a lightweight `import`. This is a significant operational difference from R's `library(HDtweedie)`.
- **Saturation limit.** The number of predictors that can enter a LASSO model saturates at `min(n, p)`.
- **AIC approximation caveat.** AIC for ridge/elastic net is approximated, not exact.
- **Not pure coordinate descent.** Uses IRLSM, which should give equivalent results but is algorithmically different.
- **Dispersion parameter note.** Regularization cannot be used simultaneously with ML-based dispersion parameter estimation.

**Verdict:** ✅ **Functionally strong, operationally heavy.** Covers all the core HDTweedie capabilities. The blocker is the JVM+H2O cluster requirement — this is not a drop-in replacement but a platform choice. In a production MLOps pipeline where H2O is already present, this is excellent. For a lightweight script replacement, it's overkill.

---

### 6. pyglmnet

**Repository:** [glm-tools/pyglmnet](https://github.com/glm-tools/pyglmnet)

Supports: gaussian, binomial, poisson, softplus, probit, gamma. **Tweedie is not in the list of supported distributions.** Has elastic net regularization with coordinate descent, but the critical family is missing.

**Verdict:** ❌ **Not a replacement.** No Tweedie support.

---

### 7. PySpark GeneralizedLinearRegression

Supports Tweedie family (`family='tweedie'`) and L1 regularization (`regParam`, `elasticNetParam=1.0`). However:
- No native solution path or lambda search — requires external `ParamGridBuilder` + `CrossValidator`
- Requires a Spark cluster
- Designed for distributed, massive-scale data — inappropriate for 20-50 predictor problems
- No coordinate descent (uses IRLS)

**Verdict:** ❌ **Not a replacement.** Architectural mismatch. Overkill for this use case, and lacks native solution path.

---

## Additional Packages Investigated

- **`tweedie` (PyPI, thequackdaddy):** Only implements Tweedie density estimation (probability distributions), not regression. No modeling.
- **`lightgbm` / `xgboost`:** Support Tweedie deviance as objective but are tree-based, not GLM/lasso.
- **`glmnet_python` (various):** All Python glmnet ports are limited to linear/logistic families.
- **`LightGBM` with Tweedie loss:** Non-linear, not applicable.

---

## Verdict: Is There a True 1:1 Python Alternative?

**Short answer: No, but glum gets you 90% of the way there.**

### The gap analysis

| Capability | HDTweedie | glum | Delta |
|---|---|---|---|
| Tweedie deviance loss | ✅ | ✅ | None |
| L1/lasso penalty | ✅ | ✅ | None |
| Elastic net | ✅ | ✅ | None |
| Coordinate descent | ✅ IRLS-BMD | ✅ IRLS-CD | Algorithmically equivalent |
| Full coefficient path matrix | ✅ Single call | ⚠️ Loop required | Minor code work |
| CV loss options (deviance/MAE/MSE) | ✅ Named options | ⚠️ Deviance default | Minor |
| Grouped lasso | ✅ | ❌ | Feature gap |
| Coefficient parity | Reference | Untested | Unknown |

### Recommendation

**Use `glum` as the primary Python alternative.** Specifically:
- `glum.TweedieDistribution(power=p)` for the exact same Tweedie family
- `l1_ratio=1.0` for pure lasso
- `solver='irls-cd'` for coordinate descent (same algorithm class as HDTweedie)
- `GeneralizedLinearRegressorCV` for lambda selection via CV

To replicate HDTweedie's full coefficient path output, add a wrapper:
```python
def tweedie_lasso_path(X, y, power, alphas, cv=5):
    """Replicate HDTweedie's solution path output."""
    coef_path = []
    for alpha in alphas:
        m = glum.GeneralizedLinearRegressor(
            family=glum.TweedieDistribution(power),
            alpha=alpha, l1_ratio=1.0, solver='irls-cd'
        )
        m.fit(X, y)
        coef_path.append(m.coef_)
    return np.array(coef_path)  # shape: (n_lambdas, n_features)
```

**If grouped lasso is required,** there is no Python alternative. HDTweedie's grouped lasso is unique; you would need to wrap the R package via `rpy2`.

**For production use with heavy infrastructure** already in place, H2O's Python API is the strongest option but requires operational commitment to the H2O platform (JVM, cluster, data upload).

---

## Sources

- [HDtweedie CRAN page](https://cran.r-project.org/web/packages/HDtweedie/index.html)
- [HDtweedie GitHub mirror (cran/HDtweedie)](https://github.com/cran/HDtweedie)
- [Qian et al. 2016 paper: Tweedie's Compound Poisson Model with Grouped Elastic Net](https://repository.rit.edu/cgi/viewcontent.cgi?article=2822&context=article)
- [glum documentation](https://glum.readthedocs.io/en/latest/glm.html)
- [glum background/algorithms](https://glum.readthedocs.io/en/latest/background.html)
- [glum getting started: Lasso model](https://glum.readthedocs.io/en/latest/getting_started/getting_started.html)
- [Quantco/glum GitHub](https://github.com/Quantco/glum)
- [scikit-learn TweedieRegressor docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html)
- [statsmodels GLM.fit_regularized docs](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.fit_regularized.html)
- [statsmodels GitHub Issue #7476: fit_regularized unstable with Tweedie](https://github.com/statsmodels/statsmodels/issues/7476)
- [H2O GLM documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html)
- [glm-tools/pyglmnet GitHub](https://github.com/glm-tools/pyglmnet)
- [python-glmnet (Civis Analytics) GitHub](https://github.com/civisanalytics/python-glmnet)
- [PySpark GeneralizedLinearRegression docs](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.GeneralizedLinearRegression.html)
