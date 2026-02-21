# H2O Tweedie + Lasso GLM: Deep Technical Assessment

**Prepared for:** Engineering Team / Actuarial Pricing Migration Decision
**Date:** 2026-02-19
**Purpose:** Exhaustive technical evaluation of H2O as a Python deployment path for a Tweedie GLM with L1 regularization, previously fitted in R with HDTweedie

---

## Executive Summary

H2O's `H2OGeneralizedLinearEstimator` with `family='tweedie'` and `alpha=1.0` (lasso) is a **functionally complete** solution for fitting Tweedie GLMs with L1 regularization in Python. It handles fixed variance power, produces a full regularization path, integrates cross-validation with lambda selection, and supports both exposure weights and log-exposure offsets. The MOJO export pathway is a real and documented feature that allows scoring without a running H2O cluster — this is the strongest argument for H2O.

However, H2O comes with a non-trivial operational tax: the JVM dependency, cluster lifecycle management, Docker image weight (~1.5 GB vs ~200 MB for a pure-Python solution), and a client/server architecture that is architecturally alien to a standard Python microservice. The MOJO scoring angle partially mitigates the deployment problem for inference, but training still requires a full H2O environment.

**Ranking for this specific use case:** H2O sits between glum and the R+Plumber approach. It is more capable than glum for full regularization path extraction and has a better "coefficient audit" story, but glum is lighter, faster to start, and better integrated with standard Python tooling. The MOJO export is a genuine differentiator if the engineering team wants Java-native scoring, but it adds Java expertise requirements and does not eliminate H2O from the training pipeline.

---

## 1. Algorithm: IRLSM vs Coordinate Descent

### What algorithm does H2O actually use?

H2O GLM offers four solvers controlled by the `solver` parameter:

| Solver | Description |
|---|---|
| `IRLSM` | Iteratively Reweighted Least Squares Method (default for small/medium p) |
| `COORDINATE_DESCENT` | IRLSM with cyclical coordinate descent in the innermost loop |
| `L_BFGS` | Limited-memory quasi-Newton; better for wide, dense datasets |
| `AUTO` | H2O selects based on data shape and parameters |

**For Tweedie with lasso (alpha=1.0) and fewer than ~500 predictors, `AUTO` will select `IRLSM`.** The documentation explicitly states IRLSM is "most efficient for tall and narrow datasets and lambda search with sparse solutions." For 20–50 predictors, this is the right choice.

The `COORDINATE_DESCENT` solver is a hybrid: IRLSM with a covariance-updates version of cyclical coordinate descent as the innermost loop. This is **not** the same algorithm as HDTweedie's IRLS-BMD (Blockwise Majorization Descent), and it is not the same as glum's IRLS with true Hessian-based coordinate descent.

### The key algorithmic distinction vs HDTweedie and glum

HDTweedie (IRLS-BMD) and glum both use the **true Hessian** in the IRLS quadratic approximation for Tweedie (1 < p < 2), rather than the expected Hessian (Fisher information matrix). The glum documentation notes this "gave huge reductions in the number of IRLS iterations" for Tweedie and Gamma models. H2O's IRLSM uses the Fisher information matrix (expected Hessian) by default, which is the older IRLS formulation.

**Practical implication:** H2O IRLSM will likely take more IRLS outer iterations than HDTweedie or glum on the same Tweedie problem, but each iteration is efficient and the convergence behavior on well-conditioned problems (20–50 predictors) will still be fast in wall-clock time.

### Will coefficients match HDTweedie?

Not exactly. Reasons for divergence:
1. Different quadratic approximation (expected vs true Hessian) changes the optimization path
2. Different lambda grid construction and spacing
3. Potential differences in standardization conventions
4. H2O adds an intercept term by default; confirm HDTweedie's behavior
5. H2O's `alpha` parameter blends L1 and L2; verify that `alpha=1.0` uses **pure** L1 (it does by documentation)

Coefficient values should converge to the same optimum at a given lambda if both are solving the same objective function with identical penalty scaling. Whether the penalty scaling is identical between H2O and HDTweedie depends on normalization conventions that are **not guaranteed to match without empirical comparison**. Expect coefficients to be consistent in sign and magnitude ordering, but not numerically identical to HDTweedie output.

---

## 2. Lambda Search / Regularization Path

### How it works

Setting `lambda_search=True` instructs H2O to compute the full regularization path, mirroring the glmnet approach:

1. H2O first computes `lambda_max` — the smallest lambda that drives all coefficients to zero (the null model)
2. H2O then constructs an exponentially decreasing sequence from `lambda_max` down to `lambda_min = lambda_min_ratio * lambda_max`
3. The default `lambda_min_ratio` is adaptive: 0.0001 when n > p, 0.01 when n < p
4. The default ratio between consecutive lambdas is 0.912
5. H2O uses **warm-starting** at each lambda step (the previous solution initializes the next), which is essential for efficient path computation

The regularization path is directly analogous to glmnet's and HDTweedie's approaches. The number of steps in the path is controlled by `nlambdas` (default 100).

### Extracting the full path

The complete solution path (all lambda values, all coefficient vectors, and deviance explained on train/validation) is accessible via:

```python
import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator

h2o.init()

model = H2OGeneralizedLinearEstimator(
    family="tweedie",
    tweedie_variance_power=1.5,   # fixed power
    tweedie_link_power=0.0,       # log link (power=0)
    alpha=1.0,                    # pure lasso
    lambda_search=True,
    nlambdas=100,
    weights_column="exposure",    # observation weights
    offset_column="log_exposure", # or use offset instead of weights
    nfolds=5,
    seed=42,
)
model.train(x=predictors, y="pure_premium", training_frame=train_h2o)

# Extract full regularization path
path = H2OGeneralizedLinearEstimator.getGLMRegularizationPath(model)
# Returns dict with keys:
# "lambdas"                  - list of lambda values
# "explained_deviance_train" - deviance explained per lambda on train
# "explained_deviance_valid" - deviance explained per lambda on validation
# "coefficients"             - list of dicts {feature_name: coef_value} per lambda
# "coefficients_std"         - standardized coefficients per lambda
```

This is the Python equivalent of `glmnet`'s coefficient matrix extraction and substantially richer than what you can get out of glum (which provides only the CV-selected model, not the full path).

### Lambda selection

H2O selects the optimal lambda based on (in order of precedence):
1. Cross-validation performance (if `nfolds` is specified)
2. Validation frame performance (if `validation_frame` is specified)
3. Training data performance (fallback)

The "best" model coefficients are at the selected lambda. The `makeGLMModel` function can reconstruct a model at any point along the path.

---

## 3. Cross-Validation Integration with Lambda Search

When `lambda_search=True` and `nfolds=N` are both set:

- H2O fits N cross-validation models **at every lambda step** along the path
- The best lambda is selected by the CV metric (default: deviance for Tweedie)
- This is the correct procedure: it avoids the information leak that would occur from selecting lambda on training deviance
- All cross-validation fold models are trained in parallel on the H2O cluster

This is functionally equivalent to glmnet's `cv.glmnet()` or HDTweedie's `HDtweedie.cv()`. The implementation is sound.

One caveat: H2O's CV with lambda search is **expensive**. For a dataset with n observations, p predictors, N folds, and K lambda values, the total fits are N*K GLM models. With 5 folds and 100 lambdas, that is 500 model fits. For small datasets (insurance pricing, n = tens of thousands), this is still fast — but the JVM initialization and data transfer overhead dominates for small datasets.

---

## 4. Fixed Tweedie Power Parameter

### Can you fix tweedie_variance_power?

**Yes, completely.** This is a direct parameter:

```python
model = H2OGeneralizedLinearEstimator(
    family="tweedie",
    tweedie_variance_power=1.5,  # your previously estimated p
    tweedie_link_power=0.0,      # 0 = log link
    ...
)
```

The parameter `tweedie_variance_power` accepts any valid value (p <= 0, p >= 1, excluding (0,1)). For insurance compound Poisson models, 1 < p < 2.

There is also a `fix_tweedie_variance_power` boolean (mentioned in Sparkling Water documentation). In standard H2O Python API, simply specifying `tweedie_variance_power` fixes it to that value by default. H2O does **not** estimate the variance power automatically unless you explicitly set up a grid search over it.

### Estimating the dispersion parameter

H2O separates variance power (p) from the dispersion parameter (phi):
- `tweedie_variance_power` (p) is fixed by the user
- The dispersion parameter can be estimated via `dispersion_parameter_method`: `"pearson"`, `"deviance"`, or `"ml"` (maximum likelihood)
- `"ml"` is recommended — it uses Newton's method with a golden section search for the learning rate

For inference workflows where p was already estimated in R, simply pass `tweedie_variance_power=p_estimated_in_R` and treat it as fixed. H2O will not re-estimate it.

---

## 5. Coefficient Extraction

### Single model coefficients

```python
# Coefficients on original scale (after de-standardization)
coefs = model.coef()           # dict: {feature_name: coefficient}

# Coefficients table with standard errors, z-values, p-values
model.coef_table()             # H2OTwoDimTable

# Convert to pandas
coef_df = model.coef_table().as_data_frame()
```

H2O GLM coefficients are reported on the **original (unstandardized) scale** by default, even when `standardize=True` (the default). The internal optimization uses standardized features for numerical stability, and H2O back-transforms the coefficients automatically.

### Full path matrix

```python
path = H2OGeneralizedLinearEstimator.getGLMRegularizationPath(model)

import pandas as pd
# Build coefficient matrix: rows = lambdas, columns = features
coef_matrix = pd.DataFrame(path["coefficients"])
coef_matrix.index = path["lambdas"]
```

This gives you the complete solution path in a structure directly comparable to HDTweedie's path output.

---

## 6. Observation Weights and Offsets

H2O GLM fully supports both:

```python
model = H2OGeneralizedLinearEstimator(
    family="tweedie",
    tweedie_variance_power=1.5,
    weights_column="exposure_col",      # per-row exposure weight
    offset_column="log_exposure_col",   # log-offset column
    ...
)
```

**Weights** (`weights_column`): Per-row weights that scale the loss function contribution of each observation. For insurance, this is typically the exposure in policy-years. The weights do not change the data frame structure — H2O uses them only during optimization.

**Offsets** (`offset_column`): Per-row bias terms added to the linear predictor before the link function. For a log-link Tweedie, `offset = log(exposure)` is the standard actuarial convention for exposure-adjusted pure premium models.

**Important constraint:** `weights_column` and `offset_column` cannot reference the same column. If you want log-exposure as offset, store it as a separate column.

**Weights vs offset for insurance:** The two approaches are not numerically equivalent. Exposure-as-weight scales loss contribution; log-exposure-as-offset shifts the linear predictor. For pure premium models, the offset approach is actuarially correct. H2O supports both, so use whichever matches the HDTweedie specification.

---

## 7. Production Deployment

### The Training Environment

Running H2O for training requires:

1. **Java Runtime Environment (JRE 17+):** H2O runs as a JVM process. You cannot avoid this for training.
2. **`h2o.init()`:** Starts a local H2O cluster (single-node JVM). Startup takes approximately 5–15 seconds.
3. **Memory:** H2O recommends 4x the dataset size. For a small insurance pricing dataset (say, 1M rows x 50 features ≈ 400 MB), `h2o.init(max_mem_size="2g")` is sufficient. Default allocation is 25% of system RAM if unspecified.
4. **The Python `h2o` package:** A REST client to the local JVM cluster, not a pure-Python library.

### Docker Image for Training

The official H2O Docker image (`h2oai/h2o-open-source-k8s`) is approximately **1.5 GB** compressed. Compare:
- glum Docker image (Python + dependencies): ~200–300 MB
- R + Plumber Docker image: ~500–800 MB depending on base image and packages
- H2O training image: ~1.5 GB

### Scoring Deployment: The MOJO Path

This is H2O's strongest production argument. A trained GLM can be exported as a **MOJO (Model Object, Optimized)**:

```python
# After training
mojo_path = model.download_mojo(path="/tmp/", get_genmodel_jar=True)
# Produces: /tmp/<model_id>.zip  (the MOJO)
#           /tmp/h2o-genmodel.jar (the scoring runtime)
```

**What a MOJO is:** A self-contained zip file (a few KB for a GLM) encoding the model structure and learned parameters.

**What h2o-genmodel.jar is:** A standalone Java library (~10–15 MB) containing all the code needed to load and score a MOJO. It has **no dependency on a running H2O cluster**.

**Java scoring (no H2O cluster required):**

```java
import hex.genmodel.MojoModel;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.prediction.RegressionModelPrediction;

MojoModel mojoModel = MojoModel.load("path/to/model.zip");
EasyPredictModelWrapper wrapper = new EasyPredictModelWrapper(mojoModel);

RowData row = new RowData();
row.put("feature_1", "0.5");
row.put("feature_2", "male");
RegressionModelPrediction pred = wrapper.predictRegression(row);
double score = pred.value;
```

**Python scoring with a running MOJO (still requires H2O cluster):**

When scoring from Python, `model.predict()` requires the H2O cluster to be active. For pure Python scoring *without* a cluster, you must either:
- Call a Java microservice wrapping the MOJO, or
- Use the H2O inference server (commercial, REST-based), or
- Use Sparkling Water's Spark integration

**The practical MOJO deployment pattern for Python shops:**

1. Train in Python using `h2o` during the offline batch training job
2. Export MOJO + `h2o-genmodel.jar`
3. Build a minimal Java Spring Boot or Quarkus scoring service (~50 MB Docker image with Alpine JRE + genmodel.jar + MOJO)
4. Python scoring service calls the Java REST endpoint

This is a two-service architecture. It solves the "no H2O cluster in production" problem but requires your engineering team to maintain a Java microservice. Whether this is better than R+Plumber depends on your team's Java comfort level.

### Docker Sizing Summary

| Deployment Pattern | Approx. Docker Image Size |
|---|---|
| H2O training image (full) | ~1.5 GB |
| H2O MOJO scoring (Java Alpine + genmodel.jar) | ~80–120 MB |
| glum (Python + scientific stack) | ~300–400 MB |
| R + Plumber + tidyverse | ~500–800 MB |
| Pure Java MOJO scorer (minimal) | ~50–80 MB |

### MOJO versioning and portability

MOJOs are versioned by H2O version. The `h2o-genmodel.jar` must match the H2O version used to produce the MOJO. MOJOs are **not** cross-version portable. This means you must pin both:
- The H2O Python training package version
- The `h2o-genmodel.jar` version used in the scoring service

Version pinning is essential and somewhat tedious because H2O releases frequently (~every few weeks for patch versions).

---

## 8. H2O vs glum vs HDTweedie: Algorithm Comparison

| Characteristic | HDTweedie (R) | glum (Python) | H2O GLM (Python) |
|---|---|---|---|
| **Algorithm family** | IRLS-BMD | IRLS with CD inner loop | IRLSM (expected Hessian) |
| **Hessian approximation** | True Hessian | True Hessian | Fisher information (expected) |
| **Tweedie objective** | Exact Tweedie deviance | Exact Tweedie deviance | Exact Tweedie deviance |
| **L1 support** | Yes (grouped elastic net) | Yes (pure lasso and elastic net) | Yes (elastic net, alpha=1.0 is lasso) |
| **Solution path** | Full path | CV-selected model only | Full path (lambda_search=True) |
| **Built-in CV** | Yes | Yes | Yes (nfolds + lambda_search) |
| **Fixed power p** | Yes | Yes | Yes |
| **Weights/offsets** | Yes | Yes | Yes |
| **Coefficient extraction** | Path matrix | Single model dict | Path matrix + pandas conversion |
| **Execution model** | Pure R | Pure Python (C extension) | Python REST client → JVM cluster |
| **Cold startup** | Seconds | Milliseconds | 5–15 seconds (JVM init) |
| **Memory footprint** | Process-level | Process-level | JVM + H2O overhead (min ~512 MB) |

### Would coefficients match HDTweedie more or less than glum?

**Glum will match HDTweedie more closely than H2O** on the same dataset at the same lambda value. Both glum and HDTweedie use the true Hessian in the quadratic approximation; H2O uses the expected Hessian. At convergence and given identical lambda values and penalty normalizations, all three should reach the same mathematical optimum (since the optimization objective is the same), but:

- The path between lambda values may differ slightly due to different initialization and step quality
- Numerical precision on edge cases (near-zero coefficients, collinear features) will differ
- For 20–50 predictors, differences will be small in practice

If coefficient-for-coefficient reproducibility vs HDTweedie is a requirement, glum is the better choice. If a full auditable path matrix is needed (which glum does not provide natively), H2O has an edge.

---

## 9. Operational Burden of Running H2O in Production

### For training (batch job)

The operational burden for an offline training job is **moderate and manageable**:

```python
import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator

# Startup (5-15 seconds; can be kept alive across runs)
h2o.init(nthreads=-1, max_mem_size="4g")

# Upload data (fast for small insurance datasets)
train_h2o = h2o.H2OFrame(df_pandas)

# Fit (fast for 20-50 predictors)
model = H2OGeneralizedLinearEstimator(
    family="tweedie",
    tweedie_variance_power=1.5,
    alpha=1.0,
    lambda_search=True,
    nfolds=5,
    weights_column="exposure",
)
model.train(x=feature_cols, y="pure_premium", training_frame=train_h2o)

# Export
model.download_mojo(path="./models/", get_genmodel_jar=True)

# Shutdown (important: prevents JVM zombie processes)
h2o.cluster().shutdown()
```

If this runs in a CI/CD pipeline or scheduled batch job, the JVM overhead is a one-time cost per training run. It is not particularly burdensome.

### JVM tuning

For a small actuarial dataset (< 10M rows, ~50 features):
- `max_mem_size="2g"` is typically sufficient
- `nthreads=-1` uses all CPU cores (appropriate for MCMC-style parallel CV)
- H2O's default JVM GC settings are usually adequate for batch training
- **Do not** rely on H2O's default 25% RAM allocation in containerized environments — always specify `max_mem_size` explicitly to avoid JVM memory sizing surprises inside Docker

### Cluster lifecycle management

H2O's cluster is **stateful** — if a JVM node crashes, the entire cluster must be restarted. For a single-node local deployment (the common case for batch training), this means:
- The training script is responsible for starting and stopping the cluster
- Crashes during training require restarting from scratch (no checkpoint/resume)
- Long-running training + CV jobs (e.g., 5-fold CV on a large path) hold the JVM alive for the full duration

For multi-node distributed deployment on Kubernetes, H2O requires a StatefulSet (not a Deployment) because nodes are not interchangeable. This is a non-trivial Kubernetes configuration requirement.

### Version pinning

H2O has a hard version coupling: the Python client, the JVM jar, and the MOJO scoring JAR must all match the same H2O version. This creates a three-way pinning requirement:
- `pip install h2o==3.46.0.7` in `requirements.txt`
- `h2o-genmodel-3.46.0.7.jar` in the scoring service
- Possibly a pinned Docker base image if using the H2O Docker container for training

H2O binary model files (`.bin`) are version-specific and cannot be loaded by a different H2O version. MOJOs are more stable across patch versions but are still not guaranteed portable across minor versions.

---

## 10. MOJO Export as a Deployment Solution: Assessment

The MOJO export angle is **real but not a complete solution** to the "avoid H2O dependency" problem. Here is a precise characterization:

### What MOJO solves

- **Inference without a running H2O cluster**: The scoring service does not need `h2o.init()`. Only `h2o-genmodel.jar` is required.
- **Low-latency single-row scoring**: MOJO scoring via `EasyPredictModelWrapper` is microseconds per row (the MOJO is essentially a compiled decision graph or, for GLM, a linear evaluation).
- **Smaller deployment artifact**: A GLM MOJO zip + genmodel.jar is ~15 MB total, far smaller than a full H2O Docker image.
- **Java ecosystem integration**: If the engineering team has existing Java services, MOJO scoring slots in naturally.

### What MOJO does NOT solve

- **Training still requires full H2O**: You cannot escape the JVM for the training step.
- **Python-native inference is not cleanly supported**: True pure-Python scoring of an H2O MOJO without a running cluster is not officially supported. The `h2o` Python package's `predict()` method requires the H2O server. There are community wrappers (e.g., `h2o_scorer` on GitHub) but these are not officially maintained.
- **Two-service architecture**: You end up with a Python training service AND a Java scoring service. This is additional operational complexity.
- **Java expertise required**: Someone on the team must be comfortable writing and maintaining a small Java scoring service.

### Comparison to the alternatives

| Deployment Pattern | Training Env | Scoring Env | Ops Complexity |
|---|---|---|---|
| R + HDTweedie + Plumber | R + Docker | R + Docker | Low (one service) |
| glum (Python) | Python | Python | Low (one service) |
| H2O (full cluster scoring) | Python + JVM | Python + JVM | Medium-High |
| H2O (MOJO Java scoring) | Python + JVM | Java (genmodel.jar only) | High (two services, two runtimes) |
| Coefficient export from R | R (offline) | Python (inference only) | Low (no retraining in Python) |

### When MOJO is a genuine game-changer

The MOJO pattern becomes compelling if:
1. Your engineering team already runs Java microservices in production
2. You need sub-millisecond scoring latency (MOJO is faster than a Python model server)
3. You want to avoid Python dependencies in the inference path entirely
4. You are already running H2O for other ML workloads (amortizing the JVM cost)

For a team with **no existing H2O or Java infrastructure**, adopting H2O MOJO export to solve a Tweedie lasso problem adds significant architectural complexity for modest benefit.

---

## 11. H2O Ranking vs the Alternatives

Given the use case:
- 20–50 predictors
- Fixed Tweedie power (previously estimated)
- L1 penalization
- Currently in R/HDTweedie
- Engineering team wants Python
- Insurance pricing (not ultra-low-latency; likely batch or near-real-time pricing)

### Ranking

**1. glum (QuantCo)** — Best pure-Python replacement
- Pure Python (no JVM), installs with `pip`, zero startup overhead
- True Hessian IRLS-CD, closest algorithmic match to HDTweedie
- Supports fixed Tweedie p, L1 lasso, weights, offsets
- Limitation: no native full solution path extraction (only CV-selected model)
- Best for: teams that want a clean Python drop-in with minimal operational footprint

**2. Coefficient export from R (inference only)**
- Exact match to HDTweedie coefficients (same code, same model)
- Zero algorithmic risk
- Limitation: no Python retraining capability
- Best for: teams where actuaries control model fitting and only the scoring API needs Python

**3. H2O (Python API + MOJO export)**
- Fully featured: full solution path, integrated CV, weights, offsets, fixed p
- MOJO export solves production scoring without a cluster
- Limitation: JVM dependency for training, 1.5 GB Docker image, two-service architecture
- Best for: teams already running H2O, or teams with Java scoring infrastructure

**4. R + Plumber + Docker**
- No algorithmic risk (exact HDTweedie fit)
- Single Docker service
- Limitation: engineering team must accept R in the serving stack
- Best for: teams where Python is a preference, not a hard requirement

### Is H2O better than glum for this use case?

**No, not for this specific use case.** The key reasons:

1. For 20–50 predictors, glum is faster and has no JVM overhead
2. glum's algorithm is a closer match to HDTweedie (true Hessian), which matters if coefficient reproducibility is important
3. glum has zero startup latency; H2O has 5–15 second JVM initialization
4. glum's Docker image is ~5x smaller
5. The primary advantage H2O has — the full solution path via `getGLMRegularizationPath` — matters less if the engineering team only cares about the CV-selected model for production scoring

H2O would be preferable over glum only if:
- You need the full solution path matrix in Python (glum does not expose it)
- You have existing H2O infrastructure
- The engineering team has Java scoring services and wants MOJO deployment

### Is the JVM dependency a dealbreaker?

For training (batch job), **no** — the JVM overhead is a one-time cost per training run and manageable with `h2o.init()` / `h2o.cluster().shutdown()`.

For inference (real-time scoring), **yes** — you do not want a full H2O cluster in the scoring path. The MOJO pattern solves this, but adds Java dependency.

---

## 12. Concrete Python Code Example

```python
import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator
import pandas as pd

# -----------------------------------------------------------------
# 1. Start H2O cluster (single-node local)
# -----------------------------------------------------------------
h2o.init(
    nthreads=-1,
    max_mem_size="2g",   # sufficient for small insurance datasets
    port=54321,
    strict_version_check=False,
)

# -----------------------------------------------------------------
# 2. Upload data
# -----------------------------------------------------------------
df = pd.read_csv("claims_data.csv")
train_h2o = h2o.H2OFrame(df)

feature_cols = [c for c in df.columns if c not in ["pure_premium", "exposure", "log_exposure"]]

# -----------------------------------------------------------------
# 3. Fit Tweedie GLM with lasso and lambda search
# -----------------------------------------------------------------
model = H2OGeneralizedLinearEstimator(
    family="tweedie",
    tweedie_variance_power=1.5,   # previously estimated p; fixed
    tweedie_link_power=0.0,       # 0.0 = log link
    alpha=1.0,                    # pure L1 lasso
    lambda_search=True,
    nlambdas=100,
    lambda_min_ratio=-1,          # adaptive (default)
    standardize=True,             # standardize features internally
    nfolds=5,
    fold_assignment="Random",
    keep_cross_validation_predictions=True,
    weights_column="exposure",    # per-row exposure weight
    # offset_column="log_exposure",  # alternative: use offset instead
    seed=42,
    solver="AUTO",                # AUTO selects IRLSM for narrow data
)
model.train(x=feature_cols, y="pure_premium", training_frame=train_h2o)

# -----------------------------------------------------------------
# 4. Inspect results
# -----------------------------------------------------------------
print("Best lambda:", model.actual_params["lambda"])
print("Deviance explained:", model.residual_deviance() / model.null_deviance())

# Coefficient table
coef_df = model.coef_table().as_data_frame()
print(coef_df)

# -----------------------------------------------------------------
# 5. Extract full regularization path
# -----------------------------------------------------------------
path = H2OGeneralizedLinearEstimator.getGLMRegularizationPath(model)
coef_matrix = pd.DataFrame(path["coefficients"])
coef_matrix.index = path["lambdas"]
deviance_df = pd.DataFrame({
    "lambda": path["lambdas"],
    "deviance_train": path["explained_deviance_train"],
    "deviance_valid": path["explained_deviance_valid"],
})

# -----------------------------------------------------------------
# 6. Export MOJO for production scoring
# -----------------------------------------------------------------
mojo_path = model.download_mojo(path="./models/", get_genmodel_jar=True)
print(f"MOJO saved to: {mojo_path}")
# Files: ./models/<model_id>.zip  (MOJO artifact, ~few KB)
#        ./models/h2o-genmodel.jar (scoring runtime, ~15 MB)

# -----------------------------------------------------------------
# 7. Shutdown
# -----------------------------------------------------------------
h2o.cluster().shutdown()
```

---

## 13. Open Questions and Caveats

1. **Penalty normalization**: H2O's lambda may not be on the same scale as HDTweedie's. The convention for how lambda scales with n and p varies across implementations. Do **not** transfer lambda values numerically from HDTweedie to H2O — re-run CV to find the optimal lambda.

2. **Tweedie deviance metric for CV**: Confirm that H2O uses Tweedie deviance (not MSE or Gaussian deviance) as the CV loss metric. The default for Tweedie family should be deviance, but verify this with `model.get_params()["stopping_metric"]`.

3. **Intercept handling**: H2O fits an intercept by default. HDTweedie's intercept convention may differ. Verify with `model.coef()["Intercept"]`.

4. **h2o-genmodel.jar version lock**: Treat the genmodel JAR as a build artifact that must be versioned alongside the MOJO. Never update one without the other.

5. **Local vs remote cluster**: The examples above use `h2o.init()` to start a local single-node cluster. For cloud batch training, H2O can connect to an existing cluster (`h2o.connect(url=...)`) but this requires H2O server infrastructure to be pre-provisioned.

6. **"Steam" and enterprise deployment**: H2O.ai offers a commercial deployment product called H2O AI Cloud (formerly Steam) for model management and REST deployment. This is separate from the open-source H2O-3 and requires a license. Do not conflate H2O open source with H2O AI Cloud enterprise features.

---

## Sources

- [H2O GLM Documentation (v3.46)](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html)
- [H2O GLM Booklet (PDF)](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/booklets/GLMBooklet.pdf)
- [H2O tweedie_variance_power parameter reference](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/tweedie_variance_power.html)
- [H2O MOJO Quick Start](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/mojo-quickstart.html)
- [H2O Productionizing Guide](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/productionizing.html)
- [H2O Saving and Loading Models](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/save-and-load-model.html)
- [H2O Kubernetes Deployment](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/cloud-integration/kubernetes.html)
- [h2oai/h2o-open-source-k8s Docker Hub](https://hub.docker.com/r/h2oai/h2o-open-source-k8s)
- [h2o-genmodel Maven Repository](https://mvnrepository.com/artifact/ai.h2o/h2o-genmodel)
- [glum motivation and algorithmic background](https://glum.readthedocs.io/en/latest/motivation.html)
- [glum benchmarks vs glmnet and H2O](https://glum.readthedocs.io/en/latest/benchmarks.html)
- [glum algorithm background](https://glum.readthedocs.io/en/latest/background.html)
- [Quantco/glum GitHub](https://github.com/Quantco/glum)
