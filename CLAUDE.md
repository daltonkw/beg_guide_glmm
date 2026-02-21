# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-directed learning project on Generalized Linear Mixed Models (GLMMs) for non-life (casualty) insurance pricing. Bridges actuarial credibility theory with modern mixed-effects modeling. Organized as nine modules progressing from GLM/credibility foundations through Bayesian hierarchical models. See `Context/project_description.md` for full module descriptions and `Context/syllabus.md` for the condensed syllabus.

## Language & Environment Setup

This is a **polyglot project** using R (primary), Python, and Mathematica:

- **R 4.5** — managed by `rv` (R version/environment manager). `rproject.toml` declares R dependencies. The `rv/` directory contains the project-local R library. `.Rprofile` activates `rv` on R startup.
- **Python 3.12** — managed by `uv`. `pyproject.toml` declares Python dependencies (PyMC, ArviZ, etc.). Virtual env in `.venv/`.
- **Mathematica** — `Notebooks/ml_and_reml_optimization.nb` (Wolfram notebook for ML/REML optimization theory).

### Installing dependencies

```bash
# R dependencies (rv must be installed separately)
# rv reads rproject.toml and installs into rv/library/
rv add

# Python dependencies
uv add
```

### Rendering Quarto notebooks

```bash
quarto render Notebooks/<name>.qmd
```

## Repository Structure

```
Context/          — Project description and syllabus (reference docs for Agents)
Data/raw/         — Source datasets (txt, xls); used by notebooks
Data/             — Also contains synthetic_lmm_data.csv (generated)
Notebooks/PMMA/   — Track 1: "Actuarial Bridge" (following CAS Monograph 14)
Notebooks/Advanced/ — Track 2: "Modern Frontier" (Advanced theory and complex features)
Models/           — Serialized fitted model objects (.rds); gitignored
Models/workcomp/  — Workers' comp model variants (brms/glmmTMB)
R/GLMGLMM_RCode/ — Reference R scripts from Zuur et al. textbook (Chapters 1-7)
R/                — R scripts
Theory/           — PDF reference material (CAS Monograph 14 on mixed models)
```

## Modeling Patterns

- **Frequentist GLMMs**: `glmmTMB` is the primary package. Models use Poisson, negative binomial, or Gamma families with random intercepts/slopes.
- **Bayesian GLMMs**: `brms` (R interface to Stan). Models are saved as `.rds` in `Models/` to avoid expensive refitting. Always check for existing cached models before refitting.
- **Data paths**: Notebooks use relative paths from the project root (e.g., `fs::path("Data", "raw", "Spiders.txt")`). R source files are loaded with `source("R/pred_boxplots_group.r")`.
- **Parallel MCMC**: `brms` models use `future::plan(multicore, workers = 4)` for parallel chains.
- **Datasets**: `CASdatasets` R package provides insurance data (e.g., `usworkcomp`). Raw data files in `Data/raw/` come from Zuur et al. textbook.

## Key R Packages

| Package | Purpose |
|---------|---------|
| `glmmTMB` | Frequentist GLMMs (primary) |
| `brms` | Bayesian GLMMs via Stan |
| `lme4` | Linear mixed models |
| `tidybayes` | Tidy posterior extraction |
| `DHARMa` | Residual diagnostics for GLMMs |
| `performance` | Model quality checks |
| `CASdatasets` | Actuarial datasets (from UQAM repo) |
| `bayesrules` | Teaching datasets (cherry blossom) |

## Conventions

- Notebooks (quarto and jupyter) are the primary deliverable format — theory (LaTeX math), code, and narrative together.
- Model objects are gitignored (`Models/` in `.gitignore`). They are cached as `.rds` files and loaded with `readRDS()` to skip long MCMC runs.
- The `R/GLMGLMM_RCode/` scripts are reference code from the Zuur textbook — not project-original code.
- For R code always use the Tidyverse style guide https://style.tidyverse.org/
- For Python code always use the google style guide https://google.github.io/styleguide/pyguide.html
