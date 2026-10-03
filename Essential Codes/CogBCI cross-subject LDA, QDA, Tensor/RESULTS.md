# CogBCI cross-subject discriminant geometry: results

[Code and usage](README.md) · [Research overview](../../README.md)

**Question:** does class-specific covariance add transferable brain-state information beyond linear discriminant position, and do interactions between the two provide a further gain?

**Source:** local report `Cross-Subjects LDA与QDA结果.pdf`, 8 pages. Methods are adapted from pages 1–4, 6, and 8; all seven numerical result tables are transcribed from the embedded table images on pages 5–7. Values document the original report and have not been recomputed.

## Experimental setting

| Item | Setting |
|---|---|
| Dataset | CogBCI, all sessions, five-second windows |
| States | Rest-EO; NBack-0/1/2; MATB-easy/medium/difficult |
| Foundation models | BIOT, LaBraM, CBraMod, EEGPT, EEGMamba |
| Embedding dimension | BIOT: all 256 dimensions; other models: reduced to 500 |
| Outer evaluation | Four folds holding out complete subjects |
| Training boundary | Centering, SVD, class statistics, covariance shrinkage, feature scaling, ridge selection, and readout fitting use training subjects only |
| Hyperparameter selection | Nested subject-heldout validation inside each outer-training split |
| Readout | Class-balanced multiclass ridge regression, identical family across all arms |
| Metric | Balanced accuracy (bACC); subject-level results are also computed |

## Feature construction

Let $\ell(x)$ and $q(x)$ denote LDA and QDA scores relative to a reference class. Define the score correction $r(x)=q(x)-\ell(x)$. With training-fitted standardization $z(\cdot)$:

| Arm | Feature map | Global seven-state dimensions | Dedicated three-state dimensions |
|---|---|---:|---:|
| F0: LD | $z(\ell)$ | 6 | 2 |
| F1: QD | $z(q)$ | 6 | 2 |
| F2: LD + QD residual | $z(\ell)\oplus z(r)$ | 12 | 4 |
| F3: tensor interaction | $z(\ell)\oplus z(r)\oplus z(\operatorname{vec}(\ell\otimes r))$ | 48 | 8 |

Here $\oplus$ means concatenation. The residual is the complete score change when moving from shared to class-specific covariance; it is **not a strictly pure quadratic polynomial term**. F3 adds explicit interactions and uses the same linear readout family rather than a separate neural-network classifier.

F1 versus F0 tests score replacement. F2 versus F0 tests the incremental correction while retaining LD information. F3 versus F2 tests the added interaction features.

## Evaluation scenarios

- **Global seven-state geometry:** fit LD/QD on all seven states. Use it for full seven-class classification or fit a new N-Back-only/MATB-only three-class readout.
- **Dedicated geometry:** fit LD/QD and the readout within a single task family. The discriminant scores have two dimensions per arm before residual and tensor expansion.
- **Strict audit:** retain the full seven-class readout and evaluate only true N-Back or MATB samples. A prediction outside the true family is an error. These scores differ from the separately fitted three-class readouts.

## Reported numerical results

Every cell below is **bACC in percent, with the original report's ± term preserved**. The PDF tables do not explicitly identify that term as a standard deviation, standard error, or confidence interval. It is therefore left unlabeled. The source also displays rounded gains relative to F0; the tables here reproduce the bACC values directly, so differences of rounded means may differ slightly from those gains.

### Global geometry → seven-class classification

Chance bACC: $1/7\approx14.29\%$.

| Model | F0: LD | F1: QD | F2: LD + residual | F3: tensor |
|---|---:|---:|---:|---:|
| BIOT | 38.90 ± 1.91 | 38.74 ± 1.55 | 38.91 ± 1.85 | 38.97 ± 1.94 |
| LaBraM | 37.81 ± 1.31 | 37.71 ± 1.19 | 37.66 ± 0.70 | 37.50 ± 0.72 |
| CBraMod | 39.16 ± 1.24 | 39.45 ± 1.33 | 39.08 ± 1.13 | 39.05 ± 1.27 |
| EEGPT | 34.56 ± 1.52 | 36.34 ± 1.49 | 34.83 ± 1.52 | 34.82 ± 1.48 |
| EEGMamba | 35.69 ± 0.93 | 36.45 ± 1.09 | 36.81 ± 1.05 | 36.93 ± 0.97 |

### Global geometry → N-Back three-class readout

Chance bACC: $1/3\approx33.33\%$.

| Model | F0 | F1 | F2 | F3 |
|---|---:|---:|---:|---:|
| BIOT | 39.47 ± 1.70 | 38.38 ± 0.77 | 39.43 ± 1.60 | 39.14 ± 1.64 |
| LaBraM | 36.99 ± 1.04 | 37.15 ± 1.05 | 36.82 ± 0.87 | 36.23 ± 0.71 |
| CBraMod | 35.67 ± 0.47 | 35.85 ± 0.76 | 35.84 ± 0.79 | 35.72 ± 0.89 |
| EEGPT | 37.54 ± 1.16 | 36.50 ± 0.86 | 37.46 ± 1.03 | 37.69 ± 0.89 |
| EEGMamba | 35.12 ± 0.61 | 35.75 ± 0.36 | 35.10 ± 0.70 | 35.32 ± 0.55 |

### Global geometry → MATB three-class readout

Chance bACC: $1/3\approx33.33\%$.

| Model | F0 | F1 | F2 | F3 |
|---|---:|---:|---:|---:|
| BIOT | 55.03 ± 0.95 | 52.72 ± 0.90 | 54.89 ± 1.26 | 55.05 ± 1.40 |
| LaBraM | 54.42 ± 1.35 | 55.04 ± 1.44 | 53.63 ± 0.85 | 55.11 ± 1.35 |
| CBraMod | 55.92 ± 0.79 | 56.13 ± 0.71 | 55.70 ± 0.96 | 55.67 ± 0.87 |
| EEGPT | 51.73 ± 1.52 | 52.77 ± 0.79 | 51.67 ± 0.76 | 51.33 ± 0.71 |
| EEGMamba | 51.67 ± 0.78 | 53.55 ± 1.03 | 52.45 ± 0.80 | 52.77 ± 0.88 |

### Dedicated geometry → N-Back three-class readout

| Model | F0 | F1 | F2 | F3 |
|---|---:|---:|---:|---:|
| BIOT | 39.24 ± 1.84 | 39.28 ± 1.86 | 39.15 ± 1.81 | 39.20 ± 1.82 |
| LaBraM | 36.33 ± 0.47 | 36.20 ± 0.68 | 36.12 ± 0.59 | 36.10 ± 0.47 |
| CBraMod | 35.98 ± 0.63 | 36.27 ± 0.71 | 36.47 ± 0.78 | 36.37 ± 0.76 |
| EEGPT | 36.68 ± 0.75 | 36.53 ± 0.84 | 36.63 ± 0.69 | 36.82 ± 0.79 |
| EEGMamba | 34.40 ± 0.39 | 35.64 ± 0.73 | 34.62 ± 0.37 | 34.93 ± 0.59 |

### Dedicated geometry → MATB three-class readout

| Model | F0 | F1 | F2 | F3 |
|---|---:|---:|---:|---:|
| BIOT | 55.57 ± 1.10 | 55.87 ± 1.15 | 55.66 ± 1.23 | 55.63 ± 1.26 |
| LaBraM | 54.17 ± 1.40 | 54.68 ± 1.38 | 54.35 ± 1.10 | 54.68 ± 1.11 |
| CBraMod | 55.94 ± 0.64 | 56.71 ± 0.68 | 56.66 ± 0.61 | 55.81 ± 0.54 |
| EEGPT | 51.59 ± 0.91 | 52.67 ± 1.24 | 52.90 ± 1.20 | 53.15 ± 1.10 |
| EEGMamba | 52.26 ± 0.68 | 53.64 ± 0.89 | 53.52 ± 0.72 | 53.64 ± 0.94 |

### Strict N-Back audit of the seven-class classifier

The classifier still predicts among all seven states. Under uniform seven-class random prediction, the corresponding correct-label rate is $1/7$, not $1/3$.

| Model | F0 | F1 | F2 | F3 |
|---|---:|---:|---:|---:|
| BIOT | 25.00 ± 1.68 | 24.11 ± 1.65 | 24.60 ± 1.28 | 24.77 ± 1.71 |
| LaBraM | 23.04 ± 2.40 | 24.85 ± 2.12 | 26.97 ± 1.94 | 26.00 ± 2.20 |
| CBraMod | 22.60 ± 1.43 | 24.49 ± 1.87 | 24.04 ± 1.72 | 23.52 ± 1.90 |
| EEGPT | 20.69 ± 2.93 | 19.93 ± 2.33 | 22.34 ± 2.71 | 25.60 ± 3.14 |
| EEGMamba | 23.25 ± 2.39 | 24.25 ± 1.76 | 26.05 ± 1.67 | 24.99 ± 1.66 |

### Strict MATB audit of the seven-class classifier

| Model | F0 | F1 | F2 | F3 |
|---|---:|---:|---:|---:|
| BIOT | 48.27 ± 2.37 | 48.43 ± 1.04 | 47.85 ± 2.25 | 48.31 ± 2.52 |
| LaBraM | 45.97 ± 1.84 | 45.60 ± 2.17 | 46.46 ± 1.52 | 45.89 ± 1.40 |
| CBraMod | 48.62 ± 2.07 | 50.06 ± 1.80 | 49.18 ± 1.74 | 47.14 ± 1.58 |
| EEGPT | 42.68 ± 1.73 | 44.99 ± 1.41 | 46.29 ± 1.19 | 44.69 ± 1.61 |
| EEGMamba | 43.16 ± 2.24 | 43.25 ± 2.48 | 44.14 ± 3.07 | 44.18 ± 2.91 |

## Reading the findings

The following observations are descriptive interpretations of the reported tables:

1. **Covariance effects depend on the model.** In global seven-class evaluation, EEGPT benefits from QD score replacement (reported +1.78 percentage points), while EEGMamba benefits from residual and tensor arms (reported +1.12 and +1.24 points). There is no uniform improvement across all models.
2. **Task families differ.** MATB three-class bACC is approximately 51%–57%, while N-Back is approximately 34%–39%. Dedicated geometry changes individual cells but does not produce a consistent large gain over global geometry.
3. **Strict audits expose family errors.** Strict N-Back scores are substantially below the separate N-Back three-class readouts. The strict metric requires both correct family assignment and correct level classification.
4. **Local gains can be larger than overall gains.** The strict N-Back audit reports +3.93 points for LaBraM F2 and +4.92 points for EEGPT F3 relative to F0, without establishing a universal tensor advantage.

## Interpretation limits

The source provides aggregate bACC tables and methods, but no p-value table, explicit ± definition, subject count, or underlying per-subject result files. These differences should not be described as statistically significant on the basis of this adaptation alone. The frozen-embedding analysis characterizes readout geometry; it does not establish improved backbone training or deployment performance.
