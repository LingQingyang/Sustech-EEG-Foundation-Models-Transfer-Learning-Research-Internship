# ds007554: static geometry, slow structure, and semantic retention

[Code and launch guide](README.md) · [Research overview](../../README.md)

**Source:** `ds007554 embedding analysis report.pdf`, 15 pages. This Markdown adaptation preserves the reported pilot results and their limitations. Numerical values come from presentation charts and tables rather than a new experiment run.

## Research question and pilot scope

The report diagnoses prerequisites for modeling Rest → Task transitions with Flow Matching. It asks whether state information exists in instantaneous embedding geometry, whether temporally slow variables can be recovered, and how much state discrimination survives inside those variables.

| Item | Setting |
|---|---|
| Dataset | ds007554; three sessions in one visit, seven task conditions |
| Five-state selection | Baseline, mental arithmetic (MA), N-back (NB), combined NBMA, Full |
| Frozen models | BIOT, LaBraM, CBraMod, EEGMamba, EEGPT |
| Outer split | Leave one session out within a subject: two sessions train, one tests |
| Reported pilot | **Three subjects:** sub-009, sub-025, sub-026 |
| Completed configurations | **171 folds**, pooled across model/rank/session settings |
| Readout scenarios | `context2`: Baseline vs pooled Task; `five_state`: five classes; `task4`: four task identities |
| Chance bACC | 0.50, 0.20, and 0.25 respectively |

The 171 fold configurations are not 171 independent subjects. Subject-level inference in this pilot has **n = 3**. The report calls for a full 22–24-subject rerun before stronger conclusions.

## A: static discriminant geometry

The presentation compares LD scores, QD scores, LD plus a quadratic residual, and an added LD/residual outer-product interaction. The readout and session splits are held fixed while the feature representation changes.

### Reported five-state bACC

Source: slide 6. Values are proportions, not percentages.

| Arm | Training bACC | Held-out bACC | Train minus held-out |
|---|---:|---:|---:|
| F0: LD | 0.520 | 0.245 | 0.275 |
| F1: QD | 0.601 | 0.244 | 0.357 |
| F2: LD + residual | 0.592 | 0.243 | 0.349 |
| F3: tensor interaction | 0.611 | 0.243 | 0.368 |

The held-out curve is nearly flat while the training scores increase. The reported F2–F0 gain is approximately zero for five-state classification and negative for `context2`. The presentation reports above-chance effects of +0.094 for `context2` and +0.020 for `task4`, with a median task4 permutation p-value of 0.375. The original slide does not assign those two headline effects to a specific arm in its text.

This pilot does not establish a robust task-identity representation in static LD/QD readouts. It also does not prove that all static information is absent: Baseline-vs-Task discrimination remains measurable, and other nonlinear readouts were not exhausted.

### Residual-definition note

Slide 5 describes orthogonalizing QD scores against LD scores, $\tilde q=q-\Pi_{\mathrm{lin}}q$. The supplied `ld_qd_feature_core_v5_0.py` instead implements $r=q-\ell$ and scales LD and residual blocks separately. The bundled C script describes the same score-subtraction construction.

These definitions are not generally equivalent. The table above remains a transcription of the presentation; it should not be treated as a newly verified reproduction of the exact bundled implementation. Resolving the original run version and metadata is necessary before making an exact method-to-number claim. The original code has been left unchanged.

## B: label-free slow-feature objective

For a standardized coordinate, define normalized jump energy

$$
r_{\mathrm{jump}}=\frac{\mathbb E[(x_{t+1}-x_t)^2]}{\operatorname{Var}(x)}.
$$

For stationary unit-variance independent samples, the reference is 2, with lag-one autocorrelation $\rho_1=1-r_{\mathrm{jump}}/2$. Slow-feature analysis minimizes derivative energy relative to signal variance through a generalized eigenvalue problem.

Five within-state systems and one Global system are fitted. State labels stratify within-state systems and temporal-order nulls; the Global SFA objective does not optimize a state classifier. The temporal-order null shuffles observations within the same state and has reported median jump energy 2.01–2.12. B freezes its Global handoff before C.

### Slowest-axis jump energy at M = 200

Source: slide 9. Approximate autocorrelation is derived from the rounded jump-energy values.

| System | Jump energy | Approximate lag-one autocorrelation |
|---|---:|---:|
| Global | 0.34 | 0.83 |
| MA | 0.53 | 0.74 |
| NB | 0.62 | 0.69 |
| NBMA | 0.66 | 0.67 |
| Full | 0.67 | 0.67 |
| Baseline | 1.86 | 0.07 |

Task-state systems are significant in roughly 75%–100% of folds against the temporal-order null, whereas Baseline is significant in only 20%–30%. **Baseline has only about 200 training windows versus 382 per task state.** At M ≥ 200, covariance identifiability is a material limitation; an absent detectable slow signal cannot be distinguished from insufficient data.

At M = 500, task-state slowest-axis jump energy rises to 1.63–1.84, while Global remains around 0.58. This dimension sensitivity is part of the reported result.

### Global versus within-state slow subspaces

Source: slide 10. Projection overlap is reported as shown in the presentation.

| State | Mean principal angle to Global | Projection overlap |
|---|---:|---:|
| Baseline | 63.1° | 0.27 |
| MA | 57.9° | 0.33 |
| NB | 57.6° | 0.34 |
| NBMA | 56.1° | 0.36 |
| Full | 56.8° | 0.35 |

| Cross-fold stability summary | Global | Within-state average |
|---|---:|---:|
| Stability factor shown after random-overlap correction | 4.68× | 2.62× |
| Mean principal angle | 52.1° | 63.0° |

The presentation accounts for the expected $20/M$ overlap between random 20-dimensional subspaces in an M-dimensional ambient space. Its results support the descriptive observation that Global slow axes differ from each state's internal slow axes and are more reproducible across sessions. Interpreting those axes specifically as state-transition dynamics remains a hypothesis subject to the confounds below.

## C: readout of the frozen Global slow space

C repeats semantic discrimination in approximately **23 slow axes**; the report's average effective dimension is **22.8**. Controls are a rank-matched PCA prefix, a fast-axis suffix, and the full reduced embedding space. B's maps are frozen and cannot be refitted by C.

The retention ratio is chance-corrected:

$$
\mathrm{retention}=\frac{\mathrm{bACC}_{\mathrm{slow}}-\mathrm{chance}}{\mathrm{bACC}_{\mathrm{ambient}}-\mathrm{chance}}.
$$

### Reported above-chance linear bACC

Source: slide 13; pooled over 171 fold configurations. These are **bACC minus chance**, not raw bACC. At `context2`, for example, 0.095 corresponds to raw bACC approximately 0.595.

| Scenario | Slow prefix (~23 axes) | PCA prefix | Fast suffix | Ambient space | Reported slow/ambient retention |
|---|---:|---:|---:|---:|---:|
| context2 | 0.095 | 0.077 | 0.006 | 0.095 | 1.01 |
| five_state | 0.038 | 0.028 | 0.007 | 0.046 | 0.84 |
| task4 | 0.008 | 0.004 | 0.006 | 0.020 | 0.38 |

Retention ratios are reproduced from the slide rather than recomputed from rounded bars. A ratio slightly above one does not mean more than 100% of all information was measured; it is a ratio of empirical chance-corrected accuracy estimates.

The report gives a training–held-out gap reduction from **0.191 to 0.098** without loss of the headline held-out discrimination. About 23 axes correspond to approximately 23%, 11%, 7.6%, and 4.6% of ambient ranks 100, 200, 300, and 500.

The strongest pilot observation is that a small Global slow subspace retains Baseline-vs-Task discrimination better than the displayed PCA and fast controls. Task4 gains remain small, and ratios based on near-zero excess accuracy require particular care.

## Integrated interpretation and next experiments

The pilot finds limited incremental value from the tested static quadratic/tensor readouts, measurable temporal structure, and a compact Global slow space that preserves the coarse Baseline-vs-Task contrast. It motivates a low-dimensional transition model; it does not demonstrate a completed task-conditional Flow Matching model.

The source identifies LaBraM and EEGPT as candidate models for a later experiment D. Those are proposals from the original report, not comparative conclusions from a new run of this repository.

## Limitations preserved from the report

1. **Three subjects:** two drive much of the observed effect; sub-009 is weaker. The 171 configurations do not increase subject-level n beyond three.
2. **Fast-control rank mismatch:** only 3.5% of folds exactly match the slow-prefix rank. A strict equal-rank rerun of C is required to separate slowness from capacity.
3. **Baseline sample size:** roughly 200 training windows make high-dimensional covariance estimation difficult. The source proposes reducing stride to increase the window count, while retaining the need to account for dependence between overlapping windows.
4. **State semantics are confounded:** Baseline occurs at file start, and tasks include periodic auditory stimulation. In this design, Rest→Task is confounded with no-sound→sound and potentially file position.
5. **Method-version alignment:** the orthogonalized residual described in the presentation differs from score subtraction in the supplied core. The original run metadata is needed for exact reproduction.

The current findings support a carefully bounded pilot narrative, with full-cohort validation, equal-rank controls, and stimulus-aware conditions needed before stronger physiological or transfer-learning claims.
