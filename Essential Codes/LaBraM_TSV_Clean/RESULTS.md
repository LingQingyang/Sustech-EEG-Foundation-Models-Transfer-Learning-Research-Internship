# LaBraM × M3CV: task-conditioned update geometry and six spectra

[Original code guide](README.md) · [Mathematical methods](METHODS.md) · [Output contracts](OUTPUTS.md) · [Research overview](../../README.md)

> **Main finding:** task updates have structured, partially shared operator geometry, but direct projection into a context span does not preserve the candidate task's fitted function. **0 of 250** projected-overlap curves reach 99% of full-model balanced accuracy.

**Source:** `LaBraM_M3CV_Six_Spectra_Academic_Report_20260908.pdf`, by Qingyang Ling, dated 2026-09-08, 22 PDF pages. This adaptation follows its parameter-to-function evidence chain, preserves the reported values, and distinguishes fitted-function retention from generalization. All source page references below refer to PDF page positions, including front matter.

## Study design

Five tasks independently start from the **same pretrained LaBraM-base checkpoint**. Each task has two independent full-fine-tuning replicates, yielding ten models. Tasks are not trained sequentially and their updates are not merged in this experiment.

| Task | Prediction target | Classes | Chance bACC |
|---|---|---:|---:|
| Rest | Eyes Closed vs Eyes Open | 2 | 0.500 |
| Motor | Foot / Right Hand / Left Hand | 3 | 0.333 |
| P300 | Non-target / Target | 2 | 0.500 |
| SSS | SSVEP / SSAEP / SSSEP | 3 | 0.333 |
| TS | VEP / AEP / SEP | 3 | 0.333 |

### Training protocol

Source: PDF pages 6–7, Tables 1–2.

| Item | Fixed setting |
|---|---|
| Model | LaBraM-base: 12 Transformer blocks, hidden dimension 200, 10 attention heads, MLP ratio 4 |
| Data | M3CV Session 1; 64 channels, 4 seconds, 200 Hz |
| Input | `[64, 800]` reshaped to `[64, 4, 200]`; CLS position 0 and EEG positions 1–64 |
| Normalization | Inputs already globally z-scored; no repeated standardization or division by 100 |
| Optimization | AdamW; backbone LR 1e-4, head LR 1e-3, weight decay 0 |
| Adam settings | β1 = 0.9, β2 = 0.999, ε = 1e-8 |
| Loss | Class-weighted cross entropy |
| Batch sizes | Train 32; evaluate 64 |
| Budget | 20 epochs for every final run; two replicates per task |
| Initialization | Shared pretrained origin; fixed model initialization seed 314159 |
| Disabled options | AMP, gradient clipping, scheduler, warm-up, early stopping |
| Evaluation boundary | **Pooled fitted samples; no held-out subject/session evaluation** |

Functional spectra test whether a reconstructed parameter subset preserves the already-fitted solution. They do not estimate unseen-subject accuracy, session transfer, or deployment performance.

## From weight updates to complete rank-one operators

For each task and replicate, compare its adapted parameters with the shared pretrained origin:

$$
\Delta W=W_{\mathrm{adapted}}-W_0=\sum_i\sigma_i u_i v_i^\top,
\qquad Z_i=u_i v_i^\top.
$$

Each $Z_i$ retains both the input and output directions. Comparing only a left or right singular vector would omit half of the update operator. Their Frobenius inner product is

$$
\langle Z_i,Z_j\rangle_F=(u_i^\top u_j)(v_i^\top v_j).
$$

This identity lets the implementation use left/right Gram products without explicitly materializing high-dimensional vectorized matrices.

| Matrix type | Shape | Matrix-space dimension | Maximum SVD rank | Positions |
|---|---|---:|---:|---:|
| Q, K, V, O | 200 × 200 | 40,000 | 200 | 48 |
| fc1 | 800 × 200 | 160,000 | 200 | 12 |
| fc2 | 200 × 800 | 160,000 | 200 | 12 |
| Total | | | | **72 per model** |

Each model has 14,400 rank-one component slots across the 72 matrices. Ten runs produce 720 decompositions and 144,000 slots. This is an analysis capacity, not the full model's parameter-space dimension.

## Heat maps and STI

Relative update energy is $H=\|\Delta W\|_F^2/\|W_0\|_F^2$. Replicates are aggregated by median in linear energy space before applying the log color scale. Heat maps localize adaptation by task, matrix type, and depth; they do not select functional cutoffs.

Singular Task Interference (STI) compares the joint input/output singular structure of five tasks. The rank budget is 40 components per task per matrix. Two replicates per task give 32 empirical replicate combinations. The matched-Haar null preserves matrix shapes, ranks, empirical singular values, rank-one construction, and aggregation level while randomizing orientations.

The report finds **72/72 matrix positions** above the matched-random 99th percentile (PDF page 14). This establishes non-random joint geometry under the specified null. Since no merge-and-evaluate stage was performed, it is not an observed model-merging accuracy drop.

## Context construction and reconstruction boundary

Single Context uses the retained functional operators of one other task. Full Context uses the union of the other four tasks' retained operators. Contexts are constructed separately at the same matrix position and combined as an orthogonal direct sum across the 72 matrices.

For a candidate atom, $p_i=\|P_C Z_i\|_F^2\in[0,1]$. Since each Single Context is contained in Full Context, its projection energy cannot exceed the corresponding Full Context projection energy.

Functional reconstruction begins with the complete adapted candidate checkpoint. Only the 72 analyzed matrices are replaced by $W_0$ plus the selected reconstruction. The fitted task head and all other adapted parameters remain in place. Thus the zero-component baseline resets those 72 matrices; it is not a completely untouched pretrained classifier.

## What the six spectra measure

Source: PDF pages 10–13, equations 12–23 and Table 5.

| Spectrum | Ordered object | Measured quantity | Interpretation |
|---|---|---|---|
| Individual Energy | Matrix-wise SVD rank prefix | Retained squared update norm | Energy concentration |
| Individual Functional | Same SVD prefix | Raw fitted-data bACC after reconstruction | Prefix sufficiency for fitted function |
| Shared Energy | Candidate functional atoms ranked by $\sigma_i^2p_i$ | Absolute cumulative shareable energy | Context expressibility |
| Shared Functional | Same shared-energy order; restore original candidate atoms | Raw fitted-data bACC | Whether shared-energy ranking restores function early |
| Principal Angle | Canonical directions ordered by principal cosine | $\cos\theta_j$ | Alignment of retained operator spans |
| Overlapped Functional | Candidate principal directions projected into context | Raw fitted-data bACC | Whether geometric projection preserves function |

### Essential distinctions

- The Individual rank fraction is applied **per matrix** using a prefix, not a global top-k over 14,400 singular values.
- Individual Energy's threshold is 99% of update energy. Individual Functional's threshold is **0.99 × the same run's full-checkpoint bACC**.
- Shared Energy's endpoint $\Gamma$ is absolute and remains in [0,1]; it is not normalized back to one.
- Shared Functional uses projection only to rank atoms. It writes back **the original candidate atoms**, so its endpoint must equal Individual Functional at the candidate cutoff.
- Overlapped Functional writes **projected directions**, preserving candidate principal coefficients. It is a direct test of functional substitutability rather than a ranking experiment.

## Reported results

### Fitted-task balanced accuracy

Source: PDF page 13, Table 6. Each range spans the task's two independent replicates; values are proportions.

| Task | Replicates | Final fitted-data bACC |
|---|---:|---:|
| Rest | 2 | 0.9849–0.9914 |
| Motor | 2 | 0.9972–0.9986 |
| P300 | 2 | 0.9992–1.0000 |
| SSS | 2 | 0.9953–0.9976 |
| TS | 2 | 0.9960–0.9972 |
| All runs | 10 | **0.9849–1.0000** |

These values show completion of fitted-task optimization and define reconstruction reference scores. They are not held-out performance results.

### Energy versus functional dimension

Source: PDF pages 14–15, Table 8. Component counts and rounded fraction ranges are reproduced as reported.

| Criterion | Reported matrix-wise rank fraction | Reported components per model | Target |
|---|---:|---:|---|
| Individual Energy cutoff | 0.67–0.75 | 9,648–10,800 | 99% update energy |
| Individual Functional cutoff | **0.04–0.22** | **576–3,168** | 99% full-model fitted bACC |

The reported functional range lies well below the energy range. Maintaining the fitted decision function requires a smaller leading operator set than reconstructing almost all update energy. This does not establish that all remaining components are useless under other metrics or evaluation data.

### Shared geometry and the projection test

Source: PDF pages 15–17.

| Observation | Reported finding | Supported conclusion |
|---|---|---|
| STI | 72/72 positions exceed matched-Haar q99 | Joint singular geometry is structured |
| Shared Energy | Real contexts differ systematically from matched-Haar contexts | Candidate functional operators have non-random context expressibility |
| Full Context | All nesting audits pass | Context union increases expressibility consistently with span inclusion |
| Principal Angle | Structured alignment versus matched random; same-task other-replicate reference also shown | Subspaces have non-random geometric overlap |
| Shared Functional | All complete-prefix endpoints match Individual Functional cutoff endpoints | Ranking and reconstruction remain separate |
| Overlapped Functional | **0/250 curves reach 0.99 × full-model bACC** | Direct context projection is insufficient for fitted-function retention under this protocol |

The source does not provide a per-task numerical principal-angle or Shared Energy summary table. This adaptation therefore retains its qualitative comparisons without inventing such values.

### Original numerical and artifact audits

Source: PDF page 14, Table 7. These are **audits reported for the original experiment**, not audits rerun from external numerical artifacts when preparing this repository.

| Audit | Reported status |
|---|---|
| Training run grid | 10/10 complete |
| Matrices analyzed | 72 per run |
| V5 declared artifacts | 7/7 matched |
| Six-spectrum artifacts | 225/226 immutable matches; the remaining item is a permitted mutable log |
| Curated display checksums | All 216 delivered files verified |
| Final six-spectrum figure collection | 31 PNG figures |

SVD reconstruction, operator orthonormality, context nesting, full-model endpoints, and Shared Functional endpoint identities were reported to pass. The source display directory included 113 PNG and 80 CSV files; this repository documents their findings in Markdown and does not upload that result directory.

## Interpretation

The evidence chain has a clear boundary: full fine-tuning fits the five tasks; updates are structured in magnitude and singular geometry; fitted function is more compressible than update energy; some candidate directions overlap with other tasks; **that overlap does not by itself make direct projection function-preserving**.

Possible explanations discussed in the report include coefficient mismatch, task-critical directions missing from context, coordination across matrices, and nonlinear coupling with other adapted parameters. These are hypotheses for follow-up interventions rather than causes established by the 0/250 result alone.

## Proposed follow-up experiments

| Aim | Proposed comparison |
|---|---|
| Coefficients versus directions | Re-optimize coefficients in a fixed context span; compare with direct projection, original candidate prefixes, and matched random spans |
| Function-aware geometry | Weight operators using gradients, Hessian-vector products, or empirical Fisher information |
| Matrix coordination | Reconstruct or ablate attention/MLP and depth groups |
| Positive and negative controls | Same-task replicate projection and functional random-span controls |
| Generalization | Repeat retention curves with held-out subjects or sessions |

The primary threshold, context rank/shape contracts, coefficient budget, validation boundary, and per-run reporting should be fixed before interpreting these follow-ups. Crossings not reached should remain explicitly labeled as such.

## Limitations

1. All functional scores use pooled fitted data; held-out generalization has not been established.
2. Two replicates per task provide only a limited estimate of optimization variance.
3. Operator geometry covers 72 Transformer matrices. The fitted head and other adapted state remain present in inference but are outside the analyzed geometry.
4. Near-degenerate singular values can make individual atom orientations unstable; cumulative and group-level results are safer than interpreting exact component rankings.
5. STI is not an observed merging degradation metric in the absence of actual model merging.
6. Frobenius projection minimizes parameter-space distance and need not preserve logits, loss, or representations.

## Relation to the supplied code

The clean package implements the six spectra in `labram_tsv/spectra.py`, with separate heatmap and STI modules and read-only plotting. The existing README documents preflight, training, audit, spectra, heatmap, STI, and plot commands. `Historical Execution Codes` contains the original earlier scripts and notes and is preserved unchanged.

The source report describes historical numerical runs. `configs/reported_results.json` and `configs/full_analysis.json` expose different clean-package analysis profiles; their random budgets and reference quantiles should be inspected before claiming an exact reproduction of historical q99 results. The report's observed audits and values have not been re-established by rerunning the package here.

## References retained from the report

1. Jiang, Zhao, and Lu. *Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI.* ICLR, 2024.
2. Huang, Hu, Chen, et al. *M3CV: A multi-subject, multi-session, and multi-task database for EEG-based biometrics challenge.* NeuroImage 264:119666, 2022.
3. Gargiulo, Crisostomi, Bucarelli, Scardapane, Silvestri, and Rodolà. *Task Singular Vectors: Reducing Task Interference in Model Merging.* arXiv:2412.00081, version 3, 2025.
4. Hamm and Lee. *Grassmann Discriminant Analysis: A Unifying View on Subspace-Based Learning.* ICML, 2008.
5. Saha, Garg, and Roy. *Gradient Projection Memory for Continual Learning.* ICLR, 2021.
6. Lin, Yang, Fan, and Zhang. *TRGP: Trust Region Gradient Projection for Continual Learning.* ICLR, 2022.
