# Numerical Methods and Scientific Boundaries

## 1. Update objects

For task (t), replicate (a), and one fixed LaBraM matrix position (m),

\[
\Delta W_{t,a,m}=W_{t,a,m}-W_{0,m}
=U_{t,a,m}\Sigma_{t,a,m}V_{t,a,m}^{\top}
=\sum_i \sigma_i Z_i,
\qquad Z_i=u_i v_i^{\top}.
\]

The immutable geometry atom is the full rank-one matrix operator (Z_i), not a left singular vector or a right singular vector alone. Its Frobenius inner product factorises as

\[
\langle u_i v_i^{\top},u_j v_j^{\top}\rangle_F
=(u_i^{\top}u_j)(v_i^{\top}v_j).
\]

The implementation therefore evaluates the exact operator Gram matrix by Hadamard products and never materialises 40,000- or 160,000-dimensional vectorisations.

All cross-task contexts are constructed **within the same matrix position**. Q in one block is never compared with K, an MLP matrix, or another block.

For a retained fraction (q\in[0,1]), each matrix retains

\[
k_m(q)=\lceil q r_m\rceil,
\]

clipped to ([0,r_m]). Thus the same (q) gives a matrix-wise rank prefix, not a global top-singular-value selection.

## 2. Individual Energy Spectrum

\[
E_{\mathrm{ind}}(q)=
\frac{\sum_m\sum_{i\le k_m(q)}\sigma_{m,i}^2}
{\sum_m\sum_i\sigma_{m,i}^2}.
\]

Energy output points are chosen at absolute vertical targets (0,.05,\ldots,1), with an explicit 0.99 target and exact endpoints. The smallest sampled natural rank breakpoint reaching 0.99 is recorded as `q_energy_99`.

## 3. Individual Functional Spectrum

For functional reconstruction, the complete adapted candidate checkpoint is loaded first. The task head and all adapted backbone parameters outside the analysed 72 matrices remain untouched. Only those 72 matrices are replaced by

\[
W_{0,m}+\sum_{i\le k_m(q)}\sigma_{m,i}Z_{m,i}.
\]

The zero-component baseline therefore means “adapted checkpoint with the analysed 72 matrices reset to (W_0),” not an entirely pretrained classifier.

Let (B_{\mathrm{full}}) be the bACC of the complete adapted candidate checkpoint. The functional target is

\[
B(q)\ge .99B_{\mathrm{full}}.
\]

The implementation uses sparse component-fraction probes to bracket absolute bACC targets, maps target estimates to natural rank breakpoints, and refines the first observed 99% crossing. `q_star` is the first **evaluated** breakpoint reaching the target; it is not presented as an exhaustive search over every possible global component subset. The exact (q=1) reconstruction must reproduce the stored full checkpoint bACC within tolerance.

## 4. Context operator span

For each context run, retain its Individual Functional cutoff (q^*\). At a fixed matrix position, concatenate its retained rank-one operators into a raw basis (C). An eigendecomposition of (C^{\top}C) produces an orthonormal operator basis (Q_C). Numerical rank is selected using the standard dimension-scaled floating-point tolerance unless `ortho_rtol` is explicitly nonzero.

For candidate component (Z_i), exact projection energy is

\[
p_i=\|P_C Z_i\|_F^2=\|Q_C^{\top}Z_i\|_2^2\in[0,1].
\]

Single Context uses one other task. Full Context uses all four other tasks, but still independently at each identical matrix position.

The `reported` profile follows aligned replicate tracks (`rep1→rep1`, `rep2→rep2`). The `full` profile enumerates all directed replicate pairings: four choices per other task for Single Context and all (2^4) Full Context combinations per candidate replicate.

## 5. Shared Energy Spectrum

Candidate components up to its own (q^*\) are ranked globally by

\[
\epsilon_i^{\mathrm{shared}}=\sigma_i^2p_i.
\]

The cumulative curve is

\[
E_{\mathrm{shared}}(M)=
\frac{\sum_{i=1}^{M}\epsilon_{(i)}^{\mathrm{shared}}}
{\sum_{i=1}^{K^*}\sigma_i^2},
\]

and its endpoint

\[
\Gamma=E_{\mathrm{shared}}(K^*)
\]

is deliberately not rescaled to one. Vertical samples are absolute shared-energy targets up to (Gamma), plus its exact endpoint.

The matched random reference preserves matrix shapes, candidate singular values, all retained ranks, and rank-one (uv^\top) structure. It replaces context left/right bases by independent Haar orthonormal bases.

## 6. Shared Functional Spectrum

Shared Functional uses the ordering from Shared Energy but reconstructs the **original candidate components**:

\[
W_{0,m}+\sum_{i\in\text{first }M}\sigma_i Z_i.
\]

It never writes (P_CZ_i). Consequently, the (M=K^*\) endpoint must equal the candidate's Individual Functional result at (q^*\). This endpoint equality is asserted numerically and is the main guard against accidentally mixing Shared Functional with projection-based reconstruction.

## 7. Principal Angle Spectrum

Let (Q_A) be the retained candidate operator basis and (Q_C) the context basis. Singular values of

\[
Q_A^{\top}Q_C=L\,\mathrm{diag}(\cos\theta_j)R^{\top}
\]

give principal-angle cosines. Per-matrix values are pooled and sorted over the whole model. Cross-task curves receive a shape/rank-matched Haar reference. The other replicate of the same task is emitted as a separate `replicate` reference curve.

This spectrum reports geometric alignment only. A large (\cos\theta\) does not establish that the projected parameter update preserves the candidate task function.

## 8. Overlapped Functional Spectrum

For candidate principal direction (a_j=Q_A L_j), its projection into the context span is

\[
P_Ca_j=\cos\theta_j\,Q_C R_j.
\]

The candidate update's coefficient along (a_j) is retained, and projected directions are accumulated in descending (\cos\theta_j) order. These projected matrices are inserted as the 72 analysed updates and evaluated with the same adapted head and non-72 checkpoint state.

Principal Angle and Overlapped Functional must be interpreted together: the first can show apparent overlap while the second shows that projection is insufficient for functional recovery. That disagreement is a substantive negative result, not a reason to delete the principal-angle analysis.

## 9. Heat map

For every run and matrix,

\[
H=\frac{\|\Delta W\|_F^2}{\|W_0\|_F^2}.
\]

Replicate medians are computed in linear energy space. The log heat map applies `log10` only after aggregation. Heat-map calculation is independent of all six spectra.

## 10. Singular Task Interference

At every matrix position, choose one replicate for each of five tasks, yielding (2^5=32) combinations. With (T=5), each task contributes

\[
k=\left\lfloor\frac{\min_t r_t}{T}\right\rfloor
\]

leading singular triplets. After concatenating (U=[U_1,\ldots,U_T]), (V=[V_1,\ldots,V_T]), and (s=[s_1;\ldots;s_T]), the reported raw statistic is

\[
\mathrm{STI}=\left\|(U^{\top}U-I)\,\mathrm{diag}(s)\,(V^{\top}V-I)\right\|_{1,\mathrm{entrywise}}.
\]

For every matrix, each matched-null draw samples one of the 32 empirical replicate combinations uniformly, preserves its five singular-value vectors, randomises only U/V orientations with independent Haar bases, and recomputes STI. The random distribution therefore matches the aggregation level of the reported matrix statistic without a redundant 32× Monte Carlo expansion.

## 11. Functional-evaluation boundary

Training and every functional reconstruction use all pooled fitted samples. They answer whether a parameter subset or projection retains the fitted task solution. They do not estimate subject-independent, session-independent, or held-out predictive performance, and the pipeline performs no model merging.
