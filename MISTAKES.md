# MISTAKES.md — Lessons Learned

## Paper-Specific Lessons (STP-Diff)

### 1. Temperature-Scaled Softmax — Multiplication vs Division
`mammalian_cell_cycle_attack.py` used `torch.softmax(theta * tau, dim=1)` — multiplication instead of division. With τ=10.0, multiplication *sharpens* the softmax (opposite of the intended smoothing effect). Fix: use `theta / tau`.
- **Affected files:** `scripts/mammalian_cell_cycle_attack.py`, `STP-GNN/Research_STP/scripts/mammalian_cell_cycle_attack.py`

### 2. No Random Seeds = Zero Reproducibility
All experiment scripts used `torch.randn(...)` without `torch.manual_seed()`. Every run produced different ε_critical, top-5 rankings, and knock-in probabilities. Fix: add `torch.manual_seed(42)` + `np.random.seed(42)` to every experiment script.
- **Design rule:** Every script that uses random initialization MUST set a fixed seed at the top.

### 3. Precision Overstated in Paper
ε_critical was reported as `2.8000` (four decimal places) but the binary search only guarantees ±0.01 precision (tolerance parameter). Report what the algorithm can actually deliver.
- **Rule:** Round to match the measurement precision. `≈2.80` not `=2.8000`.

### 4. Visualization vs Experimental Data
The attack dosage plot was a hardcoded sigmoid (`stability = 1/(1+exp(3*(x-2.80)))`), not generated from actual experiment output. This is misleading when presented without disclosure.
- **Fix:** Label synthetic figures as "Schematic" or generate from actual data.

### 5. Claimed Speedup Without Benchmark Code
The 353× speedup was reported as "measured" but no benchmark code actually measures the ImplicitSTPFunction against explicit STP at N=20. The existing benchmark script tests an MLP (unrelated).
- **Rule:** Every numerical claim in a paper must be traceable to executable code that produces it.

### 6. Temperature Annealing Described But Not Implemented
The paper described annealing τ→0 post-optimization, but zero code implements τ scheduling or annealing loops.
- **Rule:** Don't describe features in the paper that don't exist in the code.

### 7. DepMap p-value Not Computed in Code
The paper claimed `p < 10^-6` but the script only compared means and never called `ttest_ind`. The statistical test must match what the paper reports.
- **Fix:** Added `stats.ttest_ind(rb_wt['E2F'], rb_loss['E2F'], equal_var=False)` to `cell_cycle_depmap_validation.py`.

### 8. "First Demonstration" Claims
Absolute novelty claims ("the first demonstration") should always include "to our knowledge" or similar qualifier — there may be non-arxiv, non-English, or unpublished prior work.

### 9. Author Name Errors in references.bib
- Geng, Chuqiao → Chuqin Geng (LogicXGNN)
- Zhang, Xu → Xiao Zhang (Cheng STP+CNN 2025)
- **Rule:** Always verify author names against the actual paper, not secondary sources.

### 10. Missing Related Work Citations
- Mazumdar 2020 (arXiv:2008.08546) — STP Boolean control networks — directly relevant, not cited
- Cheng et al. 2023 (arXiv:2303.06295) — STP hypermatrices — extends STP foundations
