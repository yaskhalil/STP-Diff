# PROJECT — STP-Diff Research

## What
A computational framework for identifying regulatory vulnerabilities in Gene Regulatory Networks (GRNs) using differentiable Semi-Tensor Product mappings. Transforms discrete Boolean logic into differentiable manifolds for gradient-based adversarial optimization in systems biology.

Targeting publication.

## Paper
**Title:** STP-Diff: Differentiable Adversarial Discovery of Regulatory Vulnerabilities in Gene Circuits via Semi-Tensor Product Mapping

Key results:
- 1,637× measured speedup over explicit STP at N=15 (theoretical bound 52,429× at N=20)
- Identified Rb-E2F axis as primary vulnerability bottleneck in mammalian cell cycle
- Verified against DepMap Public 25Q3 CRISPR knockout data across 1000+ cell lines
- ε_critical ≈ 2.80 for p53-Mdm2 circuit

## Repo
- **Main paper + experiments:** [`github.com/yaskhalil/STP-Diff`](https://github.com/yaskhalil/STP-Diff) (this repo)
- **PyTorch engine:** `./STP-GNN/` — separate git repo (github.com/yaskhalil/STP-GNN)

## Structure
```
Research_STP/
├── papers/
│   ├── stp-diff-paper.tex           — LaTeX source
│   ├── stp-diff-paper.md            — Markdown export
│   ├── stp-diff-grn-vulnerability.md
│   └── references.bib
├── scripts/                         — 10 Python scripts
│   ├── STP_Diff_Research.py         — Core VJP engine
│   ├── run_adversarial_pipeline.py  — Main PGD pipeline
│   ├── mammalian_cell_cycle_attack.py
│   ├── cell_cycle_depmap_validation.py
│   ├── p53_stp_network.py
│   ├── stp_gradient_benchmarking.py
│   ├── attractor_collapse_viz.py
│   ├── generate_research_viz.py
│   ├── DepMap_Validation.py
│   └── n20_depmap_validation.py
├── outputs/                         — Generated figures (PNG)
├── draft.md                         — Working draft
├── cited.md                         — Draft with citations
├── AGENTS.md                        — Agent instructions
├── PROJECT.md                       — This file
├── MISTAKES.md                      — Lessons learned
└── STP-GNN/                         — PyTorch engine (separate repo)
```

## Commands
- `python scripts/run_adversarial_pipeline.py` — full p53-Mdm2 PGD attack pipeline
- `python scripts/mammalian_cell_cycle_attack.py` — 10-node cell cycle attack
- `python scripts/STP_Diff_Research.py` — core engine (research suite)
- `python scripts/stp_speedup_benchmark.py` — speedup benchmark (see §4.1)

## Random Seeds
All experiment scripts now set `torch.manual_seed(42)` + `np.random.seed(42)` for reproducibility.
