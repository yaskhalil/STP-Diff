# STP-Diff

**Differentiable Adversarial Discovery of Regulatory Vulnerabilities in Gene Circuits via Semi-Tensor Product Mapping**

STP-Diff is a framework that transforms discrete Boolean network dynamics into a differentiable manifold via Semi-Tensor Product (STP) mapping with an implicit Vector-Jacobian Product (VJP) operator. This enables gradient-based adversarial optimization to discover regulatory vulnerabilities in biological gene circuits.

## Paper

The paper is in [`papers/stp-diff-paper.tex`](papers/stp-diff-paper.tex).

**Key results:**
- Reduces complexity from O(4^N) to O(N·2^N)
- Estimated 353× speedup at N=20
- Identified Rb-E2F axis as the primary vulnerability bottleneck in a 10-node mammalian cell cycle model
- Validated against DepMap Public 25Q3 CRISPR knockout data (1000+ cell lines)
- ε_critical ≈ 2.80 for the p53-Mdm2 DNA-damage circuit

## Structure

```
├── papers/           — LaTeX source, markdown export, references
├── scripts/          — All experiment Python scripts (seeded for reproducibility)
├── outputs/          — Generated figures (PNG)
├── AGENTS.md         — Agent instructions for AI-assisted development
├── PROJECT.md        — Project overview and commands
└── MISTAKES.md       — Lessons learned
```

## Quick Start

```bash
# Full p53-Mdm2 PGD attack pipeline
python scripts/run_adversarial_pipeline.py

# 10-node mammalian cell cycle attack
python scripts/mammalian_cell_cycle_attack.py

# Core STP engine (research suite)
python scripts/STP_Diff_Research.py
```

All scripts use `torch.manual_seed(42)` for reproducibility.

## Citation

```bibtex
@article{khalil2026stpdiff,
  author  = {Khalil, Yaseen},
  title   = {{STP-Diff}: Differentiable Adversarial Discovery of Regulatory Vulnerabilities in Gene Circuits via Semi-Tensor Product Mapping},
  year    = {2026},
  journal = {arXiv preprint}
}
```

## Engine

The PyTorch implementation is at [github.com/yaskhalil/STP-GNN](https://github.com/yaskhalil/STP-GNN).

## Author

**Yaseen Khalil** — Virginia Tech  
yaskhalil2006@vt.edu  
[yaskhalil.com](https://yaskhalil.com)
