# STP-Diff: Differentiable Adversarial Discovery of Regulatory Vulnerabilities in Gene Circuits via Semi-Tensor Product Mapping

**Yaseen Khalil**  
Independent Researcher  
*yaseenkhalil.com*

---

## Abstract

The discovery of regulatory vulnerabilities in Gene Regulatory Networks (GRNs) is central to computational oncology, yet remains computationally intractable for networks exceeding 15-20 nodes due to exponential state-space explosion. We introduce STP-Diff, a framework that transforms discrete Boolean network dynamics into a differentiable manifold via Semi-Tensor Product (STP) mapping with an implicit Vector-Jacobian Product (VJP) operator. This reduces theoretical complexity from $O(4^N)$ to $O(N \cdot 2^N)$, achieving a measured 353x speedup at $N=20$ without numerical approximation. Using Projected Gradient Descent (PGD) attacks, we identify the Rb-E2F axis as the primary vulnerability bottleneck in a 10-node mammalian cell cycle model, a finding validated against CRISPR-Cas9 knockout data from the Broad Institute DepMap (25Q3, 1000+ cell lines). We further quantify network resilience through Epsilon-Critical search, establishing $\epsilon_{\text{critical}} = 2.8000$ for the p53-Mdm2 DNA-damage circuit. These results show that gradient-based adversarial optimization over algebraic state spaces can reliably expose biological control points invisible to single-node knockout analysis.

---

## 1 Introduction

The study of Gene Regulatory Networks (GRNs) has deep roots in dynamical systems theory. Jacob and Monod's operon model established the paradigm of logical regulation, later formalized by Kauffman [1969] into Random Boolean Networks (RBNs) and refined by Thomas [1973] into the logical formalism that underpins modern systems biology. Glass and Kauffman [1973] showed that even simple Boolean idealizations of biochemical networks could reproduce complex dynamical behaviors like limit cycles and multiple steady states. Albert and Othmer [2003] later demonstrated that these discrete models could make experimentally testable predictions about real biological systems, specifically the segment polarity gene network in *Drosophila*.

Boolean Network (BN) models capture the switch-like behavior of gene activation: each gene is ON (1) or OFF (0), and its next state is a deterministic Boolean function of its regulators' current states. Despite their conceptual clarity, BNs face a fundamental computational barrier. The state space of an $N$-node network scales as $2^N$, and brute-force attractor analysis enumerating all possible transitions becomes intractable beyond $N \approx 15$.

Two dominant approaches have emerged to manage this complexity:

**Differential Equation (ODE) models** replace discrete logic with continuous concentration dynamics, enabling numerical simulation. However, ODEs introduce dozens of free parameters (Michaelis constants, degradation rates, Hill coefficients) that must be measured or fit, and small parameter changes can produce qualitatively different attractor landscapes [Aldridge et al., 2006]. For most biological circuits, these parameters are unknown.

**Semi-Tensor Product (STP) methods**, pioneered by Cheng et al. [2011], provide an algebraic alternative. By representing Boolean states as canonical vectors in $\mathbb{R}^2$ and using the semi-tensor product in place of the standard matrix product, any Boolean network can be expressed as a linear system:

$$x(t+1) = L x(t)$$

where $L \in \mathbb{R}^{2^N \times 2^N}$ is the global transition matrix. This enables algebraic control theory for network analysis and intervention design [Cheng & Qi, 2010]. However, constructing $L$ explicitly requires $O(4^N)$ memory, the "matrix explosion" problem, limiting practical application to $N \lesssim 10$.

**Graph Neural Networks (GNNs)** have been applied to drug response prediction [Kipf & Welling, 2017] and cell-line sensitivity analysis. Standard message-passing architectures aggregate neighborhood information in ways that wash out the strict Boolean logic governing individual gene transitions. LogicXGNN [Geng et al., 2026] attempts to bridge this gap by extracting logical rules from trained GNNs, but does not provide a mechanism for discovering vulnerabilities through optimization.

**Probabilistic Boolean Networks (PBNs)** [Shmulevich et al., 2002] introduced stochasticity to address deterministic BN limitations, but their inference problem scales exponentially with network size and does not offer a gradient-based intervention design pathway.

**This paper presents STP-Diff**, which synthesizes elements from each paradigm. The contributions are:

1. An implicit Vector-Jacobian Product (VJP) operator that computes gradients through STP networks in $O(N \cdot 2^N)$ time and $O(2^N)$ memory, avoiding explicit construction of the global transition matrix.
2. A Temperature-Scaled Softmax relaxation that renders Boolean dynamics differentiable, enabling gradient-based adversarial attack.
3. The first demonstration of adversarial vulnerability discovery in biological networks, validated against empirical CRISPR-Cas9 knockout data.

---

## 2 Problem Statement

Let $G = (V, E)$ be a GRN with $N = |V|$ nodes. Each node $v_i \in V$ has a Boolean state $x_i(t) \in \{0, 1\}$ and a transition function $f_i: \{0,1\}^{k_i} \to \{0,1\}$, where $k_i$ is the in-degree of $v_i$. The global state at time $t$ is $x(t) \in \{0,1\}^N$.

The **regulatory vulnerability problem** is: given a target phenotype $s^* \in \{0,1\}^N$ (e.g., apoptosis, G1-arrest), find the minimal perturbation to the network's logical structure such that the attractor landscape shifts to include $s^*$ as a reachable fixed point.

In the Boolean setting, a perturbation corresponds to flipping one or more entries of the structure matrices $M_i$ that encode $f_i$. Brute-force evaluation of all possible perturbations requires $O(2^N \cdot |E|)$ simulations, prohibitive for $N > 10$.

We reformulate this as an adversarial optimization problem: given a baseline network $\Theta^{(0)}$, find $\Delta\Theta$ within an $\ell_\infty$ ball of radius $\epsilon$ that maximizes the probability of the target state after $T$ steps:

$$\max_{\Delta\Theta} \quad x_T[s^*] \quad \text{s.t.} \quad \|\Delta\Theta\|_\infty \leq \epsilon$$

This formulation is borrowed from adversarial attacks in computer vision [Szegedy et al., 2014; Madry et al., 2018], where small input perturbations cause catastrophic model failure. Here, the "input" is the network's logical structure, and the "failure" is a forced phenotypic transition.

---

## 3 Methods

### 3.1 Semi-Tensor Product Representation

Following Cheng et al. [2011], we map each Boolean state to the logical domain $\Delta_2 = \{\delta_2^1, \delta_2^2\}$, where:

$$\delta_2^1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \text{ (ON)}, \quad \delta_2^2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \text{ (OFF)}$$

The global network state is the Kronecker product of individual node states:

$$x(t) = \bigotimes_{i=1}^N x_i(t) \in \mathbb{R}^{2^N}$$

Each node $i$ has a $2 \times 2^N$ structure matrix $M_i$ encoding its Boolean logic. The global transition is:

$$x_i(t+1) = M_i x(t), \quad x(t+1) = \bigotimes_{i=1}^N x_i(t+1)$$

The explicit global transition matrix $L = M_1 \star M_2 \star \cdots \star M_N$ (where $\star$ denotes the semi-tensor product) has dimension $2^N \times 2^N$, requiring $O(4^N)$ storage. Cheng and Qi [2010] showed that STP-based control of Boolean networks is theoretically tractable for small $N$, but the $O(4^N)$ barrier prevents scaling to realistic circuit sizes.

### 3.2 Continuous Relaxation via Temperature-Scaled Softmax

To enable gradient flow, we parameterize each structure matrix using logits $\Theta_i \in \mathbb{R}^{2 \times 2^N}$ and apply a Temperature-Scaled Softmax:

$$\tilde{M}_i(j,k) = \frac{\exp(\Theta_i(j,k) / \tau)}{\sum_{j'=1}^2 \exp(\Theta_i(j',k) / \tau)}$$

At $\tau = 1$, the softmax approximates a hard max, recovering near-Boolean behavior. At $\tau = 10.0$, all entries have non-negligible gradients, allowing the optimizer to traverse the topological landscape without vanishing gradients. After optimization, we verify that $\tau$ can be annealed back to $1.0$ without qualitative change in attractor structure.

### 3.3 Implicit Vector-Jacobian Product (VJP)

The core algorithmic contribution is an implicit VJP operator that avoids constructing $L$ explicitly. We implement a custom `torch.autograd.Function` that computes the forward pass in $O(N \cdot 2^N)$ time and the backward pass using dimensional contraction.

**Forward pass.** Given $x \in \mathbb{R}^{2^N}$ and structure matrices $\{M_i\}_{i=1}^N$:

$$y_i = M_i x \quad \forall i, \qquad x' = y_1 \otimes y_2 \otimes \cdots \otimes y_N$$

Each $y_i$ is $2 \times 1$, and the Kronecker product chain produces $x' \in \mathbb{R}^{2^N}$. No $2^N \times 2^N$ matrix is ever materialized.

**Backward pass.** Let $g = \partial \mathcal{L} / \partial x'$ be the upstream gradient. We reshape $g$ into an $N$-dimensional tensor $\mathcal{G} \in \mathbb{R}^{2 \times 2 \times \cdots \times 2}$ and compute gradients for each node $i$ as:

$$\frac{\partial \mathcal{L}}{\partial y_i} = \mathcal{G}_{(i)} \cdot Y_{\neg i}$$

where $\mathcal{G}_{(i)}$ is $\mathcal{G}$ permuted to place dimension $i$ first and flattened to $2 \times 2^{N-1}$, and $Y_{\neg i} = \bigotimes_{j \neq i} y_j$. The gradient with respect to $M_i$ is then:

$$\frac{\partial \mathcal{L}}{\partial M_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot x^\top$$

**Theorem 1.** The implicit VJP forward pass requires $O(N \cdot 2^N)$ time and $O(2^N)$ memory. The backward pass also requires $O(N \cdot 2^N)$ time and $O(2^N)$ memory. This compares to $O(4^N)$ time and memory for the explicit construction.

*Proof.* The forward pass performs $N$ matrix-vector products ($O(N \cdot 2^N)$) and $N-1$ Kronecker products of exponentially growing vectors ($\sum_{k=1}^{N-1} 2^{k+1} = O(2^N)$). The backward pass performs $N$ dimensional contractions, each contracting a $2^N$-element tensor with a $2^{N-1}$-element vector, for $O(N \cdot 2^N)$ total.

| Metric | Explicit STP | STP-Diff |
|--------|-------------|------------------------|
| Time complexity | $O(4^N)$ | $O(N \cdot 2^N)$ |
| Memory complexity | $O(4^N)$ | $O(2^N)$ |
| Measured speedup at $N=20$ | 1x (baseline) | **353x** |
| Theoretical speedup at $N=30$ | 1x | $\sim 10^5\!\times$ |

**Table 1:** Complexity comparison between explicit STP and the implicit VJP approach.

### 3.4 Adversarial Attack via Projected Gradient Descent

We implement a Projected Gradient Descent (PGD) attacker following Goodfellow et al. [2015]:

$$\Theta^{(k+1)} = \Pi_{\mathcal{B}_\epsilon(\Theta^{(0)})}\left( \Theta^{(k)} - \alpha \cdot \operatorname{sgn}\left( \nabla_\Theta \mathcal{L}(\Theta^{(k)}) \right) \right)$$

where $\Pi_{\mathcal{B}_\epsilon}$ projects onto the $\ell_\infty$ ball of radius $\epsilon$ around the baseline parameters $\Theta^{(0)}$, and $\mathcal{L} = -\log x_T[s_{\text{target}}]$ drives the probability of the target state toward 1.

The attack runs for $K$ iterations with step size $\alpha = 0.1$, simulating $T = 10$ network steps per iteration. The perturbation budget $\epsilon$ controls the maximum change to any single logical entry. Small $\epsilon$ forces the attacker to find distributed, "holographic" vulnerabilities rather than brute-force overwriting single nodes.

---

## 4 Experiments

### 4.1 Computational Benchmarking

We benchmarked STP-Diff against an explicit STP implementation across network sizes $N \in \{3, 5, 7, 10, 15, 20\}$. Both implementations used identical Boolean rules for a randomly generated 10-node network. Timing was averaged over 10 forward-backward passes on an Apple M4 Max.

![Theoretical Scaling](outputs/complexity_scaling.png)
**Figure 1:** Scaling comparison. The explicit STP ($O(4^N)$) exits feasible memory at $N \approx 12$. STP-Diff's implicit VJP ($O(N \cdot 2^N)$) remains tractable through $N=20$ and beyond.

![Empirical Speedup](outputs/speedup_comparison.png)
**Figure 2:** Measured runtime at $N=20$. STP-Diff achieves 353x speedup over explicit STP while maintaining numerical parity.

### 4.2 p53-Mdm2 Circuit Resilience (N=5)

The p53-Mdm2 negative feedback loop is the primary DNA-damage response mechanism in mammalian cells. We modeled a 5-node network (p53_b1, p53_b2, Mdm2_cyt, Mdm2_nuc, DNAdam) using MaBoSS-formatted Boolean rules.

**Setup.** The target attractor represents the apoptotic state (p53_b1=1, p53_b2=1, Mdm2_cyt=0, Mdm2_nuc=0, DNAdam=1). We performed an Epsilon-Critical binary search over $\epsilon \in [0.01, 5.0]$ to find the minimum perturbation required to force the network from its healthy homeostatic attractor into the apoptotic manifold.

**Result.** The search converged to $\epsilon_{\text{critical}} = 2.8000$, a quantitative measure of the network's topological robustness. Below this threshold, the negative feedback structure absorbed the perturbation and returned to homeostasis. Above it, the apoptotic attractor became dominant.

![Attack Dosage](outputs/viz_attack_dosage.png)
**Figure 3:** Probability of apoptotic state as a function of perturbation magnitude $\epsilon$. The critical threshold $\epsilon_{\text{critical}} = 2.8000$ marks the phase transition.

**Vulnerability Decoding.** The gradient-based attack identified the Mdm2_nuc logic as the most perturbed transition. When DNAdam is present and p53_b1 is inactive, the probability of Mdm2_nuc activation shifted from $P=0.04$ (baseline, allowing repair) to $P=0.97$ (attacked, overriding repair).

| Rank | Target Node | Regulatory Context | Probability Shift |
|------|------------|-------------------|-------------------|
| 1 | Mdm2_nuc | DNAdam=1, p53_b1=0 | 0.04 to 0.97 |
| 2 | p53_b1 | Mdm2_nuc=1 | 0.12 to 0.88 |
| 3 | p53_b2 | p53_b1=1, Mdm2_nuc=0 | 0.31 to 0.76 |
| 4 | Mdm2_cyt | p53_b1=1, p53_b2=1 | 0.28 to 0.65 |
| 5 | DNAdam | p53_b1=0 | 0.09 to 0.52 |

**Table 2:** Top-5 regulatory edges modified by the PGD attack on the p53-Mdm2 network. The Mdm2_nuc hijack under DNA damage is the critical vulnerability.

### 4.3 Mammalian Cell Cycle Dismantling (N=10)

We scaled to a 10-node model of the mammalian cell cycle based on Fauré et al. [2006], comprising the core regulatory circuit: CycD, Rb, E2F, CycE, CycA, p27, Cdc20, Cdh1, UbcH10, and CycB.

**Single-node knockout analysis** (simulating classical gene perturbation experiments) showed that removing any single node failed to break the cell cycle. The network's redundant feedback structure maintained cyclin oscillations.

**PGD attack.** The distributed adversarial perturbation ($\epsilon = 2.5$) forced the network into a stable G1-arrest manifold, characterized by sustained high p27 and low CycB probability. The attack identified the **Rb-E2F axis** as the critical bottleneck, the transition most sensitive to structural perturbation.

![Vulnerability Ranking](outputs/viz_vulnerability_ranking.png)
**Figure 4:** Regulatory nodes ranked by impact on attractor stability. The Rb-E2F transition dominates.

| Node | Baseline Dependency | Post-Attack Dependency | $\Delta$ |
|------|--------------------|----------------------|---|
| **Rb** | $P=0.12$ | $P=0.89$ | **+0.77** |
| **E2F** | $P=0.31$ | $P=0.81$ | **+0.50** |
| CycE | $P=0.22$ | $P=0.48$ | +0.26 |
| p27 | $P=0.18$ | $P=0.72$ | +0.54 |
| CycB | $P=0.58$ | $P=0.09$ | -0.49 |

**Table 3:** Regulatory logic shifts in the mammalian cell cycle under PGD attack.

### 4.4 DepMap Empirical Validation

To verify that STP-Diff's *in silico* predictions reflect biological reality, we cross-referenced the identified vulnerabilities against the Broad Institute DepMap Public 25Q3 dataset, spanning CRISPR-Cas9 knockout effects across 1000+ cancer cell lines.

**Hypothesis.** If the Rb-E2F axis is the true vulnerability bottleneck, cell lines with Rb loss-of-function should exhibit hyper-dependency on E2F (lower CRISPR scores for E2F knockout).

**Setup.** We mapped each model node to its DepMap gene identifier (Rb to RB1 (5925), E2F to E2F1 (1869), CycB to CCNB1 (891)) and stratified cell lines by Rb functional status.

**Result.** Rb-loss cell lines showed a mean E2F dependency score of $-0.87$ (highly dependent), compared to $-0.31$ for Rb-wildtype lines, a statistically significant difference ($p < 10^{-6}$, two-sample $t$-test). The stratification confirms that cells with compromised Rb function become critically dependent on E2F, as predicted by the attack model.

![DepMap Correlation](outputs/cell_cycle_depmap_corr.png)
**Figure 5:** Empirical correlation matrix of the 10 cell-cycle genes across 1000+ DepMap cell lines. The Rb-E2F anti-correlation is the dominant signal.

**Knock-in validation.** The top-5 most perturbed regulatory edges identified by the PGD attack were transferred to the baseline network in isolation. The resulting network exhibited a terminal apoptosis probability of $P = 0.97$, confirming these edges are collectively sufficient to drive the phenotypic transition, demonstrating distributed causality rather than a single master switch.

---

## 5 Related Work

**Boolean Network Analysis.** Fauré et al. [2006] performed exhaustive state-space analysis of a 10-node mammalian cell cycle model, identifying stable attractors but noting the exponential blowup for larger networks. Shmulevich et al. [2002] introduced PBNs to model stochastic gene expression, though their inference cost scales exponentially with node count. Tools like GinSIM [Naldi et al., 2009] provide simulation environments but do not offer optimization-based vulnerability discovery.

**Semi-Tensor Product Methods.** Cheng et al. [2011] established the algebraic framework for STP-based BN analysis. Cheng and Qi [2010] derived controllability conditions for STP Boolean networks, showing theoretical pathways for network intervention. Cheng and Zhang [2025] recently extended STP to convolutional neural networks, but their method does not address the matrix explosion problem.

**GNNs for Systems Biology.** Kipf & Welling [2017] introduced Graph Convolutional Networks applied to drug sensitivity prediction. Geng et al. [2026] proposed LogicXGNN for extracting logical rules from trained GNNs. Neither provides a mechanism for adversarial vulnerability discovery in discrete networks.

**Adversarial Machine Learning.** Szegedy et al. [2014] first showed that imperceptible input perturbations could cause neural network misclassification. Goodfellow et al. [2015] proposed the fast gradient sign method for efficient adversarial attack generation. Madry et al. [2018] established the PGD framework as a universal first-order adversary. This paper applies these methods to biological Boolean networks for the first time, enabled by the differentiable relaxation in Section 3.2.

---

## 6 Discussion

### 6.1 Computational Scaling

The $O(N \cdot 2^N)$ complexity is a fundamental improvement over $O(4^N)$ for explicit STP, but the state vector $x(t) \in \mathbb{R}^{2^N}$ remains a hard memory constraint. At $N=20$, full state vectors require 8 MB (single precision). At $N=30$, this grows to 8 GB, and at $N=40$, 8 TB, infeasible for current hardware.

**Practical regime.** The method is practical for $N \leq 25$ on GPU hardware with 80 GB memory (e.g., NVIDIA H100). For larger networks, sparse tensor representations or Monte Carlo sampling of the Kronecker product dimensions would be necessary.

### 6.2 Limitations of the Continuous Relaxation

The Temperature-Scaled Softmax ($\tau = 10.0$) introduces controlled approximation. While we verified post-hoc that annealed models ($\tau \to 1.0$) recover the original discrete attractor landscape, there may exist pathological networks where the melted topology admits gradient shortcuts that do not correspond to realizable Boolean perturbations. All predicted vulnerabilities should be validated against the original discrete rules before experimental follow-up.

### 6.3 Biological Interpretation

The Rb-E2F axis is a well-characterized tumor suppressor pathway. Rb loss is one of the most common events in cancer, and E2F overactivation drives uncontrolled proliferation [Hanahan & Weinberg, 2011]. Our results suggest this pathway is not merely a common cancer driver but a **topological vulnerability** -- the network's attractor landscape is structurally sensitive to perturbations at this point. Rather than restoring Rb function (difficult pharmacologically), one might target the downstream E2F transcriptional program, which the attack model identifies as the critical control point.

---

## 7 Conclusion

STP-Diff shows that gradient-based adversarial optimization over algebraic state spaces can reliably identify regulatory vulnerabilities in Boolean gene networks. Combining the algebraic rigor of Semi-Tensor Products with the scalability of implicit autodiff operators, we achieve:

- **353x speedup** over explicit STP methods at $N=20$
- **Empirically validated predictions** -- the Rb-E2F vulnerability is corroborated by CRISPR-Cas9 data from 1000+ cancer cell lines
- **Quantitative resilience metrics** -- $\epsilon_{\text{critical}}$ provides a single-number measure of network robustness

The framework is general: any Boolean network expressable as a set of logical rules can be analyzed with this pipeline. Future work will extend the approach to sparse tensor representations targeting $N \geq 50$, and to multi-target adversarial objectives that model combination therapies.

---

## Acknowledgments

**Author Contributions.** **Yaseen Khalil** conceived the core methodology, developed the implicit VJP operator and the differentiable STP framework, implemented all code and experimental scripts, conducted the computational experiments and benchmarking, performed the DepMap validation analysis, interpreted the results, and prepared the manuscript. **Pavel Kraikivski** and **Agnieszka Miedlar** jointly defined the project scope and research objectives, determined the biological datasets and validation strategy, supervised the research direction, and provided critical review, mentoring, and feedback throughout the project.

**Data.** The authors thank the Broad Institute DepMap consortium (Public 25Q3) for public access to CRISPR-Cas9 knockout data.

**Declaration.** AI-assisted language tools were used for formatting and prose refinement. All technical content, analysis, and conclusions remain the authors' original work.

---

## References

1. Kauffman, S. A. (1969). Metabolic stability and epigenesis in randomly constructed genetic nets. *Journal of Theoretical Biology*, 22(3), 437-467.
2. Thomas, R. (1973). Boolean formalization of genetic control circuits. *Journal of Theoretical Biology*, 42(3), 563-585.
3. Glass, L., & Kauffman, S. A. (1973). The logical analysis of continuous, non-linear biochemical control networks. *Journal of Theoretical Biology*, 39(1), 103-129.
4. Albert, R., & Othmer, H. G. (2003). The topology of the regulatory interactions predicts the expression pattern of the segment polarity genes in *Drosophila melanogaster*. *Journal of Theoretical Biology*, 223(1), 1-18.
5. Shmulevich, I., Dougherty, E. R., Kim, S., & Zhang, W. (2002). Probabilistic Boolean Networks: a rule-based uncertainty model for gene regulatory networks. *Bioinformatics*, 18(2), 261-274.
6. Fauré, A., Naldi, A., Chaouiya, C., & Thieffry, D. (2006). Dynamical analysis of a generic Boolean model for the control of the mammalian cell cycle. *Bioinformatics*, 22(14), e124-e131.
7. Aldridge, B. B., Burke, J. M., Lauffenburger, D. A., & Sorger, P. K. (2006). Physicochemical modelling of cell signalling pathways. *Nature Cell Biology*, 8(11), 1195-1203.
8. Cheng, D., & Qi, H. (2010). Controllability of Boolean networks via semi-tensor product method. *Automatica*, 46(1), 60-69.
9. Cheng, D., Qi, H., & Li, Z. (2011). *Analysis and Control of Boolean Networks: A Semi-tensor Product Approach*. Springer.
10. Hanahan, D., & Weinberg, R. A. (2011). Hallmarks of Cancer: The Next Generation. *Cell*, 144(5), 646-674.
11. Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). Intriguing properties of neural networks. *ICLR*.
12. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. *ICLR*.
13. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.
14. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. *ICLR*.
15. Naldi, A., Berenguier, D., Fauré, A., Lopez, F., Thieffry, D., & Chaouiya, C. (2009). Logical modelling of regulatory networks with GINsim 2.3. *Biosystems*, 97(2), 134-139.
16. Geng, C., et al. (2026). LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks. *ICLR*.
17. Cheng, D., & Zhang, X. (2025). Semi-Tensor-Product Based Convolutional Neural Networks. *arXiv:2506.10407*.
18. Broad Institute. (2025). Cancer Dependency Map (DepMap) Public 25Q3 Dataset. https://depmap.org/portal/
