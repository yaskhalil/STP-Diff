"""
STP-Diff: Implicit VJP Benchmark
--------------------------------
Measures wall-clock speedup of ImplicitSTPFunction vs. explicit STP (full 2^N x 2^N matrix).
For N <= 12: both methods can run (explicit matrix fits in memory).
For N > 12: explicit matrix OOMs — reports theoretical bound and runs implicit only.

Usage:
    python scripts/stp_speedup_benchmark.py
"""
import torch
import torch.nn as nn
import numpy as np
import time
import sys

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# ImplicitSTPFunction (copied from run_adversarial_pipeline.py)
# ---------------------------------------------------------------------------
class ImplicitSTPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, M_list):
        n = M_list.shape[0]
        y_list = [torch.mm(m, x) for m in M_list]
        x_next = y_list[0]
        for i in range(1, len(y_list)):
            x_next = torch.kron(x_next, y_list[i])
        ctx.save_for_backward(x, M_list, torch.stack(y_list))
        return x_next

    @staticmethod
    def backward(ctx, grad_output):
        x, M_list, y_list = ctx.saved_tensors
        n = M_list.shape[0]
        grad_x = torch.zeros_like(x)
        grad_M = torch.zeros_like(M_list)
        g_tensor = grad_output.view([2] * n)
        for i in range(n):
            y_comp = None
            for j in range(n):
                if i == j:
                    continue
                y_comp = y_list[j] if y_comp is None else torch.kron(y_comp, y_list[j])
            permute_order = [i] + list(range(i)) + list(range(i + 1, n))
            g_reshaped = g_tensor.permute(permute_order).reshape(2, -1)
            grad_y_i = torch.mm(g_reshaped, y_comp)
            grad_M[i] = torch.mm(grad_y_i, x.t())
            grad_x += torch.mm(M_list[i].t(), grad_y_i)
        return grad_x, grad_M


# ---------------------------------------------------------------------------
# Explicit STP: build the full 2^N × 2^N transition matrix
# ---------------------------------------------------------------------------
def build_explicit_L(M_list):
    """Build the full global transition matrix L = M_1 ★ ... ★ M_N.
    Uses the STP property: L = M_1 ⊗ (I_2 ⊗ ... ⊗ I_2) + ...
    But the direct way is equivalent to:
        L = kron_product of all M_i rows
    Actually, for STP Boolean networks, L = M_1 ★ M_2 ★ ... ★ M_N
    where ★ is the semi-tensor product.
    
    For Boolean networks with structure matrices M_i (2 × 2^N),
    the global transition L is 2^N × 2^N.
    We build it column-by-column: for each state index k (0..2^N-1),
    L[:, k] = kron(M_1[:, k], M_2[:, k], ..., M_N[:, k])
    where M_i[:, k] is column k of M_i (a 2×1 vector).
    """
    n = M_list.shape[0]
    num_states = 2 ** n
    L = torch.zeros(num_states, num_states)
    for k in range(num_states):
        # Each M_i[:, k] is 2×1 → kron them to get a 2^N × 1 column
        col = M_list[0, :, k]
        for i in range(1, n):
            col = torch.kron(col, M_list[i, :, k])
        L[:, k] = col
    return L


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def benchmark_n(n, device='cpu', num_trials=10):
    """Benchmark both methods at network size n. Returns (implicit_time, explicit_time)."""
    # Fewer trials for larger N (each trial takes longer)
    if n >= 15:
        num_trials = 5
    elif n >= 20:
        num_trials = 3
    print(f"\n  N={n} ...", end=" ", flush=True)

    # Random structure matrices (ground truth Boolean → relax to logits)
    # We use random soft M_i (close to Boolean) for realistic timing
    theta = torch.randn(n, 2, 2 ** n, requires_grad=True) * 0.5
    M_list = torch.softmax(theta, dim=1)  # relaxed structure matrices, has grad_fn

    # Initial state vector (no identity matrix — avoids O(2^N × 2^N) eye creation)
    x = torch.zeros(2 ** n, 1)
    x[0] = 1.0

    # Warmup
    try:
        _ = ImplicitSTPFunction.apply(x, M_list)
    except:
        pass

    # ---- Implicit VJP timing ----
    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.perf_counter()
    for _ in range(num_trials):
        th = torch.randn(n, 2, 2 ** n, requires_grad=True) * 0.5
        Ml = torch.softmax(th, dim=1)
        x_out = ImplicitSTPFunction.apply(x, Ml)
        x_out.sum().backward()
    torch.cuda.synchronize() if device == 'cuda' else None
    t_impl = (time.perf_counter() - t0) / num_trials
    print(f"implicit={t_impl*1000:.1f}ms", end=" ", flush=True)

    # ---- Explicit STP timing (only if fits in memory) ----
    exp_speedup = None
    t_exp = None
    if n <= 15:
        # Pre-build L once (for n <= 15 the 2^n x 2^n matrix fits in memory)
        try:
            th_base = torch.randn(n, 2, 2 ** n, requires_grad=True) * 0.5
            M_base = torch.softmax(th_base, dim=1)
            Lt = build_explicit_L(M_base)
            t_build = time.perf_counter()
            Lt = build_explicit_L(M_base)
            t_build = time.perf_counter() - t_build
        except (RuntimeError, MemoryError) as e:
            print(f"explicit=OOM (build failed)")
            return t_impl, None, (4**n) / (n * 2**n)

        torch.cuda.synchronize() if device == 'cuda' else None
        t0 = time.perf_counter()
        for _ in range(num_trials):
            _ = Lt @ x
            _ = Lt.T @ x  # backward pass (L^T @ grad)
        torch.cuda.synchronize() if device == 'cuda' else None
        t_exp_fwd = (time.perf_counter() - t0) / num_trials
        
        # Total explicit time = build (one-time) + forward + backward
        t_exp = t_build + t_exp_fwd
        exp_speedup = (t_exp / t_impl) if t_impl > 0 else float('inf')
        print(f"explicit_total={t_exp*1000:.1f}ms (build={t_build*1000:.1f}ms + fwd+bwd={t_exp_fwd*1000:.1f}ms) speedup={exp_speedup:.1f}x", flush=True)
    else:
        # Theoretical: explicit requires O(4^N) which is infeasible
        exp_speedup = (4 ** n) / (n * 2 ** n)
        print(f"explicit=infeasible (4^{n}={4**n:.0e} ops) theoretical_speedup={exp_speedup:.0f}x", flush=True)

    return t_impl, t_exp, exp_speedup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on: {device}")
    print(f"{'N':>3}  {'Implicit(ms)':>13}  {'Explicit(ms)':>13}  {'Speedup':>9}")
    print("-" * 45)

    results = []
    for n in [3, 5, 7, 10, 12, 15, 20]:
        t_impl, t_exp, speedup = benchmark_n(n, device)
        impl_str = f"{t_impl*1000:.2f}" if t_impl is not None else "N/A"
        exp_str  = f"{t_exp*1000:.2f}" if t_exp is not None else "OOM"
        sp_str   = f"{speedup:.1f}x" if speedup is not None else "N/A"
        print(f"{n:3}  {impl_str:>13}  {exp_str:>13}  {sp_str:>9}")
        results.append((n, t_impl, t_exp, speedup))

    print("\nDone.")
