import numpy as np
import math
from utils.states import State, DensityState


# --- fidelity ---
def fidelity_pure_pure(psi: np.ndarray, phi: np.ndarray) -> float:
    """F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|²"""
    v1 = psi.ravel()
    v2 = phi.ravel()
    return float(np.abs(np.vdot(v1, v2)) ** 2)


def fidelity_pure_mixed(psi: np.ndarray, rho: np.ndarray) -> float:
    """F(|ψ⟩, ρ) = ⟨ψ|ρ|ψ⟩"""
    v = psi.reshape(-1, 1)
    return float(np.real((v.conj().T @ rho @ v)[0, 0]))


def fidelity_mixed_mixed(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity: F(ρ, σ) = (Tr √(√ρ σ √ρ))²"""
    sqrt_rho = _sqrt_psd(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    sqrt_inner = _sqrt_psd(inner)
    return float(np.real(np.trace(sqrt_inner)) ** 2)


def _sqrt_psd(mat: np.ndarray) -> np.ndarray:
    """Matrix square root of a PSD Hermitian matrix."""
    vals, vecs = np.linalg.eigh((mat + mat.conj().T) / 2.0)  # hermitize
    vals = np.clip(vals, 0, None)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


def fidelity(state1: State | DensityState, state2: State | DensityState) -> float:
    """Fidelity between two quantum states (pure or mixed), with type-based dispatch."""
    if isinstance(state1, State) and not isinstance(state1, DensityState) and \
            isinstance(state2, State) and not isinstance(state2, DensityState):
        return fidelity_pure_pure(state1.qubit_vector, state2.qubit_vector)

    if isinstance(state1, State) and isinstance(state2, DensityState):
        return fidelity_pure_mixed(state1.qubit_vector, state2.rho)

    if isinstance(state2, State) and isinstance(state1, DensityState):
        return fidelity_pure_mixed(state2.qubit_vector, state1.rho)

    if isinstance(state1, DensityState) and isinstance(state2, DensityState):
        return fidelity_mixed_mixed(state1.rho, state2.rho)

    raise TypeError("fidelity: expected State or DensityState objects")


def fidelities_from_arrays(states: np.ndarray, noisy_states: np.ndarray) -> np.ndarray:
    """Vectorized fidelities between pure states and density matrices."""
    v = states[..., 0]  # (n+1, dim)
    return np.real(np.einsum('bi,bij,bj->b', v.conj(), noisy_states, v))


# ---------- Haar fidelity distribution helpers ----------
def haar_fidelity_pdf(F: np.ndarray, dim: int) -> np.ndarray:
    """PDF of fidelities between two Haar-random pure states in dimension 2^n."""
    N = dim
    return (N - 1) * np.power(1 - np.clip(F, 0.0, 1.0), N - 2)


def haar_bin_masses(bin_edges: np.ndarray, dim: int) -> np.ndarray:
    """Haar bin probabilities via CDF differences on [0,1]."""
    N = dim
    a = bin_edges[:-1]
    b = bin_edges[1:]
    # CDF(F) = 1 - (1 - F)^(N-1)
    return np.power(1 - a, N - 1) - np.power(1 - b, N - 1)


# ---------- Pairwise fidelities from pure states ----------
def pairwise_fidelities_consecutive(states_1c: np.ndarray) -> np.ndarray:
    """Fidelities for consecutive pure-state pairs (0,1), (2,3), … for one circuit."""
    S, dim = states_1c.shape[0], states_1c.shape[1]
    # reshape to 2 per pair
    K = (S // 2) * 2
    psi_pairs = states_1c[:K].reshape(-1, 2, dim)
    # normalize defensively
    psi_pairs = psi_pairs / np.linalg.norm(psi_pairs, axis=2, keepdims=True)
    v1 = psi_pairs[:, 0, :]
    v2 = psi_pairs[:, 1, :]
    ip = np.sum(np.conj(v1) * v2, axis=1)
    return np.abs(ip) ** 2  # (num_pairs,)


def pairwise_uhlmann_fidelities_consecutive(rhos_1c: np.ndarray) -> np.ndarray:
    """Uhlmann fidelities for consecutive density-matrix pairs (0,1), (2,3), … for one circuit."""
    S = rhos_1c.shape[0]
    K = (S // 2) * 2
    rhos = rhos_1c[:K].reshape(-1, 2, rhos_1c.shape[1], rhos_1c.shape[2])
    out = np.empty(rhos.shape[0], dtype=float)
    for k in range(out.size):
        out[k] = fidelity_mixed_mixed(rhos[k, 0], rhos[k, 1])
    return out


# --- entropy ---
def von_neumann_entropy(state: State | DensityState, atol: float = 1e-12) -> float:
    """Von Neumann entropy S(ρ) = −Tr(ρ log₂ ρ) in bits."""
    # --- Build density matrix ρ ---
    if isinstance(state, DensityState):
        rho = np.asarray(state.rho, dtype=complex)
    elif isinstance(state, State):
        return 0.0
    else:
        raise TypeError("Expected a State or DensityState.")

    # --- Numerical  ---
    rho = 0.5 * (rho + rho.conj().T)  # enforce Hermiticity
    tr = np.trace(rho)
    if tr == 0:
        return 0.0
    rho = rho / tr  # normalize trace to 1

    # --- Eigenvalues ---
    evals = np.linalg.eigvalsh(rho).real
    evals = np.clip(evals, 0.0, None)  # clip tiny negatives
    s = evals.sum()
    if s > 0:
        evals = evals / s  # renormalize

    # --- Entropy in bits ---
    nonzero = evals > atol
    if not np.any(nonzero):
        return 0.0
    return float(-np.sum(evals[nonzero] * np.log2(evals[nonzero])))


def entropy_density_matrix(rho, atol):
    """Helper: von Neumann entropy (bits) for a single density matrix."""
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / np.trace(rho)
    evals = np.linalg.eigvalsh(rho).real
    evals = np.clip(evals, 0.0, None)
    if evals.sum() > 0:
        evals = evals / evals.sum()
    mask = evals > atol
    return -np.sum(evals[mask] * np.log2(evals[mask]))


def von_neumann_entropies_arrays(states: np.ndarray, noisy_states: np.ndarray, atol: float = 1e-12) -> tuple[
    np.ndarray, np.ndarray]:
    """Vectorized entropies: zeros for pure states and S(ρ) for noisy density matrices."""
    n_plus_1 = states.shape[0]

    # --- Pure states → always 0 entropy ---
    entropies_states = np.zeros(n_plus_1, dtype=float)

    # Apply entropy computation along the 0th axis (batch of matrices)
    entropies_noisy = np.array([entropy_density_matrix(rho, atol) for rho in noisy_states])

    return entropies_states, entropies_noisy


# ---------- frame potentials ----------
def frame_potentials_from_samples(states: np.ndarray,
                                  noisy_states: np.ndarray | None = None,
                                  t_max: int = 4) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Estimate frame potentials F^{(t)}=E[F^t] for t=1..t_max (ideal, optional noisy, and Haar baseline)."""
    C, S, dim = states.shape[0], states.shape[1], states.shape[2]
    # Pairwise fidelities for each circuit (consecutive pairs)
    fids_pure = [pairwise_fidelities_consecutive(states[c].reshape(S, dim)) for c in range(C)]

    # vector of moments 1..t_max
    tvec = np.arange(1, t_max + 1, dtype=int)

    FP_ideal = np.zeros((C, t_max), dtype=float)
    for c in range(C):
        f = fids_pure[c]
        # shape (len(f), t_max) → mean over samples
        FP_ideal[c, :] = np.mean(f[:, None] ** tvec[None, :], axis=0)

    FP_noisy = None
    if noisy_states is not None:
        FP_noisy = np.zeros((C, t_max), dtype=float)
        for c in range(C):
            f_mixed = pairwise_uhlmann_fidelities_consecutive(noisy_states[c])
            FP_noisy[c, :] = np.mean(f_mixed[:, None] ** tvec[None, :], axis=0)

    # Haar baseline: F_Haar^{(t)} = t!(N-1)! / (t+N-1)!  where N=dim
    N = dim
    FP_haar = np.array([math.factorial(t) * math.factorial(N - 1) / math.factorial(t + N - 1)
                        for t in range(1, t_max + 1)], dtype=float)

    return FP_ideal, FP_noisy, FP_haar


# compressibility
def kl_expressibility_one_circuit(states_1c: np.ndarray,
                                  noisy_states_1c: np.ndarray | None = None,
                                  num_bins: int = 75,
                                  log_base: str = "e") -> tuple[float, float | None]:
    """Histogram-based KL expressibility to Haar for a single circuit (ideal and optional noisy)."""
    # Pairwise fidelities (your functions)
    fids_ideal = pairwise_fidelities_consecutive(states_1c)
    fids_noisy = (pairwise_uhlmann_fidelities_consecutive(noisy_states_1c)
                  if noisy_states_1c is not None else None)

    # Shared bins and Haar bin masses
    dim = states_1c.shape[1]
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    q = haar_bin_masses(bin_edges, dim)
    eps = 1e-12
    q = np.clip(q, eps, 1.0)
    q /= q.sum()

    ln_conv = 1.0 if (log_base == "e") else (1.0 / np.log(2.0))

    def _kl_from_fids(fids: np.ndarray) -> float:
        counts, _ = np.histogram(fids, bins=bin_edges)
        p = counts.astype(float)
        p /= max(p.sum(), 1.0)
        mask = p > 0
        return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask])))) * ln_conv

    KL_ideal = _kl_from_fids(fids_ideal)
    KL_noisy = _kl_from_fids(fids_noisy) if fids_noisy is not None else None
    return KL_ideal, KL_noisy


def _rho_j_from_dm(rho: np.ndarray, n: int, j: int) -> np.ndarray:
    """One-qubit reduced density matrix ρ_j from an n-qubit ρ."""
    R = rho.reshape((2,) * n * 2)  # (a0,...,a_{n-1}, b0,...,b_{n-1})
    # iteratively trace out all k != j; trace high-to-low to keep axis indices stable
    for k in range(n - 1, -1, -1):
        if k == j:
            continue
        # pair (ket axis k) with (bra axis current_half + k)
        R = np.trace(R, axis1=k, axis2=R.ndim // 2 + k)
    # now R has shape (2, 2) with axes (aj, bj)
    rho_j = R
    rho_j = 0.5 * (rho_j + rho_j.conj().T)
    tr = np.trace(rho_j)
    if tr != 0:
        rho_j = rho_j / tr
    return rho_j


# ---------- entangling capability (ideal / noisy) ----------
def entangling_capability_ideal_from_states(states_1c: np.ndarray) -> float:
    """Meyer–Wallach entangling capability (average Q) from pure-state samples for one circuit."""
    S, dim = states_1c.shape[0], states_1c.shape[1]
    n = int(np.log2(dim))
    if n <= 1:
        return 0.0

    # batch normalize |psi⟩
    psi_all = states_1c.reshape(S, dim)
    norms = np.linalg.norm(psi_all, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    psi_all = psi_all / norms

    scale = 2.0 / n
    total = 0.0
    for s in range(S):
        # reshape once per sample
        psi_nd = psi_all[s].reshape((2,) * n)
        acc = 0.0
        for j in range(n):
            # put qubit j first, then flatten env
            A = np.moveaxis(psi_nd, j, 0).reshape(2, -1)  # rows r0, r1
            r0, r1 = A[0], A[1]
            n0 = np.vdot(r0, r0).real
            n1 = np.vdot(r1, r1).real
            s01 = np.vdot(r0, r1)
            purity = n0 * n0 + n1 * n1 + 2.0 * (np.abs(s01) ** 2)
            acc += (1.0 - purity)
        total += scale * acc
    return total / S


def entangling_capability_noisy_from_density(noisy_states_1c: np.ndarray) -> float:
    """Noisy entangling capability proxy (average \u007EQ) from density-matrix samples for one circuit."""
    S, dim = noisy_states_1c.shape[0], noisy_states_1c.shape[1]
    n = int(np.log2(dim))
    if n <= 1:
        return 0.0

    # batch hermitize & normalize to unit trace
    R = noisy_states_1c.astype(complex, copy=False)
    R = 0.5 * (R + np.conjugate(np.swapaxes(R, -1, -2)))
    tr = np.trace(R, axis1=-2, axis2=-1).real
    tr[tr == 0.0] = 1.0
    R = R / tr[:, None, None]

    scale = 2.0 / n
    total = 0.0
    for s in range(S):
        rho = R[s]
        acc = 0.0
        for j in range(n):
            rho_j = _rho_j_from_dm(rho, n, j)
            purity = np.einsum('ij,ji->', rho_j, rho_j).real  # Tr(ρ_j^2)
            acc += (1.0 - purity)
        total += scale * acc
    return total / S
