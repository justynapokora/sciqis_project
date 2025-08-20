import numpy as np
from utils.states import State, DensityState


# --- fidelities ---
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
    """Matrix square root of PSD Hermitian"""
    vals, vecs = np.linalg.eigh((mat + mat.conj().T) / 2.0)  # hermitize
    vals = np.clip(vals, 0, None)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


def fidelity(state1: State | DensityState, state2: State | DensityState) -> float:
    """
    Fidelity between two quantum states.
    state1, state2: State or DensityState
    """
    from utils.states import State, DensityState  # adjust path as needed

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
    """
    Vectorized fidelities between pure states and noisy density matrices.

    Args:
        states:       (n+1, dim, 1) array of pure state vectors
        noisy_states: (n+1, dim, dim) array of density matrices

    Returns:
        fidelities: (n+1,) array of fidelities
    """
    # Conjugate transpose: (n+1, 1, dim)
    bra = np.conjugate(states.transpose(0, 2, 1))

    # Compute <psi| rho |psi> for each snapshot
    vals = bra @ noisy_states @ states   # shape (n+1, 1, 1)
    return np.real(vals[:, 0, 0])


def von_neumann_entropy(state: State | DensityState, atol: float = 1e-12) -> float:
    """
    Von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ), measured in bits.

    Args:
        state: A State (pure) or DensityState (mixed).
        atol:  Numerical tolerance for treating small eigenvalues as 0.

    Returns:
        float: Entropy in bits.
    """
    # --- Build density matrix ρ ---
    if isinstance(state, DensityState):
        rho = np.asarray(state.rho, dtype=complex)
    elif isinstance(state, State):
        return 0.0
    else:
        raise TypeError("Expected a State or DensityState.")

    # --- Numerical  ---
    rho = 0.5 * (rho + rho.conj().T)     # enforce Hermiticity
    tr = np.trace(rho)
    if tr == 0:
        return 0.0
    rho = rho / tr                       # normalize trace to 1

    # --- Eigenvalues ---
    evals = np.linalg.eigvalsh(rho).real
    evals = np.clip(evals, 0.0, None)    # clip tiny negatives
    s = evals.sum()
    if s > 0:
        evals = evals / s                # renormalize

    # --- Entropy in bits ---
    nonzero = evals > atol
    if not np.any(nonzero):
        return 0.0
    return float(-np.sum(evals[nonzero] * np.log2(evals[nonzero])))


def entropy_density_matrix(rho, atol):
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / np.trace(rho)
    evals = np.linalg.eigvalsh(rho).real
    evals = np.clip(evals, 0.0, None)
    if evals.sum() > 0:
        evals = evals / evals.sum()
    mask = evals > atol
    return -np.sum(evals[mask] * np.log2(evals[mask]))


def von_neumann_entropies_arrays(states: np.ndarray, noisy_states: np.ndarray, atol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized-style von Neumann entropies for states and noisy density matrices.
    - states:       (n+1, dim, 1) array of pure statevectors
    - noisy_states: (n+1, dim, dim) array of density matrices

    Returns:
        entropies_states: (n+1,) zeros (pure states are entropy 0)
        entropies_noisy:  (n+1,) von Neumann entropies (bits)
    """
    n_plus_1 = states.shape[0]

    # --- Pure states → always 0 entropy ---
    entropies_states = np.zeros(n_plus_1, dtype=float)

    # Apply entropy computation along the 0th axis (batch of matrices)
    entropies_noisy = np.array([entropy_density_matrix(rho, atol) for rho in noisy_states])

    return entropies_states, entropies_noisy



def circuit_expressibility():
    ...


def circuit_entangling_capability():
    ...
