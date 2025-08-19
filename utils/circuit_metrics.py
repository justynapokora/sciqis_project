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


def circuit_expressibility():
    ...


def circuit_entangling_capability():
    ...
