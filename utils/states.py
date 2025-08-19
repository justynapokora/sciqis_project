from __future__ import annotations

import numpy as np


class State:
    def __init__(self, qubit_vector: np.ndarray):
        # Ensure ndarray with complex dtype
        v = np.asarray(qubit_vector, dtype=complex)

        # Accept 1D (length 2^n) or 2D column (2^n, 1); store as (2^n, 1)
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        elif v.ndim == 2 and v.shape[1] == 1:
            pass
        else:
            raise ValueError(
                f"State vector must be 1D (2^n,) or a column vector (2^n, 1); got shape {v.shape}."
            )

        size = v.shape[0]
        if size <= 0 or (size & (size - 1)) != 0:  # power-of-two check
            raise ValueError(f"Invalid state vector length: {size}. Must be 2^n.")

        self.qubit_vector = v
        self.num_of_qubits = int(np.log2(size))

    def __str__(self):
        # Work with a 1D view for iteration/printing
        flat = self.qubit_vector.ravel()
        terms = []

        for i, amplitude in enumerate(flat):
            if not np.isclose(amplitude, 0):  # skip zero amplitudes
                basis_state = format(i, f'0{self.num_of_qubits}b')  # binary label

                # Pretty-print complex amplitude compactly
                if np.isclose(amplitude.imag, 0.0):
                    amp_str = f"{amplitude.real:.3g}"
                elif np.isclose(amplitude.real, 0.0):
                    amp_str = f"{amplitude.imag:.3g}i"
                else:
                    amp_str = f"{amplitude.real:.3g}{amplitude.imag:+.3g}i"

                terms.append(f"{amp_str} |{basis_state}⟩")

        return " + ".join(terms) if terms else "0"

    def get_probabilities_str(self):
        flat = self.qubit_vector.ravel()
        lines = []

        for i, amplitude in enumerate(flat):
            prob = abs(amplitude) ** 2
            if not np.isclose(prob, 0):  # skip zero-probability states
                basis_state = format(i, f'0{self.num_of_qubits}b')
                prob_str = f"{prob:.3f}"  # keep 3 decimal places
                lines.append(f"P( |{basis_state}⟩ ) = {prob_str}")

        return "\n".join(lines) if lines else "P=0"

    def normalize_state(self):
        norm = np.linalg.norm(self.qubit_vector)
        if norm != 0:
            self.qubit_vector /= norm

    def measure_all(self, num_of_measurements=1, rng: np.random.Generator | None = None):
        if not rng:
            rng = np.random.default_rng()

        self.normalize_state()

        # Flatten to 1D and compute probabilities
        flat = self.qubit_vector.ravel()
        probs = np.abs(flat)**2

        basis_states = [
            format(i, f'0{self.num_of_qubits}b')
            for i in range(flat.size)
        ]

        return rng.choice(basis_states, size=num_of_measurements, p=probs)

    def measure_qubit(self, qubit_index, num_of_measurements=1, rng: np.random.Generator | None = None):
        """
        Measures a single qubit (logical index) in the computational basis.
        Returns: outcome (0 or 1), and the new collapsed state vector.
        """
        if not rng:
            rng = np.random.default_rng()

        proj_0 = np.zeros_like(self.qubit_vector)
        proj_1 = np.zeros_like(self.qubit_vector)

        # Convert logical qubit index (top-to-bottom) to physical bit position (little-endian)
        physical_index = self.num_of_qubits - 1 - qubit_index

        for i, amplitude in enumerate(self.qubit_vector):
            bits = np.binary_repr(i, width=self.num_of_qubits)
            if bits[physical_index] == '0':
                proj_0[i] = amplitude
            else:
                proj_1[i] = amplitude

        # Calculate measurement probabilities
        prob_0 = np.sum(np.abs(proj_0) ** 2)
        prob_1 = np.sum(np.abs(proj_1) ** 2)

        outcome = rng.choice([0, 1], size=num_of_measurements, p=[prob_0, prob_1])

        # Collapse state
        if outcome == 0:
            new_state = proj_0 / np.sqrt(prob_0)
        else:
            new_state = proj_1 / np.sqrt(prob_1)

        self.qubit_vector = new_state

        return outcome


class DensityState:
    """
    Minimal density-matrix holder for the noisy path.
    Provides:
      - apply_unitary(U)
      - apply_1q_channel(kraus_list, target_qubit)
      - normalize_dm()
      - pretty printing of populations (optional)
    """
    def __init__(self, rho: np.ndarray):
        rho = np.asarray(rho, dtype=complex)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("rho must be a square matrix.")
        if rho.shape[0] & (rho.shape[0] - 1):
            raise ValueError("rho dimension must be 2^n.")
        self.rho = rho
        self.num_of_qubits = int(np.log2(rho.shape[0]))

    @classmethod
    def from_state(cls, state: State) -> DensityState:
        # state.qubit_vector is (2^n, 1)
        psi = state.qubit_vector
        rho = psi @ psi.conj().T
        return cls(rho)

    def __str__(self, tol=1e-10, max_terms=2000):
        """
        Pretty-print:
          - If ρ is (numerically) pure: print the ket expansion (like State.__str__).
          - Else: print a sparse operator expansion Σ_ij ρ_ij |i><j|.
        Args:
          tol: threshold for considering values as zero / purity ≈ 1
          max_terms: cap number of printed |i><j| terms (to avoid huge outputs)
        """
        # --- Try pure detection via eigen-decomposition ---
        # eigh is Hermitian-safe; small negative eigvals can appear numerically
        eigvals, eigvecs = np.linalg.eigh(0.5 * (self.rho + self.rho.conj().T))
        # clip tiny negatives and renormalize tiny drift in trace
        eigvals = np.clip(eigvals.real, 0.0, None)
        if eigvals.sum() > 0:
            eigvals = eigvals / eigvals.sum()

        # purity ~ sum λ_i^2; equals 1 for pure states
        purity = float(np.sum(eigvals ** 2))
        if np.isclose(purity, 1.0, atol=1e-8):
            # dominant eigenvector is the ket (up to global phase)
            idx = int(np.argmax(eigvals))
            psi = eigvecs[:, idx]
            # fix global phase so first nonzero component is real & ≥ 0
            for c in psi:
                if not np.isclose(c, 0.0, atol=tol):
                    psi = psi * np.exp(-1j * np.angle(c))
                    break
            return State(psi.reshape(-1, 1)).__str__()

        # --- Mixed: print sparse operator expansion ---
        lines = []
        n = self.num_of_qubits
        dim = self.rho.shape[0]
        printed = 0
        for i in range(dim):
            for j in range(dim):
                val = self.rho[i, j]
                if abs(val) <= tol:
                    continue
                re, im = val.real, val.imag
                if np.isclose(im, 0.0, atol=tol):
                    amp = f"{re:.3g}"
                elif np.isclose(re, 0.0, atol=tol):
                    amp = f"{im:.3g}i"
                else:
                    amp = f"{re:.3g}{im:+.3g}i"
                ket = format(i, f'0{n}b')
                bra = format(j, f'0{n}b')
                lines.append(f"{amp} |{ket}⟩⟨{bra}|")
                printed += 1
                if printed >= max_terms:
                    remaining = dim * dim - printed
                    lines.append(f"... (+{remaining} more terms below |{tol:g}|)")
                    return " + ".join(lines)
        return " + ".join(lines) if lines else "0"

    def apply_unitary(self, U: np.ndarray):
        # ρ ← U ρ U†
        self.rho = U @ self.rho @ U.conj().T

    def _lift_1q_op(self, K1: np.ndarray, q: int) -> np.ndarray:
        """
        Lift a 2x2 operator acting on logical qubit q to full-system.
        Matches the kron ordering used elsewhere in your code.
        """
        n = self.num_of_qubits
        ops = [np.eye(2, dtype=complex) for _ in range(n)]
        ops[q] = K1
        full = np.array([[1]], dtype=complex)
        for op in ops:
            full = np.kron(op, full)
        return full

    def apply_1q_channel(self, kraus_list: list[np.ndarray], q: int):
        # ρ ← Σ K ρ K†
        accum = np.zeros_like(self.rho, dtype=complex)
        for K1 in kraus_list:
            K = self._lift_1q_op(K1, q)
            accum += K @ self.rho @ K.conj().T
        self.rho = accum

    def normalize_dm(self):
        # Numerical hygiene: enforce Hermiticity and unit trace
        self.rho = 0.5 * (self.rho + self.rho.conj().T)
        tr = np.trace(self.rho)
        if tr != 0:
            self.rho /= tr

    def get_probabilities_str(self):
        pops = np.real(np.diag(self.rho))
        lines = []
        for i, p in enumerate(pops):
            if not np.isclose(p, 0):
                b = format(i, f'0{self.num_of_qubits}b')
                lines.append(f"P(|{b}⟩) = {p:.3f}")
        return "\n".join(lines) if lines else "all ~0"

    def measure_all(self, num_of_measurements=1, rng=None):
        """
        Sample measurement outcomes from the diagonal of ρ in the
        computational basis (same convention as State.measure_all).
        """
        if rng is None:
            rng = np.random.default_rng()
        pops = np.real(np.diag(self.rho))
        pops = np.clip(pops, 0.0, 1.0)
        if pops.sum() == 0.0:
            pops = np.ones_like(pops) / pops.size
        else:
            pops = pops / pops.sum()

        basis_states = [format(i, f'0{self.num_of_qubits}b') for i in range(pops.size)]
        return rng.choice(basis_states, size=num_of_measurements, p=pops)


class States:
    def __init__(self):
        sqrt2_inv = 1 / np.sqrt(2)

        # one qubit
        self.zero = State(qubit_vector=np.array([[1], [0]], dtype=complex))
        self.one = State(qubit_vector=np.array([[0], [1]], dtype=complex))
        self.plus = State(qubit_vector=sqrt2_inv * np.array([[1], [1]], dtype=complex))
        self.minus = State(qubit_vector=sqrt2_inv * np.array([[1], [-1]], dtype=complex))
        self.i = State(qubit_vector=sqrt2_inv * np.array([[1], [1j]], dtype=complex))
        self.minus_i = State(qubit_vector=sqrt2_inv * np.array([[1], [-1j]], dtype=complex))

        # two qubits
        self.phi_plus = State(qubit_vector=sqrt2_inv * np.array([[1], [0], [0], [1]], dtype=complex))  # |Φ+⟩
        self.phi_minus = State(qubit_vector=sqrt2_inv * np.array([[1], [0], [0], [-1]], dtype=complex))  # |Φ-⟩
        self.psi_plus = State(qubit_vector=sqrt2_inv * np.array([[0], [1], [1], [0]], dtype=complex))  # |Ψ+⟩
        self.psi_minus = State(qubit_vector=sqrt2_inv * np.array([[0], [1], [-1], [0]], dtype=complex))  # |Ψ-⟩

    @staticmethod
    def generate_zero_n_qubit_state(n: int) -> State:
        if n <= 0:
            raise ValueError("n must be >= 0")

        qubit_vector = np.zeros((2**n, 1), dtype=complex)
        qubit_vector[0, 0] = 1.0
        return State(qubit_vector)

    @staticmethod
    def generate_state(states: [State]) -> State:
        if not states:
            raise ValueError("states must be a non-empty list of State")

        qubit_vector = np.array([[1]], dtype=complex)
        for state in states:
            qubit_vector = np.kron(qubit_vector, state.qubit_vector)

        return State(qubit_vector)


STATES = States()
