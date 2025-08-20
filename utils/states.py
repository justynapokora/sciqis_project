from __future__ import annotations

import numpy as np
from utils.gates import GATES


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
        """Compact ket expansion, skipping zero amplitudes."""
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

    def _full_basis_op(self, basis: str) -> np.ndarray:
        """⊗ over all qubits of the per-qubit mapping for a uniform basis ('X','Y','Z')."""
        if len(basis) != 1:
            raise ValueError("Only uniform bases supported here: basis must be one of 'X','Y','Z'.")
        Uq = _oneq_basis_op(basis)
        ops = [Uq for _ in range(self.num_of_qubits)]
        U = np.array([[1]], dtype=complex)
        for op in ops:
            U = np.kron(op, U)
        return U

    def get_probabilities_dict(self, basis: str = "Z") -> dict[str, float]:
        """
        Probabilities for measuring in a UNIFORM basis across all qubits.
        basis ∈ {'Z','X','Y'}; labels are:
          Z: '0'/'1',  X: '+/-',  Y: '+i'/'-i' per qubit, concatenated.
        """
        b = basis.upper()

        # Map to Z-basis by applying the appropriate unitary, then square amplitudes
        U = self._full_basis_op(b)
        psi_prime = (U @ self.qubit_vector).ravel()
        probs = np.abs(psi_prime) ** 2
        s = probs.sum()
        if s > 0:
            probs = probs / s  # guard tiny drift

        labels = [_label_bits(self.num_of_qubits, i, b) for i in range(psi_prime.size)]
        return dict(zip(labels, probs.astype(float)))

    def get_probabilities_str(self, print_zero_probabilities: bool = False, basis: str = "Z") -> str:
        """Pretty print non-zero probabilities for basis ∈ {'X','Y','Z'}."""
        probs = self.get_probabilities_dict(basis)
        if print_zero_probabilities:
            lines = [f"P(|{b}⟩) = {p:.5f}" for b, p in probs.items()]
        else:
            lines = [f"P(|{b}⟩) = {p:.5f}" for b, p in probs.items() if not np.isclose(p, 0, atol=1e-5)]
        return "\n".join(lines) if lines else "Invalid state: no nonzero probabilities (check normalization)"

    def normalize_state(self):
        """Normalize the state vector to unit norm."""
        norm = np.linalg.norm(self.qubit_vector)
        if norm != 0:
            self.qubit_vector /= norm

    def measure_all(self, num_of_measurements=1, rng: np.random.Generator | None = None):
        """Sample bitstrings from |ψ|² in the computational basis."""
        if not rng:
            rng = np.random.default_rng()

        self.normalize_state()

        # Flatten to 1D and compute probabilities
        flat = self.qubit_vector.ravel()
        probs = np.abs(flat) ** 2

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
    Density-matrix holder for the noisy states.
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
        """Construct ρ = |ψ⟩⟨ψ| from a pure state."""
        # state.qubit_vector is (2^n, 1)
        psi = state.qubit_vector
        rho = psi @ psi.conj().T
        return cls(rho)

    def __str__(self, tol=1e-10, max_terms=2000):
        """
        Pretty-print:
          - If ρ is (numerically) pure: print the ket expansion (like State.__str__).
          - Else: print a sparse operator expansion Σ_ij ρ_ij |i><j|.
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
        """Apply a full-system unitary: ρ ← U ρ U†."""
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
        """Apply a 1-qubit CPTP channel on qubit q via Kraus operators."""
        # ρ ← Σ K ρ K†
        accum = np.zeros_like(self.rho, dtype=complex)

        Ks = [self._lift_1q_op(K1, q) for K1 in kraus_list]
        for K in Ks:
            accum += K @ self.rho @ K.conj().T

        self.rho = accum

    def normalize_dm(self):
        """Hermitize and renormalize ρ to unit trace."""
        self.rho = 0.5 * (self.rho + self.rho.conj().T)
        tr = np.trace(self.rho)
        if tr != 0:
            self.rho /= tr

    def _full_basis_op(self, basis: str) -> np.ndarray:
        """⊗ over all qubits of the per-qubit mapping for a uniform basis ('X','Y','Z')."""
        if len(basis) != 1:
            raise ValueError("Only uniform bases supported: 'X', 'Y', or 'Z'.")
        Uq = _oneq_basis_op(basis)
        ops = [Uq for _ in range(self.num_of_qubits)]
        U = np.array([[1]], dtype=complex)
        for op in ops:
            U = np.kron(op, U)
        return U

    # ---------- dictionaries ----------
    def get_probabilities_dict(self, basis: str = "Z") -> dict[str, float]:
        """
        Probabilities for measuring in a UNIFORM basis across all qubits.
        basis ∈ {'Z','X','Y'}; labels are:
          Z: '0'/'1',  X: '+/-',  Y: '+i'/'-i' per qubit, concatenated.
        """
        b = basis.upper()

        # Rotate density matrix to Z-basis frame for the chosen basis: ρ' = U ρ U†
        U = self._full_basis_op(b)
        rho_prime = U @ self.rho @ U.conj().T
        pops = np.real(np.diag(rho_prime))

        probs = {}
        for i, p in enumerate(pops):
            probs[_label_bits(self.num_of_qubits, i, b)] = float(p)
        return probs

    def get_probabilities_str(self, print_zero_probabilities: bool = False, basis: str = "Z") -> str:
        """
        Pretty print non-zero probabilities in the chosen basis ('X','Y','Z').
        """
        probs = self.get_probabilities_dict(basis)
        if print_zero_probabilities:
            lines = [f"P(|{b}⟩) = {p:.5f}" for b, p in probs.items()]
        else:
            lines = [f"P(|{b}⟩) = {p:.5f}" for b, p in probs.items() if not np.isclose(p, 0, atol=1e-5)]
        return "\n".join(lines) if lines else "Invalid state: no nonzero probabilities (check normalization)"

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


### functions common for state and density state
def _oneq_basis_op(basis_char: str) -> np.ndarray:
    """Single-qubit unitary that maps a B-basis measurement to Z-basis measurement."""
    b = basis_char.upper()
    I = np.eye(2, dtype=complex)
    H = GATES.H.target_qubit_matrix
    S = GATES.S.target_qubit_matrix
    Sdg = S.conj().T
    if b == "Z":
        return I
    if b == "X":
        return H
    if b == "Y":
        return H @ Sdg
    raise ValueError(f"Unknown basis '{basis_char}'. Use 'X', 'Y', or 'Z'.")


def _label_bits(num_of_qubits: int, index: int, basis: str) -> str:
    """Map computational index → per-qubit label string in the requested basis."""
    b = basis.upper()
    bits = format(index, f'0{num_of_qubits}b')
    if b == "Z":
        return bits  # '0'/'1'
    if b == "X":
        return "".join("+" if c == "0" else "-" for c in bits)  # +/-
    if b == "Y":
        return "".join("+i" if c == "0" else "-i" for c in bits)  # +i/-i
    raise ValueError(f"Unknown basis '{basis}'.")


class States:
    """Convenience container for common 1-qubit and 2-qubit basis/Bell states."""
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
        """|0⟩^{⊗ n} as a State object."""
        if n <= 0:
            raise ValueError("n must be >= 0")

        qubit_vector = np.zeros((2 ** n, 1), dtype=complex)
        qubit_vector[0, 0] = 1.0
        return State(qubit_vector)

    @staticmethod
    def generate_state(states: [State]) -> State:
        """Kronecker product of provided single-/multi-qubit State objects."""
        if not states:
            raise ValueError("states must be a non-empty list of State")

        qubit_vector = np.array([[1]], dtype=complex)
        for state in states:
            qubit_vector = np.kron(qubit_vector, state.qubit_vector)

        return State(qubit_vector)


STATES = States()
