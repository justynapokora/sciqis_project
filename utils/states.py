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
        self.measured_qubits = []

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

        self.measured_qubits.append(qubit_index)

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
