import numpy as np
from dataclasses import dataclass, field

from utils.gates import ONE_QUBIT_GATES, TWO_QUBITS_GATES, CircuitGate, GATES


def base_name(name: str) -> str:
    return name.split("(", 1)[0]


# (i) depolarizing (gate infidelities)
@dataclass
class DepolarizingNoise:
    p1_1q: float = 1e-3  # default 1q depol prob per gate
    p1_2q: float = 5e-3  # default 2q depol prob per gate
    overrides: dict[str, float] = field(default_factory=dict)  # e.g. {"CNOT": 0.02, "Rz": 5e-4}

    @staticmethod
    def _validate(p: float):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability must be in [0,1], got {p}")

    def set_default_1q(self, p: float):
        self._validate(p)
        self.p1_1q = p

    def set_default_2q(self, p: float):
        self._validate(p)
        self.p1_2q = p

    def set_gate(self, gate: str, p: float):
        self._validate(p)
        self.overrides[gate] = p

    def rate_for_gate(self, gate_name: str) -> float:
        b = base_name(gate_name)
        if b in self.overrides:
            return self.overrides[b]
        if b == 'I':
            return 0
        if b in ONE_QUBIT_GATES:
            return self.p1_1q
        if b in TWO_QUBITS_GATES:
            return self.p1_2q
        # fallback: treat unknown as 1q
        return self.p1_1q

    def sample_error_gate(self, gate_name: str, qubit: int, rng: np.random.Generator) -> CircuitGate | None:
        p = self.rate_for_gate(gate_name)
        if p <= 0:
            return None
        probs = [1 - p, p / 3, p / 3, p / 3]
        choice = rng.choice(["I", "X", "Y", "Z"], p=probs)
        if choice == "I":
            return None

        return CircuitGate(GATES.GATE_DICT[choice], target_qubit=qubit)


# (ii) SPAM: state preparation & measurement X-flip errors
@dataclass
class SPAMNoise:
    # defaults if a qubit doesn't have an override
    p_prep_default: float = 1e-3  # 0.1%       # probability that preparation flips (X) the intended state
    p_meas_default: float = 2e-2  # 2%         # probability that a measurement reports X-flipped outcome

    # per-qubit overrides
    prep_overrides: dict[int, float] = field(default_factory=dict)
    meas_overrides: dict[int, float] = field(default_factory=dict)

    @staticmethod
    def _validate(p: float):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability must be in [0,1], got {p}")

    # --- mutators
    def set_default_prep(self, p: float):
        self._validate(p)
        self.p_prep_default = p

    def set_default_meas(self, p: float):
        self._validate(p)
        self.p_meas_default = p

    def set_prep_qubit(self, q: int, p: float):
        self._validate(p)
        self.prep_overrides[q] = p

    def set_meas_qubit(self, q: int, p: float):
        self._validate(p)
        self.meas_overrides[q] = p

    # --- accessors you can call where you apply prep/measure
    def p_prep(self, qubit: int) -> float:
        return self.prep_overrides.get(qubit, self.p_prep_default)

    def p_meas(self, qubit: int) -> float:
        return self.meas_overrides.get(qubit, self.p_meas_default)

    def sample_prep_error(self, qubit: int, rng: np.random.Generator) -> CircuitGate | None:
        p = self.p_prep(qubit)
        if rng.random() < p:
            return CircuitGate(GATES.X, target_qubit=qubit)
        return None

    def sample_meas_error(self, qubit: int, rng: np.random.Generator) -> CircuitGate | None:
        p = self.p_meas(qubit)
        if rng.random() < p:
            return CircuitGate(GATES.X, target_qubit=qubit)
        return None


# (iii) TRC: thermal relaxation (T1) + dephasing (T2) per qubit
@dataclass
class TRCNoise:
    # Device parameters (per paper; Θ≈0 ⇒ no excitation path by default)
    T1_default: float = 50e-6
    T2_default: float = 50e-6

    # Gate-duration table [seconds]. Nonzero defaults so TRC is effective.
    gate_durations: dict[str, float] = field(default_factory=dict)

    # Optional per-qubit overrides
    T1_overrides: dict[int, float] = field(default_factory=dict)
    T2_overrides: dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        print(self.T1_default, self.T2_default)
        # Provide realistic default durations if none supplied (you can still override)
        if not self.gate_durations:
            self.gate_durations = {
                # single-qubit
                "I": 0.0, "X": 35e-9, "Y": 35e-9, "Z": 0.0,  # virtual Z
                "H": 35e-9, "S": 0.0, "T": 0.0,
                "Rx": 35e-9, "Ry": 35e-9, "Rz": 0.0,
                # two-qubit (example superconducting scale)
                "CNOT": 250e-9, "CZ": 250e-9,
                "CRx": 250e-9, "CRy": 250e-9, "CRz": 250e-9,
                # placeholders never consume time
                "RESET": 0.0, "SPAM": 0.0, "DC": 0.0, "TRC": 0.0,
            }

    # ----------------- setters -----------------
    @staticmethod
    def _validate_time(t: float, name: str):
        if t < 0.0:
            raise ValueError(f"{name} must be ≥ 0, got {t}")

    def set_T1(self, qubit: int, T1: float):
        self._validate_time(T1, "T1")
        if T1 == 0.0:
            raise ValueError("T1 must be > 0")
        self.T1_overrides[qubit] = T1

    def set_T2(self, qubit: int, T2: float):
        self._validate_time(T2, "T2")
        if T2 == 0.0:
            raise ValueError("T2 must be > 0")
        self.T2_overrides[qubit] = T2

    def set_gate_duration(self, gate: str, seconds: float):
        self._validate_time(seconds, "gate duration")
        self.gate_durations[base_name(gate)] = seconds

    # ----------------- helpers -----------------
    def _T1(self, qubit: int) -> float:
        return self.T1_overrides.get(qubit, self.T1_default)

    def _T2(self, qubit: int) -> float:
        # Physical constraint T2 ≤ 2 T1 (paper)
        T1 = self._T1(qubit)
        T2 = self.T2_overrides.get(qubit, self.T2_default)
        return min(T2, 2.0 * T1)

    def _Tg(self, gate_name: str) -> float:
        return self.gate_durations.get(base_name(gate_name), 0.0)

    def has_time(self, gate_name: str) -> bool:
        return self._Tg(gate_name) > 0.0

    # ----------------- probabilities (paper, Θ≈0) -----------------
    def _probs_T2_le_T1(self, Tg: float, T1: float, T2: float) -> dict[str, float]:
        # Eqns from the paper (low-T), valid when T2 ≤ T1.
        pT1 = np.exp(-Tg / T1)
        pT2 = np.exp(-Tg / T2)
        p_reset = 1.0 - pT1
        # pZ as in text; guard numerical drift
        ratio = pT2 / pT1 if pT1 > 0.0 else 0.0
        p_Z = (1.0 - p_reset) * (1.0 - ratio) / 2.0
        p_Z = max(0.0, p_Z)  # clamp tiny negative due to FP error
        p_I = 1.0 - p_reset - p_Z
        # final clamp
        def clip01(x): return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        return {"p_I": clip01(p_I), "p_Z": clip01(p_Z), "p_reset": clip01(p_reset)}

    def _lambda_dephase(self, Tg: float, T1: float, T2: float) -> float:
        # Compose amplitude damping (γ) and extra pure dephasing so that coherences ~ e^{-Tg/T2}.
        # 1/T2 = 1/(2T1) + 1/Tφ  ⇒  Tφ = 1 / (1/T2 - 1/(2T1))
        denom = (1.0 / T2) - (1.0 / (2.0 * T1))
        if denom <= 0:
            return 0.0
        Tphi = 1.0 / denom
        # For a dephasing (Pauli-Z) channel: ρ -> (1-λ)ρ + λ ZρZ, off-diagonals scale by (1-2λ) = e^{-Tg/Tφ}
        return (1.0 - np.exp(-Tg / Tphi)) / 2.0

    # ----------------- sampling API -----------------
    def sample_error_gates(self, gate_name: str, qubit: int, rng: np.random.Generator) -> list[CircuitGate]:
        """
        Faithful to the paper:
        - If Tg = 0: no TRC.
        - If T2 ≤ T1: single-draw among {RESET, Z, I} with probabilities from the paper.
        - If T1 < T2 ≤ 2T1: sequential composition — first amplitude damping (RESET with γ),
          then (only if no RESET) pure dephasing as a Pauli-Z with probability λ(Tφ).
        Returns a list of CircuitGate noise operations to apply on that qubit in this time step (often 0 or 1 element).
        """
        Tg = self._Tg(gate_name)
        if Tg <= 0.0:
            return []

        T1 = self._T1(qubit)
        T2 = self._T2(qubit)
        # print("sample error gate")
        # print(Tg, T1, T2)

        # --- case 1: T2 ≤ T1  (paper's mixed channel with Kraus {√p_I I, √p_Z Z, √p_reset |0><0|})
        if T2 <= T1:
            probs = self._probs_T2_le_T1(Tg, T1, T2)
            # print(f"probs: p_I = {probs['p_I']}, p_Z = {probs['p_Z']}, p_RESET = {probs['p_reset']},")
            #
            # one multinomial draw
            choice = rng.choice(["I", "Z", "RESET"], p=[probs["p_I"], probs["p_Z"], probs["p_reset"]])
            if choice == "Z":
                return [CircuitGate(GATES.Z, target_qubit=qubit)]
            if choice == "RESET":
                return [CircuitGate(GATES.RESET, target_qubit=qubit)]
            return []

        # --- case 2: T1 < T2 ≤ 2T1  (paper uses a Choi representation; we realize the same decay via AD + dephasing)
        # amplitude damping first
        gamma = 1.0 - np.exp(-Tg / T1)  # jump (reset-to-|0>) probability
        if rng.random() < gamma:
            return [CircuitGate(GATES.RESET, target_qubit=qubit)]

        # then extra pure dephasing to fulfill T2
        lam = self._lambda_dephase(Tg, T1, T2)
        if lam > 0.0 and rng.random() < lam:
            return [CircuitGate(GATES.Z, target_qubit=qubit)]

        return []

    # Backward-compatible adapter (returns just one gate or None)
    def sample_error_gate(self, gate_name: str, qubit: int, rng: np.random.Generator):
        gates = self.sample_error_gates(gate_name, qubit, rng)
        # print(gates)

        return gates[0] if gates else None
