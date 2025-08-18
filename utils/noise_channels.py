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
    p_prep_default: float = 1e-3   # 0.1%       # probability that preparation flips (X) the intended state
    p_meas_default: float = 2e-2   # 2%         # probability that a measurement reports X-flipped outcome

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


# (iii) TRC: thermal relaxation (T1) + dephasing (T2) per qubit
@dataclass
class TRCNoise:
    # defaults if a qubit or gate has no override
    T1_default: float = 50e-6      # seconds
    T2_default: float = 50e-6      # seconds
    gate_durations: dict[str, float] = field(default_factory=dict)  # e.g. {"X": 35e-9, "CNOT": 250e-9}

    # per-qubit overrides
    T1_overrides: dict[int, float] = field(default_factory=dict)
    T2_overrides: dict[int, float] = field(default_factory=dict)

    @staticmethod
    def _validate_time(t: float, name: str):
        if t <= 0.0:
            raise ValueError(f"{name} must be > 0, got {t}")

    def set_T1(self, qubit: int, T1: float):
        self._validate_time(T1, "T1")
        self.T1_overrides[qubit] = T1

    def set_T2(self, qubit: int, T2: float):
        self._validate_time(T2, "T2")
        self.T2_overrides[qubit] = T2

    def set_gate_duration(self, gate: str, seconds: float):
        self._validate_time(seconds, "gate duration"); self.gate_durations[base_name(gate)] = seconds

    # helpers
    def _T1(self, qubit: int) -> float:
        return self.T1_overrides.get(qubit, self.T1_default)

    def _T2(self, qubit: int) -> float:
        # Clamp to the physical constraint T2 <= 2*T1 (paper notes this relation)
        T1 = self._T1(qubit)
        T2 = self.T2_overrides.get(qubit, self.T2_default)
        return min(T2, 2.0 * T1)

    def _Tg(self, gate_name: str) -> float:
        b = base_name(gate_name)
        if b in self.gate_durations:
            return self.gate_durations[b]
        # Fallback: zero-time "virtual" identity (common for frame changes), else a tiny default to avoid 0
        return 0.0

    # --- main API: per-qubit TRC probabilities for a gate
    def probs_for(self, gate_name: str, qubit: int) -> dict[str, float]:
        """
        Returns {'p_reset0', 'p_Z', 'p_I'} for this qubit due to TRC after `gate_name`.
        Model uses Θ≈0K approximation (no excitation), as in the paper.
        """
        Tg = self._Tg(gate_name)
        if Tg <= 0.0:
            # No time passes: no TRC
            return {"p_reset0": 0.0, "p_Z": 0.0, "p_I": 1.0}

        T1 = self._T1(qubit)
        T2 = self._T2(qubit)

        pT1 = np.exp(-Tg / T1)          # survival wrt T1
        pT2 = np.exp(-Tg / T2)          # survival wrt T2
        p_reset0 = 1.0 - pT1            # reset to |0>
        # guard small divisions
        ratio = pT2 / pT1 if pT1 > 0.0 else 0.0
        p_Z = (1.0 - p_reset0) * (1.0 - ratio) / 2.0
        p_I = 1.0 - p_Z - p_reset0

        # Numerical clipping
        def clip01(x): return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
        return {"p_reset0": clip01(p_reset0), "p_Z": clip01(p_Z), "p_I": clip01(p_I)}

    # convenience for 2-qubit gates (acts independently on each qubit)
    def probs_for_2q(self, gate_name: str, q_ctrl: int, q_tgt: int) -> tuple[dict[str, float], dict[str, float]]:
        return self.probs_for(gate_name, q_ctrl), self.probs_for(gate_name, q_tgt)

    # You can also ask whether we should apply TRC after a gate (e.g., skip for virtual Z / parameterized phases)
    def has_time(self, gate_name: str) -> bool:
        return self._Tg(gate_name) > 0.0

    def sample_error_gate(self, gate_name: str, qubit: int, rng: np.random.Generator) -> CircuitGate | None:
        probs = self.probs_for(gate_name, qubit)
        choice = rng.choice(
            ["I", "Z", "RESET"],
            p=[probs["p_I"], probs["p_Z"], probs["p_reset0"]],
        )
        if choice == "I":
            return None
        if choice == "Z":
            return CircuitGate(GATES.Z, target_qubit=qubit)
        if choice == "RESET":
            return CircuitGate(GATES.RESET, target_qubit=qubit)

