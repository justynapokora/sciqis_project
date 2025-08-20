import numpy as np
from dataclasses import dataclass, field
from scipy.constants import h, k

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

    # ---  Kraus builders (1-qubit) ---
    def kraus_for_1q(self, gate_name: str, qubit: int) -> list[np.ndarray]:
        p = self.rate_for_gate(gate_name)
        if p <= 0.0:
            return [np.eye(2, dtype=complex)]
        sI = np.sqrt(1.0 - p)
        sP = np.sqrt(p / 3.0)
        I = np.eye(2, dtype=complex)
        X = GATES.X.target_qubit_matrix
        Y = GATES.Y.target_qubit_matrix
        Z = GATES.Z.target_qubit_matrix
        return [sI * I, sP * X, sP * Y, sP * Z]


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

    # --- Kraus builders ---
    def kraus_prep(self, qubit: int) -> list[np.ndarray]:
        p = self.p_prep(qubit)
        s0, s1 = np.sqrt(1.0 - p), np.sqrt(p)
        I = np.eye(2, dtype=complex)
        X = GATES.X.target_qubit_matrix
        return [s0 * I, s1 * X]

    def kraus_meas(self, qubit: int) -> list[np.ndarray]:
        p = self.p_meas(qubit)
        s0, s1 = np.sqrt(1.0 - p), np.sqrt(p)
        I = np.eye(2, dtype=complex)
        X = GATES.X.target_qubit_matrix
        return [s0 * I, s1 * X]


@dataclass
class TDCNoise:
    """
    Thermal decoherence channel (relaxation + excitation + dephasing).

    Parameters held per qubit:
      - T1, T2     : times [s], with T2 ≤ 2*T1 (clamped)
      - f (freq)   : qubit frequency [Hz]
      - T (temp)   : physical temperature [K]

    Per gate:
      - Tg         : duration [s] from gate_durations

    Probabilities per gate on a qubit:
      p_T1   = exp(-Tg/T1)
      p_T2   = exp(-Tg/T2)
      p_reset = 1 - p_T1
      w_e    = 1 / (1 + exp(h f / (k_B T)))   # equilibrium excited-state pop
      (split) p_reset0 = (1 - w_e) * p_reset  # decay → |0⟩
              p_reset1 = w_e * p_reset        # excitation → |1⟩
      p_Z    = (1 - p_reset) * (1 - p_T2/p_T1) / 2
      p_I    = 1 - p_Z - p_reset0 - p_reset1

    Two regimes:
      • T2 ≤ T1     → explicit Kraus: {√p_I I, √p_Z Z, √p_reset0 |0⟩⟨0|, √p_reset1 |1⟩⟨1|}
      • T1 < T2 ≤ 2T1 → build Kraus from the Choi matrix via SVD.

    Notes:
      - Set frequencies/temperatures via setters or defaults; w_e is always computed at call time.
      - Gates with Tg = 0 return identity Kraus.
    """

    # Defaults (can be overridden per qubit)
    T1_default: float = 50e-6
    T2_default: float = 50e-6
    f_default: float = 5.0e9  # 5 GHz typical SC qubit
    temp_default: float = 0.015  # 15 mK typical dilution fridge

    # Gate-duration table [s]
    gate_durations: dict[str, float] = field(default_factory=dict)

    # Per-qubit overrides
    T1_overrides: dict[int, float] = field(default_factory=dict)
    T2_overrides: dict[int, float] = field(default_factory=dict)
    freq_overrides: dict[int, float] = field(default_factory=dict)
    temp_overrides: dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.gate_durations:
            self.gate_durations = {
                # single-qubit
                "I": 0.0, "X": 35e-9, "Y": 35e-9, "Z": 0.0,
                "H": 35e-9, "S": 0.0, "T": 0.0,
                "Rx": 35e-9, "Ry": 35e-9, "Rz": 0.0,
                # two-qubit
                "CNOT": 250e-9, "CZ": 250e-9,
                "CRx": 250e-9, "CRy": 250e-9, "CRz": 250e-9,
                # placeholders
                "RESET": 0.0, "SPAM": 0.0, "DC": 0.0, "TDC": 0.0,
            }

    # ---------- setters ----------
    def set_T1(self, q: int, T1: float):
        self.T1_overrides[q] = float(T1)

    def set_T2(self, q: int, T2: float):
        self.T2_overrides[q] = float(T2)

    def set_frequency(self, q: int, f_hz: float):
        self.freq_overrides[q] = float(f_hz)

    def set_temperature(self, q: int, T_K: float):
        self.temp_overrides[q] = float(T_K)

    def set_gate_duration(self, gate: str, seconds: float):
        self.gate_durations[base_name(gate)] = float(seconds)

    # ---------- accessors ----------
    def _T1(self, q: int) -> float:
        return self.T1_overrides.get(q, self.T1_default)

    def _T2(self, q: int) -> float:
        T1 = self._T1(q)
        T2 = self.T2_overrides.get(q, self.T2_default)
        return min(T2, 2.0 * T1)

    def _freq(self, q: int) -> float:
        return self.freq_overrides.get(q, self.f_default)

    def _temp(self, q: int) -> float:
        return self.temp_overrides.get(q, self.temp_default)

    def _Tg(self, gate_name: str) -> float:
        return self.gate_durations.get(base_name(gate_name), 0.0)

    @staticmethod
    def _we_from_f_T(freq_hz: float, temp_K: float) -> float:
        """
        Equilibrium excited-state population:
            w_e = 1 / (1 + exp(h f / (k_B T)))
        """
        x = (h * freq_hz) / (k * max(temp_K, 1e-12))  # guard T→0
        try:
            return 1.0 / (1.0 + np.exp(x))
        except OverflowError:
            return 0.0

    # ---------- main ----------
    def kraus_for(self, gate_name: str, qubit: int) -> list[np.ndarray]:
        """
        Return Kraus operators for thermal decoherence on `qubit` after `gate_name`.
        """
        Tg = self._Tg(gate_name)
        if Tg <= 0.0:
            return [np.eye(2, dtype=complex)]

        T1 = self._T1(qubit)
        T2 = self._T2(qubit)
        f = self._freq(qubit)
        T = self._temp(qubit)

        pT1 = np.exp(-Tg / T1)
        pT2 = np.exp(-Tg / T2)
        p_reset = 1.0 - pT1

        we = self._we_from_f_T(f, T)  # compute each call
        p_reset0 = (1.0 - we) * p_reset
        p_reset1 = we * p_reset

        ratio = (pT2 / pT1) if pT1 > 0.0 else 0.0
        p_Z = max(0.0, (1.0 - p_reset) * (1.0 - ratio) / 2.0)
        p_I = 1.0 - p_Z - p_reset0 - p_reset1

        # Case A: T2 ≤ T1 → explicit Kraus
        if T2 <= T1:
            I2 = np.eye(2, dtype=complex)
            Z2 = GATES.Z.target_qubit_matrix
            K_I = np.sqrt(max(0.0, p_I)) * I2
            K_Z = np.sqrt(max(0.0, p_Z)) * Z2
            K_r0 = np.sqrt(max(0.0, p_reset0)) * np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|
            K_r1 = np.sqrt(max(0.0, p_reset1)) * np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|
            return [K_I, K_Z, K_r0, K_r1]

        # Case B: T1 < T2 ≤ 2T1 → Choi matrix with excitation included
        # Build C in the standard 2×2 block form:
        #   C = [[E(|0><0|), E(|0><1|)],
        #        [E(|1><0|), E(|1><1|)]]
        #
        # where:
        #   E(|0><0|) = (1 - p_reset1) |0><0| + p_reset1 |1><1|
        #   E(|1><1|) = p_reset0 |0><0| + (1 - p_reset0) |1><1|
        #   E(|0><1|) = pT2 |0><1|,   E(|1><0|) = pT2 |1><0|
        #
        # In the computational basis { |00>, |01>, |10>, |11> } this gives:
        C = np.array([
            [1.0 - p_reset1, 0.0, 0.0, pT2],
            [0.0, p_reset1, 0.0, 0.0],
            [0.0, 0.0, p_reset0, 0.0],
            [pT2, 0.0, 0.0, 1.0 - p_reset0]
        ], dtype=complex)

        # Kraus operators from SVD (numerically stable, CPTP if C is valid)
        U, s, _ = np.linalg.svd(C)
        kraus_ops = []
        for sv, col in zip(s, U.T):
            if sv > 1e-12:
                K = (np.sqrt(sv) * col).reshape(2, 2)
                kraus_ops.append(K)
        return kraus_ops

