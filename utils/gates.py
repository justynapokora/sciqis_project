import numpy as np
from dataclasses import dataclass, field

ONE_QUBIT_FIXED_GATE_SET = np.array(["I", "X", "Y", "Z", "H", "S", "T"])
TWO_QUBITS_FIXED_GATE_SET = np.array(["CNOT", "CZ"])
FIXED_GATE_SET = np.concatenate([ONE_QUBIT_FIXED_GATE_SET, TWO_QUBITS_FIXED_GATE_SET])

ONE_QUBIT_PARAMETRISED_GATE_SET = np.array(["Rx", "Ry", "Rz"])
TWO_QUBITS_PARAMETRISED_GATE_SET = np.array(["CRx", "CRy", "CRz"])
PARAMETRISED_GATE_SET = np.concatenate([ONE_QUBIT_PARAMETRISED_GATE_SET, TWO_QUBITS_PARAMETRISED_GATE_SET])

ONE_QUBIT_GATES = np.concatenate([ONE_QUBIT_FIXED_GATE_SET, ONE_QUBIT_PARAMETRISED_GATE_SET])
TWO_QUBITS_GATES = np.concatenate([TWO_QUBITS_FIXED_GATE_SET, TWO_QUBITS_PARAMETRISED_GATE_SET])


@dataclass
class Gate:
    """Static description of a quantum gate (name, number of qubits, and matrices)."""
    name: str
    num_of_qubits: int
    target_qubit_matrix: np.ndarray
    control_qubit_matrix_0: np.ndarray = field(default_factory=lambda: np.array([]))
    control_qubit_matrix_1: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class CircuitGate:
    """A gate placed on specific qubits within a circuit (target and optional control)."""
    gate: Gate
    target_qubit: int
    control_qubit: int | None = None

    def __post_init__(self):
        """Validate that qubit indices are consistent with the gate’s number of qubits."""
        if self.target_qubit is None:
            raise ValueError(
                f"Gate '{self.gate}' requires {self.gate.num_of_qubits} qubit(s), "
                f"but no target qubit was defined"
            )

        if self.control_qubit is None:
            if self.gate.num_of_qubits != 1:
                raise ValueError(
                    f"Gate '{self.gate}' requires {self.gate.num_of_qubits} qubit(s), "
                    f"but no control qubit was defined"
                )

        else:
            if self.gate.num_of_qubits != 2:
                raise ValueError(
                    f"Gate '{self.gate}' requires {self.gate.num_of_qubits} qubit(s), "
                    f"but two qubits - target and control - are defined"
                )


class Gates:
    """Class for standard 1q/2q gates, including parameterized and controlled variants."""
    def __init__(self):
        self.I = self.init_I()

        # one qubit X Y Z H
        self.X = self.init_X()
        self.Y = self.init_Y()
        self.Z = self.init_Z()
        self.H = self.init_H()
        self.S = self.init_S()
        self.T = self.init_T()

        # parametrized - no matrix defined
        self.Rx = self.init_Rx()
        self.Ry = self.init_Ry()
        self.Rz = self.init_Rz()

        # two qubit CNOT CZ

        self.P0_matrix = np.array([
            [1, 0],
            [0, 0]
        ]).astype(complex)

        self.P1_matrix = np.array([
            [0, 0],
            [0, 1]
        ]).astype(complex)

        self.CNOT = self.init_CNOT()
        self.CZ = self.init_CZ()

        self.CRx = self.init_CRx()
        self.CRy = self.init_CRy()
        self.CRz = self.init_CRz()

        self.RESET = self.init_RESET()  # only for processing, no matrix

        self.GATE_DICT = {
            "I": self.I,
            "X": self.X,
            "Y": self.Y,
            "Z": self.Z,
            "H": self.H,
            "S": self.S,
            "T": self.T,

            "Rx": self.Rx,
            "Ry": self.Ry,
            "Rz": self.Rz,

            "CNOT": self.CNOT,
            "CZ": self.CZ,

            "CRx": self.CRx,
            "CRy": self.CRy,
            "CRz": self.CRz,

            "RESET": self.RESET
        }

        self.INIT_PARAMETRIZED_GATE_FUNC_DICT = {
            "Rx": self.init_Rx,
            "Ry": self.init_Ry,
            "Rz": self.init_Rz,

            "CRx": self.init_CRx,
            "CRy": self.init_CRy,
            "CRz": self.init_CRz,
        }

        self.INIT_PARAMETRIZED_GATE_MATRIX_FUNC_DICT = {
            "Rx": self._rx_matrix,
            "Ry": self._ry_matrix,
            "Rz": self._rz_matrix,

            "CRx": self._rx_matrix,
            "CRy": self._ry_matrix,
            "CRz": self._rz_matrix,
        }

    #### Definitions
    @staticmethod
    def init_I() -> Gate:
        """Identity gate."""
        return Gate(
            name="I",
            target_qubit_matrix=np.array([
                [1, 0],
                [0, 1]
            ]).astype(complex),
            num_of_qubits=1
        )

    @staticmethod
    def init_X() -> Gate:
        """Pauli-X gate."""
        return Gate(
            name="X",
            num_of_qubits=1,
            target_qubit_matrix=np.array([
                [0, 1],
                [1, 0]
            ]).astype(complex)
        )

    @staticmethod
    def init_Y() -> Gate:
        """Pauli-Y gate."""
        return Gate(
            name="Y",
            num_of_qubits=1,
            target_qubit_matrix=np.array([
                [0, -1j],
                [1j, 0]
            ]).astype(complex)
        )

    @staticmethod
    def init_Z() -> Gate:
        """Pauli-Z gate."""
        return Gate(
            name="Z",
            num_of_qubits=1,
            target_qubit_matrix=np.array([
                [1, 0],
                [0, -1]
            ]).astype(complex)
        )

    @staticmethod
    def init_H() -> Gate:
        """Hadamard gate."""
        return Gate(
            name="H",
            num_of_qubits=1,
            target_qubit_matrix=(1 / np.sqrt(2)) * np.array([
                [1, 1],
                [1, -1]
            ]).astype(complex)
        )

    @staticmethod
    def init_S() -> Gate:
        """Phase-S gate."""
        return Gate(
            name="S",
            num_of_qubits=1,
            target_qubit_matrix=np.array([
                [1, 0],
                [0, 1j]
            ]).astype(complex)
        )

    @staticmethod
    def init_T() -> Gate:
        """T gate (π/8 phase)."""
        return Gate(
            name="T",
            num_of_qubits=1,
            target_qubit_matrix=np.array([
                [1, 0],
                [0, np.exp(1j * np.pi / 4)]
            ]).astype(complex)
        )

    @staticmethod
    def _rx_matrix(theta):
        """Rotation-X unitary for angle θ."""
        sin = np.sin(theta / 2)
        cos = np.cos(theta / 2)
        return np.array([
            [cos, -1j * sin],
            [-1j * sin, cos]
        ]).astype(complex)

    @staticmethod
    def _ry_matrix(theta):
        """Rotation-Y unitary for angle θ."""
        sin = np.sin(theta / 2)
        cos = np.cos(theta / 2)
        return np.array([
            [cos, -sin],
            [sin, cos]
        ]).astype(complex)

    @staticmethod
    def _rz_matrix(theta):
        """Rotation-Z unitary for angle θ."""
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ]).astype(complex)

    def init_Rx(self, theta: float | None = None) -> Gate:
        """Parameterized Rx gate (placeholder if θ is None)."""
        if theta is None:
            name = "Rx"
            matrix = None
        else:
            name = f"Rx({theta:.3f})"
            matrix = self._rx_matrix(theta)

        return Gate(
            name=name,
            num_of_qubits=1,
            target_qubit_matrix=matrix
        )

    def init_Ry(self, theta: float | None = None) -> Gate:
        """Parameterized Ry gate (placeholder if θ is None)."""
        if theta is None:
            name = "Ry"
            matrix = None
        else:
            name = f"Ry({theta:.3f})"
            matrix = self._ry_matrix(theta)

        return Gate(
            name=name,
            num_of_qubits=1,
            target_qubit_matrix=matrix
        )

    def init_Rz(self, theta: float | None = None) -> Gate:
        """Parameterized Rz gate (placeholder if θ is None)."""
        if theta is None:
            name = "Rz"
            matrix = None
        else:
            name = f"Rz({theta:.3f})"
            matrix = self._rz_matrix(theta)

        return Gate(
            name=name,
            num_of_qubits=1,
            target_qubit_matrix=matrix
        )

    @staticmethod
    def init_Rk(k) -> Gate:  # dyadic rational phase gate (for Fourier transform)
        """Fixed R_k phase gate (dyadic rational phase)."""
        return Gate(
            name=f"R{k}",
            num_of_qubits=1,
            target_qubit_matrix=np.array([
                [1, 0],
                [0, np.exp(1j * 2 * np.pi / (2**k))]
            ]).astype(complex)
        )

    @staticmethod
    def init_U(gate_matrix: np.ndarray) -> Gate:
        """Arbitrary single-qubit unitary defined by a 2×2 matrix."""
        return Gate(
            name=f"U{gate_matrix.tolist()}",
            num_of_qubits=1,
            target_qubit_matrix=gate_matrix.astype(complex)
        )

    ### two qubits
    def init_CNOT(self) -> Gate:
        """Controlled-NOT gate (uses X as the target operation with P0/P1 projectors)."""
        return Gate(
            name="CNOT",
            num_of_qubits=2,
            target_qubit_matrix=self.X.target_qubit_matrix,
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )

    def init_CZ(self) -> Gate:
        """Controlled-Z gate (uses Z as the target operation with P0/P1 projectors)."""
        return Gate(
            name="CZ",
            num_of_qubits=2,
            target_qubit_matrix=self.Z.target_qubit_matrix,
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )

    def init_CRx(self, theta: float | None = None) -> Gate:
        """Controlled-Rx gate (placeholder if θ is None)."""
        if theta is None:
            name = "CRx"
            matrix = None
        else:
            name = f"CRx({theta:.3f})"
            matrix = self._rx_matrix(theta)

        return Gate(
            name=name,
            num_of_qubits=2,
            target_qubit_matrix=matrix,
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )

    def init_CRy(self, theta: float | None = None) -> Gate:
        if theta is None:
            name = "CRy"
            matrix = None
        else:
            name = f"CRy({theta:.3f})"
            matrix = self._ry_matrix(theta)

        return Gate(
            name=name,
            num_of_qubits=2,
            target_qubit_matrix=matrix,
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )

    def init_CRz(self, theta: float | None = None) -> Gate:
        if theta is None:
            name = "CRz"
            matrix = None
        else:
            name = f"CRz({theta:.3f})"
            matrix = self._rz_matrix(theta)

        return Gate(
            name=name,
            num_of_qubits=2,
            target_qubit_matrix=matrix,
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )

    def init_CRk(self, k: int) -> Gate:
        """Controlled-R_k phase gate."""
        return Gate(
            name=f"CR{k}",
            num_of_qubits=2,
            target_qubit_matrix=self.init_Rk(k).target_qubit_matrix,
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )

    def init_CU(self, gate_matrix: np.ndarray) -> Gate:
        """Controlled arbitrary single-qubit unitary."""
        return Gate(
            name=f"CU{gate_matrix.tolist()}",
            num_of_qubits=2,
            target_qubit_matrix=gate_matrix.astype(complex),
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )

    @staticmethod
    def init_RESET() -> Gate:
        """Reset placeholder gate (used for processing; no matrix action)."""
        return Gate(
            name="RESET",
            num_of_qubits=1,
            target_qubit_matrix=None  # handled specially
        )


GATES = Gates()
