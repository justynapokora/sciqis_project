import numpy as np
from dataclasses import dataclass, field


@dataclass
class Gate:
    name: str
    num_of_qubits: int
    target_qubit_matrix: np.ndarray
    control_qubit_matrix_0: np.ndarray = field(default_factory=lambda: np.array([]))
    control_qubit_matrix_1: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class CircuitGate:
    gate: Gate
    target_qubit: int
    control_qubit: int | None = None

    def __post_init__(self):

        # Validate the number of target qubits matches gate's definition
        if self.target_qubit is None:
            raise ValueError(
                f"Gate '{self.gate}' requires {self.gate.num_of_qubits} qubit(s), "
                f"but no target qubit was defined"
            )

        if self.control_qubit is None:
            if self.gate.num_of_qubits is not 1:
                raise ValueError(
                    f"Gate '{self.gate}' requires {self.gate.num_of_qubits} qubit(s), "
                    f"but no control qubit was defined"
                )

        else:
            if self.gate.num_of_qubits is not 2:
                raise ValueError(
                    f"Gate '{self.gate}' requires {self.gate.num_of_qubits} qubit(s), "
                    f"but two qubits - target and control - are defined"
                )


class Gates:
    def __init__(self):
        self.I = self.init_I()

        # one qubit X Y Z H
        self.X = self.init_X()
        self.Y = self.init_Y()
        self.Z = self.init_Z()
        self.H = self.init_H()
        self.S = self.init_S()
        self.T = self.init_T()

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

    #### Definitions
    @staticmethod
    def init_I() -> Gate:
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
        return Gate(
            name="T",
            num_of_qubits=1,
            target_qubit_matrix=np.array([
                [1, 0],
                [0, np.exp(1j * np.pi / 4)]
            ]).astype(complex)
        )

    @staticmethod
    def init_Rx(theta) -> Gate:
        sin = np.sin(theta/2)
        cos = np.cos(theta/2)

        return Gate(
            name=f"Rx({theta})",
            num_of_qubits=1,
            target_qubit_matrix=np.array([
                [cos, -1j * sin],
                [-1j * sin, cos]
            ]).astype(complex)
        )

    @staticmethod
    def init_Ry(theta) -> Gate:
        sin = np.sin(theta / 2)
        cos = np.cos(theta / 2)

        return Gate(
            name=f"Ry({theta})",
            num_of_qubits=1,
            target_qubit_matrix=np.array([
                [cos, -sin],
                [sin, cos]
            ]).astype(complex)
        )

    @staticmethod
    def init_Rz(theta) -> Gate:
        return Gate(
            name=f"Rz({theta})",
            num_of_qubits=1,
            target_qubit_matrix=np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)]
            ]).astype(complex)
        )

    @staticmethod
    def init_Rk(k) -> Gate:  # dyadic rational phase gate (Fourier transform)
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
        return Gate(
            name=f"U{gate_matrix.tolist()}",
            num_of_qubits=1,
            target_qubit_matrix=gate_matrix.astype(complex)
        )

    ### two qubits
    # @staticmethod
    def init_CNOT(self) -> Gate:
        return Gate(
            name="CNOT",
            num_of_qubits=2,
            target_qubit_matrix=self.X.target_qubit_matrix,
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )

    def init_CZ(self) -> Gate:
        return Gate(
            name="CZ",
            num_of_qubits=2,
            target_qubit_matrix=self.Z.target_qubit_matrix,
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )

    def init_CRx(self, theta) -> Gate:
        return Gate(
            name=f"CRx({theta})",
            num_of_qubits=2,
            target_qubit_matrix=self.init_Rx(theta).target_qubit_matrix,
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )

    def init_CRy(self, theta) -> Gate:
        return Gate(
            name=f"CRy({theta})",
            num_of_qubits=2,
            target_qubit_matrix=self.init_Ry(theta).target_qubit_matrix,
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )

    def init_CRz(self, theta) -> Gate:
        return Gate(
            name=f"CRz({theta})",
            num_of_qubits=2,
            target_qubit_matrix=self.init_Rz(theta).target_qubit_matrix,
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )

    def init_CRk(self, k: int) -> Gate:
        return Gate(
            name=f"CR{k}",
            num_of_qubits=2,
            target_qubit_matrix=self.init_Rk(k).target_qubit_matrix,
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )

    def init_CU(self, gate_matrix: np.ndarray) -> Gate:
        return Gate(
            name=f"CU{gate_matrix.tolist()}",
            num_of_qubits=2,
            target_qubit_matrix=gate_matrix.astype(complex),
            control_qubit_matrix_0=self.P0_matrix,
            control_qubit_matrix_1=self.P1_matrix
        )


GATES = Gates()
