from __future__ import annotations

import numpy as np
import copy

from utils.gates import CircuitGate, GATES, PARAMETRISED_GATE_SET
from utils.states import State


class Circuit:
    def __init__(self, state: State, gates: list[list[CircuitGate]], rng: np.random.Generator | None = None):
        self.initial_state: State = copy.deepcopy(state)
        self.state: State = state
        self.gates: list[list[CircuitGate]] = gates
        self.measured_qubits: list = []

        self.rng = rng
        if not rng:
            self.rng = np.random.default_rng()

    def reset_circuit(self):
        self.state = copy.deepcopy(self.initial_state)
        self.measured_qubits = []

    def simulate_circuit(self):
        print(f"initial state ({self.initial_state.num_of_qubits} qubit(s)): {self.initial_state}")

        for layer in self.gates:
            for g in layer:
                self.check_gate_validity(g)

                target_qubit_matrix = g.gate.target_qubit_matrix
                if target_qubit_matrix is None:
                    target_qubit_matrix = self.get_parameterised_gate_random_matrix(g.gate.name)

                match g.gate.num_of_qubits:
                    case 1:
                        self.apply_one_qubit_gate(g, target_qubit_matrix)
                    case 2:
                        self.apply_two_qubit_gate(g, target_qubit_matrix)
                    case default:
                        raise ValueError(
                            f"Incorrect number of qubits for gate: {g}, "
                            f"allowed number of qubits: 1, 2"
                        )

    def get_parameterised_gate_random_matrix(self, gate_name):
        base = gate_name.split("(", 1)[0]  # e.g. "Rx" from "Rx" or "Rx(…)"
        init_function = GATES.INIT_PARAMETRIZED_GATE_MATRIX_FUNC_DICT.get(base)
        if init_function is None:
            raise ValueError(f"No initialization function for param gate '{gate_name}'")

        theta = self.rng.uniform(0.0, 2.0 * np.pi)
        return init_function(theta)

    def check_gate_validity(self, g: CircuitGate):
        max_qubit_val = g.target_qubit
        if g.control_qubit is not None:
            max_qubit_val = np.max([g.target_qubit, g.control_qubit])

        if max_qubit_val > self.state.num_of_qubits:
            raise ValueError(
                f"Incorrect gate target qubits definition for gate: {g}, "
                f"number of qubits in the state: {self.state.num_of_qubits}"
            )

        if g.gate.num_of_qubits > self.state.num_of_qubits:
            raise ValueError(
            )

    def apply_one_qubit_gate(self, g: CircuitGate, target_qubit_matrix: np.ndarray):
        # gates = np.full(self.state.num_of_qubits, GATES.I.target_qubit_matrix, dtype=object)
        gates = [GATES.I.target_qubit_matrix.copy() for _ in range(self.state.num_of_qubits)]
        gates[g.target_qubit] = target_qubit_matrix

        circuit_gate = np.array([[1]], dtype=complex)
        for gate in gates:
            circuit_gate = np.kron(gate, circuit_gate)

        self.state.qubit_vector = circuit_gate @ self.state.qubit_vector

    def apply_two_qubit_gate(self, g: CircuitGate, target_qubit_matrix: np.ndarray):
        control_zero_gates = [GATES.I.target_qubit_matrix.copy() for _ in range(self.state.num_of_qubits)]
        control_one_gates = [GATES.I.target_qubit_matrix.copy() for _ in range(self.state.num_of_qubits)]

        control_zero_gates[g.control_qubit] = g.gate.control_qubit_matrix_0
        control_one_gates[g.control_qubit] = g.gate.control_qubit_matrix_1
        control_one_gates[g.target_qubit] = target_qubit_matrix

        circuit_zero_gate = np.array([[1]], dtype=complex)
        for gate in control_zero_gates:
            circuit_zero_gate = np.kron(gate, circuit_zero_gate)

        circuit_one_gate = np.array([[1]], dtype=complex)
        for gate in control_one_gates:
            circuit_one_gate = np.kron(gate, circuit_one_gate)

        circuit_gate = circuit_zero_gate + circuit_one_gate

        self.state.qubit_vector = circuit_gate @ self.state.qubit_vector
