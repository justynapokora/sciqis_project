from __future__ import annotations

import numpy as np

from utils.gates import Gate, CircuitGate, GATES
from utils.states import State


class Circuit:
    def __init__(self, state: State, gates: list[CircuitGate]):
        self.initial_state = state
        self.state: State = state
        self.gates: list[CircuitGate] = gates
        self.measured_qubits: list = []

    def simulate_circuit(self):
        print(f"initial state ({self.initial_state.num_of_qubits} qubit(s)): {self.initial_state}")

        for g in self.gates:
            self.check_gate_validity(g)

            match g.gate.num_of_qubits:
                case 1:
                    self.apply_one_qubit_gate(g)
                case 2:
                    self.apply_two_qubit_gate(g)
                case default:
                    raise ValueError(
                        f"Incorrect number of qubits for gate: {g}, "
                        f"allowed number of qubits: 1, 2"
                    )

    def check_gate_validity(self, g: CircuitGate):
        max_qubit_val = np.max(np.concatenate((g.target_qubits, g.control_qubits)))

        if max_qubit_val > self.state.num_of_qubits:
            raise ValueError(
                f"Incorrect gate target qubits definition for gate: {g}, "
                f"number of qubits in the state: {self.state.num_of_qubits}"
            )

        if g.gate.num_of_qubits > self.state.num_of_qubits:
            raise ValueError(
            )

    def apply_one_qubit_gate(self, g: CircuitGate):
        # gates = np.full(self.state.num_of_qubits, GATES.I.target_qubit_matrix, dtype=object)
        gates = [GATES.I.target_qubit_matrix.copy() for _ in range(self.state.num_of_qubits)]
        gates[g.target_qubits[0]] = g.gate.target_qubit_matrix

        circuit_gate = np.array([[1]], dtype=complex)
        for gate in gates:
            circuit_gate = np.kron(gate, circuit_gate)

        self.state.qubit_vector = circuit_gate @ self.state.qubit_vector

    def apply_two_qubit_gate(self, g: CircuitGate):
        control_zero_gates = [GATES.I.target_qubit_matrix.copy() for _ in range(self.state.num_of_qubits)]
        control_one_gates = [GATES.I.target_qubit_matrix.copy() for _ in range(self.state.num_of_qubits)]

        control_zero_gates[g.control_qubits[0]] = g.gate.control_qubit_matrix_0
        control_one_gates[g.control_qubits[0]] = g.gate.control_qubit_matrix_1
        control_one_gates[g.target_qubits[0]] = g.gate.target_qubit_matrix

        circuit_zero_gate = np.array([[1]], dtype=complex)
        for gate in control_zero_gates:
            circuit_zero_gate = np.kron(gate, circuit_zero_gate)

        circuit_one_gate = np.array([[1]], dtype=complex)
        for gate in control_one_gates:
            circuit_one_gate = np.kron(gate, circuit_one_gate)

        circuit_gate = circuit_zero_gate + circuit_one_gate

        self.state.qubit_vector = circuit_gate @ self.state.qubit_vector
