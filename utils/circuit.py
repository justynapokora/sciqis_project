from __future__ import annotations

import numpy as np
import copy

from utils.gates import CircuitGate, GATES, PARAMETRISED_GATE_SET
from utils.states import State, DensityState
from utils.noise_channels import DepolarizingNoise, SPAMNoise, TRCNoise


class Circuit:
    def __init__(self,
                 state: State,
                 gates: list[list[CircuitGate]],
                 rng: np.random.Generator | None = None,
                 depolarizing_noise: DepolarizingNoise | None = None,
                 spam_noise: SPAMNoise | None = None,
                 trc_noise: TRCNoise | None = None,
                 ):
        self.initial_state: State = copy.deepcopy(state)
        self.state: State = state
        self.noisy_dm: DensityState = DensityState.from_state(state)

        self.gates: list[list[CircuitGate]] = self.layers_to_timesteps(gates)

        self.depolarizing_noise = depolarizing_noise  # if None then do not add this noise
        self.spam_noise = spam_noise
        self.trc_noise = trc_noise

        self.measured_qubits: list = []

        self.rng = rng or np.random.default_rng()

    def reset_circuit(self):
        self.state = copy.deepcopy(self.initial_state)
        self.noisy_dm = DensityState.from_state(self.initial_state)
        self.measured_qubits = []

    @staticmethod
    def layers_to_timesteps(gates: list[list[CircuitGate]]) -> list[list[CircuitGate]]:
        """
        Split each layer into sequential timesteps.
        Each timestep contains gates that act on disjoint qubits.
        """
        gates_timesteps = []
        for layer in gates:
            remaining = list(layer)
            timesteps: list[list[CircuitGate]] = []

            while remaining:
                used = set()
                step: list[CircuitGate] = []
                next_remaining: list[CircuitGate] = []

                for g in remaining:
                    qubits = {g.target_qubit} | ({g.control_qubit} if g.control_qubit is not None else set())
                    if qubits.isdisjoint(used):     # can place in this timestep
                        step.append(g)
                        used.update(qubits)
                    else:                            # defer to a later timestep
                        next_remaining.append(g)

                timesteps.append(step)
                remaining = next_remaining

            gates_timesteps.extend(timesteps)

        return gates_timesteps

    # --- helpers to build full-system unitaries ---
    @staticmethod
    def _full_1q(n: int, q: int, Uq: np.ndarray) -> np.ndarray:
        ops = [GATES.I.target_qubit_matrix.copy() for _ in range(n)]
        ops[q] = Uq
        full = np.array([[1]], dtype=complex)
        for op in ops:
            full = np.kron(op, full)
        return full

    @staticmethod
    def _full_2q_ctrl_target(n: int, ctrl: int, tgt: int, target_U: np.ndarray, gate) -> np.ndarray:
        # same construction as apply_two_qubit_gate, but returns the full operator
        cz0 = [GATES.I.target_qubit_matrix.copy() for _ in range(n)]
        cz1 = [GATES.I.target_qubit_matrix.copy() for _ in range(n)]
        cz0[ctrl] = gate.control_qubit_matrix_0
        cz1[ctrl] = gate.control_qubit_matrix_1
        cz1[tgt] = target_U

        full0 = np.array([[1]], dtype=complex)
        for op in cz0:
            full0 = np.kron(op, full0)

        full1 = np.array([[1]], dtype=complex)
        for op in cz1:
            full1 = np.kron(op, full1)

        return full0 + full1

    def simulate_circuit(self):
        n = self.state.num_of_qubits
        # print("=== Starting simulation ===")
        # print(f"Initial state (ideal):  {self.state}")
        # print(f"Initial state (noisy):  {self.noisy_state}")

        # --- SPAM preparation as channel on noisy DM (optional) ---
        if self.spam_noise:
            for q in range(n):
                ks = self.spam_noise.kraus_prep(q)
                self.noisy_dm.apply_1q_channel(ks, q)

            if self.trc_noise:
                for q in range(n):
                    ks = self.trc_noise.kraus_for("SPAM", q)
                    self.noisy_dm.apply_1q_channel(ks, q)

        # print(f"State (ideal): {self.state}")
        # print(f"State (noisy): {self.noisy_state}")

        # --- main layers ---
        for layer in self.gates:
            all_1q = all(g.gate.num_of_qubits == 1 for g in layer)

            if all_1q:
                # Build the single parallel 1q unitary once (sampling thetas once),
                # apply to ideal statevector AND to noisy density matrix.
                ops = [GATES.I.target_qubit_matrix.copy() for _ in range(n)]
                for g in layer:
                    Uq = g.gate.target_qubit_matrix
                    if Uq is None:
                        Uq = self.get_parameterised_gate_random_matrix(g.gate.name)
                    ops[g.target_qubit] = Uq

                U = np.array([[1]], dtype=complex)
                for op in ops:
                    U = np.kron(op, U)

                # ideal vector
                self.state.qubit_vector = U @ self.state.qubit_vector
                # noisy density-matrix
                self.noisy_dm.apply_unitary(U)

                # --- channels (noisy only) ---
                # Depolarizing after each gate (paper)
                if self.depolarizing_noise:
                    for g in layer:
                        ks = self.depolarizing_noise.kraus_for_1q(g.gate.name, g.target_qubit)
                        self.noisy_dm.apply_1q_channel(ks, g.target_qubit)

                # TRC after each gate
                if self.trc_noise:
                    for g in layer:
                        ks = self.trc_noise.kraus_for(g.gate.name, g.target_qubit)
                        self.noisy_dm.apply_1q_channel(ks, g.target_qubit)

            else:
                # Mixed layer: apply gates one-by-one, but share thetas across ideal+noisy
                for g in layer:
                    self.check_gate_validity(g)
                    Uq = g.gate.target_qubit_matrix
                    if Uq is None:
                        Uq = self.get_parameterised_gate_random_matrix(g.gate.name)

                    if g.gate.num_of_qubits == 1:
                        # ideal
                        self.apply_one_qubit_gate(self.state, g, Uq)
                        # noisy
                        U = self._full_1q(n, g.target_qubit, Uq)
                        self.noisy_dm.apply_unitary(U)
                        # channels
                        if self.depolarizing_noise:
                            ks = self.depolarizing_noise.kraus_for_1q(g.gate.name, g.target_qubit)
                            self.noisy_dm.apply_1q_channel(ks, g.target_qubit)
                        if self.trc_noise:
                            ks = self.trc_noise.kraus_for(g.gate.name, g.target_qubit)
                            self.noisy_dm.apply_1q_channel(ks, g.target_qubit)

                    else:
                        # two-qubit (control, target)
                        # ideal
                        self.apply_two_qubit_gate(self.state, g, Uq)
                        # noisy
                        U = self._full_2q_ctrl_target(n, g.control_qubit, g.target_qubit, Uq, g.gate)
                        self.noisy_dm.apply_unitary(U)
                        # channels
                        # Depolarizing: target only (paper Fig. 2b)
                        if self.depolarizing_noise:
                            ks = self.depolarizing_noise.kraus_for_1q(g.gate.name, g.target_qubit)
                            self.noisy_dm.apply_1q_channel(ks, g.target_qubit)
                        # TRC: independently on both qubits
                        if self.trc_noise:
                            for q in (g.control_qubit, g.target_qubit):
                                ks = self.trc_noise.kraus_for(g.gate.name, q)
                                self.noisy_dm.apply_1q_channel(ks, q)

        # --- SPAM measurement channel (optional) ---
        if self.spam_noise:
            for q in range(n):
                ks = self.spam_noise.kraus_meas(q)
                self.noisy_dm.apply_1q_channel(ks, q)

        # finalize
        self.state.normalize_state()
        self.noisy_dm.normalize_dm()

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

        if max_qubit_val >= self.state.num_of_qubits:
            raise ValueError(
                f"Incorrect gate target qubits definition for gate: {g}, "
                f"number of qubits in the state: {self.state.num_of_qubits}"
            )

        if g.gate.num_of_qubits > self.state.num_of_qubits:
            raise ValueError(
                f"Incorrect gate: {g}, "
                f"number of qubits in the state: {self.state.num_of_qubits}"
            )

    def apply_one_qubit_gate(self, state: State, g: CircuitGate, target_qubit_matrix: np.ndarray):
        gates = [GATES.I.target_qubit_matrix.copy() for _ in range(state.num_of_qubits)]
        gates[g.target_qubit] = target_qubit_matrix
        circuit_gate = np.array([[1]], dtype=complex)
        for gate in gates:
            circuit_gate = np.kron(gate, circuit_gate)
        state.qubit_vector = circuit_gate @ state.qubit_vector

    @staticmethod
    def apply_two_qubit_gate(state: State, g: CircuitGate, target_qubit_matrix: np.ndarray):
        control_zero_gates = [GATES.I.target_qubit_matrix.copy() for _ in range(state.num_of_qubits)]
        control_one_gates = [GATES.I.target_qubit_matrix.copy() for _ in range(state.num_of_qubits)]

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

        state.qubit_vector = circuit_gate @ state.qubit_vector
