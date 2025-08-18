from __future__ import annotations

import numpy as np
import copy

from utils.gates import CircuitGate, GATES, PARAMETRISED_GATE_SET
from utils.states import State
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
        self.noisy_state: State = copy.deepcopy(state)

        self.gates: list[list[CircuitGate]] = self.layers_to_timesteps(gates)

        self.depolarizing_noise = depolarizing_noise  # if None then do not add this noise
        self.spam_noise = spam_noise
        self.trc_noise = trc_noise

        self.measured_qubits: list = []

        self.rng = rng
        if not rng:
            self.rng = np.random.default_rng()

    def reset_circuit(self):
        self.state = copy.deepcopy(self.initial_state)
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

    # def simulate_circuit(self):
    #     for layer in self.gates:
    #
    #         if all(g.gate.num_of_qubits == 1 for g in layer):
    #             # all gates in a timestep are 1 qubit gates -> they can be applied at the same time
    #             self.apply_one_qubit_gates_timestep(layer)
    #             continue
    #
    #         # mixed/controlled gates -> fall back to per-gate application
    #         for g in layer:
    #             self.check_gate_validity(g)
    #
    #             target_qubit_matrix = g.gate.target_qubit_matrix
    #             if target_qubit_matrix is None:
    #                 target_qubit_matrix = self.get_parameterised_gate_random_matrix(g.gate.name)
    #
    #             match g.gate.num_of_qubits:
    #                 case 1:
    #                     self.apply_one_qubit_gate(g, target_qubit_matrix)
    #                 case 2:
    #                     self.apply_two_qubit_gate(g, target_qubit_matrix)
    #                 case default:
    #                     raise ValueError(
    #                         f"Incorrect number of qubits for gate: {g}, "
    #                         f"allowed number of qubits: 1, 2"
    #                     )

    # def simulate_circuit(self):
    #     # SPAM prep noise at t=0
    #     if self.spam_noise:
    #         for q in range(self.noisy_state.num_of_qubits):
    #             g = self.spam_noise.sample_prep_error(q, self.rng)
    #             if g:
    #                 self.apply_one_qubit_gate(self.noisy_state, g, g.gate.target_qubit_matrix)
    #
    #     for layer in self.gates:
    #         # Ideal path (always update self.state)
    #         if all(g.gate.num_of_qubits == 1 for g in layer):
    #             self.apply_one_qubit_gates_timestep(self.state, layer)
    #         else:
    #             for g in layer:
    #                 self.check_gate_validity(g)
    #                 target_qubit_matrix = g.gate.target_qubit_matrix
    #                 if target_qubit_matrix is None:
    #                     target_qubit_matrix = self.get_parameterised_gate_random_matrix(g.gate.name)
    #
    #                 if g.gate.num_of_qubits == 1:
    #                     self.apply_one_qubit_gate(self.state, g, target_qubit_matrix)
    #                 else:
    #                     self.apply_two_qubit_gate(self.state, g, target_qubit_matrix)
    #
    #         # Noisy path (gates + noise)
    #         if all(g.gate.num_of_qubits == 1 for g in layer):
    #             self.apply_one_qubit_gates_timestep(self.noisy_state, layer)
    #             # Depol + TRC
    #             if self.depolarizing_noise:
    #                 for g in layer:
    #                     err = self.depolarizing_noise.sample_error_gate(g.gate.name, g.target_qubit, self.rng)
    #                     if err:
    #                         self.apply_one_qubit_gate(self.noisy_state, err, err.gate.target_qubit_matrix)
    #             if self.trc_noise:
    #                 for g in layer:
    #                     err = self.trc_noise.sample_error_gate(g.gate.name, g.target_qubit, self.rng)
    #                     if err:
    #                         self.apply_one_qubit_gate(self.noisy_state, err, err.gate.target_qubit_matrix)
    #         else:
    #             for g in layer:
    #                 self.check_gate_validity(g)
    #                 target_qubit_matrix = g.gate.target_qubit_matrix
    #                 if target_qubit_matrix is None:
    #                     target_qubit_matrix = self.get_parameterised_gate_random_matrix(g.gate.name)
    #
    #                 if g.gate.num_of_qubits == 1:
    #                     self.apply_one_qubit_gate(self.noisy_state, g, target_qubit_matrix)
    #                 else:
    #                     self.apply_two_qubit_gate(self.noisy_state, g, target_qubit_matrix)
    #
    #                 # Add noise after
    #                 if self.depolarizing_noise:
    #                     bname = g.gate.name.split("(")[0]
    #                     if bname == "CZ":
    #                         for q in [g.control_qubit, g.target_qubit]:
    #                             err = self.depolarizing_noise.sample_error_gate(g.gate.name, q, self.rng)
    #                             if err:
    #                                 self.apply_one_qubit_gate(self.noisy_state, err, err.gate.target_qubit_matrix)
    #                     elif g.gate.num_of_qubits == 2:
    #                         err = self.depolarizing_noise.sample_error_gate(g.gate.name, g.target_qubit, self.rng)
    #                         if err:
    #                             self.apply_one_qubit_gate(self.noisy_state, err, err.gate.target_qubit_matrix)
    #                     else:
    #                         err = self.depolarizing_noise.sample_error_gate(g.gate.name, g.target_qubit, self.rng)
    #                         if err:
    #                             self.apply_one_qubit_gate(self.noisy_state, err, err.gate.target_qubit_matrix)
    #
    #                 if self.trc_noise:
    #                     touched = [g.target_qubit] if g.gate.num_of_qubits == 1 else [g.control_qubit, g.target_qubit]
    #                     for q in touched:
    #                         err = self.trc_noise.sample_error_gate(g.gate.name, q, self.rng)
    #                         if err:
    #                             self.apply_one_qubit_gate(self.noisy_state, err, err.gate.target_qubit_matrix)

    def simulate_circuit(self):
        print("=== Starting simulation ===")
        print(f"Initial state (ideal):  {self.state}")
        print(f"Initial state (noisy):  {self.noisy_state}")

        # --- SPAM preparation noise at t=0
        if self.spam_noise:
            print("\n[SPAM Noise] Applying preparation errors...")
            for q in range(self.noisy_state.num_of_qubits):
                g = self.spam_noise.sample_prep_error(q, self.rng)
                if g:
                    print(f"  -> Qubit {q}: prep error {g.gate.name}")
                    self.apply_one_qubit_gate(self.noisy_state, g, g.gate.target_qubit_matrix)

        print(f"State (ideal): {self.state}")
        print(f"State (noisy): {self.noisy_state}")

        # --- main layers
        for layer_idx, layer in enumerate(self.gates):
            print(f"\n=== Layer {layer_idx} ===")
            print("  Ideal path:")
            if all(g.gate.num_of_qubits == 1 for g in layer):
                print(f"    -> Applying {len(layer)} parallel 1q gates {[g.gate.name for g in layer]}")
                self.apply_one_qubit_gates_timestep(self.state, layer)
            else:
                for g in layer:
                    self.check_gate_validity(g)
                    target_qubit_matrix = g.gate.target_qubit_matrix
                    if target_qubit_matrix is None:
                        target_qubit_matrix = self.get_parameterised_gate_random_matrix(g.gate.name)

                    print(f"    -> Applying {g.gate.name} (ideal) on qubits "
                          f"target={g.target_qubit} "
                          f"{'control=' + str(g.control_qubit) if g.control_qubit is not None else ''}")
                    if g.gate.num_of_qubits == 1:
                        self.apply_one_qubit_gate(self.state, g, target_qubit_matrix)
                    else:
                        self.apply_two_qubit_gate(self.state, g, target_qubit_matrix)

            print(f"State (ideal): {self.state}")
            print(f"State (noisy): {self.noisy_state}")

            # --- noisy path
            print("  Noisy path:")
            if all(g.gate.num_of_qubits == 1 for g in layer):
                print(f"    -> Applying {len(layer)} parallel 1q gates {[g.gate.name for g in layer]}")
                self.apply_one_qubit_gates_timestep(self.noisy_state, layer)

                # Depolarizing noise
                if self.depolarizing_noise:
                    for g in layer:
                        err = self.depolarizing_noise.sample_error_gate(g.gate.name, g.target_qubit, self.rng)
                        if err:
                            print(f"    -> Depol noise on q{g.target_qubit}: {err.gate.name}")
                            self.apply_one_qubit_gate(self.noisy_state, err, err.gate.target_qubit_matrix)

                # TRC noise
                if self.trc_noise:
                    for g in layer:
                        err = self.trc_noise.sample_error_gate(g.gate.name, g.target_qubit, self.rng)
                        if err:
                            print(f"    -> TRC noise on q{g.target_qubit}: {err.gate.name}")
                            self.apply_one_qubit_gate(self.noisy_state, err, err.gate.target_qubit_matrix)

            else:
                for g in layer:
                    self.check_gate_validity(g)
                    target_qubit_matrix = g.gate.target_qubit_matrix
                    if target_qubit_matrix is None:
                        target_qubit_matrix = self.get_parameterised_gate_random_matrix(g.gate.name)

                    print(f"    -> Applying {g.gate.name} (noisy) on qubits "
                          f"target={g.target_qubit} "
                          f"{'control=' + str(g.control_qubit) if g.control_qubit is not None else ''}")
                    if g.gate.num_of_qubits == 1:
                        self.apply_one_qubit_gate(self.noisy_state, g, target_qubit_matrix)
                    else:
                        self.apply_two_qubit_gate(self.noisy_state, g, target_qubit_matrix)

                    # Add noise
                    if self.depolarizing_noise:
                        bname = g.gate.name.split("(")[0]
                        if bname == "CZ":
                            for q in [g.control_qubit, g.target_qubit]:
                                err = self.depolarizing_noise.sample_error_gate(g.gate.name, q, self.rng)
                                if err:
                                    print(f"    -> Depol noise after CZ on q{q}: {err.gate.name}")
                                    self.apply_one_qubit_gate(self.noisy_state, err, err.gate.target_qubit_matrix)
                        elif g.gate.num_of_qubits == 2:
                            err = self.depolarizing_noise.sample_error_gate(g.gate.name, g.target_qubit, self.rng)
                            if err:
                                print(f"    -> Depol noise after 2q gate on target q{g.target_qubit}: {err.gate.name}")
                                self.apply_one_qubit_gate(self.noisy_state, err, err.gate.target_qubit_matrix)
                        else:
                            err = self.depolarizing_noise.sample_error_gate(g.gate.name, g.target_qubit, self.rng)
                            if err:
                                print(f"    -> Depol noise after 1q gate on q{g.target_qubit}: {err.gate.name}")
                                self.apply_one_qubit_gate(self.noisy_state, err, err.gate.target_qubit_matrix)

                    if self.trc_noise:
                        touched = [g.target_qubit] if g.gate.num_of_qubits == 1 else [g.control_qubit, g.target_qubit]
                        for q in touched:
                            err = self.trc_noise.sample_error_gate(g.gate.name, q, self.rng)
                            if err:
                                print(f"    -> TRC noise on q{q}: {err.gate.name}")
                                self.apply_one_qubit_gate(self.noisy_state, err, err.gate.target_qubit_matrix)

        print("\n=== Simulation complete ===")
        print(f"Final ideal state: {self.state}")
        print(f"Final noisy state: {self.noisy_state}")

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
            )

    # def apply_one_qubit_gate(self, g: CircuitGate, target_qubit_matrix: np.ndarray):
    #     gates = [GATES.I.target_qubit_matrix.copy() for _ in range(self.state.num_of_qubits)]
    #     gates[g.target_qubit] = target_qubit_matrix
    #
    #     if g.gate.name == "RESET":
    #         self.reset_qubit_to_zero(g.target_qubit)
    #         return
    #
    #     circuit_gate = np.array([[1]], dtype=complex)
    #     for gate in gates:
    #         circuit_gate = np.kron(gate, circuit_gate)
    #
    #     self.state.qubit_vector = circuit_gate @ self.state.qubit_vector

    def apply_one_qubit_gate(self, state: State, g: CircuitGate, target_qubit_matrix: np.ndarray):
        if g.gate.name == "RESET":
            self.reset_qubit_to_zero(state, g.target_qubit)
            return
        gates = [GATES.I.target_qubit_matrix.copy() for _ in range(state.num_of_qubits)]
        gates[g.target_qubit] = target_qubit_matrix
        circuit_gate = np.array([[1]], dtype=complex)
        for gate in gates:
            circuit_gate = np.kron(gate, circuit_gate)
        state.qubit_vector = circuit_gate @ state.qubit_vector

    def apply_one_qubit_gates_timestep(self, state: State, timestep_layer: list[CircuitGate]):
        gates = [GATES.I.target_qubit_matrix.copy() for _ in range(state.num_of_qubits)]
        for g in timestep_layer:
            target_qubit_matrix = g.gate.target_qubit_matrix
            if target_qubit_matrix is None:
                target_qubit_matrix = self.get_parameterised_gate_random_matrix(g.gate.name)
            gates[g.target_qubit] = target_qubit_matrix

        circuit_gate = np.array([[1]], dtype=complex)
        for gate in gates:
            circuit_gate = np.kron(gate, circuit_gate)
        print(circuit_gate)
        print(circuit_gate @ state.qubit_vector)
        state.qubit_vector = circuit_gate @ state.qubit_vector

    # def reset_qubit_to_zero(self, q: int):
    #     """Collapse qubit q to |0> and renormalize the state vector."""
    #     n = self.state.num_of_qubits
    #     new_state = np.zeros_like(self.state.qubit_vector)
    #     for i, amp in enumerate(self.state.qubit_vector):
    #         if ((i >> q) & 1) == 0:  # bit q is 0
    #             new_state[i] = amp
    #
    #     self.state.qubit_vector = new_state
    #     self.state.normalize_state()

    @staticmethod
    def reset_qubit_to_zero(state: State, q: int):
        """Collapse qubit q to |0> and renormalize the state vector."""
        n = state.num_of_qubits
        new_state = np.zeros_like(state.qubit_vector)
        for i, amp in enumerate(state.qubit_vector):
            if ((i >> q) & 1) == 0:  # bit q is 0
                new_state[i] = amp

        state.qubit_vector = new_state
        state.normalize_state()

    # def apply_two_qubit_gate(self, g: CircuitGate, target_qubit_matrix: np.ndarray):
    #     control_zero_gates = [GATES.I.target_qubit_matrix.copy() for _ in range(self.state.num_of_qubits)]
    #     control_one_gates = [GATES.I.target_qubit_matrix.copy() for _ in range(self.state.num_of_qubits)]
    #
    #     control_zero_gates[g.control_qubit] = g.gate.control_qubit_matrix_0
    #     control_one_gates[g.control_qubit] = g.gate.control_qubit_matrix_1
    #     control_one_gates[g.target_qubit] = target_qubit_matrix
    #
    #     circuit_zero_gate = np.array([[1]], dtype=complex)
    #     for gate in control_zero_gates:
    #         circuit_zero_gate = np.kron(gate, circuit_zero_gate)
    #
    #     circuit_one_gate = np.array([[1]], dtype=complex)
    #     for gate in control_one_gates:
    #         circuit_one_gate = np.kron(gate, circuit_one_gate)
    #
    #     circuit_gate = circuit_zero_gate + circuit_one_gate
    #
    #     self.state.qubit_vector = circuit_gate @ self.state.qubit_vector

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
