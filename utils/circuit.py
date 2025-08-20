from __future__ import annotations

import numpy as np
import copy

from utils.gates import CircuitGate, GATES
from utils.states import State, DensityState
from utils.noise_channels import DepolarizingNoise, SPAMNoise, TDCNoise
from utils.random_gates import resolve_parameters


class Circuit:
    """
    Simulator for parameterized quantum circuits (PQCs) with optional noise.

    Maintains both an ideal statevector (`State`) and a noisy density matrix
    (`DensityState`). Gates may be parameterized; parameters are sampled via
    `resolve_parameters` on initialization and whenever the circuit is reset.

    Noise model support:
        - DepolarizingNoise (per gate/qubit as used in the paper)
        - SPAMNoise (preparation/measurement channels)
        - TDCNoise (time-dependent channel, applied after gates or SPAM steps)
    """

    def __init__(self,
                 state: State,
                 gates: list[list[CircuitGate]],
                 rng: np.random.Generator | None = None,
                 depolarizing_noise: DepolarizingNoise | None = None,
                 spam_noise: SPAMNoise | None = None,
                 tdc_noise: TDCNoise | None = None,
                 ):
        """
        Initialize a circuit with an initial state, a layered gate list,
        and optional noise channels.
        """

        self.rng = rng or np.random.default_rng()

        self.initial_state: State = copy.deepcopy(state)
        self.state: State = state
        self.noisy_dm: DensityState = DensityState.from_state(state)

        self.base_gates: list[list[CircuitGate]] = self.layers_to_timesteps(gates)
        self.gates: list[list[CircuitGate]] = resolve_parameters(self.base_gates, self.rng)

        self.depolarizing_noise = depolarizing_noise  # if None then do not add this noise
        self.spam_noise = spam_noise
        self.tdc_noise = tdc_noise

    def reset_circuit(self):
        """
        Reset the circuit to the initial state and resample gate parameters.
        """

        # reset states back to initial state
        self.state = copy.deepcopy(self.initial_state)
        self.noisy_dm = DensityState.from_state(self.initial_state)

        # reset parametrized gates (sample new parameters for parametrized gates)
        self.gates: list[list[CircuitGate]] = resolve_parameters(self.base_gates, self.rng)

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
        """
        Build the full-system unitary for a 1-qubit gate `Uq` acting on qubit `q`
        in an `n`-qubit system via Kronecker products.
        """
        ops = [GATES.I.target_qubit_matrix.copy() for _ in range(n)]
        ops[q] = Uq
        full = np.array([[1]], dtype=complex)
        for op in ops:
            full = np.kron(op, full)
        return full

    @staticmethod
    def _full_2q_ctrl_target(n: int, ctrl: int, tgt: int, target_U: np.ndarray, gate) -> np.ndarray:
        """
        Build the full-system controlled operation for a 2-qubit gate.

        Constructs |0><0|_ctrl ⊗ I + |1><1|_ctrl ⊗ U_target on qubit `tgt`,
        embedded into an `n`-qubit system.
        """
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

    def simulate_circuit_n_rounds(self, n: int = 1):
        """
        Run the circuit for `n` rounds and save states after each round.
        The 0th snapshot is the initial state before any round.
        """

        if n < 1:
            raise ValueError("Number of rounds must be at least 1")

        dim = 2 ** self.state.num_of_qubits

        # Preallocate arrays (include initial snapshot at index 0)
        states = np.zeros((n + 1, dim, 1), dtype=complex)
        noisy_states = np.zeros((n + 1, dim, dim), dtype=complex)

        # Save initial state
        states[0, :, :] = self.state.qubit_vector
        noisy_states[0, :, :] = self.noisy_dm.rho

        # Unified loop: first_round for k==1, last_round for k==n
        for k in range(1, n + 1):
            self.simulate_circuit(first_round=(k == 1), last_round=(k == n))
            states[k, :, :] = self.state.qubit_vector
            noisy_states[k, :, :] = self.noisy_dm.rho

        return states, noisy_states

    def simulate_circuit(self, first_round: bool = True, last_round: bool = True):
        """
        Simulate a single full pass over all timesteps (one *round*).

        Applies gates to both the ideal statevector and the noisy density matrix.
        Optional noise channels are applied as follows:
            - SPAM preparation: before main layers (if `first_round` and present)
            - Depolarizing/TDC: after each gate (as documented inline)
            - SPAM measurement: after main layers (if `last_round` and present)
        """
        n = self.state.num_of_qubits

        # --- SPAM preparation as channel on noisy DM (optional) ---
        if self.spam_noise and first_round:
            for q in range(n):
                ks = self.spam_noise.kraus_prep(q)
                self.noisy_dm.apply_1q_channel(ks, q)

            if self.tdc_noise:
                for q in range(n):
                    ks = self.tdc_noise.kraus_for("SPAM", q)
                    self.noisy_dm.apply_1q_channel(ks, q)

        # --- main layers ---
        for layer in self.gates:
            all_1q = all(g.gate.num_of_qubits == 1 for g in layer)

            if all_1q:
                # Build the single parallel 1q unitary once (sampling thetas once),
                # apply to ideal statevector AND to noisy density matrix.
                ops = [GATES.I.target_qubit_matrix.copy() for _ in range(n)]
                for g in layer:
                    Uq = g.gate.target_qubit_matrix
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
                if self.tdc_noise:
                    for g in layer:
                        ks = self.tdc_noise.kraus_for(g.gate.name, g.target_qubit)
                        self.noisy_dm.apply_1q_channel(ks, g.target_qubit)

            else:
                # Mixed layer: apply gates one-by-one, but share thetas across ideal+noisy
                for g in layer:
                    self.check_gate_validity(g)
                    Uq = g.gate.target_qubit_matrix

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
                        if self.tdc_noise:
                            ks = self.tdc_noise.kraus_for(g.gate.name, g.target_qubit)
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
                        if self.tdc_noise:
                            for q in (g.control_qubit, g.target_qubit):
                                ks = self.tdc_noise.kraus_for(g.gate.name, q)
                                self.noisy_dm.apply_1q_channel(ks, q)

        # --- SPAM measurement channel (optional) ---
        if self.spam_noise and last_round:
            for q in range(n):
                ks = self.spam_noise.kraus_meas(q)
                self.noisy_dm.apply_1q_channel(ks, q)

        # finalize
        self.state.normalize_state()
        self.noisy_dm.normalize_dm()

    def check_gate_validity(self, g: CircuitGate):
        """
        Validate that the gate's qubit indices are within range and consistent.
        """
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

    @staticmethod
    def apply_one_qubit_gate(state: State, g: CircuitGate, target_qubit_matrix: np.ndarray):
        """
        Apply a single-qubit operation to the ideal statevector.
        """
        gates = [GATES.I.target_qubit_matrix.copy() for _ in range(state.num_of_qubits)]
        gates[g.target_qubit] = target_qubit_matrix
        circuit_gate = np.array([[1]], dtype=complex)
        for gate in gates:
            circuit_gate = np.kron(gate, circuit_gate)
        state.qubit_vector = circuit_gate @ state.qubit_vector

    @staticmethod
    def apply_two_qubit_gate(state: State, g: CircuitGate, target_qubit_matrix: np.ndarray):
        """
        Apply a controlled two-qubit operation to the ideal statevector.

        Builds the block-controlled unitary
            |0><0|_ctrl ⊗ I + |1><1|_ctrl ⊗ U_target
        embedded into the full system, then left-multiplies `state.qubit_vector`.
        """
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
