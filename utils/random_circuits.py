import numpy as np

from utils.circuit import Circuit
from utils.states import STATES, State
from utils.gates import GATES, CircuitGate, Gate, TWO_QUBITS_GATES, ONE_QUBIT_GATES


def sample_theta(rng):
    return rng.uniform(0.0, 2.0 * np.pi)


def random_pairs(rng, n: int) -> list[tuple[int, int]]:
    """
    For n qubits, return ~2 partners for each qubit in one layer.
    - For even n: each qubit gets exactly 2 different partners.
    - For odd n: some qubits will repeat a partner to fill out the list.
    """
    pairs = []
    for q in range(n):
        # Pick two partners different from q
        partners = rng.choice([p for p in range(n) if p != q], size=min(2, n - 1), replace=False)
        for p in partners:
            pair = (q, p)
            # Avoid adding duplicate in reverse order
            if (p, q) not in pairs:
                pairs.append(pair)
    return pairs


def sample_1q_gates(rng, num_of_qubits: int, repeats: int, gate_set: np.ndarray):
    gates = rng.choice(gate_set, size=repeats * num_of_qubits, replace=True)
    gates = gates.reshape(repeats, num_of_qubits)

    circuit_gates: list[CircuitGate] = []

    for r in range(repeats):
        for q in range(num_of_qubits):
            gate_name = str(gates[r, q])
            gate = GATES.GATE_DICT[gate_name]
            circuit_gates.append(CircuitGate(gate=gate, target_qubit=q))

    return circuit_gates


def sample_2q_gates(rng, num_of_qubits: int, repeats: int, gate_set: np.ndarray):
    circuit_gates: list[CircuitGate] = []

    if num_of_qubits < 2 or repeats <= 0:
        return circuit_gates

    for _ in range(repeats):
        pairs = random_pairs(rng, num_of_qubits)
        names = rng.choice(gate_set, size=len(pairs), replace=True)
        for (a, b), name in zip(pairs, names):
            gate_name = str(name)
            gate = GATES.GATE_DICT[gate_name]
            circuit_gates.append(CircuitGate(gate=gate, control_qubit=a, target_qubit=b))
    return circuit_gates


def sample_mixed_gates(rng, num_of_qubits: int, repeats: int, gate_set: np.ndarray)-> list[CircuitGate]:
    """
    Mixed layer(s): gate_set may contain both 1q and 2q gates.
    - For each repeat (layer), sample ~num_of_qubits gate names up front.
    - Assign gates to a shuffled pool of free qubits, consuming 1 or 2 qubits per gate.
    - Each qubit is used at most once per layer (disjoint assignment).
    - If a 2q gate is drawn but no partner remains, fall back to a 1q gate from the set (or I).
    - Leftover sampled gate names (if we run out of qubits) are ignored.
    """
    circuit_gates: list[CircuitGate] = []

    for _ in range(repeats):
        # sample ~n gate names for this layer; leftovers are fine (ignored)
        names = list(map(str, rng.choice(gate_set, size=num_of_qubits, replace=True)))
        name_idx = 0

        # shuffled pool of free qubits
        free = list(rng.permutation(num_of_qubits))

        while free and name_idx < len(names):
            q = free.pop(0)
            name = names[name_idx]
            name_idx += 1

            if name in TWO_QUBITS_GATES:
                if free:
                    # choose a random partner among remaining free qubits
                    j = int(rng.integers(0, len(free)))
                    partner = free.pop(j)
                    gate = GATES.GATE_DICT[name]
                    # randomize direction for controlled gates (CZ symmetric anyway)
                    ctrl, tgt = (q, partner)
                    if name in {"CNOT", "CRx", "CRy", "CRz"} and rng.random() < 0.5:
                        ctrl, tgt = tgt, ctrl
                    circuit_gates.append(CircuitGate(gate=gate, control_qubit=ctrl, target_qubit=tgt))
                else:
                    # no partner left → prefer a 1q from the same set, else identity
                    oneq_candidates = [g for g in gate_set if str(g) in ONE_QUBIT_GATES]
                    if oneq_candidates:
                        oneq_name = str(rng.choice(oneq_candidates))
                        circuit_gates.append(CircuitGate(gate=GATES.GATE_DICT[oneq_name], target_qubit=q))
                    else:
                        circuit_gates.append(CircuitGate(gate=GATES.I, target_qubit=q))
            else:
                # 1q gate on this (possibly non-neighbor) qubit
                circuit_gates.append(CircuitGate(gate=GATES.GATE_DICT[name], target_qubit=q))

        # any leftover sampled names are ignored; each qubit used at most once

    return circuit_gates


def sample_random_gates(
        num_of_qubits: int,
        layers: list,  # e.g. [(type = 1q/2q/mixed, repeats, gate_set), ("mixed", 1, PARAMETRISED_GATE_SET)]
        rng_seed: int | None = None

):

    rng = np.random.default_rng(rng_seed)

    circuit_gates = []

    for layer_type, repeats, gate_set in layers:
        layer_type = str(layer_type).strip().lower()
        repeats = int(repeats)
        gate_set = np.array(gate_set)  # accept lists or arrays

        if layer_type not in {"1q", "2q", "mixed"}:
            raise ValueError(f"Unknown layer type '{layer_type}'. Use '1q', '2q', or 'mixed'.")

        for _ in range(repeats):
            if layer_type == "1q":
                circuit_gates.extend(sample_1q_gates(rng, num_of_qubits, repeats, gate_set))

            elif layer_type == "2q":
                circuit_gates.extend(sample_2q_gates(rng, num_of_qubits, repeats, gate_set))

            else:
                circuit_gates.extend(sample_mixed_gates(rng, num_of_qubits, repeats, gate_set))

    return circuit_gates


def resolve_parameters(circuit_gates, seed: int | None = None):
    rng = np.random.default_rng(seed)

    for cg in circuit_gates:
        if cg.gate.target_qubit_matrix is None:
            base = cg.gate.name.split("(", 1)[0]  # e.g. "Rx" from "Rx" or "Rx(…)"
            init_function = GATES.INIT_PARAMETRIZED_GATE_FUNC_DICT.get(base)
            if init_function is None:
                raise ValueError(f"No initialization function for param gate '{cg.gate.name}'")

            theta = rng.uniform(0.0, 2.0*np.pi)
            cg.gate = init_function(theta)
