import numpy as np
from utils.gates import ONE_QUBIT_FIXED_GATE_SET, TWO_QUBITS_FIXED_GATE_SET, FIXED_GATE_SET, \
    ONE_QUBIT_PARAMETRISED_GATE_SET, TWO_QUBITS_PARAMETRISED_GATE_SET, PARAMETRISED_GATE_SET, \
    ONE_QUBIT_GATES, TWO_QUBITS_GATES
from utils.states import State, STATES
from utils.circuit import Circuit

from utils.draw_circuit import print_circuit_gates_info, save_circuit_drawing

from utils.random_gates import sample_random_gates, resolve_parameters

np.set_printoptions(precision=3, suppress=True)

# define rng
RNG = np.random.default_rng(4242)

# state = STATES.generate_state([
#     STATES.zero,
#     STATES.plus
# ])
#
# print(state)

##### example 1
#
# circuit = Circuit(
#     state=STATES.generate_zero_n_qubit_state(4),
#     gates=[[
#
#         CircuitGate(GATES.X, target_qubit=2),
#         CircuitGate(GATES.init_Rx(np.pi / 4), target_qubit=2),
#         # CircuitGate(GATES.Y, np.array([0])),
#         # CircuitGate(GATES.X, np.array([1])),
#         #
#         CircuitGate(GATES.H, target_qubit=2),
#         CircuitGate(GATES.H, target_qubit=0),
#
#         CircuitGate(GATES.CNOT, target_qubit=0, control_qubit=2),
#         CircuitGate(GATES.CNOT, target_qubit=1, control_qubit=2),
#         CircuitGate(GATES.CNOT, target_qubit=2, control_qubit=1),
#         CircuitGate(GATES.CNOT, target_qubit=3, control_qubit=0)
#
#
#
#     ]]
# )
#
# circuit.simulate_circuit()
#
# # print(f" res : {circuit.state.qubit_vector}")
# print(f" res: {circuit.state}")
# # print(f"measurement: {circuit.state.measure_all()}")
#
#
# print(circuit.state.measure_qubit(1))
# print(circuit.state)
# print(circuit.state.measure_qubit(2))
# print(circuit.state)
#
# print(circuit.state.measure_qubit(0))
# print(circuit.state)
# print(circuit.state.measure_qubit(3))
# print(circuit.state)
# print(circuit.state.get_probabilities_str())
#
#
#


sampled_circuit_gates = sample_random_gates(num_of_qubits=4,
                                            layers=[
                                                ("1q", 1, ONE_QUBIT_FIXED_GATE_SET),
                                                ("1q", 1, ONE_QUBIT_PARAMETRISED_GATE_SET),
                                                ("2q", 1, TWO_QUBITS_GATES),
                                                ("mixed", 1, PARAMETRISED_GATE_SET),
                                            ],
                                            rng=RNG)

print_circuit_gates_info(sampled_circuit_gates)
save_circuit_drawing(sampled_circuit_gates, 4, "circuit.png")

# resolve_parameters(sampled_circuit_gates, RNG)
# draw_circuit(sampled_circuit_gates, 4, "circuit.png")
# print_circuit_gates_info(sampled_circuit_gates)

circuit = Circuit(
    state=STATES.generate_zero_n_qubit_state(4),
    gates=sampled_circuit_gates
)

circuit.simulate_circuit()
print(f" res: {circuit.state}")
print(f" init state: {circuit.initial_state}")
print(f"measurement: {circuit.state.measure_all(rng=RNG)}")
