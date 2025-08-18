import numpy as np
from utils.gates import ONE_QUBIT_FIXED_GATE_SET, TWO_QUBITS_FIXED_GATE_SET, FIXED_GATE_SET, \
    ONE_QUBIT_PARAMETRISED_GATE_SET, TWO_QUBITS_PARAMETRISED_GATE_SET, PARAMETRISED_GATE_SET, \
    ONE_QUBIT_GATES, TWO_QUBITS_GATES, CircuitGate, GATES
from utils.states import State, STATES
from utils.circuit import Circuit
from utils.draw_circuit import print_circuit_gates_info, save_circuit_drawing
from utils.random_gates import sample_random_gates, resolve_parameters
from utils.noise_channels import DepolarizingNoise, TRCNoise, SPAMNoise

np.set_printoptions(precision=3, suppress=True)

# define rng
RNG = np.random.default_rng(4242)


num_of_qubits = 4
sampled_circuit_gates = sample_random_gates(num_of_qubits=num_of_qubits,
                                            layers=[
                                                ("1q", 3, ONE_QUBIT_FIXED_GATE_SET),
                                                # ("1q", 1, ONE_QUBIT_PARAMETRISED_GATE_SET),
                                                ("2q", 2, TWO_QUBITS_FIXED_GATE_SET),
                                                # ("mixed", 1, PARAMETRISED_GATE_SET),
                                            ],
                                            rng=RNG)


print_circuit_gates_info(sampled_circuit_gates)

save_circuit_drawing(sampled_circuit_gates, num_of_qubits, "circuit.png")
save_circuit_drawing(sampled_circuit_gates, num_of_qubits, "circuit_noisy.png", True, True, True)
save_circuit_drawing(sampled_circuit_gates, num_of_qubits, "circuit_noisy_dc.png", depolarizing_noise=True)
save_circuit_drawing(sampled_circuit_gates, num_of_qubits, "circuit_noisy_trc.png", trc_noise=True)
save_circuit_drawing(sampled_circuit_gates, num_of_qubits, "circuit_noisy_spam.png", spam_noise=True)

# resolve_parameters(sampled_circuit_gates, RNG)
# draw_circuit(sampled_circuit_gates, num_of_qubits, "circuit.png")

circuit = Circuit(
    state=STATES.generate_zero_n_qubit_state(num_of_qubits),
    gates=sampled_circuit_gates,
    rng=RNG,
    depolarizing_noise=DepolarizingNoise(p1_1q=0.05, p1_2q=0.1),   # 5% and 10% depolarizing
    spam_noise=SPAMNoise(p_prep_default=0.1, p_meas_default=0.1),  # 10% prep error, 20% meas error
    trc_noise=TRCNoise(T1_default=5e-6, T2_default=8e-6)   # much shorter T1/T2 → more decay
)

# print_circuit_gates_info(circuit.gates)


circuit.simulate_circuit()
print(f" res      : {circuit.state}")
print(f" res noisy: {circuit.noisy_state}")
print(f" init state: {circuit.initial_state}")
print(f"measurement  : {circuit.state.measure_all(rng=RNG)}")
print(f"measurement n: {circuit.noisy_state.measure_all(rng=RNG)}")
