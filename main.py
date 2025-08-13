import numpy as np
from utils.gates import Gates, CircuitGate, GATES
from utils.states import State, STATES
from utils.circuit import Circuit

np.set_printoptions(precision=3, suppress=True)

# state = STATES.generate_state([
#     STATES.zero,
#     STATES.plus
# ])
#
# print(state)

##### example 1
#
circuit = Circuit(
    state=STATES.generate_zero_n_qubit_state(4),
    gates=[

        CircuitGate(GATES.X, target_qubit=2),
        CircuitGate(GATES.init_Rx(np.pi / 4), target_qubit=2),
        # CircuitGate(GATES.Y, np.array([0])),
        # CircuitGate(GATES.X, np.array([1])),
        #
        CircuitGate(GATES.H, target_qubit=2),
        CircuitGate(GATES.H, target_qubit=0),

        CircuitGate(GATES.CNOT, target_qubit=0, control_qubit=2),
        CircuitGate(GATES.CNOT, target_qubit=1, control_qubit=2),
        CircuitGate(GATES.CNOT, target_qubit=2, control_qubit=1),
        CircuitGate(GATES.CNOT, target_qubit=3, control_qubit=0)



    ]
)

circuit.simulate_circuit()

# print(f" res : {circuit.state.qubit_vector}")
print(f" res: {circuit.state}")
# print(f"measurement: {circuit.state.measure_all()}")


print(circuit.state.measure_qubit(1))
print(circuit.state)
print(circuit.state.measure_qubit(2))
print(circuit.state)

print(circuit.state.measure_qubit(0))
print(circuit.state)
print(circuit.state.measure_qubit(3))
print(circuit.state)
print(circuit.state.get_probabilities_str())




