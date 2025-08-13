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

        CircuitGate(GATES.X, np.array([2])),
        # CircuitGate(GATES.Z, np.array([2])),
        # CircuitGate(GATES.Y, np.array([0])),
        # CircuitGate(GATES.X, np.array([1])),
        #
        CircuitGate(GATES.H, np.array([2])),
        CircuitGate(GATES.H, np.array([0])),

        CircuitGate(GATES.CNOT, target_qubits=np.array([0]), control_qubits=np.array([2])),  # control, target
        CircuitGate(GATES.CNOT, target_qubits=np.array([1]), control_qubits=np.array([2])),  # control, target
        CircuitGate(GATES.CNOT, target_qubits=np.array([2]), control_qubits=np.array([1])),  # control, target
        CircuitGate(GATES.CNOT, target_qubits=np.array([3]), control_qubits=np.array([0])),  # control, target



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




