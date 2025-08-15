from utils.gates import CircuitGate
from qutip_qip.circuit import QubitCircuit
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def print_circuit_gates_info(circuit_gates: list[CircuitGate]):
    for cg in circuit_gates:
        gate_str = f"{cg.gate.name}, target: {cg.target_qubit}"
        if cg.control_qubit is not None:
            gate_str += f", control: {cg.control_qubit}"
        print(gate_str)


# qc.add_measurement("M0", targets=1, classical_store=0)

def draw_circuit(circuit_gates, num_qubits, filepath):
    qc = QubitCircuit(num_qubits)

    for cg in circuit_gates:
        name = cg.gate.name.split('(')[0]  # strip params
        t = cg.target_qubit
        c = cg.control_qubit

        # 1-qubit gates
        if c is None:
            qc.add_gate(name, targets=[t])

        # 2-qubit gates
        else:
            qc.add_gate(name, targets=[t], controls=[c])

    qc.draw(
        renderer="matplotlib"
    )
    # Draw the circuit (QuTiP will plot into the current Matplotlib figure
    # Get the active figure and save it
    fig = plt.gcf()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
