import matplotlib
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from utils.gates import CircuitGate


matplotlib.use("Agg")


def print_circuit_gates_info(circuit_gates: list[list[CircuitGate]]):
    for layer in circuit_gates:
        print("-"*30)
        for cg in layer:
            gate_str = f"{cg.gate.name}, target: {cg.target_qubit}"
            if cg.control_qubit is not None:
                gate_str += f", control: {cg.control_qubit}"
            print(gate_str)


def build_qiskit_circuit(circuit_gates: list[list[CircuitGate]], num_qubits: int):
    """
    Build a Qiskit circuit from `circuit_gates` and save its diagram to `filepath`.
    Gate names expected (base part): I, X, Y, Z, H, P, T, Rx(θ), Ry(θ), Rz(θ),
                                     CNOT, CZ, CRx(θ), CRy(θ), CRz(θ)
    """
    qc = QuantumCircuit(num_qubits)
    theta = Parameter('θ')

    for layer in circuit_gates:
        for cg in layer:
            name = cg.gate.name.split("(", 1)[0]
            theta_str = cg.gate.name[len(name) + 1:-1]
            if theta_str:
                theta = Parameter(theta_str)

            t = cg.target_qubit
            c = cg.control_qubit

            if c is None:
                # 1-qubit
                if name == "I":
                    qc.id(t)
                elif name == "X":
                    qc.x(t)
                elif name == "Y":
                    qc.y(t)
                elif name == "Z":
                    qc.z(t)
                elif name == "H":
                    qc.h(t)
                elif name == "S":
                    qc.s(t)
                elif name == "T":
                    qc.t(t)
                elif name == "Rx":
                    qc.rx(theta, t)
                elif name == "Ry":
                    qc.ry(theta, t)
                elif name == "Rz":
                    qc.rz(theta, t)
                else:
                    raise ValueError(f"Unsupported gate: {name}")
            else:
                # 2-qubit (note: Qiskit uses (control, target) order)
                if name == "CNOT":
                    qc.cx(c, t)
                elif name == "CZ":
                    qc.cz(c, t)
                elif name == "CRx":
                    qc.crx(theta, c, t)
                elif name == "CRy":
                    qc.cry(theta, c, t)
                elif name == "CRz":
                    qc.crz(theta, c, t)
                else:
                    raise ValueError(f"Unsupported gate: {name}")

        qc.barrier()

    return qc


def draw_circuit(circuit_gates: list[list[CircuitGate]], num_qubits: int):
    qc = build_qiskit_circuit(circuit_gates, num_qubits)
    fig = qc.draw(
            output="mpl",  # returns a Matplotlib Figure
            plot_barriers=False,
            style="bw",
            initial_state=True
    )

    plt.show()


def save_circuit_drawing(circuit_gates: list[list[CircuitGate]], num_qubits: int, filepath: str):
    qc = build_qiskit_circuit(circuit_gates, num_qubits)

    # Draw with Matplotlib and save
    fig = qc.draw(
        output="mpl",  # returns a Matplotlib Figure
        plot_barriers=False,
        style="bw",
        initial_state=True
    )

    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
