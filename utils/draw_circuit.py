import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Instruction

from utils.gates import CircuitGate


def print_circuit_gates_info(circuit_gates: list[list[CircuitGate]]):
    """Print a readable summary of each layer’s gates and their qubit indices."""
    for layer in circuit_gates:
        print("-" * 30)
        for cg in layer:
            gate_str = f"{cg.gate.name}, target: {cg.target_qubit}"
            if cg.control_qubit is not None:
                gate_str += f", control: {cg.control_qubit}"
            print(gate_str)


def build_qiskit_circuit(
        circuit_gates: list[list["CircuitGate"]],
        num_qubits: int,
        depolarizing_noise: bool = False,
        spam_noise: bool = False,
        tdc_noise: bool = False
):
    """
    Build a Qiskit circuit from `circuit_gates` and draw placeholder noise gates.

    Noise drawing conventions (visual only):
      - 'SPAM' boxes are drawn as their own 1q gate on each qubit.
      - 'DC' (depolarizing) is drawn after each gate: 1q -> on its target;
          2q asymmetric (CNOT/CRx/CRy/CRz) -> on target only;
          2q symmetric (CZ) -> on both qubits.
      - 'TDC' is drawn after DC on every qubit touched by the layer.
      - Measurement SPAM is a separate final timestep (own layer).
      - All placeholders are zero-action `Instruction`s (for visualization only).
    """
    qc = QuantumCircuit(num_qubits)
    theta_param = Parameter('θ')

    # visual-only placeholders
    inst_SPAM = Instruction(name="SPAM", num_qubits=1, num_clbits=0, params=[])
    inst_DC = Instruction(name="DC", num_qubits=1, num_clbits=0, params=[])
    inst_TDC = Instruction(name="TDC", num_qubits=1, num_clbits=0, params=[])

    # --- pre-circuit: SPAM, and RDC only if SPAM exists (no TDC at time 0 by itself)
    pre_noise_inserted = False
    if spam_noise:
        for q in range(num_qubits):
            qc.append(inst_SPAM, [q])
        if tdc_noise:
            for q in range(num_qubits):
                qc.append(inst_TDC, [q])
        pre_noise_inserted = True
    if pre_noise_inserted:
        qc.barrier()

    # --- main layers
    for i, layer in enumerate(circuit_gates):
        touched_qubits = set()
        dc_targets = set()
        twoq_controls_needing_I = set()  # for asymmetric 2q DC alignment

        # 1) algorithmic gates
        for cg in layer:
            name = cg.gate.name.split("(", 1)[0]
            theta_str = cg.gate.name[len(name) + 1:-1] if "(" in cg.gate.name else ""
            theta = float(theta_str) if theta_str else theta_param

            t = cg.target_qubit
            c = cg.control_qubit

            if c is None:
                # 1-qubit gates
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

                touched_qubits.add(t)
                if depolarizing_noise and name != "I":
                    dc_targets.add(t)

            else:
                # 2-qubit (Qiskit order: control, target)
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

                touched_qubits.update([c, t])

                if depolarizing_noise:
                    # if name == "CZ":
                    #     # symmetric: DC on both
                    #     dc_targets.update([c, t])
                    # else:
                    # asymmetric: DC only on target; add I on control for alignment
                    dc_targets.add(t)
                    # twoq_controls_needing_I.add(c)

        # 2) DC (and alignment I for asymmetric 2q)
        if depolarizing_noise:
            for q in sorted(dc_targets):
                qc.append(inst_DC, [q])
            for c in sorted(twoq_controls_needing_I):
                qc.id(c)

        # 3) TDC after DC on all touched qubits (no TDC column before first layer unless SPAM existed)
        if tdc_noise:
            for q in sorted(touched_qubits):
                qc.append(inst_TDC, [q])

        # 4) barrier between layers (not after the last)
        if i < len(circuit_gates) - 1:
            qc.barrier()

    # --- measurement SPAM as its own final timestep
    if spam_noise:
        qc.barrier()
        for q in range(num_qubits):
            qc.append(inst_SPAM, [q])

    return qc


def get_circuit_plot(
        circuit_gates: list[list[CircuitGate]],
        num_qubits: int,
        depolarizing_noise=False,
        spam_noise=False,
        tdc_noise=False,
        plot_barriers=True
):
    """Render the circuit to a Matplotlib figure using Qiskit’s drawer."""
    qc = build_qiskit_circuit(circuit_gates, num_qubits, depolarizing_noise, spam_noise, tdc_noise)
    fig = qc.draw(
        output="mpl",  # returns a Matplotlib Figure
        plot_barriers=plot_barriers,
        style="bw",
        initial_state=True
    )
    return fig


def save_circuit_drawing(
        circuit_gates: list[list[CircuitGate]],
        num_qubits: int,
        filepath: str,
        depolarizing_noise=False,
        spam_noise=False,
        tdc_noise=False,
        plot_barriers=True
):
    """Save a static circuit diagram (Matplotlib) to disk."""
    qc = build_qiskit_circuit(circuit_gates, num_qubits, depolarizing_noise, spam_noise, tdc_noise)

    # Draw with Matplotlib and save
    fig = qc.draw(
        output="mpl",  # returns a Matplotlib Figure
        plot_barriers=plot_barriers,
        style="bw",
        initial_state=True
    )

    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
