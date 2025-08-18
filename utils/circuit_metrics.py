import numpy as np


def fidelity(state1, state2):
    v1 = state1.qubit_vector.ravel()
    v2 = state2.qubit_vector.ravel()
    return np.abs(np.vdot(v1, v2))**2


def circuit_expressibility():
    ...


def circuit_entangling_capability():
    ...

