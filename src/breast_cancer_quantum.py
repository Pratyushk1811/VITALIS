import pennylane as qml
from pennylane import numpy as np


# =========================
# QUANTUM CONFIGURATION
# =========================

N_QUBITS = 6
N_FEATURES = 30
N_LAYERS = N_FEATURES // N_QUBITS


# =========================
# QUANTUM DEVICE
# =========================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# =========================
# DATA RE-UPLOADING FEATURE MAP
# =========================

@qml.qnode(dev)
def breast_cancer_feature_map(x):

    for layer in range(N_LAYERS):

        start = layer * N_QUBITS
        end = start + N_QUBITS

        # Encode 6 features into 6 qubits
        for qubit in range(N_QUBITS):
            qml.RY(
                x[start + qubit],
                wires=qubit
            )

        # Entangle neighbouring qubits
        for qubit in range(N_QUBITS - 1):
            qml.CNOT(
                wires=[qubit, qubit + 1]
            )

    return qml.state()


# =========================
# TEST
# =========================

if __name__ == "__main__":

    sample = np.zeros(N_FEATURES)

    state = breast_cancer_feature_map(sample)

    print("Quantum feature map generated successfully.")
    print("Number of features:", N_FEATURES)
    print("Number of qubits:", N_QUBITS)
    print("Number of re-uploading layers:", N_LAYERS)
    print("State vector length:", len(state))

    print("\nCircuit:")
    print(qml.draw(breast_cancer_feature_map)(sample))