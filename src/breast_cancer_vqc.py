import pennylane as qml
from pennylane import numpy as np


# ============================================================
# VQC CONFIGURATION
# ============================================================

N_QUBITS = 6
N_FEATURES = 30
N_LAYERS = N_FEATURES // N_QUBITS


# ============================================================
# QUANTUM DEVICE
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# VARIATIONAL QUANTUM CIRCUIT
# ============================================================

@qml.qnode(dev)
def vqc_circuit(x, weights):

    for layer in range(N_LAYERS):

        start = layer * N_QUBITS

        # ----------------------------------------------------
        # Data encoding
        # ----------------------------------------------------

        for qubit in range(N_QUBITS):

            qml.RY(
                x[start + qubit],
                wires=qubit
            )

        # ----------------------------------------------------
        # Trainable rotations
        # ----------------------------------------------------

        for qubit in range(N_QUBITS):

            qml.RY(
                weights[layer, qubit],
                wires=qubit
            )

            qml.RZ(
                weights[layer, qubit + N_QUBITS],
                wires=qubit
            )

        # ----------------------------------------------------
        # Entanglement
        # ----------------------------------------------------

        for qubit in range(N_QUBITS - 1):

            qml.CNOT(
                wires=[qubit, qubit + 1]
            )

    # Measure one qubit
    return qml.expval(
        qml.PauliZ(0)
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample = np.zeros(N_FEATURES)

    weights = np.zeros(
        (N_LAYERS, 2 * N_QUBITS),
        requires_grad=True
    )

    output = vqc_circuit(
        sample,
        weights
    )

    print("VQC circuit generated successfully.")
    print("Number of features:", N_FEATURES)
    print("Number of qubits:", N_QUBITS)
    print("Number of data-reuploading layers:", N_LAYERS)
    print("Trainable parameters:", weights.size)
    print("Circuit output:", output)

    print("\nCircuit:")
    print(
        qml.draw(vqc_circuit)(
            sample,
            weights
        )
    )