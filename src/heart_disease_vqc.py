import pennylane as qml
from pennylane import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

N_QUBITS = 6
N_FEATURES = 6

# Three variational layers.
#
# Every layer re-uploads all six input features.
#
# 6 features
# × 2 trainable rotations (RY + RZ)
# × 3 layers
# = 36 trainable parameters

N_LAYERS = 3


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

        # ----------------------------------------------------
        # Data re-uploading
        # ----------------------------------------------------

        for qubit in range(N_QUBITS):

            qml.RY(
                x[qubit],
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
                wires=[
                    qubit,
                    qubit + 1
                ]
            )

    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------

    return qml.expval(
        qml.PauliZ(0)
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample = np.zeros(
        N_FEATURES
    )

    weights = np.zeros(
        (
            N_LAYERS,
            2 * N_QUBITS
        ),
        requires_grad=True
    )

    output = vqc_circuit(
        sample,
        weights
    )

    print("=" * 60)
    print("HEART DISEASE VQC V2")
    print("=" * 60)

    print(
        "Number of features:",
        N_FEATURES
    )

    print(
        "Number of qubits:",
        N_QUBITS
    )

    print(
        "Re-uploading layers:",
        N_LAYERS
    )

    print(
        "Trainable parameters:",
        weights.size
    )

    print(
        "Circuit output:",
        output
    )

    print("\nCircuit:")

    print(
        qml.draw(vqc_circuit)(
            sample,
            weights
        )
    )