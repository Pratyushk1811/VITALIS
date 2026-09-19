import pennylane as qml
from pennylane import numpy as np


# =========================
# 1. CREATE QUANTUM DEVICE
# =========================

n_qubits = 6

dev = qml.device(
    "default.qubit",
    wires=n_qubits
)


# =========================
# 2. QUANTUM CIRCUIT
# =========================

@qml.qnode(dev)
def quantum_circuit(features):

    # Encode one feature into each qubit
    for i in range(n_qubits):
        qml.RY(features[i], wires=i)

    # Entangle neighbouring qubits
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    # Measure the quantum state
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# =========================
# 3. TEST INPUT
# =========================

features = np.array([
    0.5,
    1.0,
    1.5,
    0.7,
    0.3,
    1.2
])


# =========================
# 4. RUN CIRCUIT
# =========================

result = quantum_circuit(features)

print("Quantum circuit output:")
print(result)


# =========================
# 5. DISPLAY CIRCUIT
# =========================

print("\nQuantum circuit:")
print(qml.draw(quantum_circuit)(features))