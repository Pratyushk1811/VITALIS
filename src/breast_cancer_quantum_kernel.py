from pennylane import numpy as np

from src.breast_cancer_quantum import breast_cancer_feature_map


def quantum_kernel(x1, x2):
    """
    Compute the quantum kernel between two samples.

    K(x1, x2) = |<phi(x1) | phi(x2)>|^2
    """

    state1 = breast_cancer_feature_map(x1)
    state2 = breast_cancer_feature_map(x2)

    overlap = np.vdot(state1, state2)

    kernel_value = np.abs(overlap) ** 2

    return float(kernel_value)


if __name__ == "__main__":

    # Two simple test inputs
    patient_a = np.zeros(30)
    patient_b = np.ones(30)

    similarity = quantum_kernel(
        patient_a,
        patient_b
    )

    print("Quantum kernel similarity:", similarity)

    print("Self-similarity A:", quantum_kernel(patient_a, patient_a))
    print("Self-similarity B:", quantum_kernel(patient_b, patient_b))