import numpy as np

from backend.breast_cancer_pipeline import (
    breast_cancer_feature_map,
    predict_qsvm,
    predict_vqc,
    predict_breast_cancer,
)

from backend.explainability_service import (
    explain_breast_cancer,
    explain_heart,
)


print("=" * 60)
print("VITALIS QUANTUM BACKEND TEST")
print("=" * 60)


# ============================================================
# TEST 1 — QUANTUM FEATURE MAP
# ============================================================

print("\n[1] Quantum feature map")

x = np.linspace(0, 1, 30)

state = breast_cancer_feature_map(x)

print("State shape:", state.shape)
print("State norm:", np.linalg.norm(state))

assert state.shape == (64,)
assert np.isclose(np.linalg.norm(state), 1.0)

print("PASS")


# ============================================================
# TEST 2 — ALL 30 FEATURES ACTUALLY REACH THE CIRCUIT
# ============================================================

print("\n[2] 30-feature encoding")

x1 = np.zeros(30)

x2 = np.zeros(30)
x2[0] = 0.5

x3 = np.zeros(30)
x3[29] = 0.5

state_1 = breast_cancer_feature_map(x1)
state_2 = breast_cancer_feature_map(x2)
state_3 = breast_cancer_feature_map(x3)

feature_1_changes_state = not np.allclose(
    state_1,
    state_2
)

feature_30_changes_state = not np.allclose(
    state_1,
    state_3
)

print(
    "Feature 1 changes state:",
    feature_1_changes_state
)

print(
    "Feature 30 changes state:",
    feature_30_changes_state
)

assert feature_1_changes_state
assert feature_30_changes_state

print("PASS")


# ============================================================
# TEST 3 — BREAST QSVM
# ============================================================

print("\n[3] Breast-cancer QSVM")

patient = {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,

    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave_points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193,

    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019.0,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave_points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189,
}

full_result = predict_breast_cancer(patient)

qsvm_result = full_result["models"]["qsvm"]

print("QSVM prediction:", qsvm_result["prediction"])
print(
    "Class 0 probability:",
    qsvm_result["class_0_probability"]
)
print(
    "Class 1 probability:",
    qsvm_result["class_1_probability"]
)

assert qsvm_result["prediction"] in [0, 1]
assert 0 <= qsvm_result["class_0_probability"] <= 1
assert 0 <= qsvm_result["class_1_probability"] <= 1

print("PASS")


# ============================================================
# TEST 4 — BREAST VQC
# ============================================================

print("\n[4] Breast-cancer VQC")

vqc_result = full_result["models"]["vqc"]

print("VQC prediction:", vqc_result["prediction"])
print(
    "Class 0 probability:",
    vqc_result["class_0_probability"]
)
print(
    "Class 1 probability:",
    vqc_result["class_1_probability"]
)

assert vqc_result["prediction"] in [0, 1]
assert 0 <= vqc_result["class_0_probability"] <= 1
assert 0 <= vqc_result["class_1_probability"] <= 1

print("PASS")


# ============================================================
# TEST 5 — COMPLETE BREAST PIPELINE
# ============================================================

print("\n[5] Complete breast-cancer pipeline")

print(
    "Prediction:",
    full_result["prediction"]
)

print(
    "Consensus:",
    full_result["consensus"]
)

print(
    "Models:",
    list(full_result["models"].keys())
)

expected_models = {
    "logistic_regression",
    "random_forest",
    "rbf_svm",
    "qsvm",
    "vqc",
}

assert set(full_result["models"].keys()) == expected_models

print("PASS")


# ============================================================
# TEST 6 — BREAST EXPLAINABILITY
# ============================================================

print("\n[6] Breast-cancer explainability")

explanation = explain_breast_cancer(
    patient
)

print(
    "Classical methods:",
    list(explanation["classical"].keys())
)

print(
    "Quantum methods:",
    list(explanation["quantum"].keys())
)

assert set(
    explanation["quantum"].keys()
) == {
    "qsvm",
    "vqc",
}

assert len(
    explanation["quantum"]["qsvm"]
) == 30

assert len(
    explanation["quantum"]["vqc"]
) == 30

print("PASS")


# ============================================================
# TEST 7 — CHECK WHETHER LATE FEATURES CAN INFLUENCE
# EXPLANATION
# ============================================================

print("\n[7] Quantum explanation feature coverage")

qsvm_features = {
    item["feature"]
    for item
    in explanation["quantum"]["qsvm"]
}

vqc_features = {
    item["feature"]
    for item
    in explanation["quantum"]["vqc"]
}

assert len(qsvm_features) == 30
assert len(vqc_features) == 30

print(
    "QSVM features returned:",
    len(qsvm_features)
)

print(
    "VQC features returned:",
    len(vqc_features)
)

print("PASS")


# ============================================================
# TEST 8 — HEART EXPLAINABILITY
# ============================================================

print("\n[8] Heart-disease explainability")

heart_patient = {
    "cp": 2,
    "thal": 3,
    "thalach": 150,
    "oldpeak": 1.2,
    "ca": 0,
    "age": 54,
}

heart_result = explain_heart(
    heart_patient
)

print(
    "Classical methods:",
    list(
        heart_result["classical"].keys()
    )
)

print(
    "Quantum methods:",
    list(
        heart_result["quantum"].keys()
    )
)

assert set(
    heart_result["classical"].keys()
) == {
    "logistic_regression",
    "random_forest",
    "rbf_svm",
}

assert set(
    heart_result["quantum"].keys()
) == {
    "qsvm",
    "vqc",
}

print("PASS")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)