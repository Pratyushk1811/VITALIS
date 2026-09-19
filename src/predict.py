from src.classical_inference import predict_classical
from src.quantum_inference import predict_quantum


# ============================================================
# VITALIS UNIFIED INFERENCE
# ============================================================

def analyze_patient(patient):
    """
    Run all classical and quantum models for one patient.

    Expected input:

    {
        "cp": 2,
        "thal": 3,
        "thalach": 150,
        "oldpeak": 1.2,
        "ca": 0,
        "age": 54
    }
    """

    # --------------------------------------------------------
    # Run classical models
    # --------------------------------------------------------

    classical_results = predict_classical(patient)

    # --------------------------------------------------------
    # Run quantum model
    # --------------------------------------------------------

    quantum_result = predict_quantum(patient)

    # --------------------------------------------------------
    # Combine all model results
    # --------------------------------------------------------

    predictions = {
        "lr": classical_results["lr"],
        "rf": classical_results["rf"],
        "svm": classical_results["svm"],
        "qsvm": quantum_result
    }

    # --------------------------------------------------------
    # Calculate model consensus
    # --------------------------------------------------------

    prediction_values = [
        predictions["lr"]["prediction"],
        predictions["rf"]["prediction"],
        predictions["svm"]["prediction"],
        predictions["qsvm"]["prediction"]
    ]

    class_0_count = prediction_values.count(0)
    class_1_count = prediction_values.count(1)

    if class_1_count > class_0_count:
        unified_prediction = 1
        consensus_count = class_1_count
    else:
        unified_prediction = 0
        consensus_count = class_0_count

    # --------------------------------------------------------
    # Human-readable consensus
    # --------------------------------------------------------

    consensus_text = (
        f"{consensus_count} out of "
        f"{len(prediction_values)} models agree on "
        f"the classification."
    )

    # --------------------------------------------------------
    # Return unified result
    # --------------------------------------------------------

    return {
        "predictions": predictions,

        "unifiedPrediction": unified_prediction,

        "consensus": {
            "count": consensus_count,
            "total": len(prediction_values),
            "text": consensus_text
        }
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_patient = {
        "cp": 2,
        "thal": 3,
        "thalach": 150,
        "oldpeak": 1.2,
        "ca": 0,
        "age": 54
    }

    print("=" * 55)
    print("VITALIS UNIFIED INFERENCE")
    print("=" * 55)

    print("\nPatient:")

    for feature, value in test_patient.items():
        print(f"{feature}: {value}")

    # --------------------------------------------------------
    # Run complete analysis
    # --------------------------------------------------------

    result = analyze_patient(test_patient)

    # ========================================================
    # MODEL RESULTS
    # ========================================================

    print("\n" + "-" * 55)
    print("MODEL RESULTS")
    print("-" * 55)

    for model_name, model_result in result["predictions"].items():

        print(f"\n{model_name.upper()}")

        print(
            "Prediction:",
            model_result["prediction"]
        )

        # Classical models return "score".
        # QSVM returns separate probabilities.

        if "score" in model_result:

            print(
                "Probability:",
                f"{model_result['score']:.4f}"
            )

        else:

            print(
                "Class 0 Probability:",
                f"{model_result['class_0_probability']:.4f}"
            )

            print(
                "Class 1 Probability:",
                f"{model_result['class_1_probability']:.4f}"
            )

    # ========================================================
    # UNIFIED RESULT
    # ========================================================

    print("\n" + "-" * 55)
    print("UNIFIED RESULT")
    print("-" * 55)

    print(
        "Prediction:",
        result["unifiedPrediction"]
    )

    print(
        "Consensus:",
        result["consensus"]["text"]
    )

    print("\n" + "=" * 55)