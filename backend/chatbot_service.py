import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set."
    )

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"


SYSTEM_PROMPT = """
You are the AI explanation assistant for VITALIS.

VITALIS is a hybrid classical-quantum machine learning
platform for disease screening and research benchmarking.

Your job is to explain the results produced by VITALIS.

IMPORTANT RULES:

1. Never replace the actual VITALIS prediction.
2. Never invent patient measurements, probabilities,
   model outputs, benchmark results, or clinical facts.
3. Treat supplied VITALIS model outputs as the source of
   truth for the prediction being explained.
4. Clearly distinguish model prediction, probability,
   consensus, feature importance, and feature sensitivity.
5. Never describe a model prediction as a medical diagnosis.
6. Never claim that a feature causes a disease.
7. Quantum models are research models. Do not claim
   demonstrated quantum advantage.
8. If models disagree, explicitly mention the disagreement.
9. If information is insufficient, say so.
10. Keep explanations concise and understandable.
11. VITALIS does not replace professional medical evaluation.
"""


def explain_prediction(
    disease: str,
    patient_data: dict,
    prediction_result: dict,
    explainability_result: dict | None = None,
) -> str:

    prompt = f"""
Explain the following VITALIS screening result.

DISEASE MODULE:
{disease}

PATIENT INPUT:
{patient_data}

VITALIS PREDICTION:
{prediction_result}

VITALIS EXPLAINABILITY OUTPUT:
{explainability_result}

Use this structure:

1. Result
2. Model consensus
3. What influenced the prediction
4. Classical vs quantum model behavior
5. Important limitation

Do not invent information that is not present.
Do not describe the result as a medical diagnosis.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=700,
    )

    return response.choices[0].message.content