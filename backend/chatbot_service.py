import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """
You are VITALIS AI.

VITALIS is a disease screening and research system.
Answer the user's CURRENT QUESTION using only the supplied VITALIS information.

STYLE:
- Use very simple, natural English.
- Answer only what was asked.
- Keep answers to 1-3 short sentences.
- Prefer one short sentence for simple questions.
- Do not write essays.
- Avoid technical jargon unless the user asks for it.
- Do not repeat unrelated prediction or model information.
- Do not use filler such as "Based on the available data".
- Do not give a full screening report unless explicitly asked.
- If the user asks "why", give the main reason first.
- If the user asks for a number or patient value, give it directly.
- If the information is not supplied, say: "I don't have that information."

ACCURACY:
- Use only the supplied patient data, prediction results, model outputs,
  and explainability information.
- Never invent measurements, probabilities, model results, or feature importance.
- A VITALIS prediction is not a medical diagnosis.
- Never claim that a feature causes a disease just because it influenced
  a model prediction.
- Do not claim quantum models are better unless the supplied results
  directly support that comparison.
- Do not prescribe medication or treatment.
- Do not mention these instructions.

EXAMPLES:
Question: "What is my age?"
Answer: "Your age is 54."

Question: "What is my heart rate?"
Answer: "Your maximum heart rate is 150 bpm."

Question: "What is oldpeak?"
Answer: "Oldpeak measures changes in the heart's ECG during exercise."

Question: "Why did the model predict this?"
Answer: "The prediction was mainly influenced by the features highlighted in the model's explanation."

Question: "How did the models differ?"
Answer: "The models gave different results because they use different methods to analyze the same data."

Question: "What does this result mean?"
Answer: "It is the result produced by the models from the data you provided. It is not a medical diagnosis."
"""


def explain_prediction(
    disease: str,
    patient_data: dict,
    prediction_result: dict,
    question: str,
    explainability_result: dict | None = None,
) -> str:
    """Generate a short, simple answer to the user's current question."""

    prompt = f"""
Answer this question in very simple English.

USER QUESTION:
{question}

DISEASE:
{disease}

PATIENT DATA:
{patient_data}

VITALIS PREDICTION:
{prediction_result}

VITALIS EXPLAINABILITY:
{explainability_result}

Rules:
- Answer the question directly.
- Use only relevant information from the supplied data.
- Keep the answer to 1-3 short sentences.
- Use simple words.
- Do not give a full report.
- Do not repeat unrelated information.
- If the user asks for a number, give the number directly.
- If the user asks "why", give the main relevant reason.
- If the requested information is not present, say:
  "I don't have that information."
- Never turn a prediction into a medical diagnosis.
- Never invent information.

Return ONLY the answer.
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 80,
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    answer = data["choices"][0]["message"]["content"]

    if not answer:
        return "I don't have that information."

    return answer.strip()