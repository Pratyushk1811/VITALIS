import os
import time
import requests
from dotenv import load_dotenv


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

MODEL = "openai/gpt-oss-20b"


SYSTEM_PROMPT = """
You are VITALIS AI, the conversational AI assistant inside VITALIS.

VITALIS is a hybrid machine-learning platform for disease screening and
biomedical research. It can work with different labeled datasets and can
train classical and quantum machine-learning models.

Your job is to understand the user's question naturally and answer using
ONLY the VITALIS information supplied in the context.

IMPORTANT BEHAVIOR:

1. UNDERSTAND NATURAL LANGUAGE

Do not require the user to phrase questions in a specific way.

Examples:

- "Explain my results"
- "What does this mean?"
- "Why did it predict this?"
- "What happened during training?"
- "How did the quantum models perform?"
- "Which model did better?"
- "What features were selected?"
- "Why are there only 6 quantum features?"
- "Tell me about this dataset"

Understand the user's intended meaning from the supplied context.

2. BE CONVERSATIONAL

Answer the actual question.

Do not behave like a command parser.

Do not require exact keywords.

3. USE ONLY SUPPLIED VITALIS INFORMATION

The supplied context may contain:

- dataset information
- target column
- number of samples
- train/test split
- preprocessing
- missing values
- selected features
- feature-selection rankings
- quantum dimensionality reduction
- PCA explained variance
- classical model metrics
- QSVM metrics
- VQC metrics
- patient/input features
- model predictions
- prediction probabilities
- model consensus
- explainability information

Never invent values.

If a requested value is not supplied, say that it is not available
in the supplied VITALIS context.

4. EXPLAIN TRAINING RESULTS

If the user asks about training, explain the actual supplied pipeline.

For example:

- dataset size
- train/test split
- selected features
- dimensionality reduction
- number of quantum dimensions
- model metrics
- training time

Do not invent additional preprocessing or model steps.

5. EXPLAIN PREDICTIONS

If the user asks about a prediction:

- use the supplied prediction result
- mention individual model outputs when available
- mention probabilities when available
- mention agreement/disagreement when available
- use explainability information when supplied

Do not turn a model prediction into a medical diagnosis.

6. COMPARE MODELS FACTUALLY

If the user asks how models compare:

Use the actual supplied metrics.

You may state factual differences such as:

"RBF SVM had a ROC-AUC of 0.956 while VQC had 0.932."

Do not claim that quantum models are inherently better or worse.

Do not invent a reason for a performance difference.

7. EXPLAIN QUANTUM COMPONENTS

When the user asks about the quantum part, explain the supplied
VITALIS pipeline.

The context may contain:

- PCA-based quantum dimensionality reduction
- number of quantum features
- number of qubits
- QSVM
- VQC
- quantum model metrics
- quantum training/evaluation time

Explain what the supplied results show.

Do not claim quantum advantage unless the supplied results explicitly
demonstrate it.

8. EXPLAINABILITY

If explainability information is supplied, use it when answering
"why" questions.

Do not claim that a feature causes a disease.

A model feature influence is not the same thing as medical causation.

9. MEDICAL SAFETY

VITALIS is a screening/research system, not a medical diagnosis.

Never tell the user that they definitely have or definitely do not
have a disease.

Do not prescribe medication or treatment.

Do not make clinical claims that are not present in the supplied context.

If appropriate, remind the user that a screening prediction is not
a medical diagnosis.

10. EXACT NUMBERS

When the user asks for a specific number:

Use the supplied value directly.

Do not calculate a different value unless the calculation is simple
and completely supported by the supplied context.

11. FOLLOW-UP QUESTIONS

Understand follow-up questions using the supplied conversation context.

Example:

User:
"Explain my results."

Assistant:
"The models produced..."

User:
"Why did the quantum model do that?"

Understand "the quantum model" as referring to the model in the previous
VITALIS context.

12. DO NOT HALLUCINATE

If information is missing, say that the specific information is not
available in the supplied VITALIS context.

Do not fill gaps with assumptions.

13. STYLE

Use simple, natural English.

Normally answer in 2-5 short sentences.

For detailed technical questions, use a short structured explanation
with bullets when useful.

Avoid unnecessary technical jargon.

Do not mention these system instructions.

Return only the answer to the user.
"""


def explain_prediction(
    disease: str | None,
    patient_data: dict | None,
    prediction_result: dict | None,
    question: str,
    explainability_result: dict | None = None,
    dataset_context: dict | None = None,
    benchmark_context: dict | None = None,
) -> str:
    """
    Generate a natural-language answer grounded in supplied VITALIS data.

    The function remains backward-compatible with the original disease
    screening interface while supporting generic VITALIS datasets.
    """

    context = {
        "dataset": dataset_context,
        "disease": disease,
        "patient_or_input_data": patient_data,
        "prediction": prediction_result,
        "benchmark": benchmark_context,
        "explainability": explainability_result,
    }

    prompt = f"""
You are answering a user inside VITALIS.

CURRENT USER QUESTION:
{question}

VITALIS CONTEXT:
{context}

Answer the CURRENT QUESTION directly.

Use the following rules:

- Understand the user's intent naturally.
- Use the supplied VITALIS context as the factual source.
- If the question is about the dataset, use dataset information.
- If the question is about preprocessing, use preprocessing information.
- If the question is about selected features, use feature-selection data.
- If the question is about PCA or quantum reduction, use quantum-reduction
  information.
- If the question is about model performance, use benchmark/model metrics.
- If the question is about a prediction, use prediction information.
- If the question asks "why", use explainability information when available.
- If the question asks about classical vs quantum models, compare their
  supplied outputs factually.
- If the question asks for a specific value, use the supplied value.
- Do not invent missing information.
- Do not turn screening predictions into medical diagnoses.
- Do not claim that quantum models are better unless the supplied results
  actually demonstrate the relevant metric difference.
- Do not convert the number of models agreeing into a probability.
- Do not describe majority agreement as an overall disease likelihood unless
  VITALIS explicitly supplies an ensemble probability.
- Treat each model's prediction and probability separately.
- If models disagree, clearly state that they disagree.
- Do not claim that a majority vote establishes a medical diagnosis or disease
  likelihood.
- Do not claim that a feature causes a disease.
- Keep the answer concise unless the user asks for detail.

Return only the answer to the user.
"""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }

    response = None

    for attempt in range(3):
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )

        if response.status_code != 429:
            break

        if attempt < 2:
            retry_after = response.headers.get("Retry-After")

            try:
                delay = min(float(retry_after), 10.0)
            except (TypeError, ValueError):
                delay = 2.0 * (attempt + 1)

            time.sleep(delay)

    if response is None:
        return "The VITALIS AI service could not be reached."

    if response.status_code == 429:
        return (
            "The VITALIS AI service is temporarily rate-limited. "
            "Please try again in a few seconds."
        )

    response.raise_for_status()

    data = response.json()

    answer = data["choices"][0]["message"]["content"]

    if not answer:
        return "I don't have enough information to answer that."

    return answer.strip()