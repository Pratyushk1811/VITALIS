import {
  useEffect,
  useMemo,
  useState,
  type ChangeEvent,
  type KeyboardEvent,
} from 'react'
import { useNavigate } from 'react-router-dom'
import './screening.css'

const API =
  import.meta.env.VITE_API_URL ||
  localStorage.getItem('VITALIS_API_BASE_URL') ||
  'https://vitalis-production-17a9.up.railway.app'

type PageMode =
  | 'experimental'
  | 'personal'

type Disease =
  | 'heart'
  | 'breast_cancer'

type DatasetSummary = {
  rows: number
  columns: number
  column_names: string[]
  numeric_columns?: string[]
  categorical_columns?: string[]
  missing_values?: number
  duplicate_rows?: number
  target_column?: string | null
}

type PredictionModel = {
  prediction?: number | string
  class_0_probability?: number | null
  class_1_probability?: number | null
  family?: string
}

type ExperimentalPrediction = {
  disease?: string
  prediction?: number | string
  prediction_label?: string
  models?: Record<string, PredictionModel>
  consensus?: {
    agreeing_models?: number
    total_models?: number
    percentage?: number
    status?: string
  } | null
  [key: string]: any
}

type TrainingResponse = {
  session_id: string
  status: string
  dataset: {
    rows: number
    columns: number
    target_column: string
  }
  selected_features?: string[]
  quantum_dimensions?: number
  training_result?: Record<string, any>
}

type PredictionResponse = {
  session_id: string
  models: Record<string, PredictionModel>
  consensus?: {
    prediction?: number
    class_0_votes?: number
    class_1_votes?: number
    total_models?: number
    agreeing_models?: number
  } | null
  features_used?: string[]
  quantum_features?: string[]
}

type BenchmarkRow = {
  model: string
  family?: string
  accuracy?: number
  precision?: number
  sensitivity?: number
  specificity?: number
  f1_score?: number
  roc_auc?: number
}

type BenchmarkResponse = {
  session_id: string
  benchmark?: {
    models?: BenchmarkRow[]
  }
}

type ChatMessage = {
  id: number
  role: 'user' | 'assistant'
  content: string
}

type ExplanationItem = {
  feature: string
  importance: number
  direction?: string
}

type ExperimentalExplainability = {
  disease?: string
  feature_count?: number
  classical?: Record<string, ExplanationItem[]>
  quantum?: Record<string, ExplanationItem[]>
  methodology?: Record<string, string>
}

const breastFeatures = [
  'radius_mean',
  'texture_mean',
  'perimeter_mean',
  'area_mean',
  'smoothness_mean',
  'compactness_mean',
  'concavity_mean',
  'concave_points_mean',
  'symmetry_mean',
  'fractal_dimension_mean',

  'radius_se',
  'texture_se',
  'perimeter_se',
  'area_se',
  'smoothness_se',
  'compactness_se',
  'concavity_se',
  'concave_points_se',
  'symmetry_se',
  'fractal_dimension_se',

  'radius_worst',
  'texture_worst',
  'perimeter_worst',
  'area_worst',
  'smoothness_worst',
  'compactness_worst',
  'concavity_worst',
  'concave_points_worst',
  'symmetry_worst',
  'fractal_dimension_worst',
]

const heartFields = [
  {
    key: 'age',
    label: 'Age',
    type: 'number',
  },
  {
    key: 'cp',
    label: 'Chest Pain Type',
    type: 'select',
    options: [
      {
        value: 1,
        label: '1 — Typical angina',
      },
      {
        value: 2,
        label: '2 — Atypical angina',
      },
      {
        value: 3,
        label: '3 — Non-anginal pain',
      },
      {
        value: 4,
        label: '4 — Asymptomatic',
      },
    ],
  },
  {
    key: 'thalach',
    label: 'Maximum Heart Rate',
    type: 'number',
  },
  {
    key: 'oldpeak',
    label: 'ST Depression',
    type: 'number',
  },
  {
    key: 'ca',
    label: 'Major Vessels',
    type: 'select',
    options: [
      {
        value: 0,
        label: '0',
      },
      {
        value: 1,
        label: '1',
      },
      {
        value: 2,
        label: '2',
      },
      {
        value: 3,
        label: '3',
      },
    ],
  },
  {
    key: 'thal',
    label: 'Thalassemia',
    type: 'select',
    options: [
      {
        value: 3,
        label: '3 — Normal',
      },
      {
        value: 6,
        label: '6 — Fixed defect',
      },
      {
        value: 7,
        label: '7 — Reversible defect',
      },
    ],
  },
]

async function jsonApi(
  path: string,
  options: RequestInit = {},
) {
  const headers = new Headers(options.headers)

  if (
    options.body &&
    !(options.body instanceof FormData) &&
    !headers.has('Content-Type')
  ) {
    headers.set(
      'Content-Type',
      'application/json',
    )
  }

  const response = await fetch(
    `${API}${path}`,
    {
      ...options,
      headers,
    },
  )

  let data: any = {}

  try {
    data = await response.json()
  } catch {
    data = {}
  }

  if (!response.ok) {
    throw new Error(
      data?.detail ||
        data?.message ||
        `Request failed (${response.status})`,
    )
  }

  return data
}

function formatModelName(
  name: string,
) {
  const names: Record<
    string,
    string
  > = {
    logistic_regression:
      'Logistic Regression',
    random_forest:
      'Random Forest',
    rbf_svm: 'RBF SVM',
    qsvm: 'QSVM',
    vqc: 'VQC',
  }

  return names[name] || name
}

function familyFor(
  name: string,
  model?: PredictionModel | BenchmarkRow,
) {
  if (model?.family) {
    return model.family
  }

  const lower =
    name.toLowerCase()

  if (
    lower.includes('quantum') ||
    lower.includes('qsvm') ||
    lower.includes('vqc')
  ) {
    return 'quantum'
  }

  return 'classical'
}

function percentage(
  value?: number | null,
) {
  if (
    value === undefined ||
    value === null ||
    Number.isNaN(Number(value))
  ) {
    return '—'
  }

  const number = Number(value)

  return `${
    number <= 1
      ? (number * 100).toFixed(2)
      : number.toFixed(2)
  }%`
}

function modelInitial(name: string) {
  return formatModelName(name)
    .split(' ')
    .map((part) => part[0])
    .join('')
    .slice(0, 3)
    .toUpperCase()
}

function modelAccent(name: string) {
  return familyFor(name) === 'quantum'
    ? 'QUANTUM'
    : 'CLASSICAL'
}

function renderPredictionLabel(
  prediction?: number | string | null,
) {
  if (
    prediction === undefined ||
    prediction === null
  ) {
    return '—'
  }

  return String(prediction)
}

function initialTarget(
  columns: string[],
) {
  const preferred = [
    'target',
    'label',
    'class',
    'outcome',
    'diagnosis',
    'result',
  ]

  const exact =
    columns.find((column) =>
      preferred.includes(
        column.toLowerCase(),
      ),
    )

  if (exact) {
    return exact
  }

  return (
    columns.find((column) => {
      const lower =
        column.toLowerCase()

      return (
        lower.includes('target') ||
        lower.includes('label') ||
        lower.includes('class') ||
        lower.includes('outcome')
      )
    }) || ''
  )
}

export default function ScreeningPage() {
  const navigate = useNavigate()

  const [pageMode, setPageMode] =
    useState<PageMode>(
      'experimental',
    )

  const [disease, setDisease] =
    useState<Disease>('heart')

  const [experimentalInput, setExperimentalInput] =
    useState<
      Record<string, number | string>
    >({
      age: '',
      cp: 1,
      thalach: '',
      oldpeak: '',
      ca: 0,
      thal: 3,
    })

  const [breastInput, setBreastInput] =
    useState<
      Record<string, string>
    >(
      Object.fromEntries(
        breastFeatures.map(
          (feature) => [
            feature,
            '',
          ],
        ),
      ),
    )

  const [experimentalResult, setExperimentalResult] =
    useState<ExperimentalPrediction | null>(
      null,
    )

  const [experimentalExplainability, setExperimentalExplainability] =
    useState<ExperimentalExplainability | null>(null)

  const [experimentalExplainabilityLoading, setExperimentalExplainabilityLoading] =
    useState(false)

  const [experimentalLoading, setExperimentalLoading] =
    useState(false)

  const [experimentalError, setExperimentalError] =
    useState('')

  const [experimentalChatMessages, setExperimentalChatMessages] =
    useState<ChatMessage[]>([])

  const [experimentalChatInput, setExperimentalChatInput] =
    useState('')

  const [experimentalChatLoading, setExperimentalChatLoading] =
    useState(false)

  const [experimentalChatError, setExperimentalChatError] =
    useState('')

  const [file, setFile] =
    useState<File | null>(null)

  const [summary, setSummary] =
    useState<DatasetSummary | null>(
      null,
    )

  const [preview, setPreview] =
    useState<
      Record<string, unknown>[]
    >([])

  const [targetColumn, setTargetColumn] =
    useState('')

  const [quantumComponents, setQuantumComponents] =
    useState(6)

  const [testSize, setTestSize] =
    useState(0.2)

  const [training, setTraining] =
    useState<TrainingResponse | null>(
      null,
    )

  const [sessionId, setSessionId] =
    useState('')

  const [selectedFeatures, setSelectedFeatures] =
    useState<string[]>([])

  const [predictionFeatures, setPredictionFeatures] =
    useState<
      Record<string, unknown>
    >({})

  const [prediction, setPrediction] =
    useState<PredictionResponse | null>(
      null,
    )

  const [benchmark, setBenchmark] =
    useState<BenchmarkResponse | null>(
      null,
    )

  const [uploadLoading, setUploadLoading] =
    useState(false)

  const [trainingLoading, setTrainingLoading] =
    useState(false)

  const [trainingElapsed, setTrainingElapsed] =
    useState(0)

  const [predicting, setPredicting] =
    useState(false)

  const [personalError, setPersonalError] =
    useState('')

  const [personalStage, setPersonalStage] =
    useState<
      'upload'
      | 'explore'
      | 'train'
      | 'predict'
    >('upload')

  const [chatMessages, setChatMessages] =
    useState<ChatMessage[]>([])

  const [chatInput, setChatInput] =
    useState('')

  const [chatLoading, setChatLoading] =
    useState(false)

  const [chatError, setChatError] =
    useState('')

  const [messageId, setMessageId] =
    useState(0)

  useEffect(() => {
    if (!trainingLoading) {
      setTrainingElapsed(0)
      return
    }

    const started = Date.now()

    const timer = window.setInterval(() => {
      setTrainingElapsed(
        Math.floor((Date.now() - started) / 1000),
      )
    }, 1000)

    return () => {
      window.clearInterval(timer)
    }
  }, [trainingLoading])

  const previewColumns = useMemo(
    () =>
      summary?.column_names ||
      (preview.length
        ? Object.keys(
            preview[0],
          )
        : []),
    [
      summary,
      preview,
    ],
  )

  /*
   * ============================================================
   * EXPERIMENTAL DATASETS
   * ============================================================
   */

  const updateExperimentalInput = (
    key: string,
    value: string,
  ) => {
    setExperimentalInput(
      (previous) => ({
        ...previous,
        [key]:
          value === ''
            ? ''
            : Number(value),
      }),
    )
  }

  const updateBreastInput = (
    key: string,
    value: string,
  ) => {
    setBreastInput(
      (previous) => ({
        ...previous,
        [key]: value,
      }),
    )
  }

  const runExperimentalPrediction =
    async () => {
      setExperimentalError('')
      setExperimentalResult(null)
      setExperimentalExplainability(null)
      setExperimentalExplainabilityLoading(false)
      setExperimentalChatMessages([])
      setExperimentalChatInput('')
      setExperimentalChatError('')

      const missing =
        disease === 'heart'
          ? heartFields
              .filter(
                (field) =>
                  experimentalInput[
                    field.key
                  ] === '',
              )
              .map(
                (field) =>
                  field.label,
              )
          : breastFeatures.filter(
              (feature) =>
                breastInput[
                  feature
                ] === '',
            )

      if (missing.length) {
        setExperimentalError(
          `Please enter: ${missing.join(', ')}`,
        )

        return
      }

      setExperimentalLoading(true)

      try {
        const body =
          disease === 'heart'
            ? {
                age: Number(
                  experimentalInput
                    .age,
                ),
                cp: Number(
                  experimentalInput
                    .cp,
                ),
                thalach:
                  Number(
                    experimentalInput
                      .thalach,
                  ),
                oldpeak:
                  Number(
                    experimentalInput
                      .oldpeak,
                  ),
                ca: Number(
                  experimentalInput
                    .ca,
                ),
                thal: Number(
                  experimentalInput
                    .thal,
                ),
              }
            : Object.fromEntries(
                breastFeatures.map(
                  (feature) => [
                    feature,
                    Number(
                      breastInput[
                        feature
                      ],
                    ),
                  ],
                ),
              )

        const endpoint =
          disease === 'heart'
            ? '/predict/heart'
            : '/predict/breast-cancer'

        const result =
          await jsonApi(
            endpoint,
            {
              method: 'POST',
              body: JSON.stringify(
                body,
              ),
            },
          )

        setExperimentalResult(result)

        // Explainability is supplementary: a failed explanation must never
        // remove or break an otherwise successful prediction.
        setExperimentalExplainabilityLoading(true)
        try {
          const explainEndpoint =
            disease === 'heart'
              ? '/explain/heart'
              : '/explain/breast-cancer'

          const explanation = await jsonApi(
            explainEndpoint,
            {
              method: 'POST',
              body: JSON.stringify(body),
            },
          )

          setExperimentalExplainability(
            explanation as ExperimentalExplainability,
          )
        } catch {
          setExperimentalExplainability(null)
        } finally {
          setExperimentalExplainabilityLoading(false)
        }
      } catch (error) {
        setExperimentalError(
          error instanceof Error
            ? error.message
            : 'Prediction failed.',
        )
      } finally {
        setExperimentalLoading(
          false,
        )
      }
    }

  /*
   * ============================================================
   * PERSONAL DATASET — UPLOAD
   * ============================================================
   */

  const handleFile = async (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const selected =
      event.target.files?.[0]

    if (!selected) {
      return
    }

    setPersonalError('')

    if (
      !selected.name
        .toLowerCase()
        .endsWith('.csv')
    ) {
      setPersonalError(
        'Please upload a CSV file.',
      )

      return
    }

    setFile(selected)
    setSummary(null)
    setPreview([])
    setTraining(null)
    setSessionId('')
    setPrediction(null)
    setBenchmark(null)
    setSelectedFeatures([])
    setPredictionFeatures({})

    setUploadLoading(true)

    try {
      const formData =
        new FormData()

      formData.append(
        'file',
        selected,
      )

      const result =
        await jsonApi(
          '/upload/dataset',
          {
            method: 'POST',
            body: formData,
          },
        )

      const nextSummary =
        result.summary as DatasetSummary

      setSummary(
        nextSummary,
      )

      setPreview(
        Array.isArray(
          result.preview,
        )
          ? result.preview
          : [],
      )

      setTargetColumn(
        initialTarget(
          nextSummary.column_names ||
            [],
        ),
      )

      setPersonalStage(
        'explore',
      )
    } catch (error) {
      setFile(null)

      setPersonalError(
        error instanceof Error
          ? error.message
          : 'Dataset upload failed.',
      )
    } finally {
      setUploadLoading(false)

      event.target.value = ''
    }
  }

  /*
   * ============================================================
   * PERSONAL DATASET — TRAIN
   * ============================================================
   */
    const trainDataset = async () => {
    if (!file) {
      setPersonalError('Please select a CSV file first.')
      return
    }

    if (!targetColumn) {
      setPersonalError('Please select a target column.')
      return
    }

    setPersonalError('')
    setTrainingLoading(true)
    setTraining(null)
    setPrediction(null)
    setBenchmark(null)
    setSessionId(null)
    setSelectedFeatures([])
    setPredictionFeatures({})

    try {
      const formData = new FormData()
      formData.append('file', file)

      const query = new URLSearchParams({
        target_column: targetColumn,
        quantum_components: String(quantumComponents),
        test_size: String(testSize),
        random_state: '42',
      })

      const response = await fetch(
        `${API}/train/dataset?${query.toString()}`,
        {
          method: 'POST',
          body: formData,
        },
      )

      let result: any = {}

      try {
        result = await response.json()
      } catch {
        result = {}
      }

      if (!response.ok) {
        throw new Error(
          result?.detail ||
            `Training failed (${response.status})`,
        )
      }

      const initial = result as TrainingResponse

      if (!initial.session_id) {
        throw new Error('Training session was not created.')
      }

      setSessionId(initial.session_id)
      setTraining(initial)

      const pollIntervalMs = 2000
      const maxPolls = 600

      let trained: TrainingResponse | null = null

      for (
        let attempt = 0;
        attempt < maxPolls;
        attempt += 1
      ) {
        await new Promise<void>((resolve) => {
          window.setTimeout(resolve, pollIntervalMs)
        })

        const session =
          (await jsonApi(
            `/train/session/${initial.session_id}`,
          )) as TrainingResponse

        setTraining(session)

        if (session.status === 'trained') {
          trained = session
          break
        }

        if (session.status === 'failed') {
          const errorMessage =
            (session as any)?.error ||
            (session as any)?.training_result?.error ||
            'Training failed.'

          throw new Error(String(errorMessage))
        }
      }

      if (!trained) {
        throw new Error(
          'Training is taking longer than expected. The backend may still be processing the session.',
        )
      }

      const features = trained.selected_features || []
      setSelectedFeatures(features)

      const firstRow = preview[0] || {}
      const values: Record<string, unknown> = {}

      for (const feature of features) {
        values[feature] = firstRow[feature] ?? ''
      }

      setPredictionFeatures(values)

      try {
        const benchmarkResult = await jsonApi(
          `/benchmark/session/${trained.session_id}`,
        )

        setBenchmark(
          benchmarkResult as BenchmarkResponse,
        )
      } catch {
        // Benchmark can be loaded after prediction as well.
      }

      setPersonalStage('predict')
    } catch (error) {
      setPersonalError(
        error instanceof Error
          ? error.message
          : 'Training failed.',
      )
    } finally {
      setTrainingLoading(false)
    }
  }
  
  /*
   * ============================================================
   * PERSONAL DATASET — PREDICT
   * ============================================================
   */

  const updatePredictionFeature =
    (
      feature: string,
      value: string,
    ) => {
      const numeric =
        Number(value)

      setPredictionFeatures(
        (previous) => ({
          ...previous,
          [feature]:
            value === ''
              ? ''
              : Number.isFinite(
                    numeric,
                  )
                ? numeric
                : value,
        }),
      )
    }

  const runPersonalPrediction =
    async () => {
      setPersonalError('')

      if (!sessionId) {
        setPersonalError(
          'Train the dataset first.',
        )

        return
      }

      const missing =
        selectedFeatures.filter(
          (feature) =>
            predictionFeatures[
              feature
            ] === '' ||
            predictionFeatures[
              feature
            ] === undefined ||
            predictionFeatures[
              feature
            ] === null,
        )

      if (missing.length) {
        setPersonalError(
          `Missing values for: ${missing.join(', ')}`,
        )

        return
      }

      setPredicting(true)

      try {
        const result =
          await jsonApi(
            '/predict/dataset',
            {
              method: 'POST',
              body: JSON.stringify({
                session_id:
                  sessionId,
                features:
                  predictionFeatures,
              }),
            },
          )

        setPrediction(
          result as PredictionResponse,
        )

        try {
          const benchmarkResult =
            await jsonApi(
              `/benchmark/session/${sessionId}`,
            )

          setBenchmark(
            benchmarkResult as BenchmarkResponse,
          )
        } catch {
          // Keep prediction usable.
        }
      } catch (error) {
        setPersonalError(
          error instanceof Error
            ? error.message
            : 'Prediction failed.',
        )
      } finally {
        setPredicting(false)
      }
    }

  /*
   * ============================================================
   * AI ASSISTANT
   * ============================================================
   */

  const askAssistant =
    async (
      question: string,
    ) => {
      const trimmed =
        question.trim()

      if (
        !trimmed ||
        !prediction ||
        !sessionId
      ) {
        return
      }

      setChatError('')
      setChatInput('')
      setChatLoading(true)

      const userMessage: ChatMessage =
        {
          id: messageId + 1,
          role: 'user',
          content: trimmed,
        }

      setMessageId(
        (value) =>
          value + 1,
      )

      setChatMessages(
        (previous) => [
          ...previous,
          userMessage,
        ],
      )

      try {
        const result =
          await jsonApi(
            '/chat',
            {
              method: 'POST',
              body: JSON.stringify({
                question:
                  trimmed,
                session_id:
                  sessionId,
                patient_data:
                  predictionFeatures,
                prediction_result:
                  prediction,
              }),
            },
          )

        const assistantMessage:
          ChatMessage = {
          id: messageId + 2,
          role: 'assistant',
          content:
            result?.explanation ||
            'No explanation was returned.',
        }

        setMessageId(
          (value) =>
            value + 1,
        )

        setChatMessages(
          (previous) => [
            ...previous,
            assistantMessage,
          ],
        )
      } catch (error) {
        setChatError(
          error instanceof Error
            ? error.message
            : 'Assistant request failed.',
        )
      } finally {
        setChatLoading(
          false,
        )
      }
    }

  const handleChatKeyDown =
    (
      event: KeyboardEvent<HTMLInputElement>,
    ) => {
      if (
        event.key === 'Enter'
      ) {
        event.preventDefault()

        void askAssistant(
          chatInput,
        )
      }
    }

  /*
   * ============================================================
   * SHARED UI
   * ============================================================
   */

  const renderTopNavigation =
    () => (
      <div
        style={{
          display: 'flex',
          gap: 8,
          marginBottom: 28,
          borderBottom:
            '1px solid #e2e5ea',
          paddingBottom: 12,
        }}
      >
        <button
          type="button"
          className={
            pageMode ===
            'experimental'
              ? 'screening-primary'
              : 'screening-secondary'
          }
          onClick={() => {
            setPageMode(
              'experimental',
            )
            setExperimentalError(
              '',
            )
          }}
        >
          Experimental Datasets
        </button>

        <button
          type="button"
          className={
            pageMode === 'personal'
              ? 'screening-primary'
              : 'screening-secondary'
          }
          onClick={() => {
            setPageMode(
              'personal',
            )
            setPersonalError('')
          }}
        >
          Your Dataset
        </button>
      </div>
    )

  /*
   * ============================================================
   * EXPERIMENTAL PAGE
   * ============================================================
   */

  const renderDiseaseSelector =
    () => (
      <div
        style={{
          display: 'grid',
          gridTemplateColumns:
            'repeat(auto-fit, minmax(280px, 1fr))',
          gap: 16,
          marginTop: 22,
        }}
      >
        <button
          type="button"
          onClick={() => {
            setDisease('heart')
            setExperimentalResult(null)
            setExperimentalExplainability(null)
            setExperimentalError(
              '',
            )
            setExperimentalChatMessages([])
            setExperimentalChatInput('')
            setExperimentalChatError('')
          }}
          style={{
            textAlign: 'left',
            padding: 22,
            border:
              disease ===
              'heart'
                ? '2px solid #2859d7'
                : '1px solid #e2e5ea',
            borderRadius: 8,
            background:
              disease ===
              'heart'
                ? '#f7f9ff'
                : '#fff',
            cursor: 'pointer',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent:
                'space-between',
              gap: 10,
            }}
          >
            <div>
              <h2
                style={{
                  margin: 0,
                }}
              >
                Cardiovascular Disease
              </h2>

              <div
                className="screening-muted"
                style={{
                  marginTop: 6,
                  fontSize: 12,
                }}
              >
                Experimental benchmark
                dataset
              </div>
            </div>

            <span className="screening-badge">
              Available
            </span>
          </div>

          <p
            className="screening-muted"
            style={{
              marginTop: 16,
              lineHeight: 1.6,
            }}
          >
            Predict using the validated
            VITALIS cardiovascular
            classical and quantum model
            set.
          </p>

          <div
            className="screening-form-grid"
            style={{
              marginTop: 12,
            }}
          >
            <div className="screening-field">
              <label className="screening-field-label">
                Dataset
              </label>
              <div className="screening-input">
                Heart Disease
              </div>
            </div>

            <div className="screening-field">
              <label className="screening-field-label">
                Models
              </label>
              <div className="screening-input">
                3 classical + 2 quantum
              </div>
            </div>
          </div>
        </button>

        <button
          type="button"
          onClick={() => {
            setDisease(
              'breast_cancer',
            )
            setExperimentalResult(null)
            setExperimentalExplainability(null)
            setExperimentalError(
              '',
            )
            setExperimentalChatMessages([])
            setExperimentalChatInput('')
            setExperimentalChatError('')
          }}
          style={{
            textAlign: 'left',
            padding: 22,
            border:
              disease ===
              'breast_cancer'
                ? '2px solid #2859d7'
                : '1px solid #e2e5ea',
            borderRadius: 8,
            background:
              disease ===
              'breast_cancer'
                ? '#f7f9ff'
                : '#fff',
            cursor: 'pointer',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent:
                'space-between',
              gap: 10,
            }}
          >
            <div>
              <h2
                style={{
                  margin: 0,
                }}
              >
                Breast Cancer
              </h2>

              <div
                className="screening-muted"
                style={{
                  marginTop: 6,
                  fontSize: 12,
                }}
              >
                Experimental benchmark
                dataset
              </div>
            </div>

            <span className="screening-badge">
              Available
            </span>
          </div>

          <p
            className="screening-muted"
            style={{
              marginTop: 16,
              lineHeight: 1.6,
            }}
          >
            Predict using the validated
            VITALIS breast-cancer
            classical and quantum model
            set.
          </p>

          <div
            className="screening-form-grid"
            style={{
              marginTop: 12,
            }}
          >
            <div className="screening-field">
              <label className="screening-field-label">
                Dataset
              </label>
              <div className="screening-input">
                Breast Cancer
              </div>
            </div>

            <div className="screening-field">
              <label className="screening-field-label">
                Models
              </label>
              <div className="screening-input">
                Classical + Quantum
              </div>
            </div>
          </div>
        </button>
      </div>
    )

  const renderHeartForm =
    () => (
      <div className="screening-panel">
        <div className="screening-form-head">
          <div>
            <h2>
              CARDIOVASCULAR PREDICTION
            </h2>

            <div
              className="screening-muted"
              style={{
                fontSize: 12,
                marginTop: 3,
              }}
            >
              Enter the features required
              by the validated cardiovascular
              model set.
            </div>
          </div>

          <span className="screening-badge">
            Experimental
          </span>
        </div>

        <div
          className="screening-form-grid"
          style={{
            marginTop: 18,
          }}
        >
          {heartFields.map(
            (field) => (
              <div
                className="screening-field"
                key={field.key}
              >
                <label className="screening-field-label">
                  {field.label}
                </label>

                {field.type ===
                'select' ? (
                  <select
                    className="screening-select"
                    value={String(
                      experimentalInput[
                        field.key
                      ],
                    )}
                    onChange={(
                      event,
                    ) =>
                      updateExperimentalInput(
                        field.key,
                        event.target
                          .value,
                      )
                    }
                  >
                    {field.options?.map(
                      (
                        option,
                      ) => (
                        <option
                          key={
                            option.value
                          }
                          value={
                            option.value
                          }
                        >
                          {
                            option.label
                          }
                        </option>
                      ),
                    )}
                  </select>
                ) : (
                  <input
                    className="screening-input"
                    type="number"
                    step="any"
                    value={String(
                      experimentalInput[
                        field.key
                      ] ?? '',
                    )}
                    onChange={(
                      event,
                    ) =>
                      updateExperimentalInput(
                        field.key,
                        event.target
                          .value,
                      )
                    }
                  />
                )}
              </div>
            ),
          )}
        </div>

        <div className="screening-actions">
          <button
            type="button"
            className="screening-primary"
            onClick={
              runExperimentalPrediction
            }
            disabled={
              experimentalLoading
            }
          >
            {experimentalLoading
              ? 'Running Models...'
              : 'Run Prediction'}
          </button>
        </div>
      </div>
    )

  const renderBreastForm =
    () => (
      <div className="screening-panel">
        <div className="screening-form-head">
          <div>
            <h2>
              BREAST CANCER PREDICTION
            </h2>

            <div
              className="screening-muted"
              style={{
                fontSize: 12,
                marginTop: 3,
              }}
            >
              Enter the 30 biomedical
              measurements required by the
              validated breast-cancer model.
            </div>
          </div>

          <span className="screening-badge">
            Experimental
          </span>
        </div>

        <div
          className="screening-form-grid"
          style={{
            marginTop: 18,
          }}
        >
          {breastFeatures.map(
            (feature) => (
              <div
                className="screening-field"
                key={feature}
              >
                <label className="screening-field-label">
                  {feature}
                </label>

                <input
                  className="screening-input"
                  type="number"
                  step="any"
                  value={
                    breastInput[
                      feature
                    ]
                  }
                  onChange={(
                    event,
                  ) =>
                    updateBreastInput(
                      feature,
                      event.target
                        .value,
                    )
                  }
                />
              </div>
            ),
          )}
        </div>

        <div className="screening-actions">
          <button
            type="button"
            className="screening-primary"
            onClick={
              runExperimentalPrediction
            }
            disabled={
              experimentalLoading
            }
          >
            {experimentalLoading
              ? 'Running Models...'
              : 'Run Prediction'}
          </button>
        </div>
      </div>
    )

  const askExperimentalAssistant = async (
    question: string,
  ) => {
    const trimmed = question.trim()

    if (!trimmed || !experimentalResult) {
      return
    }

    setExperimentalChatError('')
    setExperimentalChatInput('')
    setExperimentalChatLoading(true)

    const userMessage: ChatMessage = {
      id: messageId + 1,
      role: 'user',
      content: trimmed,
    }

    setMessageId((value) => value + 1)
    setExperimentalChatMessages((previous) => [
      ...previous,
      userMessage,
    ])

    const patientData =
      disease === 'heart'
        ? {
            age: Number(experimentalInput.age),
            cp: Number(experimentalInput.cp),
            thalach: Number(experimentalInput.thalach),
            oldpeak: Number(experimentalInput.oldpeak),
            ca: Number(experimentalInput.ca),
            thal: Number(experimentalInput.thal),
          }
        : Object.fromEntries(
            breastFeatures.map((feature) => [
              feature,
              Number(breastInput[feature]),
            ]),
          )

    try {
      const result = await jsonApi('/chat', {
        method: 'POST',
        body: JSON.stringify({
          question: trimmed,
          disease,
          patient_data: patientData,
          prediction_result: experimentalResult,
        }),
      })

      const assistantMessage: ChatMessage = {
        id: messageId + 2,
        role: 'assistant',
        content:
          result?.explanation ||
          'No explanation was returned.',
      }

      setMessageId((value) => value + 1)
      setExperimentalChatMessages((previous) => [
        ...previous,
        assistantMessage,
      ])
    } catch (error) {
      setExperimentalChatError(
        error instanceof Error
          ? error.message
          : 'Assistant request failed.',
      )
    } finally {
      setExperimentalChatLoading(false)
    }
  }

  const handleExperimentalChatKeyDown = (
    event: KeyboardEvent<HTMLInputElement>,
  ) => {
    if (event.key === 'Enter') {
      event.preventDefault()
      void askExperimentalAssistant(
        experimentalChatInput,
      )
    }
  }

  const renderExperimentalResults =
    () => {
      if (!experimentalResult) {
        return null
      }

      const models =
        experimentalResult.models ||
        {}

      const classicalModels =
        Object.entries(models).filter(
          ([name, model]) =>
            familyFor(name, model) !== 'quantum',
        )

      const quantumModels =
        Object.entries(models).filter(
          ([name, model]) =>
            familyFor(name, model) === 'quantum',
        )

      const consensus =
        experimentalResult.consensus

      const predictionValue =
        Number(experimentalResult.prediction)

      const predictionLabel =
        experimentalResult.prediction_label ||
        (predictionValue === 1
          ? 'Higher likelihood of cardiovascular disease'
          : predictionValue === 0
            ? 'Lower likelihood of cardiovascular disease'
            : 'Model consensus unavailable')

      const modelPredictions = Object.values(models)
        .map((model) => Number(model.prediction))
        .filter((value) => Number.isFinite(value))

      const totalModels =
        consensus?.total_models ??
        modelPredictions.length

      const class0Votes =
        consensus?.class_0_votes ??
        modelPredictions.filter((value) => value === 0).length

      const class1Votes =
        consensus?.class_1_votes ??
        modelPredictions.filter((value) => value === 1).length

      const agreeingModels =
        consensus?.agreeing_models ??
        Math.max(class0Votes, class1Votes)

      const agreementPercentage =
        totalModels > 0
          ? (agreeingModels / totalModels) * 100
          : 0

      const renderModelCard = (
        name: string,
        model: PredictionModel,
      ) => {
        const quantum =
          familyFor(name, model) === 'quantum'

        const class0 = Number(
          model.class_0_probability ?? 0,
        )
        const class1 = Number(
          model.class_1_probability ?? 0,
        )
        const total = class0 + class1

        const class0Width =
          total > 0 ? (class0 / total) * 100 : 0
        const class1Width =
          total > 0 ? (class1 / total) * 100 : 0

        return (
          <div
            key={name}
            style={{
              border: quantum
                ? '1px solid #c8d7f5'
                : '1px solid #e1e6ee',
              borderRadius: 12,
              padding: 17,
              background: quantum ? '#f8fbff' : '#fff',
              boxShadow: '0 1px 2px rgba(24, 39, 75, 0.03)',
              minWidth: 0,
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'space-between',
                gap: 12,
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  minWidth: 0,
                }}
              >
                <div
                  style={{
                    width: 34,
                    height: 34,
                    flexShrink: 0,
                    borderRadius: 8,
                    display: 'grid',
                    placeItems: 'center',
                    background: quantum ? '#e9f0ff' : '#f0f3f7',
                    color: '#2859d7',
                    fontSize: 10,
                    fontWeight: 800,
                    letterSpacing: '0.04em',
                  }}
                >
                  {modelInitial(name)}
                </div>

                <div style={{ minWidth: 0 }}>
                  <div
                    style={{
                      fontWeight: 700,
                      fontSize: 13,
                      color: '#1d2940',
                    }}
                  >
                    {formatModelName(name)}
                  </div>

                  <div
                    style={{
                      marginTop: 3,
                      fontSize: 10,
                      letterSpacing: '0.08em',
                      color: quantum ? '#2859d7' : '#7b879b',
                      fontWeight: 700,
                    }}
                  >
                    {quantum ? 'QUANTUM' : 'CLASSICAL'}
                  </div>
                </div>
              </div>

              <span
                style={{
                  flexShrink: 0,
                  padding: '4px 7px',
                  borderRadius: 5,
                  border: '1px solid #d9e0ea',
                  background: '#fff',
                  color: '#66748a',
                  fontSize: 10,
                  fontWeight: 700,
                }}
              >
                CLASS {model.prediction ?? '—'}
              </span>
            </div>

            <div style={{ marginTop: 20 }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'baseline',
                  gap: 10,
                }}
              >
                <span
                  style={{
                    fontSize: 10,
                    color: '#8792a5',
                    letterSpacing: '0.08em',
                    fontWeight: 700,
                  }}
                >
                  CLASS PROBABILITY
                </span>

                <span
                  style={{
                    fontSize: 12,
                    color: '#26344c',
                    fontWeight: 700,
                  }}
                >
                  {percentage(model.class_1_probability)} class 1
                </span>
              </div>

              <div
                style={{
                  display: 'flex',
                  height: 7,
                  marginTop: 9,
                  borderRadius: 999,
                  overflow: 'hidden',
                  background: '#edf1f6',
                }}
              >
                <div
                  style={{
                    width: `${class0Width}%`,
                    background: '#b9c3d1',
                  }}
                />
                <div
                  style={{
                    width: `${class1Width}%`,
                    background: '#2859d7',
                  }}
                />
              </div>

              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginTop: 8,
                  fontSize: 10,
                  color: '#7c889b',
                }}
              >
                <span>
                  Class 0 <strong>{percentage(model.class_0_probability)}</strong>
                </span>
                <span>
                  Class 1 <strong>{percentage(model.class_1_probability)}</strong>
                </span>
              </div>
            </div>
          </div>
        )
      }

      const renderExplainabilityFamily = (
        title: string,
        groups: Record<string, ExplanationItem[]> = {},
        quantum: boolean,
      ) => (
        <div
          style={{
            border: quantum ? '1px solid #cbdaf7' : '1px solid #e1e6ee',
            borderRadius: 12,
            padding: 18,
            background: quantum ? '#f8fbff' : '#fff',
            minWidth: 0,
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12,
              marginBottom: 15,
            }}
          >
            <div>
              <div
                style={{
                  fontSize: 10,
                  letterSpacing: '0.1em',
                  fontWeight: 800,
                  color: quantum ? '#2859d7' : '#68758a',
                }}
              >
                {title}
              </div>
              <div
                style={{
                  marginTop: 4,
                  fontSize: 11,
                  color: '#7c889b',
                }}
              >
                Top local feature signals
              </div>
            </div>

            <span
              style={{
                fontSize: 9,
                fontWeight: 800,
                letterSpacing: '0.08em',
                color: quantum ? '#2859d7' : '#7b879b',
                padding: '5px 7px',
                borderRadius: 5,
                background: quantum ? '#eaf1ff' : '#f1f3f6',
              }}
            >
              {quantum ? 'QML' : 'CLASSICAL'}
            </span>
          </div>

          <div style={{ display: 'grid', gap: 14 }}>
            {Object.entries(groups).map(([modelName, rawItems]) => {
              const items = [...(rawItems || [])]
                .sort(
                  (a, b) =>
                    Number(b.importance || 0) -
                    Number(a.importance || 0),
                )
                .slice(0, 3)

              const maxImportance = Math.max(
                ...items.map((item) => Number(item.importance || 0)),
                0.0001,
              )

              return (
                <div key={modelName}>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: 10,
                      marginBottom: 8,
                    }}
                  >
                    <strong
                      style={{
                        fontSize: 11,
                        color: '#26344c',
                      }}
                    >
                      {formatModelName(modelName)}
                    </strong>
                    <span
                      style={{
                        fontSize: 9,
                        color: '#a0a9b7',
                        letterSpacing: '0.06em',
                      }}
                    >
                      {items.length} FEATURES
                    </span>
                  </div>

                  <div style={{ display: 'grid', gap: 8 }}>
                    {items.map((item) => {
                      const importance = Number(item.importance || 0)
                      const width = Math.max(
                        4,
                        Math.min(100, (importance / maxImportance) * 100),
                      )
                      const direction =
                        item.direction === 'risk_increasing'
                          ? '↑'
                          : item.direction === 'risk_decreasing'
                            ? '↓'
                            : ''

                      return (
                        <div key={`${modelName}-${item.feature}`}>
                          <div
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              gap: 10,
                              marginBottom: 4,
                            }}
                          >
                            <span
                              style={{
                                fontSize: 10,
                                color: '#526078',
                                fontWeight: 600,
                              }}
                            >
                              {item.feature}
                              {direction && (
                                <span
                                  style={{
                                    marginLeft: 4,
                                    color:
                                      direction === '↑'
                                        ? '#2859d7'
                                        : '#68758a',
                                    fontWeight: 800,
                                  }}
                                >
                                  {direction}
                                </span>
                              )}
                            </span>

                            <span
                              style={{
                                fontSize: 10,
                                color: '#26344c',
                                fontWeight: 700,
                                fontVariantNumeric: 'tabular-nums',
                              }}
                            >
                              {importance.toFixed(4)}
                            </span>
                          </div>

                          <div
                            style={{
                              height: 6,
                              borderRadius: 999,
                              background: '#edf1f6',
                              overflow: 'hidden',
                            }}
                          >
                            <div
                              style={{
                                width: `${width}%`,
                                height: '100%',
                                borderRadius: 999,
                                background: quantum ? '#2859d7' : '#8796ad',
                              }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )

      return (
        <section className="screening-section">
          <div className="screening-panel">
            <div
              className="screening-form-head"
              style={{ alignItems: 'flex-start' }}
            >
              <div>
                <h2>MODEL OUTPUTS</h2>
                <div
                  className="screening-muted"
                  style={{ fontSize: 12, marginTop: 4 }}
                >
                  Independent predictions from the classical and quantum model families.
                </div>
              </div>
              <span className="screening-success">COMPLETE</span>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'minmax(0, 1.65fr) minmax(230px, 0.75fr)',
                gap: 12,
                marginTop: 18,
              }}
            >
              <div
                style={{
                  border: '1px solid #dfe5ed',
                  borderRadius: 12,
                  padding: 18,
                  background: '#fff',
                }}
              >
                <div
                  style={{
                    fontSize: 10,
                    letterSpacing: '0.09em',
                    color: '#7d899d',
                    fontWeight: 800,
                  }}
                >
                  ENSEMBLE CONSENSUS
                </div>

                <div
                  style={{
                    marginTop: 7,
                    fontSize: 20,
                    lineHeight: 1.3,
                    fontWeight: 750,
                    color: '#1c2940',
                  }}
                >
                  {predictionLabel}
                </div>

                <div
                  className="screening-muted"
                  style={{ marginTop: 7, fontSize: 11 }}
                >
                  Combined output from the available model predictions.
                </div>
              </div>

              <div
                style={{
                  border: '1px solid #c8d7f5',
                  borderRadius: 12,
                  padding: 18,
                  background: '#f8fbff',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    gap: 10,
                  }}
                >
                  <span
                    style={{
                      fontSize: 10,
                      letterSpacing: '0.09em',
                      color: '#6f7f98',
                      fontWeight: 800,
                    }}
                  >
                    MODEL AGREEMENT
                  </span>
                  <span
                    style={{
                      fontSize: 10,
                      color: '#2859d7',
                      fontWeight: 800,
                    }}
                  >
                    {agreementPercentage.toFixed(0)}%
                  </span>
                </div>

                <div
                  style={{
                    marginTop: 9,
                    display: 'flex',
                    alignItems: 'baseline',
                    gap: 7,
                  }}
                >
                  <strong style={{ fontSize: 26, color: '#1c2940' }}>
                    {agreeingModels}
                  </strong>
                  <span style={{ color: '#7c889b', fontSize: 12 }}>
                    / {totalModels} models agree
                  </span>
                </div>

                <div
                  style={{
                    height: 6,
                    marginTop: 10,
                    borderRadius: 999,
                    overflow: 'hidden',
                    background: '#e4ebf8',
                  }}
                >
                  <div
                    style={{
                      width: `${Math.min(100, agreementPercentage)}%`,
                      height: '100%',
                      borderRadius: 999,
                      background: '#2859d7',
                    }}
                  />
                </div>
              </div>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
                gap: 10,
                marginTop: 12,
              }}
            >
              {[
                ['Consensus', `Class ${consensus?.prediction ?? predictionValue ?? '—'}`],
                ['Class 0 votes', String(class0Votes)],
                ['Class 1 votes', String(class1Votes)],
                ['Models agreeing', `${agreeingModels} / ${totalModels}`],
              ].map(([label, value]) => (
                <div
                  key={label}
                  style={{
                    minHeight: 72,
                    padding: '13px 14px',
                    border: '1px solid #e1e6ee',
                    borderRadius: 10,
                    background: '#fbfcfe',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    alignItems: 'center',
                    textAlign: 'center',
                  }}
                >
                  <div
                    style={{
                      fontSize: 9,
                      letterSpacing: '0.08em',
                      fontWeight: 800,
                      color: '#8994a6',
                    }}
                  >
                    {label.toUpperCase()}
                  </div>
                  <div
                    style={{
                      marginTop: 6,
                      fontSize: 15,
                      lineHeight: 1.2,
                      fontWeight: 750,
                      color: '#24324a',
                      fontVariantNumeric: 'tabular-nums',
                    }}
                  >
                    {value}
                  </div>
                </div>
              ))}
            </div>

            {classicalModels.length > 0 && (
              <div style={{ marginTop: 26 }}>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    marginBottom: 10,
                  }}
                >
                  <span
                    style={{
                      fontSize: 10,
                      letterSpacing: '0.1em',
                      fontWeight: 800,
                      color: '#68758a',
                    }}
                  >
                    CLASSICAL MODELS
                  </span>
                  <span style={{ height: 1, flex: 1, background: '#e6eaf0' }} />
                </div>

                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
                    gap: 10,
                  }}
                >
                  {classicalModels.map(([name, model]) =>
                    renderModelCard(name, model),
                  )}
                </div>
              </div>
            )}

            {quantumModels.length > 0 && (
              <div style={{ marginTop: 22 }}>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    marginBottom: 10,
                  }}
                >
                  <span
                    style={{
                      fontSize: 10,
                      letterSpacing: '0.1em',
                      fontWeight: 800,
                      color: '#2859d7',
                    }}
                  >
                    QUANTUM MODELS
                  </span>
                  <span style={{ height: 1, flex: 1, background: '#dce5f7' }} />
                </div>

                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                    gap: 10,
                  }}
                >
                  {quantumModels.map(([name, model]) =>
                    renderModelCard(name, model),
                  )}
                </div>
              </div>
            )}

            <section
              style={{
                marginTop: 28,
                paddingTop: 24,
                borderTop: '1px solid #e2e6ed',
              }}
            >
              <div
                className="screening-form-head"
                style={{ alignItems: 'flex-start' }}
              >
                <div>
                  <h2>MODEL EXPLAINABILITY</h2>
                  <div
                    className="screening-muted"
                    style={{ fontSize: 12, marginTop: 4 }}
                  >
                    Local feature influence and sensitivity for this specific observation.
                  </div>
                </div>
                <span className="screening-badge">Research</span>
              </div>

              <div
                className="screening-notice"
                style={{
                  marginTop: 14,
                  fontSize: 11,
                  lineHeight: 1.55,
                }}
              >
                Higher bars indicate stronger model influence or sensitivity. These values are not causal effects or medical risk percentages.
              </div>

              {experimentalExplainabilityLoading && (
                <div
                  style={{
                    marginTop: 14,
                    padding: 18,
                    border: '1px solid #dfe5ed',
                    borderRadius: 10,
                    textAlign: 'center',
                    color: '#718096',
                    fontSize: 12,
                  }}
                >
                  Computing feature-level explanations…
                </div>
              )}

              {!experimentalExplainabilityLoading && experimentalExplainability && (
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                    gap: 12,
                    marginTop: 14,
                  }}
                >
                  {renderExplainabilityFamily(
                    'CLASSICAL MODEL INFLUENCE',
                    experimentalExplainability.classical || {},
                    false,
                  )}
                  {renderExplainabilityFamily(
                    'QUANTUM MODEL SENSITIVITY',
                    experimentalExplainability.quantum || {},
                    true,
                  )}
                </div>
              )}

              {!experimentalExplainabilityLoading && !experimentalExplainability && (
                <div className="screening-notice" style={{ marginTop: 14 }}>
                  Feature-level explanation data was not available for this run. The model outputs above remain available.
                </div>
              )}

              {experimentalExplainability?.methodology && (
                <div
                  style={{
                    marginTop: 12,
                    padding: 15,
                    border: '1px solid #e1e6ee',
                    borderRadius: 10,
                    background: '#fbfcfe',
                  }}
                >
                  <div
                    style={{
                      fontSize: 9,
                      letterSpacing: '0.1em',
                      fontWeight: 800,
                      color: '#7d899d',
                      marginBottom: 10,
                    }}
                  >
                    EXPLANATION METHODS
                  </div>

                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                      gap: 8,
                    }}
                  >
                    {Object.entries(experimentalExplainability.methodology).map(
                      ([modelName, method]) => (
                        <div
                          key={modelName}
                          style={{
                            padding: '9px 10px',
                            borderRadius: 7,
                            background: '#fff',
                            border: '1px solid #e7ebf1',
                          }}
                        >
                          <div
                            style={{
                              fontSize: 10,
                              fontWeight: 750,
                              color: '#33415a',
                            }}
                          >
                            {formatModelName(modelName)}
                          </div>
                          <div
                            className="screening-muted"
                            style={{
                              marginTop: 3,
                              fontSize: 10,
                              lineHeight: 1.45,
                            }}
                          >
                            {method}
                          </div>
                        </div>
                      ),
                    )}
                  </div>
                </div>
              )}
            </section>

            <div
              className="screening-panel"
              style={{
                marginTop: 24,
                padding: 18,
                border: '1px solid #dfe5ed',
                boxShadow: 'none',
                background: '#fff',
              }}
            >
              <div className="screening-form-head">
                <div>
                  <h2>VITALIS AI</h2>
                  <div
                    className="screening-muted"
                    style={{ fontSize: 12, marginTop: 4 }}
                  >
                    Ask about this prediction, model agreement, or classical-versus-quantum behavior.
                  </div>
                </div>
                <span className="screening-badge">Assistant</span>
              </div>

              <div
                style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: 8,
                  marginTop: 16,
                }}
              >
                {[
                  'Why did the models predict this?',
                  'How did the classical and quantum models differ?',
                  'What does the model agreement mean?',
                ].map((question) => (
                  <button
                    type="button"
                    className="screening-secondary"
                    key={question}
                    onClick={() =>
                      void askExperimentalAssistant(question)
                    }
                    disabled={experimentalChatLoading}
                  >
                    {question}
                  </button>
                ))}
              </div>

              {experimentalChatMessages.length > 0 && (
                <div
                  style={{
                    display: 'grid',
                    gap: 10,
                    marginTop: 16,
                    maxHeight: 320,
                    overflowY: 'auto',
                    paddingRight: 2,
                  }}
                >
                  {experimentalChatMessages.map((message) => (
                    <div
                      key={message.id}
                      style={{
                        padding: 12,
                        border: '1px solid #e2e5ea',
                        borderRadius: 9,
                        background: message.role === 'assistant' ? '#f8fbff' : '#fff',
                      }}
                    >
                      <div
                        style={{
                          fontSize: 9,
                          letterSpacing: '0.08em',
                          fontWeight: 800,
                          color: message.role === 'assistant' ? '#2859d7' : '#7d899d',
                        }}
                      >
                        {message.role === 'user' ? 'YOU' : 'VITALIS AI'}
                      </div>
                      <div
                        style={{
                          marginTop: 5,
                          lineHeight: 1.6,
                          fontSize: 12,
                          color: '#33415a',
                          whiteSpace: 'pre-wrap',
                        }}
                      >
                        {message.content}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div
                style={{
                  display: 'flex',
                  gap: 8,
                  marginTop: 16,
                  alignItems: 'stretch',
                }}
              >
                <input
                  className="screening-input"
                  style={{ flex: 1, minWidth: 0 }}
                  placeholder="Ask about this prediction..."
                  value={experimentalChatInput}
                  onChange={(event) =>
                    setExperimentalChatInput(event.target.value)
                  }
                  onKeyDown={handleExperimentalChatKeyDown}
                />
                <button
                  type="button"
                  className="screening-primary"
                  onClick={() =>
                    void askExperimentalAssistant(experimentalChatInput)
                  }
                  disabled={
                    experimentalChatLoading ||
                    !experimentalChatInput.trim()
                  }
                >
                  {experimentalChatLoading ? '...' : 'Ask'}
                </button>
              </div>

              {experimentalChatError && (
                <div
                  className="screening-notice"
                  style={{ marginTop: 12 }}
                >
                  {experimentalChatError}
                </div>
              )}
            </div>

            <div
              className="screening-notice"
              style={{ marginTop: 20 }}
            >
              Model outputs are experimental benchmark results. They support comparison of model behavior and are not a medical diagnosis.
            </div>
          </div>
        </section>
      )
    }

  const renderExperimentalPage =
    () => (
      <>
        <section className="screening-intro">
          <h1>
            EXPERIMENTAL DATASETS
          </h1>

          <p>
            Explore VITALIS using validated
            biomedical disease datasets. These
            experiments demonstrate the complete
            classical-versus-quantum prediction
            workflow.
          </p>

          <div className="screening-notice">
            Select a disease dataset, enter an
            observation, and compare the outputs
            of the trained classical and quantum
            models.
          </div>
        </section>

        <section className="screening-section">
          <div className="screening-panel">
            <div className="screening-form-head">
              <div>
                <h2>
                  SELECT EXPERIMENT
                </h2>

                <div
                  className="screening-muted"
                  style={{
                    fontSize: 12,
                    marginTop: 3,
                  }}
                >
                  VITALIS currently provides
                  cardiovascular and breast-cancer
                  experimental modules.
                </div>
              </div>
            </div>

            {renderDiseaseSelector()}
          </div>
        </section>

        <section className="screening-section">
          {disease ===
          'heart'
            ? renderHeartForm()
            : renderBreastForm()}
        </section>

        {experimentalError && (
          <section className="screening-section">
            <div className="screening-notice">
              {experimentalError}
            </div>
          </section>
        )}

        {renderExperimentalResults()}
      </>
    )

  /*
   * ============================================================
   * PERSONAL DATASET PAGE
   * ============================================================
   */

  const renderPersonalNavigation =
    () => (
      <div
        style={{
          display: 'flex',
          gap: 7,
          flexWrap: 'wrap',
          marginTop: 20,
        }}
      >
        {[
          [
            'upload',
            '1. Upload',
          ],
          [
            'explore',
            '2. Explore',
          ],
          [
            'train',
            '3. Train',
          ],
          [
            'predict',
            '4. Predict',
          ],
        ].map(
          ([key, label]) => {
            const stage =
              key as typeof personalStage

            const disabled =
              stage ===
                'explore' &&
              !summary
                ? true
                : stage ===
                      'train' &&
                    !summary
                  ? true
                  : stage ===
                        'predict' &&
                      !training
                    ? true
                    : false

            return (
              <button
                key={key}
                type="button"
                className={
                  personalStage ===
                  stage
                    ? 'screening-primary'
                    : 'screening-secondary'
                }
                disabled={
                  disabled
                }
                onClick={() =>
                  setPersonalStage(
                    stage,
                  )
                }
              >
                {label}
              </button>
            )
          },
        )}
      </div>
    )

  const renderPersonalUpload =
    () => (
      <section className="screening-section">
        <div className="screening-panel">
          <div className="screening-form-head">
            <div>
              <h2>
                UPLOAD YOUR DATASET
              </h2>

              <div
                className="screening-muted"
                style={{
                  fontSize: 12,
                  marginTop: 3,
                }}
              >
                Upload a labeled biomedical CSV
                for a new VITALIS training
                experiment.
              </div>
            </div>
          </div>

          <label
            style={{
              display: 'block',
              marginTop: 20,
              padding: 28,
              border:
                '1px dashed #bfc7d6',
              borderRadius: 8,
              cursor: 'pointer',
              textAlign: 'center',
            }}
          >
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={
                handleFile
              }
              style={{
                display: 'none',
              }}
            />

            <strong>
              {uploadLoading
                ? 'Reading dataset...'
                : 'Choose CSV dataset'}
            </strong>

            <div
              className="screening-muted"
              style={{
                marginTop: 7,
                fontSize: 12,
              }}
            >
              VITALIS will profile the
              structure before training.
            </div>
          </label>

          {file && (
            <div
              className="screening-form-grid"
              style={{
                marginTop: 18,
              }}
            >
              <div className="screening-field">
                <label className="screening-field-label">
                  File
                </label>

                <div className="screening-input">
                  {file.name}
                </div>
              </div>

              <div className="screening-field">
                <label className="screening-field-label">
                  Size
                </label>

                <div className="screening-input">
                  {(
                    file.size /
                    1024
                  ).toFixed(1)}{' '}
                  KB
                </div>
              </div>
            </div>
          )}
        </div>
      </section>
    )

  const renderPersonalExplore =
    () => {
      if (!summary) {
        return null
      }

      return (
        <section className="screening-section">
          <div className="screening-panel">
            <div className="screening-form-head">
              <div>
                <h2>
                  UNDERSTAND YOUR DATA
                </h2>

                <div
                  className="screening-muted"
                  style={{
                    fontSize: 12,
                    marginTop: 3,
                  }}
                >
                  Dataset profiling and target
                  selection happen before any
                  model training.
                </div>
              </div>

              <span className="screening-success">
                Profiled
              </span>
            </div>

            <div
              className="screening-form-grid"
              style={{
                marginTop: 18,
              }}
            >
              <div className="screening-field">
                <label className="screening-field-label">
                  Rows
                </label>

                <div className="screening-input">
                  {summary.rows}
                </div>
              </div>

              <div className="screening-field">
                <label className="screening-field-label">
                  Columns
                </label>

                <div className="screening-input">
                  {summary.columns}
                </div>
              </div>

              <div className="screening-field">
                <label className="screening-field-label">
                  Numeric
                </label>

                <div className="screening-input">
                  {
                    summary
                      .numeric_columns
                      ?.length ??
                    0
                  }
                </div>
              </div>

              <div className="screening-field">
                <label className="screening-field-label">
                  Categorical
                </label>

                <div className="screening-input">
                  {
                    summary
                      .categorical_columns
                      ?.length ??
                    0
                  }
                </div>
              </div>

              <div className="screening-field">
                <label className="screening-field-label">
                  Missing values
                </label>

                <div className="screening-input">
                  {
                    summary
                      .missing_values ??
                    0
                  }
                </div>
              </div>

              <div className="screening-field">
                <label className="screening-field-label">
                  Duplicate rows
                </label>

                <div className="screening-input">
                  {
                    summary
                      .duplicate_rows ??
                    0
                  }
                </div>
              </div>
            </div>

            <div
              className="screening-field"
              style={{
                marginTop: 20,
              }}
            >
              <label className="screening-field-label">
                Target / label column
              </label>

              <select
                className="screening-select"
                value={
                  targetColumn
                }
                onChange={(
                  event,
                ) =>
                  setTargetColumn(
                    event.target
                      .value,
                  )
                }
              >
                <option value="">
                  Select target
                </option>

                {summary.column_names.map(
                  (column) => (
                    <option
                      key={column}
                      value={column}
                    >
                      {column}
                    </option>
                  ),
                )}
              </select>
            </div>

            {preview.length >
              0 && (
              <div
                className="screening-table-wrap"
                style={{
                  marginTop: 20,
                }}
              >
                <table className="screening-review-table">
                  <thead>
                    <tr>
                      {previewColumns.map(
                        (
                          column,
                        ) => (
                          <th
                            key={
                              column
                            }
                          >
                            {
                              column
                            }
                          </th>
                        ),
                      )}
                    </tr>
                  </thead>

                  <tbody>
                    {preview.map(
                      (
                        row,
                        index,
                      ) => (
                        <tr
                          key={
                            index
                          }
                        >
                          {previewColumns.map(
                            (
                              column,
                            ) => (
                              <td
                                key={
                                  column
                                }
                              >
                                {String(
                                  row[
                                    column
                                  ] ??
                                    '—',
                                )}
                              </td>
                            ),
                          )}
                        </tr>
                      ),
                    )}
                  </tbody>
                </table>
              </div>
            )}

            <div className="screening-actions">
              <button
                type="button"
                className="screening-primary"
                disabled={
                  !targetColumn
                }
                onClick={() =>
                  setPersonalStage(
                    'train',
                  )
                }
              >
                Continue to Training
              </button>
            </div>
          </div>
        </section>
      )
    }

  const renderPersonalTrain =
    () => {
      if (!summary) {
        return null
      }

      return (
        <section className="screening-section">
          <div className="screening-panel">
            <div className="screening-form-head">
              <div>
                <h2>
                  TRAIN HYBRID PIPELINE
                </h2>

                <div
                  className="screening-muted"
                  style={{
                    fontSize: 12,
                    marginTop: 3,
                  }}
                >
                  Classical preprocessing,
                  feature selection, quantum
                  reduction, and model training.
                </div>
              </div>
            </div>

            <div
              className="screening-form-grid"
              style={{
                marginTop: 18,
              }}
            >
              <div className="screening-field">
                <label className="screening-field-label">
                  Dataset
                </label>

                <div className="screening-input">
                  {file?.name}
                </div>
              </div>

              <div className="screening-field">
                <label className="screening-field-label">
                  Target
                </label>

                <div className="screening-input">
                  {targetColumn}
                </div>
              </div>

              <div className="screening-field">
                <label className="screening-field-label">
                  Quantum dimensions
                </label>

                <input
                  className="screening-input"
                  type="number"
                  min={1}
                  max={20}
                  value={
                    quantumComponents
                  }
                  onChange={(
                    event,
                  ) =>
                    setQuantumComponents(
                      Math.max(
                        1,
                        Math.min(
                          20,
                          Number(
                            event
                              .target
                              .value,
                          ) ||
                            1,
                        ),
                      ),
                    )
                  }
                />
              </div>

              <div className="screening-field">
                <label className="screening-field-label">
                  Test split
                </label>

                <select
                  className="screening-select"
                  value={
                    testSize
                  }
                  onChange={(
                    event,
                  ) =>
                    setTestSize(
                      Number(
                        event
                          .target
                          .value,
                      ),
                    )
                  }
                >
                  <option value={0.2}>
                    20%
                  </option>

                  <option value={0.25}>
                    25%
                  </option>

                  <option value={0.3}>
                    30%
                  </option>
                </select>
              </div>
            </div>

            <div
              className="screening-notice"
              style={{
                marginTop: 18,
              }}
            >
              <strong>
                Pipeline:
              </strong>{' '}
              data cleaning → feature
              selection → quantum-compatible
              dimensionality reduction →
              classical + quantum models →
              evaluation.
            </div>

            <div className="screening-actions">
              <button
                type="button"
                className="screening-primary"
                onClick={
                  trainDataset
                }
                disabled={
                  trainingLoading ||
                  !targetColumn
                }
              >
                {trainingLoading
                  ? 'Training...'
                  : 'Train VITALIS Models'}
              </button>
            </div>

            {trainingLoading && (
              <div
                style={{
                  marginTop: 22,
                  border: '1px solid #dfe4ec',
                  borderRadius: 10,
                  padding: 20,
                  background: '#fbfcfe',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    gap: 16,
                    marginBottom: 18,
                  }}
                >
                  <div>
                    <div
                      style={{
                        fontSize: 11,
                        letterSpacing: '0.12em',
                        fontWeight: 700,
                        color: '#2859d7',
                      }}
                    >
                      HYBRID COMPUTATION ACTIVE
                    </div>

                    <div
                      style={{
                        marginTop: 5,
                        fontSize: 14,
                        fontWeight: 600,
                      }}
                    >
                      VITALIS is processing the dataset
                    </div>

                    <div
                      className="screening-muted"
                      style={{
                        marginTop: 4,
                        fontSize: 12,
                      }}
                    >
                      Classical preprocessing and quantum
                      model computation are being executed.
                    </div>
                  </div>

                  <div
                    style={{
                      minWidth: 72,
                      textAlign: 'right',
                      fontVariantNumeric: 'tabular-nums',
                      fontSize: 12,
                    }}
                  >
                    {trainingElapsed}s
                  </div>
                </div>

                <div
                  style={{
                    display: 'grid',
                    gap: 8,
                  }}
                >
                  {[
                    {
                      title: 'DATA INGESTION',
                      detail: 'Dataset loaded',
                      state: 'complete',
                    },
                    {
                      title: 'PREPROCESSING',
                      detail: 'Cleaning · encoding · normalization',
                      state: 'complete',
                    },
                    {
                      title: 'FEATURE INTELLIGENCE',
                      detail: 'Feature selection',
                      state: 'complete',
                    },
                    {
                      title: 'QUANTUM REDUCTION',
                      detail: `${selectedFeatures.length || '—'} selected → ${quantumComponents} quantum dimensions`,
                      state: 'complete',
                    },
                    {
                      title: 'CLASSICAL BASELINES',
                      detail: 'Logistic Regression · Random Forest · RBF SVM',
                      state: 'active',
                    },
                    {
                      title: 'QUANTUM MODELS',
                      detail: 'QSVM · VQC',
                      state: 'active',
                    },
                  ].map((step) => (
                    <div
                      key={step.title}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 12,
                        padding: '11px 12px',
                        border: '1px solid #e5e9f0',
                        borderRadius: 7,
                        background: '#fff',
                      }}
                    >
                      <div
                        style={{
                          width: 24,
                          height: 24,
                          borderRadius: '50%',
                          display: 'grid',
                          placeItems: 'center',
                          flexShrink: 0,
                          border:
                            step.state === 'complete'
                              ? '1px solid #2859d7'
                              : '1px solid #8fa4d7',
                          color: '#2859d7',
                          fontSize: 11,
                          fontWeight: 700,
                        }}
                      >
                        {step.state === 'complete' ? '✓' : '◉'}
                      </div>

                      <div style={{ minWidth: 0 }}>
                        <div
                          style={{
                            fontSize: 11,
                            fontWeight: 700,
                            letterSpacing: '0.06em',
                          }}
                        >
                          {step.title}
                        </div>

                        <div
                          className="screening-muted"
                          style={{
                            marginTop: 2,
                            fontSize: 11,
                          }}
                        >
                          {step.detail}
                        </div>
                      </div>

                      <div
                        style={{
                          marginLeft: 'auto',
                          fontSize: 10,
                          fontWeight: 700,
                          color: '#2859d7',
                          letterSpacing: '0.05em',
                        }}
                      >
                        {step.state === 'complete'
                          ? 'READY'
                          : 'RUNNING'}
                      </div>
                    </div>
                  ))}
                </div>

                <div
                  className="screening-muted"
                  style={{
                    marginTop: 14,
                    fontSize: 11,
                  }}
                >
                  Quantum kernel computation can take
                  substantially longer than classical
                  preprocessing. Do not close this page
                  while training is running.
                </div>
              </div>
            )}

            {training && (
              <div
                style={{
                  marginTop: 22,
                }}
              >
                <div className="screening-form-head">
                  <div>
                    <h2>
                      TRAINING COMPLETE
                    </h2>
                  </div>

                  <span className="screening-success">
                    Ready
                  </span>
                </div>

                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns:
                      'repeat(auto-fit, minmax(180px, 1fr))',
                    gap: 10,
                    marginTop: 18,
                  }}
                >
                  <div
                    style={{
                      border: '1px solid #e2e5ea',
                      borderRadius: 8,
                      padding: 14,
                    }}
                  >
                    <div className="screening-field-label">
                      DATASET
                    </div>

                    <div
                      style={{
                        marginTop: 6,
                        fontWeight: 600,
                      }}
                    >
                      {file?.name || 'Dataset'}
                    </div>
                  </div>

                  <div
                    style={{
                      border: '1px solid #e2e5ea',
                      borderRadius: 8,
                      padding: 14,
                    }}
                  >
                    <div className="screening-field-label">
                      SAMPLES
                    </div>

                    <div
                      style={{
                        marginTop: 6,
                        fontWeight: 600,
                      }}
                    >
                      {training.dataset.rows}
                    </div>
                  </div>

                  <div
                    style={{
                      border: '1px solid #e2e5ea',
                      borderRadius: 8,
                      padding: 14,
                    }}
                  >
                    <div className="screening-field-label">
                      SELECTED FEATURES
                    </div>

                    <div
                      style={{
                        marginTop: 6,
                        fontWeight: 600,
                      }}
                    >
                      {selectedFeatures.length}
                    </div>
                  </div>

                  <div
                    style={{
                      border: '1px solid #2d61d8',
                      borderRadius: 8,
                      padding: 14,
                      background: '#f7f9ff',
                    }}
                  >
                    <div className="screening-field-label">
                      QUANTUM SPACE
                    </div>

                    <div
                      style={{
                        marginTop: 6,
                        fontWeight: 600,
                        color: '#2859d7',
                      }}
                    >
                      {training.quantum_dimensions ??
                        quantumComponents}{' '}
                      dimensions
                    </div>
                  </div>

                  <div
                    style={{
                      border: '1px solid #e2e5ea',
                      borderRadius: 8,
                      padding: 14,
                    }}
                  >
                    <div className="screening-field-label">
                      TEST SPLIT
                    </div>

                    <div
                      style={{
                        marginTop: 6,
                        fontWeight: 600,
                      }}
                    >
                      {(testSize * 100).toFixed(0)}%
                    </div>
                  </div>

                  <div
                    style={{
                      border: '1px solid #e2e5ea',
                      borderRadius: 8,
                      padding: 14,
                    }}
                  >
                    <div className="screening-field-label">
                      QUANTUM QUBITS
                    </div>

                    <div
                      style={{
                        marginTop: 6,
                        fontWeight: 600,
                      }}
                    >
                      {training.quantum_dimensions ??
                        quantumComponents}
                    </div>
                  </div>
                </div>

                <div
                  style={{
                    marginTop: 18,
                    padding: 14,
                    borderRadius: 8,
                    background: '#f7f9ff',
                    border: '1px solid #dbe4f8',
                  }}
                >
                  <div
                    style={{
                      fontSize: 11,
                      fontWeight: 700,
                      letterSpacing: '0.08em',
                      color: '#2859d7',
                    }}
                  >
                    PIPELINE READY
                  </div>

                  <div
                    className="screening-muted"
                    style={{
                      marginTop: 5,
                      fontSize: 12,
                      lineHeight: 1.6,
                    }}
                  >
                    The trained classical and quantum models
                    are now available for prediction and
                    held-out performance analysis.
                  </div>
                </div>

                <div className="screening-actions">
                  <button
                    type="button"
                    className="screening-primary"
                    onClick={() =>
                      setPersonalStage(
                        'predict',
                      )
                    }
                  >
                    Continue to Prediction
                  </button>
                </div>
              </div>
            )}
          </div>
        </section>
      )
    }

  const renderPersonalPredict =
    () => {
      if (
        !training ||
        !selectedFeatures.length
      ) {
        return null
      }

      return (
        <>
          <section className="screening-section">
            <div className="screening-panel">
              <div className="screening-form-head">
                <div>
                  <h2>
                    PREDICT ON YOUR DATA
                  </h2>

                  <div
                    className="screening-muted"
                    style={{
                      fontSize: 12,
                      marginTop: 3,
                    }}
                  >
                    Enter a new observation using
                    the features selected during
                    training.
                  </div>
                </div>
              </div>

              <div
                className="screening-form-grid"
                style={{
                  marginTop: 18,
                }}
              >
                {selectedFeatures.map(
                  (feature) => (
                    <div
                      className="screening-field"
                      key={
                        feature
                      }
                    >
                      <label className="screening-field-label">
                        {feature}
                      </label>

                      <input
                        className="screening-input"
                        type="number"
                        step="any"
                        value={String(
                          predictionFeatures[
                            feature
                          ] ??
                            '',
                        )}
                        onChange={(
                          event,
                        ) =>
                          updatePredictionFeature(
                            feature,
                            event
                              .target
                              .value,
                          )
                        }
                      />
                    </div>
                  ),
                )}
              </div>

              <div className="screening-actions">
                <button
                  type="button"
                  className="screening-primary"
                  onClick={
                    runPersonalPrediction
                  }
                  disabled={
                    predicting
                  }
                >
                  {predicting
                    ? 'Running Models...'
                    : 'Run Prediction'}
                </button>
              </div>
            </div>
          </section>

          {prediction && (
            <>
              <section className="screening-section">
                <div className="screening-panel">
                  <div className="screening-form-head">
                    <div>
                      <h2>
                        MODEL OUTPUTS
                      </h2>

                      <div
                        className="screening-muted"
                        style={{
                          fontSize: 12,
                          marginTop: 3,
                        }}
                      >
                        Independent predictions from the
                        trained classical and quantum models.
                      </div>
                    </div>

                    <span className="screening-success">
                      COMPLETE
                    </span>
                  </div>

                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns:
                        'repeat(auto-fit, minmax(220px, 1fr))',
                      gap: 12,
                      marginTop: 18,
                    }}
                  >
                    {Object.entries(
                      prediction.models || {},
                    ).map(([name, model]) => {
                      const quantum =
                        familyFor(name, model) ===
                        'quantum'

                      const probability =
                        model.class_1_probability

                      return (
                        <div
                          key={name}
                          style={{
                            border: quantum
                              ? '1px solid #b9c9ed'
                              : '1px solid #e2e5ea',
                            borderRadius: 9,
                            padding: 16,
                            background: quantum
                              ? '#f8faff'
                              : '#fff',
                          }}
                        >
                          <div
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: 10,
                            }}
                          >
                            <div
                              style={{
                                width: 34,
                                height: 34,
                                borderRadius: 7,
                                display: 'grid',
                                placeItems: 'center',
                                background: quantum
                                  ? '#eaf0ff'
                                  : '#f1f3f6',
                                color: '#2859d7',
                                fontSize: 10,
                                fontWeight: 800,
                              }}
                            >
                              {modelInitial(name)}
                            </div>

                            <div>
                              <div
                                style={{
                                  fontWeight: 700,
                                  fontSize: 13,
                                }}
                              >
                                {formatModelName(name)}
                              </div>

                              <div
                                className="screening-muted"
                                style={{
                                  marginTop: 2,
                                  fontSize: 10,
                                  letterSpacing: '0.06em',
                                }}
                              >
                                {modelAccent(name)}
                              </div>
                            </div>
                          </div>

                          <div
                            style={{
                              marginTop: 18,
                            }}
                          >
                            <div className="screening-field-label">
                              PREDICTION
                            </div>

                            <div
                              style={{
                                marginTop: 5,
                                fontSize: 20,
                                fontWeight: 700,
                              }}
                            >
                              {renderPredictionLabel(
                                model.prediction,
                              )}
                            </div>
                          </div>

                          {probability !== undefined &&
                            probability !== null && (
                              <div
                                style={{
                                  marginTop: 14,
                                }}
                              >
                                <div
                                  style={{
                                    display: 'flex',
                                    justifyContent:
                                      'space-between',
                                    fontSize: 10,
                                  }}
                                >
                                  <span className="screening-muted">
                                    MODEL PROBABILITY
                                  </span>

                                  <strong>
                                    {percentage(
                                      probability,
                                    )}
                                  </strong>
                                </div>

                                <div
                                  style={{
                                    height: 5,
                                    marginTop: 7,
                                    borderRadius: 999,
                                    background: '#e7ebf2',
                                    overflow: 'hidden',
                                  }}
                                >
                                  <div
                                    style={{
                                      width: `${Math.max(
                                        0,
                                        Math.min(
                                          100,
                                          Number(
                                            probability,
                                          ) <= 1
                                            ? Number(
                                                probability,
                                              ) * 100
                                            : Number(
                                                probability,
                                              ),
                                        ),
                                      )}%`,
                                      height: '100%',
                                      background: '#2859d7',
                                    }}
                                  />
                                </div>
                              </div>
                            )}
                        </div>
                      )
                    })}
                  </div>

                  <div
                    className="screening-notice"
                    style={{
                      marginTop: 16,
                    }}
                  >
                    Model outputs are predictions generated
                    by the trained research pipeline. They
                    should not be interpreted as a medical
                    diagnosis.
                  </div>
                </div>
              </section>

              {benchmark?.benchmark
                ?.models &&
                benchmark
                  .benchmark
                  .models
                  .length >
                  0 && (
                  <section className="screening-section">
                    <div className="screening-panel">
                      <div className="screening-form-head">
                        <div>
                          <h2>
                            PERFORMANCE BENCHMARK
                          </h2>

                          <div
                            className="screening-muted"
                            style={{
                              fontSize: 12,
                              marginTop: 3,
                            }}
                          >
                            Evaluation of the trained
                            models on the held-out
                            test data.
                          </div>
                        </div>
                      </div>

                      <div
                        className="screening-table-wrap"
                        style={{
                          marginTop: 16,
                        }}
                      >
                        <table className="screening-review-table">
                          <thead>
                            <tr>
                              <th>
                                Model
                              </th>

                              <th>
                                Family
                              </th>

                              <th>
                                Accuracy
                              </th>

                              <th>
                                Precision
                              </th>

                              <th>
                                Sensitivity
                              </th>

                              <th>
                                Specificity
                              </th>

                              <th>
                                F1
                              </th>

                              <th>
                                ROC-AUC
                              </th>
                            </tr>
                          </thead>

                          <tbody>
                            {benchmark.benchmark.models.map(
                              (
                                model,
                              ) => (
                                <tr
                                  key={
                                    model.model
                                  }
                                >
                                  <td>
                                    <strong>
                                      {formatModelName(
                                        model.model,
                                      )}
                                    </strong>
                                  </td>

                                  <td>
                                    <span className="screening-badge">
                                      {familyFor(
                                        model.model,
                                        model,
                                      ) ===
                                      'quantum'
                                        ? 'Quantum'
                                        : 'Classical'}
                                    </span>
                                  </td>

                                  <td>
                                    {percentage(
                                      model.accuracy,
                                    )}
                                  </td>

                                  <td>
                                    {percentage(
                                      model.precision,
                                    )}
                                  </td>

                                  <td>
                                    {percentage(
                                      model.sensitivity,
                                    )}
                                  </td>

                                  <td>
                                    {percentage(
                                      model.specificity,
                                    )}
                                  </td>

                                  <td>
                                    {percentage(
                                      model.f1_score,
                                    )}
                                  </td>

                                  <td>
                                    {percentage(
                                      model.roc_auc,
                                    )}
                                  </td>
                                </tr>
                              ),
                            )}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </section>
                )}

              <section className="screening-section">
                <div className="screening-panel">
                  <div className="screening-form-head">
                    <div>
                      <h2>
                        VITALIS AI
                      </h2>

                      <div
                        className="screening-muted"
                        style={{
                          fontSize: 12,
                          marginTop: 3,
                        }}
                      >
                        Ask about this research
                        session and its model
                        outputs.
                      </div>
                    </div>
                  </div>

                  <div
                    style={{
                      display: 'flex',
                      flexWrap:
                        'wrap',
                      gap: 8,
                      marginTop: 16,
                    }}
                  >
                    {[
                      'Why did the models predict this?',
                      'How did the classical and quantum models differ?',
                      'What influenced the prediction most?',
                    ].map(
                      (
                        question,
                      ) => (
                        <button
                          type="button"
                          className="screening-secondary"
                          key={
                            question
                          }
                          onClick={() =>
                            void askAssistant(
                              question,
                            )
                          }
                          disabled={
                            chatLoading
                          }
                        >
                          {
                            question
                          }
                        </button>
                      ),
                    )}
                  </div>

                  {chatMessages.length >
                    0 && (
                    <div
                      style={{
                        display:
                          'grid',
                        gap: 10,
                        marginTop: 18,
                      }}
                    >
                      {chatMessages.map(
                        (
                          message,
                        ) => (
                          <div
                            key={
                              message.id
                            }
                            style={{
                              padding: 12,
                              border:
                                '1px solid #e2e5ea',
                              borderRadius: 8,
                            }}
                          >
                            <div className="screening-field-label">
                              {message.role ===
                              'user'
                                ? 'YOU'
                                : 'VITALIS'}
                            </div>

                            <div
                              style={{
                                marginTop: 5,
                                lineHeight:
                                  1.6,
                                whiteSpace:
                                  'pre-wrap',
                              }}
                            >
                              {
                                message.content
                              }
                            </div>
                          </div>
                        ),
                      )}
                    </div>
                  )}

                  <div
                    style={{
                      display:
                        'flex',
                      gap: 8,
                      marginTop: 16,
                    }}
                  >
                    <input
                      className="screening-input"
                      style={{
                        flex: 1,
                      }}
                      placeholder="Ask about the prediction..."
                      value={
                        chatInput
                      }
                      onChange={(
                        event,
                      ) =>
                        setChatInput(
                          event
                            .target
                            .value,
                        )
                      }
                      onKeyDown={
                        handleChatKeyDown
                      }
                    />

                    <button
                      type="button"
                      className="screening-primary"
                      onClick={() =>
                        void askAssistant(
                          chatInput,
                        )
                      }
                      disabled={
                        chatLoading ||
                        !chatInput.trim()
                      }
                    >
                      {chatLoading
                        ? '...'
                        : 'Ask'}
                    </button>
                  </div>

                  {chatError && (
                    <div
                      className="screening-notice"
                      style={{
                        marginTop: 12,
                      }}
                    >
                      {chatError}
                    </div>
                  )}
                </div>
              </section>
            </>
          )}
        </>
      )
    }

  const downloadReferenceDataset = (
    dataset: 'heart' | 'breast_cancer',
  ) => {
    const endpoint =
      dataset === 'heart'
        ? '/datasets/heart'
        : '/datasets/breast-cancer'

    const link = document.createElement('a')
    link.href = `${API}${endpoint}`
    link.download =
      dataset === 'heart'
        ? 'heart_cleaned.csv'
        : 'breast_cancer.csv'
    document.body.appendChild(link)
    link.click()
    link.remove()
  }

  const renderPersonalPage =
    () => (
      <>
        <section className="screening-intro">
          <h1>
            YOUR DATASET
          </h1>

          <p>
            Train the VITALIS hybrid
            classical-quantum pipeline on your
            own labeled biomedical data, evaluate
            the models, and then run predictions
            using the trained session.
          </p>

          <div className="screening-notice">
            Your dataset stays in the research
            workflow: upload → understand → train
            → benchmark → predict.
          </div>

          <section
            className="screening-section"
            style={{
              marginTop: 18,
              marginBottom: 0,
            }}
          >
            <div className="screening-panel">
              <div className="screening-form-head">
                <div>
                  <h2>
                    VITALIS REFERENCE DATASETS
                  </h2>

                  <div
                    className="screening-muted"
                    style={{
                      fontSize: 12,
                      marginTop: 3,
                    }}
                  >
                    Download the datasets used by the
                    experimental modules and run them
                    through your own VITALIS workflow.
                  </div>
                </div>
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns:
                    'repeat(auto-fit, minmax(240px, 1fr))',
                  gap: 12,
                  marginTop: 16,
                }}
              >
                <div
                  style={{
                    border: '1px solid #e2e5ea',
                    borderRadius: 8,
                    padding: 16,
                    background: '#fff',
                  }}
                >
                  <strong>
                    Cardiovascular Disease
                  </strong>
                  <div
                    className="screening-muted"
                    style={{
                      marginTop: 5,
                      fontSize: 12,
                    }}
                  >
                    Heart disease reference dataset
                  </div>
                  <button
                    type="button"
                    className="screening-secondary"
                    style={{ marginTop: 12 }}
                    onClick={() =>
                      downloadReferenceDataset('heart')
                    }
                  >
                    ↓ Download Heart Disease CSV
                  </button>
                </div>

                <div
                  style={{
                    border: '1px solid #e2e5ea',
                    borderRadius: 8,
                    padding: 16,
                    background: '#fff',
                  }}
                >
                  <strong>
                    Breast Cancer
                  </strong>
                  <div
                    className="screening-muted"
                    style={{
                      marginTop: 5,
                      fontSize: 12,
                    }}
                  >
                    Breast-cancer reference dataset
                  </div>
                  <button
                    type="button"
                    className="screening-secondary"
                    style={{ marginTop: 12 }}
                    onClick={() =>
                      downloadReferenceDataset('breast_cancer')
                    }
                  >
                    ↓ Download Breast Cancer CSV
                  </button>
                </div>
              </div>
            </div>
          </section>

          {renderPersonalNavigation()}
        </section>

        {personalError && (
          <section className="screening-section">
            <div className="screening-notice">
              {personalError}
            </div>
          </section>
        )}

        {personalStage ===
          'upload' &&
          renderPersonalUpload()}

        {personalStage ===
          'explore' &&
          renderPersonalExplore()}

        {personalStage ===
          'train' &&
          renderPersonalTrain()}

        {personalStage ===
          'predict' &&
          renderPersonalPredict()}
      </>
    )

  return (
    <div className="screening-page">
      <header className="screening-header">
        <div className="screening-brand">
          VITALIS
        </div>

        <button
          type="button"
          className="screening-back"
          onClick={() =>
            navigate('/')
          }
        >
          ← Back to VITALIS
        </button>
      </header>

      <main className="screening-main">
        {renderTopNavigation()}

        {pageMode ===
          'experimental'
          ? renderExperimentalPage()
          : renderPersonalPage()}
      </main>

      <footer
        className="screening-muted"
        style={{
          textAlign: 'center',
          padding:
            '28px 20px 40px',
          fontSize: 11,
        }}
      >
        VITALIS · Hybrid Quantum-Classical
        Biomedical Research Platform
      </footer>
    </div>
  )
}