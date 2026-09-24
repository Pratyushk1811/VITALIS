import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { useNavigate } from 'react-router-dom'
import './screening.css'

const API =
  import.meta.env.VITE_API_URL ||
  localStorage.getItem('VITALIS_API_BASE_URL') ||
  'http://127.0.0.1:8000'

type Disease = 'heart' | 'breast_cancer'
type DataSource = 'manual' | 'upload'
type PatientData = Record<string, number>

type ModelOutput = {
  prediction?: number
  class_0_probability?: number
  class_1_probability?: number
}

type ScreeningResult = {
  prediction?: number
  prediction_label?: string
  consensus?: {
    agreeing_models?: number
    total_models?: number
    percentage?: number
    status?: string
  }
  models?: Record<string, ModelOutput>
}

type BenchmarkModel = {
  model: string
  family?: string
  accuracy?: number
  precision?: number
  sensitivity?: number
  specificity?: number
  f1_score?: number
  roc_auc?: number
}

type BenchmarkData = {
  heart?: {
    models?: BenchmarkModel[]
  }
  breast_cancer?: {
    models?: BenchmarkModel[]
  }
}

type ChatMessage = {
  id: number
  role: 'user' | 'assistant'
  content: string
}

const featureNames = [
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

const cancerDefaults = [
  17.99,
  10.38,
  122.8,
  1001,
  0.118,
  0.277,
  0.3,
  0.147,
  0.242,
  0.07871,
  1.095,
  0.9053,
  8.589,
  153.4,
  0.006399,
  0.04904,
  0.05373,
  0.01587,
  0.03003,
  0.006193,
  25.38,
  17.33,
  184.6,
  2019,
  0.1622,
  0.6656,
  0.7119,
  0.2654,
  0.4601,
  0.1189,
]

const initialCancerData: PatientData = Object.fromEntries(
  featureNames.map((name, index) => [
    name,
    cancerDefaults[index],
  ]),
)

const modelOrder = [
  'logistic_regression',
  'random_forest',
  'rbf_svm',
  'qsvm',
  'vqc',
]

const modelNames: Record<string, string> = {
  logistic_regression: 'Logistic Regression',
  random_forest: 'Random Forest',
  rbf_svm: 'RBF SVM',
  qsvm: 'QSVM',
  vqc: 'VQC',
}

async function api(
  path: string,
  options: RequestInit = {},
) {
  const response = await fetch(`${API}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  })

  let data: any

  try {
    data = await response.json()
  } catch {
    data = {}
  }

  if (!response.ok) {
    throw new Error(
      data.detail || `Request failed (${response.status})`,
    )
  }

  return data
}

function predictionText(value?: number) {
  return value === 1
    ? 'Higher likelihood'
    : 'Lower likelihood'
}

function labelFor(field: string, value: number) {
  const maps: Record<
    string,
    Record<number, string>
  > = {
    cp: {
      1: 'Typical angina',
      2: 'Atypical angina',
      3: 'Non-anginal pain',
      4: 'Asymptomatic',
    },
    thal: {
      3: 'Normal',
      6: 'Fixed defect',
      7: 'Reversible defect',
    },
  }

  return maps[field]?.[value] ?? String(value)
}

export default function ScreeningPage() {
  const navigate = useNavigate()

  const resultsRef =
    useRef<HTMLElement | null>(null)

  const chatMessagesRef =
    useRef<HTMLDivElement | null>(null)

  const chatInputRef =
    useRef<HTMLTextAreaElement | null>(null)

  const messageIdRef = useRef(0)

  const [disease, setDisease] =
    useState<Disease>('heart')

  const [dataSource, setDataSource] =
    useState<DataSource>('manual')

  const [heartData, setHeartData] =
    useState<PatientData>({
      age: 54,
      cp: 2,
      thalach: 150,
      oldpeak: 1.2,
      ca: 0,
      thal: 3,
    })

  const [cancerData, setCancerData] =
    useState<PatientData>(initialCancerData)

  const [file, setFile] =
    useState<File | null>(null)

  const [running, setRunning] =
    useState(false)

  const [progressText, setProgressText] =
    useState('Running models...')

  const [formError, setFormError] =
    useState('')

  const [result, setResult] =
    useState<ScreeningResult | null>(null)

  const [benchmark, setBenchmark] =
    useState<BenchmarkData | null>(null)

  /* -----------------------------
     CHAT STATE
  ----------------------------- */

  const [chatMessages, setChatMessages] =
    useState<ChatMessage[]>([])

  const [assistantError, setAssistantError] =
    useState('')

  const [assistantLoading, setAssistantLoading] =
    useState(false)

  const [chatInput, setChatInput] =
    useState('')

  const patientData = useMemo(
    () =>
      disease === 'heart'
        ? heartData
        : cancerData,
    [disease, heartData, cancerData],
  )

  const validHeart = useMemo(() => {
    const d = heartData

    return (
      Number.isFinite(d.age) &&
      Number.isFinite(d.thalach) &&
      Number.isFinite(d.oldpeak) &&
      [1, 2, 3, 4].includes(d.cp) &&
      [3, 6, 7].includes(d.thal) &&
      [0, 1, 2, 3].includes(d.ca)
    )
  }, [heartData])

  /* -----------------------------
     CHAT AUTO SCROLL
  ----------------------------- */

  useEffect(() => {
    const container = chatMessagesRef.current

    if (!container) return

    container.scrollTo({
      top: container.scrollHeight,
      behavior: 'smooth',
    })
  }, [chatMessages, assistantLoading])

  /* -----------------------------
     INPUT HELPERS
  ----------------------------- */

  const updateHeart = (
    key: string,
    value: number,
  ) => {
    setHeartData((previous) => ({
      ...previous,
      [key]: value,
    }))
  }

  const updateCancer = (
    key: string,
    value: number,
  ) => {
    setCancerData((previous) => ({
      ...previous,
      [key]: value,
    }))
  }

  /* -----------------------------
     RESET
  ----------------------------- */

  const resetResults = () => {
    setResult(null)
    setBenchmark(null)
    setChatMessages([])
    setChatInput('')
    setAssistantError('')
    setAssistantLoading(false)
  }

  const changeDisease = (
    next: Disease,
  ) => {
    setDisease(next)
    setFormError('')
    resetResults()
  }

  /* -----------------------------
     REVIEW
  ----------------------------- */

  const renderReview = () => {
    if (disease === 'heart') {
      const d = heartData

      return (
        <table className="screening-review-table">
          <thead>
            <tr>
              <th>Age</th>
              <th>Chest Pain Type</th>
              <th>Max Heart Rate</th>
              <th>ST Depression</th>
              <th>Major Vessels</th>
              <th>Thalassemia</th>
            </tr>
          </thead>

          <tbody>
            <tr>
              <td>{d.age}</td>

              <td>
                {labelFor('cp', d.cp)} ({d.cp})
              </td>

              <td>
                {d.thalach} bpm
              </td>

              <td>
                {d.oldpeak}
              </td>

              <td>
                {d.ca}
              </td>

              <td>
                {labelFor('thal', d.thal)} ({d.thal})
              </td>
            </tr>
          </tbody>
        </table>
      )
    }

    return (
      <div
        className="screening-muted"
        style={{ fontSize: 12 }}
      >
        30 numerical features are ready
        for review and submission.
      </div>
    )
  }

  /* -----------------------------
     MODEL OUTPUTS
  ----------------------------- */

  const renderModels = () => {
    if (!result?.models) {
      return (
        <tr>
          <td
            colSpan={4}
            className="screening-muted"
          >
            No model outputs returned.
          </td>
        </tr>
      )
    }

    const rows = modelOrder
      .map((key, index) => {
        const model =
          result.models?.[key]

        if (!model) return null

        const p0 =
          Number(
            model.class_0_probability || 0,
          ) * 100

        const p1 =
          Number(
            model.class_1_probability || 0,
          ) * 100

        const isQuantum = index >= 3

        return (
          <tr
            key={key}
            className="model-output-row"
          >
            <td className="model-output-name">
              <strong>
                {modelNames[key]}
              </strong>
            </td>

            <td className="model-output-family">
              <span className="screening-badge">
                {isQuantum
                  ? 'Quantum'
                  : 'Classical'}
              </span>
            </td>

            <td className="model-output-prediction">
              {predictionText(
                model.prediction,
              )}
            </td>

            <td className="model-output-probabilities">
              <div className="probability-header">
                Class probabilities
              </div>

              <div className="probability-row">
                <span className="probability-label">
                  Class 0
                </span>

                <div className="probability-track">
                  <div
                    className="probability-fill class-zero"
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(100, p0),
                      )}%`,
                    }}
                  />
                </div>

                <span className="probability-value">
                  {p0.toFixed(2)}%
                </span>
              </div>

              <div className="probability-row">
                <span className="probability-label">
                  Class 1
                </span>

                <div className="probability-track">
                  <div
                    className="probability-fill class-one"
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(100, p1),
                      )}%`,
                    }}
                  />
                </div>

                <span className="probability-value">
                  {p1.toFixed(2)}%
                </span>
              </div>
            </td>
          </tr>
        )
      })
      .filter(Boolean)

    return rows.length ? (
      rows
    ) : (
      <tr>
        <td
          colSpan={4}
          className="screening-muted"
        >
          No model outputs returned.
        </td>
      </tr>
    )
  }

  /* -----------------------------
     BENCHMARK
  ----------------------------- */

  const renderBenchmark = () => {
    const data =
      disease === 'heart'
        ? benchmark?.heart
        : benchmark?.breast_cancer

    if (!data?.models?.length) {
      return (
        <tr>
          <td
            colSpan={8}
            className="screening-muted"
          >
            No benchmark data returned.
          </td>
        </tr>
      )
    }

    return data.models.map((model) => (
      <tr key={model.model}>
        <td>
          <strong>
            {model.model}
          </strong>
        </td>

        <td>
          <span className="screening-badge">
            {model.family === 'quantum'
              ? 'Quantum'
              : 'Classical'}
          </span>
        </td>

        <td className="screening-num">
          {Number(
            model.accuracy ?? 0,
          ).toFixed(2)}
          %
        </td>

        <td className="screening-num">
          {Number(
            model.precision ?? 0,
          ).toFixed(2)}
          %
        </td>

        <td className="screening-num">
          {Number(
            model.sensitivity ?? 0,
          ).toFixed(2)}
          %
        </td>

        <td className="screening-num">
          {Number(
            model.specificity ?? 0,
          ).toFixed(2)}
          %
        </td>

        <td className="screening-num">
          {Number(
            model.f1_score ?? 0,
          ).toFixed(2)}
          %
        </td>

        <td className="screening-num">
          {Number(
            model.roc_auc ?? 0,
          ).toFixed(2)}
          %
        </td>
      </tr>
    ))
  }

  /* -----------------------------
     SCREENING
  ----------------------------- */

  const runScreening = async () => {
    setFormError('')

    if (
      disease === 'heart' &&
      !validHeart
    ) {
      setFormError(
        'Please check the cardiovascular inputs.',
      )

      return
    }

    if (dataSource === 'upload') {
      setFormError(
        'Report extraction is not enabled yet. Use Manual Input for the working screening flow.',
      )

      return
    }

    resetResults()

    setRunning(true)

    setProgressText(
      'Running screening models...',
    )

    try {
      const endpoint =
        disease === 'heart'
          ? '/predict/heart'
          : '/predict/breast-cancer'

      const screeningResult =
        await api(endpoint, {
          method: 'POST',
          body: JSON.stringify(
            patientData,
          ),
        })

      setResult(screeningResult)

      setProgressText(
        'Screening complete.',
      )

      const benchmarkResult =
        await api('/benchmark').catch(
          () => null,
        )

      if (benchmarkResult) {
        setBenchmark(
          benchmarkResult,
        )
      }

      requestAnimationFrame(() => {
        resultsRef.current?.scrollIntoView(
          {
            behavior: 'smooth',
            block: 'start',
          },
        )
      })
    } catch (error) {
      setFormError(
        error instanceof Error
          ? error.message
          : 'Screening could not be completed.',
      )
    } finally {
      setRunning(false)
    }
  }

  /* -----------------------------
     CHAT
  ----------------------------- */

  const ask = async (
    question: string,
  ) => {
    const trimmed =
      question.trim()

    if (!trimmed) return

    if (!result) {
      setAssistantError(
        'Run a screening first.',
      )

      return
    }

    if (assistantLoading) return

    setAssistantError('')
    setChatInput('')

    const userMessage: ChatMessage = {
      id: ++messageIdRef.current,
      role: 'user',
      content: trimmed,
    }

    setChatMessages(
      (previous) => [
        ...previous,
        userMessage,
      ],
    )

    setAssistantLoading(true)

    try {
      const response = await api(
        '/chat',
        {
          method: 'POST',
          body: JSON.stringify({
            disease,
            patient_data: patientData,
            prediction_result:
              result,
            question: trimmed,
          }),
        },
      )

      const answer =
        response.explanation ||
        'No explanation returned.'

      const assistantMessage:
        ChatMessage = {
        id: ++messageIdRef.current,
        role: 'assistant',
        content: answer,
      }

      setChatMessages(
        (previous) => [
          ...previous,
          assistantMessage,
        ],
      )
    } catch (error) {
      setAssistantError(
        error instanceof Error
          ? error.message
          : 'Assistant unavailable.',
      )
    } finally {
      setAssistantLoading(false)

      requestAnimationFrame(() => {
        chatInputRef.current?.focus()
      })
    }
  }

  const handleChatKeyDown = (
    event: React.KeyboardEvent<HTMLTextAreaElement>,
  ) => {
    if (
      event.key === 'Enter' &&
      !event.shiftKey
    ) {
      event.preventDefault()

      const value =
        event.currentTarget.value.trim()

      if (value) {
        ask(value)
      }
    }
  }

  /* -----------------------------
     FILE
  ----------------------------- */

  const handleFile = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const selected =
      event.target.files?.[0] || null

    setFile(selected)
  }

  const removeFile = () => {
    setFile(null)
  }

  /* -----------------------------
     NEW SCREENING
  ----------------------------- */

  const runAnother = () => {
    resetResults()

    setFormError('')


    window.scrollTo({
      top: 0,
      behavior: 'smooth',
    })
  }

  const consensus =
    result?.consensus

  const consensusPercentage =
    Math.max(
      0,
      Math.min(
        100,
        Number(
          consensus?.percentage ?? 0,
        ),
      ),
    )

  /* -----------------------------
     RENDER
  ----------------------------- */

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
        {/* INTRO */}

        <section className="screening-intro">
          <h1>SCREENING</h1>

          <p>
            Run a model-based screening
            using patient data and compare
            the resulting model outputs.
          </p>

          <div className="screening-notice">
            VITALIS is a research and
            screening platform. Results are
            model outputs and are not a
            medical diagnosis.
          </div>
        </section>

        {/* CONTROLS */}

        <section className="screening-section">
          <div className="screening-controls">
            <div className="screening-field">
              <label
                className="screening-field-label"
                htmlFor="disease"
              >
                Disease
              </label>

              <select
                id="disease"
                className="screening-select"
                value={disease}
                onChange={(event) =>
                  changeDisease(
                    event.target
                      .value as Disease,
                  )
                }
              >
                <option value="heart">
                  Cardiovascular Disease
                </option>

                <option value="breast_cancer">
                  Breast Cancer
                </option>
              </select>
            </div>

            <div className="screening-field">
              <label className="screening-field-label">
                Data source
              </label>

              <div className="screening-tabs">
                <button
                  type="button"
                  className={`screening-tab ${
                    dataSource ===
                    'manual'
                      ? 'active'
                      : ''
                  }`}
                  onClick={() =>
                    setDataSource(
                      'manual',
                    )
                  }
                >
                  Manual Input
                </button>

                <button
                  type="button"
                  className={`screening-tab ${
                    dataSource ===
                    'upload'
                      ? 'active'
                      : ''
                  }`}
                  onClick={() =>
                    setDataSource(
                      'upload',
                    )
                  }
                >
                  Upload Report
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* MANUAL INPUT */}

        {dataSource === 'manual' && (
          <section className="screening-section">
            {disease === 'heart' ? (
              <div>
                <div className="screening-form-head">
                  <div>
                    <h2>
                      Cardiovascular inputs
                    </h2>

                    <div
                      className="screening-muted"
                      style={{
                        fontSize: 12,
                        marginTop: 3,
                      }}
                    >
                      Six model features
                    </div>
                  </div>
                </div>

                <div className="screening-form-grid">
                  <div className="screening-field">
                    <label className="screening-field-label">
                      Age{' '}
                      <span className="screening-key">
                        age
                      </span>
                    </label>

                    <input
                      className="screening-input"
                      type="number"
                      min="1"
                      value={
                        heartData.age
                      }
                      onChange={(event) =>
                        updateHeart(
                          'age',
                          Number(
                            event.target
                              .value,
                          ),
                        )
                      }
                    />

                    <small>
                      Age in years
                    </small>
                  </div>

                  <div className="screening-field">
                    <label className="screening-field-label">
                      Chest Pain Type{' '}
                      <span className="screening-key">
                        cp
                      </span>
                    </label>

                    <select
                      className="screening-select"
                      value={
                        heartData.cp
                      }
                      onChange={(event) =>
                        updateHeart(
                          'cp',
                          Number(
                            event.target
                              .value,
                          ),
                        )
                      }
                    >
                      <option value="1">
                        1 — Typical angina
                      </option>

                      <option value="2">
                        2 — Atypical angina
                      </option>

                      <option value="3">
                        3 — Non-anginal pain
                      </option>

                      <option value="4">
                        4 — Asymptomatic
                      </option>
                    </select>
                  </div>

                  <div className="screening-field">
                    <label className="screening-field-label">
                      Maximum Heart Rate
                      Achieved{' '}
                      <span className="screening-key">
                        thalach
                      </span>
                    </label>

                    <input
                      className="screening-input"
                      type="number"
                      value={
                        heartData.thalach
                      }
                      onChange={(event) =>
                        updateHeart(
                          'thalach',
                          Number(
                            event.target
                              .value,
                          ),
                        )
                      }
                    />

                    <small>
                      Peak heart rate
                      during testing
                    </small>
                  </div>

                  <div className="screening-field">
                    <label className="screening-field-label">
                      ST Depression
                      (Oldpeak){' '}
                      <span className="screening-key">
                        oldpeak
                      </span>
                    </label>

                    <input
                      className="screening-input"
                      type="number"
                      step="0.1"
                      value={
                        heartData.oldpeak
                      }
                      onChange={(event) =>
                        updateHeart(
                          'oldpeak',
                          Number(
                            event.target
                              .value,
                          ),
                        )
                      }
                    />

                    <small>
                      Exercise-induced
                      ST depression
                    </small>
                  </div>

                  <div className="screening-field">
                    <label className="screening-field-label">
                      Number of Major
                      Vessels{' '}
                      <span className="screening-key">
                        ca
                      </span>
                    </label>

                    <select
                      className="screening-select"
                      value={
                        heartData.ca
                      }
                      onChange={(event) =>
                        updateHeart(
                          'ca',
                          Number(
                            event.target
                              .value,
                          ),
                        )
                      }
                    >
                      <option value="0">
                        0
                      </option>

                      <option value="1">
                        1
                      </option>

                      <option value="2">
                        2
                      </option>

                      <option value="3">
                        3
                      </option>
                    </select>
                  </div>

                  <div className="screening-field">
                    <label className="screening-field-label">
                      Thalassemia{' '}
                      <span className="screening-key">
                        thal
                      </span>
                    </label>

                    <select
                      className="screening-select"
                      value={
                        heartData.thal
                      }
                      onChange={(event) =>
                        updateHeart(
                          'thal',
                          Number(
                            event.target
                              .value,
                          ),
                        )
                      }
                    >
                      <option value="3">
                        3 — Normal
                      </option>

                      <option value="6">
                        6 — Fixed defect
                      </option>

                      <option value="7">
                        7 — Reversible defect
                      </option>
                    </select>
                  </div>
                </div>
              </div>
            ) : (
              <div>
                <div className="screening-form-head">
                  <div>
                    <h2>
                      Breast cancer inputs
                    </h2>

                    <div
                      className="screening-muted"
                      style={{
                        fontSize: 12,
                        marginTop: 3,
                      }}
                    >
                      30 WDBC model features
                    </div>
                  </div>
                </div>

                <div className="screening-form-grid">
                  {featureNames.map(
                    (name, index) => (
                      <div
                        className="screening-field"
                        key={name}
                      >
                        <label className="screening-field-label">
                          {name}{' '}
                          <span className="screening-key">
                            {index + 1}
                          </span>
                        </label>

                        <input
                          className="screening-input"
                          type="number"
                          step="any"
                          value={
                            cancerData[
                              name
                            ]
                          }
                          onChange={(event) =>
                            updateCancer(
                              name,
                              Number(
                                event.target
                                  .value,
                              ),
                            )
                          }
                        />
                      </div>
                    ),
                  )}
                </div>
              </div>
            )}

            {/* REVIEW */}

            <div className="screening-review">
              <div className="screening-form-head">
                <h2>
                  Review input
                </h2>

                <span className="screening-success">
                  Valid input
                </span>
              </div>

              {renderReview()}

              <div className="screening-actions">
                <button
                  type="button"
                  className="screening-primary"
                  onClick={
                    runScreening
                  }
                  disabled={running}
                >
                  {running
                    ? 'Running Screening...'
                    : 'Run Screening'}
                </button>

                <button
                  type="button"
                  className="screening-secondary"
                  onClick={() =>
                    window.scrollTo(
                      {
                        top: 0,
                        behavior:
                          'smooth',
                      },
                    )
                  }
                >
                  Edit inputs
                </button>
              </div>
            </div>

            {running && (
              <div className="screening-progress">
                <div>
                  {progressText}
                </div>

                <div className="screening-progress-line">
                  <div />
                </div>
              </div>
            )}

            {formError && (
              <div className="screening-error">
                {formError}
              </div>
            )}
          </section>
        )}

        {/* UPLOAD */}

        {dataSource === 'upload' && (
          <section className="screening-section">
            <div
              className="screening-upload"
              onClick={() => {
                document
                  .getElementById(
                    'screening-file-input',
                  )
                  ?.click()
              }}
            >
              <strong>
                Upload medical report
              </strong>

              <span>
                Drop a PDF or image here,
                or{' '}
                <label
                  htmlFor="screening-file-input"
                  style={{
                    color:
                      'var(--screen-blue)',
                    cursor:
                      'pointer',
                    textDecoration:
                      'underline',
                  }}
                  onClick={(event) =>
                    event.stopPropagation()
                  }
                >
                  choose a file
                </label>
                .
              </span>

              <input
                id="screening-file-input"
                type="file"
                accept=".pdf,.jpg,.jpeg,.png"
                onChange={
                  handleFile
                }
              />
            </div>

            {file && (
              <div className="screening-file-row">
                <div>
                  <strong>
                    {file.name}
                  </strong>

                  <div className="screening-muted">
                    {file.type ||
                      'File'}{' '}
                    ·{' '}
                    {(
                      file.size /
                      1024 /
                      1024
                    ).toFixed(2)}{' '}
                    MB
                  </div>
                </div>

                <button
                  type="button"
                  onClick={
                    removeFile
                  }
                >
                  Remove
                </button>
              </div>
            )}

            <div className="screening-future">
              Automatic document
              extraction is not enabled
              yet. A report must eventually
              be converted into structured
              features and reviewed before
              being sent to the screening
              models.
            </div>

            {formError && (
              <div className="screening-error">
                {formError}
              </div>
            )}
          </section>
        )}

        {/* RESULTS */}

        {result && (
          <section
            className="screening-results"
            ref={resultsRef}
          >
            <div className="screening-results-layout">
              <div className="screening-results-main">
                <div className="screening-result-top">
                  <div className="screening-result-box">
                    <h3>SCREENING RESULT</h3>
                    <div className="screening-result-label">
                      {(result.prediction_label ||
                        predictionText(result.prediction)).toUpperCase()}
                    </div>
                    <div
                      className="screening-muted"
                      style={{ fontSize: 12, marginTop: 5 }}
                    >
                      Model-based screening prediction. Not a medical
                      diagnosis.
                    </div>
                  </div>

                  <div className="screening-consensus">
                    <h3>MODEL CONSENSUS</h3>
                    <div className="screening-consensus-number">
                      {consensus?.agreeing_models ?? 0} /{" "}
                      {consensus?.total_models ?? 0}
                    </div>
                    <div className="screening-muted" style={{ fontSize: 12 }}>
                      {`models agree · ${Number(
                        consensus?.percentage ?? 0,
                      ).toFixed(1)}% · ${consensus?.status ?? ""}`}
                    </div>
                    <div className="screening-bar">
                      <div style={{ width: `${consensusPercentage}%` }} />
                    </div>
                  </div>
                </div>

                <div className="screening-panel">
                  <div className="screening-form-head">
                    <div>
                      <h2>MODEL OUTPUTS</h2>
                      <div className="screening-muted" style={{ fontSize: 12 }}>
                        Individual predictions and class probabilities returned
                        by the backend.
                      </div>
                    </div>
                  </div>
                  <div className="screening-table-wrap">
                    <table className="screening-model-table model-output-table">
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th>Family</th>
                          <th>Prediction</th>
                          <th>Probability distribution</th>
                        </tr>
                      </thead>
                      <tbody>{renderModels()}</tbody>
                    </table>
                  </div>
                </div>

                <div className="screening-panel">
                  <div className="screening-form-head">
                    <div>
                      <h2>BENCHMARK</h2>
                      <div className="screening-muted" style={{ fontSize: 12 }}>
                        Evaluation metrics for the implemented models.
                      </div>
                    </div>
                  </div>
                  <div className="screening-table-wrap">
                    <table className="screening-bench-table">
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th>Class</th>
                          <th>Accuracy</th>
                          <th>Precision</th>
                          <th>Sensitivity</th>
                          <th>Specificity</th>
                          <th>F1</th>
                          <th>AUC</th>
                        </tr>
                      </thead>
                      <tbody>{renderBenchmark()}</tbody>
                    </table>
                  </div>
                </div>

                <div className="screening-actions">
                  <button
                    type="button"
                    className="screening-secondary"
                    onClick={runAnother}
                  >
                    Run Another Screening
                  </button>
                </div>
              </div>

              <aside
                className="vitalis-chat-card"
                aria-label="VITALIS AI assistant"
              >
                <div className="vitalis-chat-card-header">
                  <div>
                    <div className="vitalis-chat-title">VITALIS AI</div>
                    <div className="vitalis-chat-status">
                      <span className="vitalis-chat-status-dot" />
                      Screening context ready
                    </div>
                  </div>
                </div>

                <div
                  className="vitalis-chat-messages"
                  ref={chatMessagesRef}
                >
                  {chatMessages.length === 0 && (
                    <div className="vitalis-chat-welcome">
                      <div className="vitalis-chat-welcome-mark">AI</div>
                      <div className="vitalis-chat-welcome-title">
                        Ask about this screening
                      </div>
                      <div className="vitalis-chat-welcome-text">
                        Ask about the model outputs, the implemented
                        approaches, or factors reflected in this result.
                      </div>
                      <div className="vitalis-chat-prompts">
                        {[
                          "Why did the models predict this?",
                          "How did the classical and quantum models differ?",
                          "What influenced the prediction most?",
                        ].map((question) => (
                          <button
                            type="button"
                            className="vitalis-chat-prompt"
                            key={question}
                            onClick={() => ask(question)}
                            disabled={assistantLoading}
                          >
                            {question}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {chatMessages.map((message) => (
                    <div
                      key={message.id}
                      className={`vitalis-chat-message ${
                        message.role === "user" ? "user" : "assistant"
                      }`}
                    >
                      <div className="vitalis-chat-message-label">
                        {message.role === "user" ? "YOU" : "VITALIS AI"}
                      </div>
                      <div className="vitalis-chat-bubble">
                        {message.content}
                      </div>
                    </div>
                  ))}

                  {assistantLoading && (
                    <div className="vitalis-chat-message assistant">
                      <div className="vitalis-chat-message-label">
                        VITALIS AI
                      </div>
                      <div className="vitalis-chat-bubble vitalis-chat-typing">
                        <span />
                        <span />
                        <span />
                      </div>
                    </div>
                  )}

                  {assistantError && (
                    <div className="vitalis-chat-error">
                      {assistantError}
                    </div>
                  )}
                </div>

                <div className="vitalis-chat-input-area">
                  <textarea
                    ref={chatInputRef}
                    value={chatInput}
                    onChange={(event) => setChatInput(event.target.value)}
                    onKeyDown={handleChatKeyDown}
                    placeholder="Ask about this result..."
                    rows={1}
                    disabled={assistantLoading}
                  />
                  <button
                    type="button"
                    className="vitalis-chat-send"
                    onClick={() => ask(chatInput)}
                    disabled={assistantLoading || !chatInput.trim()}
                    aria-label="Send message"
                  >
                    ↑
                  </button>
                </div>

                <div className="vitalis-chat-hint">
                  Enter to send · Shift + Enter for a new line
                </div>
              </aside>
            </div>
          </section>
        )}

      <footer className="screening-footer">
        <span>
          VITALIS · research screening
          interface
        </span>

        <span>
          Classical and quantum models
          are presented without ranking.
        </span>
      </footer>
      </main>
    </div>
  )
}