import { useNavigate } from 'react-router-dom'
import './research.css'

type Metric = {
  name: string
  description: string
}

const metrics: Metric[] = [
  { name: 'Accuracy', description: 'Overall proportion of correctly classified samples.' },
  { name: 'Precision', description: 'How often positive predictions correspond to positive samples.' },
  { name: 'Sensitivity / Recall', description: 'Ability to identify positive samples.' },
  { name: 'Specificity', description: 'Ability to identify negative samples.' },
  { name: 'F1 Score', description: 'Harmonic mean of precision and recall.' },
  { name: 'ROC-AUC', description: 'Threshold-independent discrimination measure.' },
]

const classicalModels = [
  {
    code: 'LR',
    name: 'Logistic Regression',
    text: 'Interpretable linear baseline for establishing a classical reference point.',
  },
  {
    code: 'RF',
    name: 'Random Forest',
    text: 'Nonlinear tree-based ensemble used to capture interactions in the feature space.',
  },
  {
    code: 'RBF-SVM',
    name: 'RBF-SVM',
    text: 'Classical kernel benchmark using a radial-basis-function kernel.',
  },
]

const quantumModels = [
  {
    code: 'QSVM',
    name: 'Quantum Support Vector Machine',
    text: 'Classical SVM classification using a kernel estimated from a quantum feature map.',
  },
  {
    code: 'VQC',
    name: 'Variational Quantum Classifier',
    text: 'Variational circuit model included in the current cardiovascular screening pipeline.',
  },
]

const references = [
  {
    title: 'Quantum-enhanced feature spaces',
    authors: 'Havlíček et al. (2019)',
    text: 'Foundational work on quantum feature spaces and quantum-kernel learning.',
    href: 'https://www.nature.com/articles/s41567-019-0551-1',
  },
  {
    title: 'Quantum machine learning for digital health',
    authors: 'Systematic review',
    text: 'Reviews the evidence and open questions surrounding QML applications in digital health.',
    href: 'https://www.nature.com/articles/s41746-025-01597-z',
  },
  {
    title: 'Predictive analysis of heart disease using quantum-assisted ML',
    authors: 'Applied cardiovascular QML literature',
    text: 'Provides context for evaluating quantum-assisted approaches on cardiovascular prediction tasks.',
    href: 'https://link.springer.com/article/10.1007/s42452-025-06944-z',
  },
]

function SectionLabel({ number, children }: { number: string; children: React.ReactNode }) {
  return (
    <div className="research-section-label">
      <span>{number}</span>
      <span>{children}</span>
    </div>
  )
}

function Arrow() {
  return <span className="research-arrow">→</span>
}

export default function ResearchPage() {
  const navigate = useNavigate()

  return (
    <div className="research-page">
      <header className="research-nav">
        <button className="research-logo" onClick={() => navigate('/')}>
          VITALIS
        </button>

        <div className="research-nav-right">
          <span className="research-nav-current">RESEARCH</span>
          <button className="research-screening-link" onClick={() => navigate('/screening')}>
            SCREENING <Arrow />
          </button>
        </div>
      </header>

      <main>
        <section className="research-hero">
          <div className="research-hero-grid" />
          <div className="research-hero-orbit research-orbit-one" />
          <div className="research-hero-orbit research-orbit-two" />

          <div className="research-hero-copy">
            <div className="research-kicker">VITALIS / RESEARCH</div>

            <h1>
              Where does
              <br />
              <em>intelligence</em>
              <br />
              become useful?
            </h1>

            <p>
              VITALIS investigates classical and quantum machine-learning approaches
              under common experimental conditions for disease-screening problems.
            </p>

            <div className="research-hero-meta">
              <span>01</span>
              <span>EXPERIMENTAL PLATFORM</span>
              <span>CLASSICAL ↔ QUANTUM</span>
            </div>
          </div>

          <div className="research-hero-side">
            <div className="research-vertical-text">COMPUTATIONAL SCREENING / 2026</div>
          </div>
        </section>

        <section className="research-section research-question">
          <SectionLabel number="01">RESEARCH QUESTION</SectionLabel>

          <div className="research-two-column">
            <div>
              <h2>
                Can a quantum-kernel approach provide competitive performance compared
                with established classical models under identical conditions?
              </h2>
            </div>

            <div className="research-copy">
              <p>
                The platform is structured around comparison rather than a predetermined
                conclusion. Classical and quantum approaches receive the same underlying
                problem definition, common preprocessing and shared evaluation metrics.
              </p>

              <p>
                The purpose is to make the computational differences visible and testable:
                model outputs, benchmark metrics, agreement and explainability are treated
                as separate layers.
              </p>
            </div>
          </div>
        </section>

        <section className="research-section research-method">
          <SectionLabel number="02">EXPERIMENTAL METHOD</SectionLabel>

          <div className="research-pipeline">
            <div className="research-pipeline-node">
              <span>01</span>
              <strong>BIOMEDICAL DATA</strong>
              <small>Structured disease datasets</small>
            </div>
            <div className="research-pipeline-line" />
            <div className="research-pipeline-node">
              <span>02</span>
              <strong>PREPROCESSING</strong>
              <small>Cleaning · encoding · scaling</small>
            </div>
            <div className="research-pipeline-line" />
            <div className="research-pipeline-node">
              <span>03</span>
              <strong>FEATURE VECTOR</strong>
              <small>Validated model representation</small>
            </div>
            <div className="research-pipeline-line" />
            <div className="research-pipeline-node">
              <span>04</span>
              <strong>ML / QML</strong>
              <small>Parallel computational paths</small>
            </div>
            <div className="research-pipeline-line" />
            <div className="research-pipeline-node">
              <span>05</span>
              <strong>EVALUATION</strong>
              <small>Common metrics and comparison</small>
            </div>
          </div>

          <div className="research-method-note">
            <span>FAIR COMPARISON</span>
            <p>Same dataset · same stratified split · common preprocessing · common metrics.</p>
          </div>
        </section>

        <section className="research-section research-data">
          <SectionLabel number="03">CURRENT DATASET</SectionLabel>

          <div className="research-data-grid">
            <div className="research-data-main">
              <div className="research-eyebrow">CARDIOVASCULAR PROOF OF CONCEPT</div>
              <h2>UCI Heart Disease</h2>
              <p>
                The current cardiovascular proof of concept uses a public UCI heart-disease
                dataset with 303 samples and a six-feature representation.
              </p>

              <div className="research-feature-list">
                {['cp', 'thal', 'thalach', 'oldpeak', 'ca', 'age'].map((feature, index) => (
                  <div key={feature} className="research-feature">
                    <span>0{index + 1}</span>
                    <strong>{feature}</strong>
                  </div>
                ))}
              </div>
            </div>

            <div className="research-stat-stack">
              <div>
                <strong>303</strong>
                <span>samples</span>
              </div>
              <div>
                <strong>6</strong>
                <span>selected features</span>
              </div>
              <div>
                <strong>80 / 20</strong>
                <span>stratified train / test split</span>
              </div>
              <div>
                <strong>42</strong>
                <span>random state</span>
              </div>
            </div>
          </div>
        </section>

        <section className="research-section research-models">
          <SectionLabel number="04">MODEL ARCHITECTURE</SectionLabel>

          <div className="research-model-intro">
            <h2>Two computational paths.<br />One evaluation framework.</h2>
            <p>
              VITALIS keeps the classical and quantum pathways separate while evaluating
              their outputs within the same experimental framework.
            </p>
          </div>

          <div className="research-model-columns">
            <div className="research-model-column">
              <div className="research-column-tag">CLASSICAL / 03</div>

              {classicalModels.map((model) => (
                <article className="research-model-card" key={model.code}>
                  <span>{model.code}</span>
                  <h3>{model.name}</h3>
                  <p>{model.text}</p>
                </article>
              ))}
            </div>

            <div className="research-model-column research-quantum-column">
              <div className="research-column-tag">QUANTUM / 02</div>

              {quantumModels.map((model) => (
                <article className="research-model-card" key={model.code}>
                  <span>{model.code}</span>
                  <h3>{model.name}</h3>
                  <p>{model.text}</p>
                </article>
              ))}

              <div className="research-circuit">
                <div className="research-circuit-title">CURRENT QUANTUM PATH</div>
                <div className="research-circuit-flow">
                  <span>6 QUBITS</span>
                  <i />
                  <span>RY</span>
                  <i />
                  <span>CNOT</span>
                  <i />
                  <span>KERNEL</span>
                  <i />
                  <span>QSVM</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="research-section research-evaluation">
          <SectionLabel number="05">COMPARATIVE EVALUATION</SectionLabel>

          <div className="research-evaluation-head">
            <h2>The experiment is the evidence.</h2>
            <p>
              VITALIS evaluates the models using a shared metric vocabulary rather than
              relying on a single headline score.
            </p>
          </div>

          <div className="research-metrics">
            {metrics.map((metric, index) => (
              <article key={metric.name}>
                <span>0{index + 1}</span>
                <h3>{metric.name}</h3>
                <p>{metric.description}</p>
              </article>
            ))}
          </div>

          <div className="research-benchmark-note">
            <span>NO PREDETERMINED WINNER</span>
            <p>
              QSVM is tested against classical baselines; the platform does not assume
              that the quantum approach is superior.
            </p>
          </div>
        </section>

        <section className="research-section research-repro">
          <SectionLabel number="06">REPRODUCIBILITY</SectionLabel>

          <div className="research-repro-layout">
            <div>
              <h2>
                From input
                <br />
                to evidence.
              </h2>
            </div>

            <div className="research-repro-list">
              <div><span>01</span><p>Dataset-specific preprocessing is kept explicit.</p></div>
              <div><span>02</span><p>Model implementations are separated into disease pipelines.</p></div>
              <div><span>03</span><p>Benchmarking is performed through a common service layer.</p></div>
              <div><span>04</span><p>Prediction, explainability and language-model interaction remain separate layers.</p></div>
              <div><span>05</span><p>Future validation can introduce repeated cross-validation and independent datasets.</p></div>
            </div>
          </div>
        </section>

        <section className="research-section research-limitations">
          <SectionLabel number="07">LIMITATIONS / NEXT VALIDATION</SectionLabel>

          <div className="research-limit-grid">
            <div className="research-limit">
              <span>CURRENT</span>
              <h3>What exists today</h3>
              <ul>
                <li>Public 303-sample cardiovascular benchmark</li>
                <li>Simulator-based quantum execution</li>
                <li>Modular FastAPI + React architecture</li>
                <li>Two disease-screening modules in the current platform</li>
              </ul>
            </div>

            <div className="research-limit research-limit-next">
              <span>NEXT</span>
              <h3>What must be tested next</h3>
              <ul>
                <li>Repeated cross-validation</li>
                <li>Independent and higher-dimensional biomedical datasets</li>
                <li>Stronger generalization testing</li>
                <li>Hardware experimentation when access permits</li>
              </ul>
            </div>
          </div>

          <div className="research-claim">
            <strong>CURRENT CLAIM</strong>
            <p>
              VITALIS is a working research proof of concept. The current implementation
              should not be presented as evidence of clinical generalization or demonstrated
              quantum advantage.
            </p>
          </div>
        </section>

        <section className="research-section research-references">
          <SectionLabel number="08">RESEARCH / REFERENCES</SectionLabel>

          <div className="research-reference-head">
            <h2>From established methods<br />to an experimental platform.</h2>
            <p>
              The references below provide methodological context for quantum feature
              spaces, quantum kernels and healthcare-oriented QML research.
            </p>
          </div>

          <div className="research-reference-grid">
            {references.map((reference, index) => (
              <a
                className="research-reference"
                href={reference.href}
                target="_blank"
                rel="noreferrer"
                key={reference.title}
              >
                <span>0{index + 1}</span>
                <div>
                  <small>{reference.authors}</small>
                  <h3>{reference.title}</h3>
                  <p>{reference.text}</p>
                </div>
                <Arrow />
              </a>
            ))}
          </div>
        </section>

        <section className="research-final">
          <div className="research-final-grid" />
          <div className="research-final-copy">
            <div className="research-kicker">THE OPEN QUESTION</div>
            <h2>
              Does quantum
              <br />
              <em>add value?</em>
            </h2>
            <p>
              VITALIS is built to measure that question rather than answer it in advance.
            </p>

            <div className="research-final-actions">
              <button onClick={() => navigate('/screening')}>
                ENTER SCREENING <Arrow />
              </button>
              <button className="research-secondary-action" onClick={() => navigate('/')}>
                BACK TO VITALIS
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
