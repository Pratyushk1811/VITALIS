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
  ['LR', 'Logistic Regression', 'Interpretable linear baseline for establishing a classical reference point.'],
  ['RF', 'Random Forest', 'Nonlinear tree ensemble used to capture interactions in the feature space.'],
  ['RBF-SVM', 'RBF-SVM', 'Classical kernel benchmark using a radial-basis-function kernel.'],
]

const quantumModels = [
  ['QSVM', 'Quantum Support Vector Machine', 'SVM classification using a kernel estimated from a quantum feature map.'],
  ['VQC', 'Variational Quantum Classifier', 'Variational circuit model included in the cardiovascular screening pipeline.'],
]

const references = [
  ['01', 'Havlíček et al. (2019)', 'Quantum-enhanced feature spaces', 'Foundational work on quantum feature spaces and quantum-kernel learning.', 'https://www.nature.com/articles/s41567-019-0551-1'],
  ['02', 'Systematic review', 'Quantum machine learning for digital health', 'Reviews evidence and open questions surrounding QML applications in digital health.', 'https://www.nature.com/articles/s41746-025-01597-z'],
  ['03', 'Applied cardiovascular QML literature', 'Predictive analysis of heart disease using quantum-assisted ML', 'Context for evaluating quantum-assisted approaches on cardiovascular prediction tasks.', 'https://link.springer.com/article/10.1007/s42452-025-06944-z'],
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
  return <span className="research-arrow">↗</span>
}

export default function ResearchPage() {
  const navigate = useNavigate()

  return (
    <div className="research-page">
      <header className="research-nav">
        <button className="research-logo" onClick={() => navigate('/')}>
          VITALIS
        </button>

        <div className="research-nav-center">
          <span>RESEARCH / 2026</span>
        </div>

        <button className="research-screening-link" onClick={() => navigate('/screening')}>
          ENTER SCREENING <Arrow />
        </button>
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
              A research-facing view of the VITALIS experimental platform:
              classical and quantum machine-learning approaches evaluated under
              common conditions for disease-screening problems.
            </p>

            <div className="research-hero-actions">
              <button onClick={() => document.getElementById('research-question')?.scrollIntoView({ behavior: 'smooth' })}>
                EXPLORE THE METHOD <Arrow />
              </button>
              <button className="research-quiet-action" onClick={() => navigate('/screening')}>
                RUN AN EXPERIMENT
              </button>
            </div>
          </div>

          <div className="research-hero-visual" aria-hidden="true">
            <div className="research-visual-core">
              <span>VITALIS</span>
              <small>COMMON EVALUATION SPACE</small>
            </div>
            <div className="research-visual-ring ring-a" />
            <div className="research-visual-ring ring-b" />
            <div className="research-visual-line line-a" />
            <div className="research-visual-line line-b" />
            <div className="research-visual-node node-a">DATA</div>
            <div className="research-visual-node node-b">ML</div>
            <div className="research-visual-node node-c">QML</div>
            <div className="research-visual-node node-d">EVIDENCE</div>
          </div>
        </section>

        <section id="research-question" className="research-section research-question">
          <SectionLabel number="01">RESEARCH QUESTION</SectionLabel>

          <div className="research-question-grid">
            <div>
              <div className="research-index">01 / QUESTION</div>
              <h2>Can quantum approaches remain competitive when the comparison is genuinely controlled?</h2>
            </div>

            <div className="research-copy-card">
              <p>
                VITALIS is structured around comparison rather than a predetermined
                conclusion. Classical and quantum approaches receive the same
                underlying problem definition, common preprocessing and shared
                evaluation metrics.
              </p>
              <p>
                Model outputs, benchmark metrics, agreement and explainability are
                kept as separate layers so the computational behaviour remains visible.
              </p>
            </div>
          </div>
        </section>

        <section className="research-section research-method">
          <SectionLabel number="02">EXPERIMENTAL METHOD</SectionLabel>

          <div className="research-method-head">
            <div>
              <div className="research-index">PIPELINE / 05 STAGES</div>
              <h2>One prepared representation. Two computational paths.</h2>
            </div>
            <p>
              The pipeline makes the transformation from biomedical data to
              comparable model outputs explicit.
            </p>
          </div>

          <div className="research-pipeline">
            {[
              ['01', 'BIOMEDICAL DATA', 'Structured disease datasets'],
              ['02', 'PREPROCESSING', 'Cleaning · encoding · scaling'],
              ['03', 'FEATURE VECTOR', 'Validated model representation'],
              ['04', 'ML / QML', 'Parallel computational paths'],
              ['05', 'EVALUATION', 'Common metrics and comparison'],
            ].map(([n, title, text], i) => (
              <div className="research-pipeline-step" key={n}>
                <span>{n}</span>
                <strong>{title}</strong>
                <small>{text}</small>
                {i < 4 && <i />}
              </div>
            ))}
          </div>

          <div className="research-method-note">
            <span>FAIR COMPARISON</span>
            <p>Same dataset · same split · common preprocessing · common metrics.</p>
          </div>
        </section>

        <section className="research-section research-data">
          <SectionLabel number="03">CURRENT DATASET</SectionLabel>

          <div className="research-data-grid">
            <div className="research-data-main">
              <div className="research-eyebrow">CARDIOVASCULAR PROOF OF CONCEPT</div>
              <h2>UCI Heart Disease</h2>
              <p>
                The current cardiovascular proof of concept uses the public UCI
                heart-disease dataset and a compact feature representation for
                the experimental screening pipeline.
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

            <div className="research-stat-panel">
              <div><strong>303</strong><span>original samples</span></div>
              <div><strong>6</strong><span>selected features</span></div>
              <div><strong>80 / 20</strong><span>train / test split</span></div>
              <div><strong>42</strong><span>random state</span></div>
            </div>
          </div>
        </section>

        <section className="research-section research-models">
          <SectionLabel number="04">MODEL ARCHITECTURE</SectionLabel>

          <div className="research-model-intro">
            <div>
              <div className="research-index">CLASSICAL ↔ QUANTUM</div>
              <h2>Two model families. One evaluation framework.</h2>
            </div>
            <p>
              VITALIS keeps the computational pathways distinct while exposing
              their outputs through the same benchmark layer.
            </p>
          </div>

          <div className="research-model-columns">
            <div className="research-model-column">
              <div className="research-column-tag">CLASSICAL / 03</div>
              {classicalModels.map(([code, name, description]) => (
                <article className="research-model-card" key={code}>
                  <span>{code}</span>
                  <h3>{name}</h3>
                  <p>{description}</p>
                </article>
              ))}
            </div>

            <div className="research-model-column research-quantum-column">
              <div className="research-column-tag">QUANTUM / 02</div>
              {quantumModels.map(([code, name, description]) => (
                <article className="research-model-card" key={code}>
                  <span>{code}</span>
                  <h3>{name}</h3>
                  <p>{description}</p>
                </article>
              ))}

              <div className="research-circuit">
                <div className="research-circuit-title">CURRENT QUANTUM PATH</div>
                <div className="research-circuit-flow">
                  <span>6 QUBITS</span><i /><span>RY</span><i /><span>CNOT</span><i /><span>QSVM / VQC</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="research-section research-evaluation">
          <SectionLabel number="05">COMPARATIVE EVALUATION</SectionLabel>

          <div className="research-evaluation-head">
            <div>
              <div className="research-index">METRIC LAYER</div>
              <h2>The experiment is the evidence.</h2>
            </div>
            <p>
              VITALIS uses a shared metric vocabulary instead of relying on a
              single headline score.
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

          <div className="research-benchmark-table">
            <div className="research-benchmark-module">
              <div className="research-column-tag">CARDIOVASCULAR / n = 61</div>
              <div className="research-benchmark-row research-benchmark-head-row"><span>MODEL</span><span>ACC</span><span>PREC</span><span>SENS</span><span>SPEC</span><span>F1</span><span>AUC</span></div>
              {[
                ['Logistic Regression','85.00%','88.00%','78.57%','90.62%','83.02%','94.20%'],
                ['Random Forest','83.33%','84.62%','78.57%','87.50%','81.48%','94.03%'],
                ['RBF-SVM','86.67%','91.67%','78.57%','93.75%','84.62%','94.75%'],
                ['QSVM','83.33%','90.91%','71.43%','93.75%','80.00%','92.86%'],
                ['VQC','78.33%','85.71%','64.29%','90.62%','73.47%','75.95%'],
              ].map((row) => <div className="research-benchmark-row" key={row[0]}>{row.map((cell, i) => <span key={i}>{cell}</span>)}</div>)}
            </div>

            <div className="research-benchmark-module">
              <div className="research-column-tag">BREAST CANCER / n = 114</div>
              <div className="research-benchmark-row research-benchmark-head-row"><span>MODEL</span><span>ACC</span><span>PREC</span><span>SENS</span><span>SPEC</span><span>F1</span><span>AUC</span></div>
              {[
                ['Logistic Regression','96.49%','97.50%','92.86%','98.61%','95.12%','99.60%'],
                ['Random Forest','96.49%','100.00%','90.48%','100.00%','95.00%','99.42%'],
                ['RBF-SVM','97.37%','100.00%','92.86%','100.00%','96.30%','99.47%'],
                ['QSVM','97.37%','97.56%','95.24%','98.61%','96.39%','98.94%'],
                ['VQC','94.74%','97.37%','88.10%','98.61%','92.50%','99.11%'],
              ].map((row) => <div className="research-benchmark-row" key={row[0]}>{row.map((cell, i) => <span key={i}>{cell}</span>)}</div>)}
            </div>
          </div>

          <div className="research-benchmark-note">
            <span>NO PREDETERMINED WINNER</span>
            <p>These are the documented benchmark results for the two prototype disease modules. They describe the measured experiment and do not establish clinical generalization or quantum advantage.</p>
          </div>
        </section>

        <section className="research-section research-repro">
          <SectionLabel number="06">REPRODUCIBILITY</SectionLabel>

          <div className="research-repro-layout">
            <div>
              <div className="research-index">FROM INPUT → EVIDENCE</div>
              <h2>Every layer has a defined job.</h2>
            </div>

            <div className="research-repro-list">
              {[
                'Dataset-specific preprocessing is kept explicit.',
                'Model implementations are separated into disease pipelines.',
                'Benchmarking is performed through a common service layer.',
                'Prediction, explainability and language-model interaction remain separate layers.',
                'Future validation can introduce repeated cross-validation and independent datasets.',
              ].map((item, index) => (
                <div key={item}>
                  <span>0{index + 1}</span>
                  <p>{item}</p>
                </div>
              ))}
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
                <li>Two public disease-screening benchmarks</li>
                <li>Simulator-based quantum execution</li>
                <li>Modular FastAPI + React architecture</li>
                <li>Two disease-screening modules</li>
              </ul>
            </div>

            <div className="research-limit research-limit-next">
              <span>NEXT</span>
              <h3>What must be tested next</h3>
              <ul>
                <li>Repeated cross-validation</li>
                <li>Independent and higher-dimensional datasets</li>
                <li>Stronger generalization testing</li>
                <li>Hardware experimentation when access permits</li>
              </ul>
            </div>
          </div>

          <div className="research-claim">
            <strong>CURRENT CLAIM</strong>
            <p>
              VITALIS is a working research proof of concept. The current
              implementation should not be presented as evidence of clinical
              generalization or demonstrated quantum advantage.
            </p>
          </div>
        </section>

        <section className="research-section research-references">
          <SectionLabel number="08">RESEARCH / REFERENCES</SectionLabel>

          <div className="research-reference-head">
            <div>
              <div className="research-index">METHOD / CONTEXT</div>
              <h2>From established methods to an experimental platform.</h2>
            </div>
            <p>
              Methodological context for quantum feature spaces, quantum kernels
              and healthcare-oriented QML research.
            </p>
          </div>

          <div className="research-reference-grid">
            {references.map(([number, authors, title, description, href]) => (
              <a className="research-reference" href={href} target="_blank" rel="noreferrer" key={title}>
                <span>{number}</span>
                <div>
                  <small>{authors}</small>
                  <h3>{title}</h3>
                  <p>{description}</p>
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
            <h2>Does quantum <em>add value?</em></h2>
            <p>VITALIS is built to measure that question rather than answer it in advance.</p>

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
