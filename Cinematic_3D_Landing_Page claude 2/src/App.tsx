import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

// ── Design tokens ─────────────────────────────────────────────────────────────
const SERIF: React.CSSProperties = { fontFamily: "'Cormorant Garamond', Georgia, serif" };
const MONO: React.CSSProperties = { fontFamily: "'DM Mono', monospace", fontWeight: 300 };
const OFF_WHITE = '#D6D2C9';
const MINT = '#6BB8AC';
const VIOLET = '#9278C4';

const IMG = {
  data: 'https://images.unsplash.com/photo-1785920223184-4933ba5a4b81?w=1920&q=88&fit=crop&auto=format',
  human: 'https://images.unsplash.com/photo-1763198302243-51142ba5b24a?w=1920&q=88&fit=crop&auto=format',
  cardio: 'https://images.unsplash.com/photo-1743767587923-5857b19e60dd?w=1920&q=88&fit=crop&auto=format',
  cancer: 'https://images.unsplash.com/photo-1576086213369-97a306d36557?w=1920&q=88&fit=crop&auto=format',
  compute: 'https://images.unsplash.com/photo-1780729996054-80dca2de1354?w=1920&q=88&fit=crop&auto=format',
  converge: 'https://images.unsplash.com/photo-1668681919287-7367677cdc4c?w=1920&q=88&fit=crop&auto=format',
  explain: 'https://images.unsplash.com/photo-1782915892765-a7a3f11441ba?w=1920&q=88&fit=crop&auto=format',
  future: 'https://images.unsplash.com/photo-1673272454443-9787c5f375b8?w=1920&q=88&fit=crop&auto=format',
};

// ── Animation helper — returns inline transition styles ───────────────────────
function fadeUp(v: boolean, delay = 0): React.CSSProperties {
  return {
    opacity: v ? 1 : 0,
    transform: v ? 'translateY(0px)' : 'translateY(30px)',
    transition: `opacity 0.85s cubic-bezier(0.22,1,0.36,1) ${delay}s, transform 0.85s cubic-bezier(0.22,1,0.36,1) ${delay}s`,
  };
}

function fadeIn(v: boolean, delay = 0): React.CSSProperties {
  return {
    opacity: v ? 1 : 0,
    transition: `opacity 0.80s ease ${delay}s`,
  };
}

function slideRight(v: boolean, delay = 0): React.CSSProperties {
  return {
    opacity: v ? 1 : 0,
    transform: v ? 'translateX(0px)' : 'translateX(-20px)',
    transition: `opacity 0.75s ease ${delay}s, transform 0.75s cubic-bezier(0.22,1,0.36,1) ${delay}s`,
  };
}

function imgReveal(v: boolean): React.CSSProperties {
  return {
    opacity: v ? 1 : 0,
    transform: v ? 'scale(1.02)' : 'scale(1.08)',
    transition: 'opacity 1.4s cubic-bezier(0.22,1,0.36,1), transform 1.6s cubic-bezier(0.22,1,0.36,1)',
  };
}

// ── IntersectionObserver hook ─────────────────────────────────────────────────
function useInView(threshold = 0.12) {
  const ref = useRef<HTMLElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) {
        setVisible(true);
        obs.disconnect();
      }
    }, { threshold });

    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);

  return [ref, visible] as const;
}

// ── Tiny helpers ──────────────────────────────────────────────────────────────
function Label({ children, op = 0.40 }: { children: React.ReactNode; op?: number }) {
  return (
    <span
      style={{
        ...MONO,
        fontSize: '0.78rem',
        letterSpacing: '0.18em',
        opacity: op,
        textTransform: 'uppercase' as const,
      }}
    >
      {children}
    </span>
  );
}

function SectionNum({
  title,
  v,
  delay = 0,
}: {
  n?: string;
  title: string;
  v: boolean;
  delay?: number;
}) {
  return (
    <div style={{ ...slideRight(v, delay) }}>
      <Label op={0.30}>{title}</Label>
    </div>
  );
}

// ── Section wrapper with bg image ─────────────────────────────────────────────
function Sec({
  children,
  imgSrc,
  imgPos = 'center',
  overlay,
  id,
}: {
  children: (v: boolean, ref: React.RefObject<HTMLElement>) => React.ReactNode;
  imgSrc?: string;
  imgPos?: string;
  overlay?: string;
}) {
  const [ref, v] = useInView();

  return (
    <section
      id={id}
      ref={ref as React.RefObject<HTMLElement>}
      style={{
        position: 'relative',
        height: '100vh',
        minHeight: 580,
        overflow: 'hidden',
      }}
    >
      {imgSrc && (
        <img
          src={imgSrc}
          alt=""
          aria-hidden
          style={{
            position: 'absolute',
            inset: '-6% 0',
            width: '100%',
            height: '112%',
            objectFit: 'cover',
            objectPosition: imgPos,
            ...imgReveal(v),
          }}
        />
      )}

      <div
        style={{
          position: 'absolute',
          inset: 0,
          background:
            overlay ??
            'linear-gradient(160deg,rgba(8,11,14,0.90) 0%,rgba(8,11,14,0.60) 50%,rgba(8,11,14,0.82) 100%)',
        }}
      />

      <div style={{ position: 'relative', zIndex: 1, height: '100%' }}>
        {children(v, ref as React.RefObject<HTMLElement>)}
      </div>
    </section>
  );
}

// ── Benchmark figure ──────────────────────────────────────────────────────────
function BenchFigure({ v }: { v: boolean }) {
  const models = [
    ['LR', 0.8500, MINT],
    ['RF', 0.8333, MINT],
    ['RBF-SVM', 0.8667, MINT],
    ['QSVM', 0.8333, VIOLET],
    ['VQC', 0.7833, VIOLET],
  ] as const;

  const W = 560;
  const H = 270;
  const PL = 54;
  const PR = 22;
  const PT = 28;
  const PB = 58;
  const plotW = W - PL - PR;
  const plotH = H - PT - PB;
  const yMin = 0.78;
  const yMax = 0.90;
  const gap = 16;
  const barW = (plotW - gap * (models.length - 1)) / models.length;
  const y = (value: number) => PT + (1 - (value - yMin) / (yMax - yMin)) * plotH;

  return (
    <div style={{ width: '100%', maxWidth: W }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }} role="img" aria-label="Current cardiovascular experiment accuracy comparison">
        {[0.80, 0.82, 0.84, 0.86, 0.88, 0.90].map((tick) => (
          <g key={tick}>
            <line
              x1={PL}
              x2={W - PR}
              y1={y(tick)}
              y2={y(tick)}
              stroke="rgba(214,210,201,0.07)"
              strokeWidth="0.7"
            />
            <text
              x={PL - 8}
              y={y(tick) + 4}
              textAnchor="end"
              style={{ ...MONO, fontSize: '8px', fill: 'rgba(214,210,201,0.30)' }}
            >
              {tick.toFixed(2)}
            </text>
          </g>
        ))}

        {models.map(([name, value, color], i) => {
          const x = PL + i * (barW + gap);
          const top = y(value);
          return (
            <g key={name}>
              <rect
                x={x}
                y={v ? top : PT + plotH}
                width={barW}
                height={v ? PT + plotH - top : 0}
                fill={color}
                opacity={0.22}
                style={{ transition: `all 0.9s cubic-bezier(0.22,1,0.36,1) ${0.18 + i * 0.08}s` }}
              />
              <rect
                x={x}
                y={v ? top : PT + plotH}
                width={barW}
                height={v ? PT + plotH - top : 0}
                fill="none"
                stroke={color}
                strokeWidth="0.8"
                opacity={0.60}
                style={{ transition: `all 0.9s cubic-bezier(0.22,1,0.36,1) ${0.18 + i * 0.08}s` }}
              />
              <text
                x={x + barW / 2}
                y={v ? top - 9 : PT + plotH - 9}
                textAnchor="middle"
                style={{ ...MONO, fontSize: '8px', fill: color, opacity: 0.72 }}
              >
                {(value * 100).toFixed(1)}%
              </text>
              <text
                x={x + barW / 2}
                y={H - PB + 18}
                textAnchor="middle"
                style={{ ...MONO, fontSize: '8px', fill: 'rgba(214,210,201,0.40)', letterSpacing: '0.06em' }}
              >
                {name}
              </text>
            </g>
          );
        })}

        <text
          x={PL}
          y={15}
          style={{ ...MONO, fontSize: '8px', fill: 'rgba(214,210,201,0.30)', letterSpacing: '0.12em' }}
        >
          VALIDATED CARDIOVASCULAR BENCHMARK · ACCURACY
        </text>
      </svg>

      <div style={{ display: 'flex', gap: '1.5rem', marginTop: 4 }}>
        <span style={{ ...MONO, fontSize: '0.58rem', color: MINT, opacity: 0.48, letterSpacing: '0.10em' }}>
          CLASSICAL · LR / RF / RBF-SVM
        </span>
        <span style={{ ...MONO, fontSize: '0.58rem', color: VIOLET, opacity: 0.48, letterSpacing: '0.10em' }}>
          QUANTUM · QSVM / VQC
        </span>
      </div>
      <div style={{ ...MONO, fontSize: '0.56rem', opacity: 0.22, letterSpacing: '0.08em', marginTop: 7 }}>
        CARDIOVASCULAR · n = 61 · 80/20 STRATIFIED SPLIT · RANDOM STATE 42
      </div>
      <div style={{ ...MONO, fontSize: '0.56rem', opacity: 0.28, letterSpacing: '0.08em', marginTop: 7 }}>
        BREAST CANCER · n = 114 · BEST ACCURACY: 97.37% · QSVM / RBF-SVM
      </div>
    </div>
  );
}

// ── Feature rows ──────────────────────────────────────────────────────────────
function FeatureRows({ v }: { v: boolean }) {
  const rows = [
    { f: 'FEATURE INFLUENCE', val: 'LOCAL' },
    { f: 'MODEL BEHAVIOUR', val: 'COMPARE' },
    { f: 'SENSITIVITY', val: 'INSPECT' },
    { f: 'PREDICTION', val: 'CONTEXT' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      <div
        style={{
          ...MONO,
          fontSize: '0.74rem',
          opacity: 0.28,
          letterSpacing: '0.16em',
          marginBottom: 10,
        }}
      >
        INSPECTABLE OUTPUTS
      </div>

      {rows.map((r, ri) => (
        <div
          key={r.f}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
            padding: '9px 0',
            borderBottom: '1px solid rgba(214,210,201,0.06)',
            ...fadeUp(v, 0.35 + ri * 0.10),
          }}
        >
          <span
            style={{
              ...MONO,
              fontSize: '0.66rem',
              opacity: 0.48,
              letterSpacing: '0.08em',
            }}
          >
            {r.f}
          </span>
          <span
            style={{
              ...MONO,
              fontSize: '0.60rem',
              color: MINT,
              opacity: 0.56,
              letterSpacing: '0.12em',
            }}
          >
            {r.val}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const navigate = useNavigate();

  return (
    <div style={{ background: '#080B0E' }}>
      {/* ── Fixed nav ────────────────────────────────────────────────────── */}
      <nav
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          zIndex: 200,
          padding: '2vh 5vw',
          background:
            'linear-gradient(to bottom,rgba(8,11,14,0.70) 0%,transparent 100%)',
          pointerEvents: 'none',
        }}
      >
        <span
          style={{
            ...MONO,
            fontSize: '1.05rem',
            letterSpacing: '0.28em',
            color: OFF_WHITE,
            opacity: 0.72,
          }}
        >
          VITALIS
        </span>
      </nav>

      {/* ── 01 DATA ──────────────────────────────────────────────────────── */}
      <Sec
        imgSrc={IMG.data}
        overlay="linear-gradient(135deg,rgba(8,11,14,0.93) 0%,rgba(8,11,14,0.72) 45%,rgba(8,11,14,0.58) 100%)"
      >
        {(v) => (
          <>
            <div
              style={{
                position: 'absolute',
                bottom: '8vh',
                left: '5vw',
                maxWidth: 600,
              }}
            >
              <h1
                style={{
                  ...SERIF,
                  fontWeight: 400,
                  fontSize: 'clamp(2.8rem,6.5vw,5.8rem)',
                  lineHeight: 1.08,
                  letterSpacing: '-0.020em',
                  color: OFF_WHITE,
                  marginBottom: '1.6rem',
                  ...fadeUp(v, 0.18),
                }}
              >
                Hybrid Intelligence
                <br />
                for Early
                <br />
                Disease Screening
              </h1>

              <p
                style={{
                  fontSize: 'clamp(1rem,1.6vw,1.15rem)',
                  opacity: 0.46,
                  lineHeight: 1.72,
                  maxWidth: 380,
                  marginBottom: '3rem',
                  ...fadeUp(v, 0.30),
                }}
              >
                Classical and quantum machine learning
                <br />
                for early disease screening.
              </p>

              <div style={{ ...fadeUp(v, 0.42) }}>
                <button
                  onClick={() => navigate('/screening')}
                  style={{
                    background: 'transparent',
                    border: `1px solid ${MINT}`,
                    color: MINT,
                    padding: '16px 44px',
                    ...MONO,
                    fontSize: '0.88rem',
                    letterSpacing: '0.18em',
                    cursor: 'pointer',
                  }}
                >
                  EXPLORE PLATFORM →
                </button>
              </div>
            </div>
          </>
        )}
      </Sec>

      {/* ── 02 HUMAN ─────────────────────────────────────────────────────── */}
      <Sec
        imgSrc={IMG.human}
        imgPos="center top"
        overlay="linear-gradient(180deg,rgba(8,11,14,0.78) 0%,rgba(8,11,14,0.38) 40%,rgba(8,11,14,0.80) 100%)"
      >
        {(v) => (
          <>
            <div
              style={{
                position: 'absolute',
                bottom: '8vh',
                left: '5vw',
                maxWidth: 520,
              }}
            >
              <h2
                style={{
                  ...SERIF,
                  fontWeight: 400,
                  fontSize: 'clamp(2.4rem,5.8vw,5.2rem)',
                  lineHeight: 1.10,
                  letterSpacing: '-0.018em',
                  color: OFF_WHITE,
                  marginBottom: '1.4rem',
                  ...fadeUp(v, 0.20),
                }}
              >
                Every patient
                <br />
                is more than
                <br />
                a dataset.
              </h2>

              <p
                style={{
                  fontSize: 'clamp(0.90rem,1.4vw,1.05rem)',
                  opacity: 0.42,
                  lineHeight: 1.70,
                  maxWidth: 340,
                  ...fadeUp(v, 0.34),
                }}
              >
                A human story behind
                <br />
                every data point.
              </p>
            </div>
          </>
        )}
      </Sec>

      {/* ── 03 HOW VITALIS WORKS ─────────────────────────────────────────── */}
      <Sec
        overlay="radial-gradient(circle at 76% 48%,rgba(107,184,172,0.08) 0%,rgba(8,11,14,0) 34%),radial-gradient(circle at 88% 64%,rgba(146,120,196,0.08) 0%,rgba(8,11,14,0) 30%),linear-gradient(180deg,rgba(8,11,14,0.96) 0%,rgba(8,11,14,0.99) 100%)"
      >
        {(v) => (
          <>
            {/* ── Abstract data field ─────────────────────────────────────── */}
            <svg
              aria-hidden="true"
              viewBox="0 0 900 720"
              preserveAspectRatio="xMidYMid slice"
              style={{
                position: 'absolute',
                right: '-4vw',
                top: '50%',
                width: '62vw',
                minWidth: 680,
                height: '92%',
                transform: `translateY(-50%) ${v ? 'scale(1)' : 'scale(1.04)'}`,
                opacity: v ? 0.92 : 0,
                transition:
                  'opacity 1.4s ease, transform 1.8s cubic-bezier(0.22,1,0.36,1)',
                pointerEvents: 'none',
              }}
            >
              <defs>
                <radialGradient id="vitalisDataGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor={MINT} stopOpacity="0.16" />
                  <stop offset="45%" stopColor={MINT} stopOpacity="0.035" />
                  <stop offset="100%" stopColor={MINT} stopOpacity="0" />
                </radialGradient>
                <linearGradient id="vitalisFlow" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor={MINT} stopOpacity="0" />
                  <stop offset="45%" stopColor={MINT} stopOpacity="0.35" />
                  <stop offset="100%" stopColor={VIOLET} stopOpacity="0.22" />
                </linearGradient>
              </defs>

              <ellipse cx="540" cy="360" rx="310" ry="270" fill="url(#vitalisDataGlow)" />

              {/* faint data-field grid */}
              {[120, 210, 300, 390, 480, 570, 660].map((y) => (
                <line
                  key={`gy-${y}`}
                  x1="150"
                  x2="860"
                  y1={y}
                  y2={y}
                  stroke="rgba(214,210,201,0.045)"
                  strokeWidth="1"
                />
              ))}
              {[190, 280, 370, 460, 550, 640, 730, 820].map((x) => (
                <line
                  key={`gx-${x}`}
                  x1={x}
                  x2={x}
                  y1="70"
                  y2="650"
                  stroke="rgba(214,210,201,0.035)"
                  strokeWidth="1"
                />
              ))}

              {/* flowing information paths */}
              <path
                d="M90 215 C245 150, 315 300, 450 270 S670 180, 860 255"
                fill="none"
                stroke="url(#vitalisFlow)"
                strokeWidth="1.4"
                opacity="0.7"
              />
              <path
                d="M110 470 C240 390, 330 505, 455 445 S690 360, 860 455"
                fill="none"
                stroke="url(#vitalisFlow)"
                strokeWidth="1"
                opacity="0.55"
              />
              <path
                d="M190 110 C310 250, 420 210, 510 345 S710 535, 835 585"
                fill="none"
                stroke="rgba(146,120,196,0.20)"
                strokeWidth="1"
                opacity="0.75"
              />

              {/* nodes: raw data → representation → model space */}
              {[
                [150, 215, 3.2, MINT],
                [220, 184, 2.2, MINT],
                [278, 260, 2.8, MINT],
                [345, 222, 2.0, MINT],
                [404, 288, 3.6, MINT],
                [468, 265, 2.4, MINT],
                [530, 330, 4.2, VIOLET],
                [592, 292, 2.6, VIOLET],
                [650, 350, 3.4, VIOLET],
                [718, 320, 2.1, VIOLET],
                [770, 390, 3.2, VIOLET],
                [825, 350, 2.1, VIOLET],
                [245, 455, 2.2, MINT],
                [330, 490, 3.0, MINT],
                [410, 435, 2.1, MINT],
                [500, 470, 3.0, VIOLET],
                [610, 425, 2.2, VIOLET],
                [700, 475, 3.4, VIOLET],
              ].map(([cx, cy, r, color], i) => (
                <g key={`node-${i}`}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={Number(r) * 4}
                    fill={String(color)}
                    opacity="0.035"
                  />
                  <circle
                    cx={cx}
                    cy={cy}
                    r={r}
                    fill="none"
                    stroke={String(color)}
                    strokeWidth="1"
                    opacity="0.52"
                  />
                  <circle
                    cx={cx}
                    cy={cy}
                    r="1.1"
                    fill={String(color)}
                    opacity="0.75"
                  />
                </g>
              ))}

              {/* central transformation field */}
              <circle
                cx="530"
                cy="330"
                r="82"
                fill="none"
                stroke="rgba(146,120,196,0.12)"
                strokeWidth="1"
                strokeDasharray="3 8"
              />
              <circle
                cx="530"
                cy="330"
                r="122"
                fill="none"
                stroke="rgba(107,184,172,0.08)"
                strokeWidth="1"
                strokeDasharray="2 12"
              />

              <text
                x="530"
                y="326"
                textAnchor="middle"
                fill="rgba(214,210,201,0.38)"
                style={{ ...MONO, fontSize: '11px', letterSpacing: '0.18em' }}
              >
                REPRESENTATION
              </text>
              <text
                x="530"
                y="345"
                textAnchor="middle"
                fill="rgba(214,210,201,0.18)"
                style={{ ...MONO, fontSize: '8px', letterSpacing: '0.12em' }}
              >
                QUANTUM-READY SPACE
              </text>

              {/* model branch labels */}
              <text
                x="665"
                y="165"
                fill="rgba(107,184,172,0.48)"
                style={{ ...MONO, fontSize: '9px', letterSpacing: '0.16em' }}
              >
                CLASSICAL
              </text>
              <text
                x="665"
                y="180"
                fill="rgba(214,210,201,0.20)"
                style={{ ...MONO, fontSize: '8px', letterSpacing: '0.10em' }}
              >
                LR · RF · SVM
              </text>

              <text
                x="665"
                y="555"
                fill="rgba(146,120,196,0.52)"
                style={{ ...MONO, fontSize: '9px', letterSpacing: '0.16em' }}
              >
                QUANTUM
              </text>
              <text
                x="665"
                y="570"
                fill="rgba(214,210,201,0.20)"
                style={{ ...MONO, fontSize: '8px', letterSpacing: '0.10em' }}
              >
                QSVM · VQC
              </text>

              <text
                x="105"
                y="590"
                fill="rgba(214,210,201,0.18)"
                style={{ ...MONO, fontSize: '8px', letterSpacing: '0.14em' }}
              >
                BIOMEDICAL SIGNAL
              </text>
            </svg>

            {/* ── Content ──────────────────────────────────────────────────── */}
            <div
              style={{
                position: 'relative',
                zIndex: 2,
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                padding: '8vh 5vw 7vh',
              }}
            >
              <div style={{ maxWidth: 650 }}>
                <div style={{ ...fadeUp(v, 0.10), marginBottom: '1.1rem' }}>
                  <Label op={0.34}>THE VITALIS PIPELINE</Label>
                </div>

                <h2
                  style={{
                    ...SERIF,
                    fontWeight: 400,
                    fontSize: 'clamp(2.4rem,5.2vw,4.7rem)',
                    lineHeight: 1.06,
                    letterSpacing: '-0.020em',
                    color: OFF_WHITE,
                    margin: 0,
                    ...fadeUp(v, 0.18),
                  }}
                >
                  From biomedical data
                  <br />
                  to explainable prediction.
                </h2>

                <p
                  style={{
                    fontSize: 'clamp(0.88rem,1.4vw,1rem)',
                    opacity: 0.42,
                    lineHeight: 1.70,
                    maxWidth: 510,
                    marginTop: '1.3rem',
                    ...fadeUp(v, 0.30),
                  }}
                >
                  VITALIS prepares a common biomedical representation for
                  classical and quantum-enhanced models, then evaluates their
                  behaviour under the same framework.
                </p>
              </div>

              <div
                style={{
                  position: 'relative',
                  zIndex: 3,
                  display: 'grid',
                  gridTemplateColumns: 'repeat(5,minmax(0,1fr))',
                  maxWidth: 1080,
                  width: '100%',
                  marginTop: '4vh',
                  background: 'rgba(8,11,14,0.56)',
                  backdropFilter: 'blur(8px)',
                  WebkitBackdropFilter: 'blur(8px)',
                  borderTop: '1px solid rgba(214,210,201,0.10)',
                  borderBottom: '1px solid rgba(214,210,201,0.08)',
                }}
              >
                {[
                  {
                    n: '01',
                    title: 'DATA',
                    lines: ['Biomedical', 'observations'],
                    accent: MINT,
                  },
                  {
                    n: '02',
                    title: 'PREPROCESS',
                    lines: ['Clean · encode', 'scale · impute'],
                    accent: MINT,
                  },
                  {
                    n: '03',
                    title: 'SELECT + REDUCE',
                    lines: ['Feature selection', 'PCA representation'],
                    accent: MINT,
                  },
                  {
                    n: '04',
                    title: 'MODEL',
                    lines: ['Classical + quantum', 'LR · RF · SVM · QSVM · VQC'],
                    accent: VIOLET,
                  },
                  {
                    n: '05',
                    title: 'EVALUATE',
                    lines: ['Benchmark', 'explain · compare'],
                    accent: MINT,
                  },
                ].map((step, i) => (
                  <div
                    key={step.n}
                    style={{
                      position: 'relative',
                      minHeight: 132,
                      padding: '1.15rem 1rem',
                      borderLeft: '1px solid rgba(214,210,201,0.07)',
                      background:
                        i === 3
                          ? 'rgba(146,120,196,0.065)'
                          : 'rgba(8,11,14,0.18)',
                      ...fadeUp(v, 0.34 + i * 0.08),
                    }}
                  >
                    <div
                      style={{
                        ...MONO,
                        fontSize: '0.60rem',
                        letterSpacing: '0.16em',
                        color: step.accent,
                        opacity: 0.72,
                        marginBottom: '1rem',
                      }}
                    >
                      {step.n}
                    </div>

                    <div
                      style={{
                        ...MONO,
                        fontSize: '0.66rem',
                        letterSpacing: '0.12em',
                        color: OFF_WHITE,
                        opacity: 0.62,
                        marginBottom: '0.65rem',
                      }}
                    >
                      {step.title}
                    </div>

                    <div
                      style={{
                        fontFamily: "'Inter',sans-serif",
                        fontSize: '0.74rem',
                        fontWeight: 300,
                        color: OFF_WHITE,
                        opacity: 0.40,
                        lineHeight: 1.65,
                      }}
                    >
                      {step.lines.map((line) => (
                        <div key={line}>{line}</div>
                      ))}
                    </div>

                    {i < 4 && (
                      <div
                        aria-hidden
                        style={{
                          position: 'absolute',
                          right: -8,
                          top: '50%',
                          width: 16,
                          height: 1,
                          background:
                            i === 2
                              ? 'rgba(146,120,196,0.52)'
                              : 'rgba(107,184,172,0.32)',
                          zIndex: 4,
                        }}
                      />
                    )}
                  </div>
                ))}
              </div>

              <div
                style={{
                  display: 'flex',
                  justifyContent: 'center',
                  marginTop: '2.4vh',
                  ...fadeUp(v, 0.78),
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '1.35rem',
                    padding: '0.75rem 1.2rem',
                    background: 'rgba(8,11,14,0.42)',
                    backdropFilter: 'blur(6px)',
                    borderTop: '1px solid rgba(214,210,201,0.07)',
                    borderBottom: '1px solid rgba(214,210,201,0.07)',
                  }}
                >
                  <span
                    style={{
                      ...MONO,
                      fontSize: '0.60rem',
                      letterSpacing: '0.15em',
                      color: MINT,
                      opacity: 0.72,
                    }}
                  >
                    OUTPUT
                  </span>
                  {['Prediction', 'Consensus', 'Explainability'].map((item, i) => (
                    <div
                      key={item}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '1.35rem',
                      }}
                    >
                      {i > 0 && (
                        <span
                          style={{
                            width: 18,
                            height: 1,
                            background: 'rgba(214,210,201,0.18)',
                          }}
                        />
                      )}
                      <span
                        style={{
                          fontFamily: "'Inter',sans-serif",
                          fontSize: '0.74rem',
                          fontWeight: 300,
                          color: OFF_WHITE,
                          opacity: 0.44,
                        }}
                      >
                        {item}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <p
                style={{
                  ...MONO,
                  textAlign: 'center',
                  fontSize: '0.60rem',
                  letterSpacing: '0.12em',
                  opacity: 0.24,
                  marginTop: '1.7vh',
                  ...fadeIn(v, 0.94),
                }}
              >
                QUANTUM MODELS ARE EVALUATED WITHIN A HYBRID CLASSICAL–QUANTUM PIPELINE
              </p>
            </div>
          </>
        )}
      </Sec>

      {/* ── 04 CARDIOVASCULAR ────────────────────────────────────────────── */}
      <Sec
        imgSrc={IMG.cardio}
        imgPos="center center"
        overlay="linear-gradient(150deg,rgba(8,11,14,0.85) 0%,rgba(8,11,14,0.42) 52%,rgba(8,11,14,0.75) 100%)"
      >
        {(v) => (
          <>
            <div
              style={{
                position: 'absolute',
                bottom: '8vh',
                left: '5vw',
                maxWidth: 480,
              }}
            >
              <h2
                style={{
                  ...SERIF,
                  fontWeight: 400,
                  fontSize: 'clamp(2.8rem,6.5vw,5.8rem)',
                  lineHeight: 1.02,
                  letterSpacing: '-0.020em',
                  color: OFF_WHITE,
                  marginBottom: '1.6rem',
                  ...fadeUp(v, 0.18),
                }}
              >
                Cardiovascular
              </h2>

              <div
                style={{
                  display: 'flex',
                  gap: '2.8rem',
                  marginBottom: '1.4rem',
                  ...fadeUp(v, 0.30),
                }}
              >
                <Label op={0.55}>6 FEATURES</Label>
                <Label op={0.55}>5 MODELS</Label>
              </div>

              <p
                style={{
                  fontSize: 'clamp(0.90rem,1.4vw,1.05rem)',
                  opacity: 0.40,
                  lineHeight: 1.70,
                  maxWidth: 340,
                  marginBottom: '1.6rem',
                  ...fadeUp(v, 0.40),
                }}
              >
                Extracting meaningful patterns
                <br />
                from complex cardiovascular data.
              </p>

              <button
                onClick={() => navigate('/screening?disease=heart')}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: OFF_WHITE,
                  opacity: 0.34,
                  ...MONO,
                  fontSize: '0.70rem',
                  letterSpacing: '0.14em',
                  cursor: 'pointer',
                  padding: 0,
                  textDecoration: 'underline',
                  textUnderlineOffset: 5,
                  ...fadeUp(v, 0.50),
                }}
              >
                Explore ——→
              </button>
            </div>
          </>
        )}
      </Sec>

      {/* ── 05 BREAST CANCER ─────────────────────────────────────────────── */}
      <Sec
        imgSrc={IMG.cancer}
        imgPos="center center"
        overlay="linear-gradient(150deg,rgba(8,11,14,0.88) 0%,rgba(8,11,14,0.45) 48%,rgba(8,11,14,0.78) 100%)"
      >
        {(v) => (
          <>
            <div
              style={{
                position: 'absolute',
                bottom: '8vh',
                left: '5vw',
                maxWidth: 480,
              }}
            >
              <h2
                style={{
                  ...SERIF,
                  fontWeight: 400,
                  fontSize: 'clamp(2.8rem,6.5vw,5.8rem)',
                  lineHeight: 1.02,
                  letterSpacing: '-0.020em',
                  color: OFF_WHITE,
                  marginBottom: '1.6rem',
                  ...fadeUp(v, 0.18),
                }}
              >
                Breast Cancer
              </h2>

              <div
                style={{
                  display: 'flex',
                  gap: '2.8rem',
                  marginBottom: '1.4rem',
                  ...fadeUp(v, 0.30),
                }}
              >
                <Label op={0.55}>30 FEATURES</Label>
                <Label op={0.55}>5 MODELS</Label>
              </div>

              <p
                style={{
                  fontSize: 'clamp(0.90rem,1.4vw,1.05rem)',
                  opacity: 0.40,
                  lineHeight: 1.70,
                  maxWidth: 340,
                  marginBottom: '1.6rem',
                  ...fadeUp(v, 0.40),
                }}
              >
                Multi-dimensional analysis
                <br />
                for early detection.
              </p>

              <button
                onClick={() => navigate('/screening?disease=breast_cancer')}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: OFF_WHITE,
                  opacity: 0.34,
                  ...MONO,
                  fontSize: '0.70rem',
                  letterSpacing: '0.14em',
                  cursor: 'pointer',
                  padding: 0,
                  textDecoration: 'underline',
                  textUnderlineOffset: 5,
                  ...fadeUp(v, 0.50),
                }}
              >
                Explore ——→
              </button>
            </div>
          </>
        )}
      </Sec>

      {/* ── 06 YOUR DATASET ──────────────────────────────────────────────── */}
      <Sec
        overlay="radial-gradient(circle at 78% 42%,rgba(107,184,172,0.10) 0%,rgba(8,11,14,0) 32%),radial-gradient(circle at 88% 68%,rgba(146,120,196,0.09) 0%,rgba(8,11,14,0) 30%),linear-gradient(150deg,rgba(8,11,14,0.98) 0%,rgba(8,11,14,0.94) 100%)"
      >
        {(v) => (
          <>
            <div
              style={{
                position: 'absolute',
                inset: '10vh 5vw 8vh',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
              }}
            >
              <div style={{ maxWidth: 760 }}>
                <div style={{ ...fadeUp(v, 0.10), marginBottom: '1.1rem' }}>
                  <Label op={0.34}>YOUR DATASET</Label>
                </div>

                <h2
                  style={{
                    ...SERIF,
                    fontWeight: 400,
                    fontSize: 'clamp(2.5rem,5.5vw,5rem)',
                    lineHeight: 1.06,
                    letterSpacing: '-0.020em',
                    color: OFF_WHITE,
                    margin: 0,
                    ...fadeUp(v, 0.18),
                  }}
                >
                  Bring your own
                  <br />
                  biomedical data.
                </h2>

                <p
                  style={{
                    fontSize: 'clamp(0.92rem,1.5vw,1.08rem)',
                    opacity: 0.42,
                    lineHeight: 1.72,
                    maxWidth: 520,
                    marginTop: '1.4rem',
                    ...fadeUp(v, 0.30),
                  }}
                >
                  Upload a labeled CSV and use the same VITALIS workflow to
                  profile, prepare, train, benchmark, predict, and inspect your
                  dataset.
                </p>
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(6,minmax(0,1fr))',
                  maxWidth: 1050,
                  width: '100%',
                  marginTop: '5vh',
                  borderTop: '1px solid rgba(214,210,201,0.09)',
                  borderBottom: '1px solid rgba(214,210,201,0.07)',
                  background: 'rgba(8,11,14,0.42)',
                  backdropFilter: 'blur(8px)',
                  WebkitBackdropFilter: 'blur(8px)',
                }}
              >
                {[
                  ['01', 'UPLOAD', 'CSV'],
                  ['02', 'PROFILE', 'DATA'],
                  ['03', 'PREPARE', 'FEATURES'],
                  ['04', 'TRAIN', 'HYBRID'],
                  ['05', 'BENCHMARK', 'MODELS'],
                  ['06', 'PREDICT', 'EXPLAIN'],
                ].map(([n, title, sub], i) => (
                  <div
                    key={n}
                    style={{
                      minHeight: 112,
                      padding: '1rem 0.85rem',
                      borderLeft: '1px solid rgba(214,210,201,0.07)',
                      ...fadeUp(v, 0.38 + i * 0.07),
                    }}
                  >
                    <div
                      style={{
                        ...MONO,
                        fontSize: '0.58rem',
                        letterSpacing: '0.16em',
                        color: i === 3 ? VIOLET : MINT,
                        opacity: 0.72,
                        marginBottom: '0.85rem',
                      }}
                    >
                      {n}
                    </div>
                    <div
                      style={{
                        ...MONO,
                        fontSize: '0.64rem',
                        letterSpacing: '0.12em',
                        color: OFF_WHITE,
                        opacity: 0.62,
                        marginBottom: 7,
                      }}
                    >
                      {title}
                    </div>
                    <div
                      style={{
                        ...MONO,
                        fontSize: '0.58rem',
                        letterSpacing: '0.10em',
                        opacity: 0.28,
                      }}
                    >
                      {sub}
                    </div>
                  </div>
                ))}
              </div>

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1.8rem',
                  marginTop: '2.8vh',
                  ...fadeUp(v, 0.84),
                }}
              >
                <button
                  onClick={() => navigate('/screening')}
                  style={{
                    background: 'transparent',
                    border: `1px solid ${MINT}`,
                    color: MINT,
                    padding: '14px 32px',
                    ...MONO,
                    fontSize: '0.74rem',
                    letterSpacing: '0.15em',
                    cursor: 'pointer',
                  }}
                >
                  OPEN DATASET WORKSPACE →
                </button>

                <span
                  style={{
                    ...MONO,
                    fontSize: '0.62rem',
                    letterSpacing: '0.12em',
                    opacity: 0.24,
                  }}
                >
                  LABELED BIOMEDICAL CSV
                </span>
              </div>
            </div>
          </>
        )}
      </Sec>

      {/* ── 06 MORE CONDITIONS ───────────────────────────────────────────── */}
      <Sec
        overlay="linear-gradient(180deg,rgba(8,11,14,1) 0%,rgba(8,11,14,0.97) 100%)"
      >
        {(v) => (
          <>
            <div
              style={{
                position: 'absolute',
                inset: '11vh 5vw 8vh',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                gap: '5vh',
              }}
            >
              <div style={{ maxWidth: 680 }}>
                <h2
                  style={{
                    ...SERIF,
                    fontWeight: 400,
                    fontSize: 'clamp(2.2rem,5.2vw,4.6rem)',
                    lineHeight: 1.10,
                    letterSpacing: '-0.018em',
                    color: OFF_WHITE,
                    marginBottom: '1.4rem',
                    ...fadeUp(v, 0.18),
                  }}
                >
                  The framework is built
                  <br />
                  to scale.
                </h2>

                <p
                  style={{
                    fontSize: 'clamp(1rem,1.6vw,1.15rem)',
                    opacity: 0.42,
                    lineHeight: 1.72,
                    maxWidth: 460,
                    ...fadeUp(v, 0.30),
                  }}
                >
                  Cardiovascular and breast cancer are the first two modules.
                  More disease domains are being integrated using the same
                  classical–quantum evaluation pipeline.
                </p>
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(4,1fr)',
                  gap: '1px',
                  maxWidth: 760,
                  background: 'rgba(214,210,201,0.06)',
                  ...fadeUp(v, 0.40),
                }}
              >
                {[
                  { name: 'Diabetes', status: 'In development', color: MINT },
                  { name: 'Stroke', status: 'Planned', color: 'rgba(214,210,201,0.30)' },
                  { name: "Alzheimer's", status: 'Planned', color: 'rgba(214,210,201,0.30)' },
                  { name: 'Lung Cancer', status: 'Planned', color: 'rgba(214,210,201,0.30)' },
                  { name: 'Kidney Disease', status: 'Planned', color: 'rgba(214,210,201,0.30)' },
                  { name: 'Liver Disease', status: 'Planned', color: 'rgba(214,210,201,0.30)' },
                ].map((d, di) => (
                  <div
                    key={d.name}
                    style={{
                      background: 'rgba(8,11,14,1)',
                      padding: '1.8rem 1.6rem',
                      borderLeft: `1px solid ${
                        di === 0
                          ? 'rgba(107,184,172,0.18)'
                          : 'rgba(214,210,201,0.05)'
                      }`,
                      ...fadeUp(v, 0.44 + di * 0.06),
                    }}
                  >
                    <div
                      style={{
                        ...MONO,
                        fontSize: '0.70rem',
                        letterSpacing: '0.18em',
                        color: d.color,
                        marginBottom: 10,
                        opacity: di === 0 ? 0.85 : 0.50,
                      }}
                    >
                      {d.status.toUpperCase()}
                    </div>

                    <div
                      style={{
                        fontFamily: "'Inter',sans-serif",
                        fontSize: 'clamp(1rem,1.5vw,1.1rem)',
                        fontWeight: 300,
                        color: OFF_WHITE,
                        opacity: di === 0 ? 0.85 : 0.45,
                      }}
                    >
                      {d.name}
                    </div>
                  </div>
                ))}

                <div
                  style={{
                    gridColumn: 'span 2',
                    background: 'rgba(8,11,14,1)',
                    borderLeft: '1px dashed rgba(107,184,172,0.20)',
                    padding: '1.8rem 1.6rem',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    ...fadeUp(v, 0.74),
                  }}
                >
                  <div
                    style={{
                      ...MONO,
                      fontSize: '0.70rem',
                      letterSpacing: '0.18em',
                      color: MINT,
                      opacity: 0.60,
                      marginBottom: 10,
                    }}
                  >
                    + EXPANDING
                  </div>

                  <div
                    style={{
                      fontFamily: "'Inter',sans-serif",
                      fontSize: 'clamp(0.88rem,1.3vw,0.98rem)',
                      fontWeight: 300,
                      color: OFF_WHITE,
                      opacity: 0.55,
                      lineHeight: 1.5,
                    }}
                  >
                    More disease domains run through the same framework — not
                    yet built, not a limit either.
                  </div>
                </div>
              </div>

              <p
                style={{
                  ...MONO,
                  fontSize: '0.74rem',
                  opacity: 0.22,
                  letterSpacing: '0.14em',
                  ...fadeUp(v, 0.85),
                }}
              >
                SAME MODELS · SAME EVALUATION · SAME REPRODUCIBILITY STANDARDS
              </p>
            </div>
          </>
        )}
      </Sec>

      {/* ── 07 COMPUTATION ──────────────────────────────────────────────── */}
      <Sec
        imgSrc={IMG.compute}
        imgPos="right center"
        overlay="linear-gradient(100deg,rgba(8,11,14,0.96) 0%,rgba(8,11,14,0.80) 38%,rgba(8,11,14,0.48) 80%,rgba(8,11,14,0.62) 100%)"
      >
        {(v) => (
          <>
            <div
              style={{
                position: 'absolute',
                top: '18vh',
                right: '5vw',
                textAlign: 'right',
                ...fadeIn(v, 0.50),
              }}
            >
              {['MODELS', 'ALGORITHMS', 'SIMULATION'].map((a) => (
                <div
                  key={a}
                  style={{
                    ...MONO,
                    fontSize: '0.72rem',
                    letterSpacing: '0.22em',
                    opacity: 0.18,
                    marginBottom: 18,
                  }}
                >
                  {a}
                </div>
              ))}
            </div>

            <div
              style={{
                position: 'absolute',
                bottom: '8vh',
                left: '5vw',
                maxWidth: 600,
              }}
            >
              <div
                style={{
                  display: 'flex',
                  gap: '4rem',
                  marginBottom: '2.8rem',
                  ...fadeUp(v, 0.18),
                }}
              >
                <div>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '1.2rem',
                      marginBottom: '1rem',
                    }}
                  >
                    <span
                      style={{
                        fontFamily: "'Inter',sans-serif",
                        fontSize: 'clamp(0.95rem,1.8vw,1.3rem)',
                        fontWeight: 300,
                        color: OFF_WHITE,
                        opacity: 0.68,
                      }}
                    >
                      Classical
                    </span>

                    <div
                      style={{
                        width: 36,
                        height: 1,
                        background: 'rgba(214,210,201,0.20)',
                      }}
                    />
                  </div>

                  {['Logistic Regression', 'Random Forest', 'RBF SVM'].map((m) => (
                    <div
                      key={m}
                      style={{
                        ...MONO,
                        fontSize: '0.76rem',
                        opacity: 0.36,
                        lineHeight: 2.4,
                      }}
                    >
                      {m}
                    </div>
                  ))}
                </div>

                <div
                  style={{
                    width: 1,
                    background: 'rgba(214,210,201,0.07)',
                    alignSelf: 'stretch',
                  }}
                />

                <div>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '1.2rem',
                      marginBottom: '1rem',
                    }}
                  >
                    <span
                      style={{
                        fontFamily: "'Inter',sans-serif",
                        fontSize: 'clamp(0.95rem,1.8vw,1.3rem)',
                        fontWeight: 300,
                        color: OFF_WHITE,
                        opacity: 0.68,
                      }}
                    >
                      Quantum
                    </span>
                  </div>

                  {['QSVM', 'VQC'].map((m) => (
                    <div
                      key={m}
                      style={{
                        ...MONO,
                        fontSize: '0.76rem',
                        opacity: 0.36,
                        lineHeight: 2.4,
                      }}
                    >
                      {m}
                    </div>
                  ))}
                </div>
              </div>

              <h2
                style={{
                  ...SERIF,
                  fontWeight: 400,
                  fontSize: 'clamp(1.8rem,4vw,3.4rem)',
                  lineHeight: 1.18,
                  letterSpacing: '-0.014em',
                  color: OFF_WHITE,
                  ...fadeUp(v, 0.34),
                }}
              >
                Two computational
                <br />
                paradigms. One evaluation
                <br />
                framework.
              </h2>
            </div>
          </>
        )}
      </Sec>

      {/* ── 08 CONVERGENCE ───────────────────────────────────────────────── */}
      <Sec
        imgSrc={IMG.converge}
        imgPos="center center"
        overlay="linear-gradient(180deg,rgba(8,11,14,0.82) 0%,rgba(8,11,14,0.48) 45%,rgba(8,11,14,0.80) 100%)"
      >
        {(v) => (
          <>
            <div
              style={{
                position: 'absolute',
                bottom: '8vh',
                left: '5vw',
                maxWidth: 560,
              }}
            >
              <h2
                style={{
                  ...SERIF,
                  fontWeight: 400,
                  fontSize: 'clamp(2.4rem,5.8vw,5.2rem)',
                  lineHeight: 1.10,
                  letterSpacing: '-0.018em',
                  color: OFF_WHITE,
                  marginBottom: '2.2rem',
                  ...fadeUp(v, 0.18),
                }}
              >
                Different models.
                <br />
                One screening
                <br />
                workflow.
              </h2>

              {['Prediction', 'Consensus', 'Explainability', 'Benchmarking'].map(
                (k, i) => (
                  <div
                    key={k}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 14,
                      marginBottom: 14,
                      ...fadeUp(v, 0.32 + i * 0.10),
                    }}
                  >
                    <div
                      style={{
                        width: 20,
                        height: 1,
                        background: 'rgba(214,210,201,0.22)',
                      }}
                    />
                    <span
                      style={{
                        fontFamily: "'Inter',sans-serif",
                        fontSize: 'clamp(0.85rem,1.4vw,1rem)',
                        opacity: 0.40,
                        fontWeight: 300,
                      }}
                    >
                      {k}
                    </span>
                  </div>
                ),
              )}
            </div>
          </>
        )}
      </Sec>

      {/* ── 09 BENCHMARK ─────────────────────────────────────────────────── */}
      <Sec
        id="benchmarks"
        overlay="linear-gradient(180deg,rgba(8,11,14,1) 0%,rgba(8,11,14,0.97) 100%)"
      >
        {(v) => (
          <>
            <div
              style={{
                position: 'absolute',
                inset: '11vh 5vw 8vh',
                display: 'flex',
                gap: '5vw',
                alignItems: 'flex-end',
              }}
            >
              <div style={{ flex: 1, minWidth: 0, ...fadeUp(v, 0.20) }}>
                <BenchFigure v={v} />
              </div>

              <div style={{ maxWidth: 300, paddingBottom: '0.4rem' }}>
                <h2
                  style={{
                    ...SERIF,
                    fontWeight: 400,
                    fontSize: 'clamp(1.8rem,3.8vw,3.2rem)',
                    lineHeight: 1.14,
                    letterSpacing: '-0.014em',
                    color: OFF_WHITE,
                    marginBottom: '1.2rem',
                    ...fadeUp(v, 0.30),
                  }}
                >
                  Performance
                  <br />
                  across models.
                </h2>

                <p
                  style={{
                    fontSize: 'clamp(0.88rem,1.4vw,1rem)',
                    opacity: 0.38,
                    lineHeight: 1.70,
                    marginBottom: '2rem',
                    ...fadeUp(v, 0.42),
                  }}
                >
                  A fair comparison under
                  <br />
                  identical evaluation conditions.
                </p>

                <button
                  onClick={() =>
                    document.getElementById('benchmarks')?.scrollIntoView({
                      behavior: 'smooth',
                      block: 'start',
                    })
                  }
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: OFF_WHITE,
                    opacity: 0.42,
                    ...MONO,
                    fontSize: '0.68rem',
                    letterSpacing: '0.14em',
                    cursor: 'pointer',
                    padding: 0,
                    textDecoration: 'underline',
                    textUnderlineOffset: 5,
                    ...fadeUp(v, 0.54),
                  }}
                >
                  View details ——→
                </button>
              </div>
            </div>
          </>
        )}
      </Sec>

      {/* ── 10 EXPLAINABILITY ────────────────────────────────────────────── */}
      <Sec
        id="explainability"
        imgSrc={IMG.explain}
        imgPos="left center"
        overlay="linear-gradient(90deg,rgba(8,11,14,0.50) 0%,rgba(8,11,14,0.80) 48%,rgba(8,11,14,0.96) 100%)"
      >
        {(v) => (
          <>
            <div
              style={{
                position: 'absolute',
                inset: '10vh 5vw 8vh',
                display: 'flex',
                alignItems: 'flex-end',
                gap: '5vw',
              }}
            >
              <div style={{ flex: 1 }} />

              <div style={{ maxWidth: 400 }}>
                <h2
                  style={{
                    ...SERIF,
                    fontWeight: 400,
                    fontSize: 'clamp(1.8rem,4vw,3.4rem)',
                    lineHeight: 1.14,
                    letterSpacing: '-0.014em',
                    color: OFF_WHITE,
                    marginBottom: '1.4rem',
                    ...fadeUp(v, 0.18),
                  }}
                >
                  Prediction
                  <br />
                  should be
                  <br />
                  inspectable.
                </h2>

                <p
                  style={{
                    fontSize: 'clamp(0.88rem,1.4vw,1rem)',
                    opacity: 0.38,
                    lineHeight: 1.70,
                    marginBottom: '2rem',
                    ...fadeUp(v, 0.30),
                  }}
                >
                  Understanding model behaviour
                  <br />
                  for trusted decisions.
                </p>

                <FeatureRows v={v} />

                <button
                  onClick={() => {
                    document.getElementById('benchmarks')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                  }}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: OFF_WHITE,
                    opacity: 0.28,
                    marginTop: '1.8rem',
                    ...MONO,
                    fontSize: '0.68rem',
                    letterSpacing: '0.14em',
                    cursor: 'pointer',
                    padding: 0,
                    textDecoration: 'underline',
                    textUnderlineOffset: 5,
                    ...fadeUp(v, 0.90),
                  }}
                >
                  Explore analysis ——→
                </button>
              </div>
            </div>
          </>
        )}
      </Sec>

      {/* ── 11 RESEARCH ──────────────────────────────────────────────────── */}
      <Sec overlay="rgba(8,11,14,1)">
        {(v) => (
          <>
            <div
              style={{
                position: 'absolute',
                inset: '11vh 5vw 8vh',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'flex-end',
                gap: '4vh',
              }}
            >
              <p
                style={{
                  ...SERIF,
                  fontWeight: 300,
                  fontStyle: 'italic',
                  fontSize: 'clamp(1.3rem,2.8vw,2.4rem)',
                  lineHeight: 1.40,
                  letterSpacing: '-0.008em',
                  color: OFF_WHITE,
                  opacity: 0.80,
                  maxWidth: 720,
                  ...fadeUp(v, 0.18),
                }}
              >
                Can quantum-kernel and variational quantum approaches provide
                competitive performance against established classical ML models
                under identical evaluation conditions?
              </p>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit,minmax(170px,1fr))',
                  gap: '0 2rem',
                  maxWidth: 760,
                  ...fadeUp(v, 0.34),
                }}
              >
                {[
                  ['Dataset', 'Cardiovascular · Breast Cancer'],
                  ['Feature Engineering', 'Clinical · Engineered Features'],
                  ['Model Architecture', 'Classical ML · Quantum ML'],
                  ['Evaluation', 'Stratified holdout · Multiple Metrics'],
                  ['Reproducibility', 'Code · Data · Methodology'],
                ].map(([k, val]) => (
                  <div
                    key={k}
                    style={{
                      borderTop: '1px solid rgba(214,210,201,0.07)',
                      paddingTop: 14,
                      paddingBottom: 14,
                    }}
                  >
                    <div
                      style={{
                        ...MONO,
                        fontSize: '0.74rem',
                        opacity: 0.26,
                        letterSpacing: '0.14em',
                        marginBottom: 6,
                      }}
                    >
                      {k.toUpperCase()}
                    </div>

                    <div
                      style={{
                        fontSize: '0.82rem',
                        opacity: 0.46,
                        fontWeight: 300,
                      }}
                    >
                      {val}
                    </div>
                  </div>
                ))}
              </div>

              <button
                onClick={() => navigate('/research')}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: OFF_WHITE,
                  opacity: 0.26,
                  ...MONO,
                  fontSize: '0.68rem',
                  letterSpacing: '0.14em',
                  cursor: 'pointer',
                  padding: 0,
                  textDecoration: 'underline',
                  textUnderlineOffset: 5,
                  alignSelf: 'flex-start',
                  ...fadeUp(v, 0.50),
                }}
              >
                Read the full research ——→
              </button>
            </div>
          </>
        )}
      </Sec>

      {/* ── 12 FUTURE ────────────────────────────────────────────────────── */}
      <Sec
        imgSrc={IMG.future}
        imgPos="center center"
        overlay="linear-gradient(180deg,rgba(8,11,14,0.88) 0%,rgba(8,11,14,0.52) 50%,rgba(8,11,14,0.84) 100%)"
      >
        {(v) => (
          <>
            <div
              style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                textAlign: 'center',
              }}
            >
              <h2
                style={{
                  ...SERIF,
                  fontWeight: 300,
                  fontSize: 'clamp(3.5rem,9vw,8rem)',
                  letterSpacing: '0.18em',
                  lineHeight: 1.0,
                  color: OFF_WHITE,
                  marginBottom: '2.4rem',
                  ...fadeUp(v, 0.18),
                }}
              >
                VITALIS
              </h2>

              <p
                style={{
                  fontFamily: "'Inter',sans-serif",
                  fontSize: 'clamp(0.95rem,1.8vw,1.2rem)',
                  fontWeight: 300,
                  opacity: 0.40,
                  lineHeight: 1.60,
                  marginBottom: '3rem',
                  ...fadeUp(v, 0.32),
                }}
              >
                Explore the intelligence
                <br />
                behind the prediction.
              </p>

              <button
                onClick={() => navigate('/screening')}
                style={{
                  background: 'transparent',
                  border: `1px solid ${MINT}`,
                  color: MINT,
                  padding: '14px 40px',
                  ...MONO,
                  fontSize: '0.80rem',
                  letterSpacing: '0.15em',
                  cursor: 'pointer',
                  ...fadeUp(v, 0.46),
                }}
              >
                Enter Screening ——→
              </button>
            </div>

            <div
              style={{
                textAlign: 'center',
                padding: '0 0 4vh',
                opacity: 0.20,
                ...fadeIn(v, 0.70),
              }}
            >
              <Label op={1}>Science for a Healthier Tomorrow.</Label>
            </div>
          </>
        )}
      </Sec>
    </div>
  );
}
