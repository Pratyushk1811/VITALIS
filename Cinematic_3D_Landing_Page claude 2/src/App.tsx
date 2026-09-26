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
  const metrics = ['ACC', 'PREC', 'SENS', 'SPEC', 'F1', 'AUC'];

  const cls: [number, number, number, number, number][] = [
    [0.965, 0.950, 0.978, 0.925, 0.992],
    [0.958, 0.938, 0.972, 0.908, 0.986],
    [0.922, 0.898, 0.944, 0.868, 0.962],
    [0.983, 0.972, 0.993, 0.954, 1.00],
    [0.940, 0.918, 0.957, 0.892, 0.972],
    [0.991, 0.981, 0.998, 0.966, 1.00],
  ];

  const qnt: [number, number, number, number, number][] = [
    [0.934, 0.910, 0.952, 0.878, 0.970],
    [0.894, 0.868, 0.918, 0.836, 0.944],
    [0.878, 0.848, 0.906, 0.816, 0.933],
    [0.970, 0.952, 0.983, 0.929, 0.995],
    [0.885, 0.858, 0.912, 0.828, 0.937],
    [0.968, 0.949, 0.983, 0.924, 0.996],
  ];

  const W = 500;
  const H = 240;
  const PT = 16;
  const PR = 20;
  const PB = 40;
  const PL = 36;

  const bW = (W - PL - PR) / metrics.length;
  const boxW = bW * 0.25;

  const yLo = 0.80;
  const yHi = 1.02;

  const py = (val: number) =>
    PT + (1 - (val - yLo) / (yHi - yLo)) * (H - PT - PB);

  const cx = (i: number, side: 0 | 1) =>
    PL + bW * i + bW * (side === 0 ? 0.25 : 0.54);

  function Box({
    d,
    i,
    side,
    color,
    delay,
  }: {
    d: [number, number, number, number, number];
    i: number;
    side: 0 | 1;
    color: string;
    delay: number;
  }) {
    const [med, q1, q3, wl, wh] = d;
    const x = cx(i, side);
    const bTop = py(q3);
    const bBot = py(q1);
    const bH = Math.max(2, bBot - bTop);

    return (
      <g>
        <line
          x1={x}
          x2={x}
          y1={py(wl)}
          y2={py(wh)}
          stroke={color}
          strokeWidth={0.8}
          opacity={0.30}
        />

        <rect
          x={x - boxW / 2}
          y={bTop}
          width={boxW}
          height={bH}
          fill={color}
          opacity={v ? 0.22 : 0}
          style={{ transition: `opacity 0.7s ease ${delay}s` }}
        />

        <rect
          x={x - boxW / 2}
          y={bTop}
          width={boxW}
          height={bH}
          fill="none"
          stroke={color}
          strokeWidth={0.7}
          opacity={v ? 0.55 : 0}
          style={{ transition: `opacity 0.7s ease ${delay}s` }}
        />

        <line
          x1={x - boxW / 2}
          x2={x + boxW / 2}
          y1={py(med)}
          y2={py(med)}
          stroke={color}
          strokeWidth={1.4}
          opacity={v ? 0.85 : 0}
          style={{ transition: `opacity 0.7s ease ${delay + 0.1}s` }}
        />
      </g>
    );
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }}>
      {[0.82, 0.86, 0.90, 0.94, 0.98].map((val) => (
        <g key={val}>
          <line
            x1={PL}
            x2={W - PR}
            y1={py(val)}
            y2={py(val)}
            stroke="rgba(214,210,201,0.06)"
            strokeWidth={0.5}
          />
          <text
            x={PL - 5}
            y={py(val) + 4}
            textAnchor="end"
            style={{
              ...MONO,
              fontSize: '7.5px',
              fill: 'rgba(214,210,201,0.28)',
            }}
          >
            {val.toFixed(2)}
          </text>
        </g>
      ))}

      {metrics.map((m, i) => (
        <text
          key={m}
          x={PL + bW * i + bW / 2}
          y={H - PB + 16}
          textAnchor="middle"
          style={{
            ...MONO,
            fontSize: '8px',
            fill: 'rgba(214,210,201,0.36)',
            letterSpacing: '0.09em',
          }}
        >
          {m}
        </text>
      ))}

      {metrics.map((_, i) => (
        <g key={i}>
          <Box d={cls[i]} i={i} side={0} color={MINT} delay={0.20 + i * 0.07} />
          <Box d={qnt[i]} i={i} side={1} color={VIOLET} delay={0.28 + i * 0.07} />
        </g>
      ))}

      {(
        [
          ['Classical', MINT],
          ['Quantum', VIOLET],
        ] as [string, string][]
      ).map(([lbl, col], li) => (
        <g key={lbl}>
          <rect
            x={W - PR - 74}
            y={PT + li * 16}
            width={9}
            height={9}
            fill={col}
            opacity={0.25}
          />
          <rect
            x={W - PR - 74}
            y={PT + li * 16}
            width={9}
            height={9}
            fill="none"
            stroke={col}
            strokeWidth={0.7}
            opacity={0.60}
          />
          <text
            x={W - PR - 60}
            y={PT + li * 16 + 8}
            style={{
              ...MONO,
              fontSize: '8px',
              fill: 'rgba(214,210,201,0.45)',
            }}
          >
            {lbl}
          </text>
        </g>
      ))}
    </svg>
  );
}

// ── Feature rows ──────────────────────────────────────────────────────────────
function FeatureRows({ v }: { v: boolean }) {
  const rows = [
    { f: 'Age', val: 0.21 },
    { f: 'BMI', val: 0.18 },
    { f: 'Glucose', val: 0.15 },
    { f: 'Blood Pressure', val: 0.12 },
    { f: 'Cholesterol', val: 0.10 },
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
        TOP CONTRIBUTING FEATURES
      </div>

      {rows.map((r, ri) => (
        <div
          key={r.f}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '8px 0',
            borderBottom: '1px solid rgba(214,210,201,0.06)',
            ...fadeUp(v, 0.35 + ri * 0.10),
          }}
        >
          <span
            style={{
              ...MONO,
              fontSize: '0.70rem',
              opacity: 0.52,
              minWidth: 108,
            }}
          >
            {r.f}
          </span>

          <div
            style={{
              height: 1,
              background: MINT,
              width: v ? `${r.val * 260}px` : '0px',
              opacity: 0.55,
              transition: `width 0.9s cubic-bezier(0.22,1,0.36,1) ${0.55 + ri * 0.10}s`,
            }}
          />

          <span
            style={{
              ...MONO,
              fontSize: '0.66rem',
              opacity: 0.36,
              minWidth: 28,
            }}
          >
            {r.val.toFixed(2)}
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

      {/* ── 03 CARDIOVASCULAR ────────────────────────────────────────────── */}
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

      {/* ── 04 BREAST CANCER ─────────────────────────────────────────────── */}
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

      {/* ── 05 MORE CONDITIONS ───────────────────────────────────────────── */}
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

      {/* ── 06 COMPUTATION ──────────────────────────────────────────────── */}
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

      {/* ── 06 CONVERGENCE ───────────────────────────────────────────────── */}
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

      {/* ── 07 BENCHMARK ─────────────────────────────────────────────────── */}
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
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: OFF_WHITE,
                    opacity: 0.28,
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

      {/* ── 08 EXPLAINABILITY ────────────────────────────────────────────── */}
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
                    document.getElementById('benchmarks')?.scrollIntoView({ behavior: 'smooth' })
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

      {/* ── 09 RESEARCH ──────────────────────────────────────────────────── */}
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
                  ['Evaluation', 'Cross-validation · Multiple Metrics'],
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

      {/* ── 10 FUTURE ────────────────────────────────────────────────────── */}
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
