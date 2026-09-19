import { useEffect, useRef, MutableRefObject } from 'react';
import * as THREE from 'three';

const N = 4500;

function gauss(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// fill helper — Box-Muller Gaussian cluster
function fillCluster(
  arr: Float32Array, idx: number,
  cx: number, cy: number, cz: number,
  rx: number, ry: number, rz: number,
  count: number,
): number {
  const end = Math.min(idx + count, N);
  while (idx < end) {
    arr[idx * 3]     = cx + gauss() * rx;
    arr[idx * 3 + 1] = cy + gauss() * ry;
    arr[idx * 3 + 2] = cz + gauss() * rz;
    idx++;
  }
  return idx;
}

// ── Phase generators ──────────────────────────────────────────────────────────

function genScattered(): Float32Array {
  const arr = new Float32Array(N * 3);
  const clusters = [
    { cx: -14, cy:  9, cz: -4, n: 700, sx: 7,   sy: 5,   sz: 5   },
    { cx:  12, cy: -5, cz:  4, n: 550, sx: 6,   sy: 4,   sz: 4   },
    { cx: -4,  cy:-12, cz: -6, n: 620, sx: 5,   sy: 4,   sz: 4   },
    { cx:  18, cy:  3, cz: -2, n: 450, sx: 4.5, sy: 3,   sz: 3.5 },
    { cx: -20, cy:  1, cz:  8, n: 380, sx: 4,   sy: 2.5, sz: 3   },
    { cx:   5, cy: 14, cz:  2, n: 320, sx: 3.5, sy: 3,   sz: 2.5 },
  ];
  let idx = 0;
  for (const c of clusters) {
    idx = fillCluster(arr, idx, c.cx, c.cy, c.cz, c.sx, c.sy, c.sz, c.n);
  }
  while (idx < N) {
    const r = 18 + Math.random() * 20;
    const phi   = Math.acos(2 * Math.random() - 1);
    const theta = Math.random() * Math.PI * 2;
    arr[idx * 3]     = r * Math.sin(phi) * Math.cos(theta) * 1.4;
    arr[idx * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.9;
    arr[idx * 3 + 2] = r * Math.cos(phi);
    idx++;
  }
  return arr;
}

function genFlowing(): Float32Array {
  const arr = new Float32Array(N * 3);
  const streams = 14;
  for (let i = 0; i < N; i++) {
    const t = i / N;
    const s = Math.floor(t * streams);
    const u = (t * streams) % 1;
    const offset = (s - streams / 2) * 2.0;
    arr[i * 3]     = (u - 0.5) * 55;
    arr[i * 3 + 1] = Math.sin(u * Math.PI * 3 + s * 0.55) * 4 + offset + gauss() * 0.4;
    arr[i * 3 + 2] = Math.cos(u * Math.PI * 2 + s * 0.3) * 2.5 + gauss() * 0.4;
  }
  return arr;
}

/**
 * Full-body X-ray skeleton: skull, vertebrae, ribcage, clavicles, scapulae,
 * pelvis, long bones, hands, and feet — all traced from anatomical bone outlines.
 * Particles follow bone edges; minimal filler keeps the X-ray feel intact.
 */
function genHumanSilhouette(): Float32Array {
  const arr = new Float32Array(N * 3);
  let idx = 0;

  // Drop a particle with small jitter; z compressed for flat X-ray look
  const pt = (x: number, y: number, jx = 0.09, jy = 0.06) => {
    if (idx >= N) return;
    arr[idx * 3]     = x + (Math.random() - 0.5) * jx;
    arr[idx * 3 + 1] = y + (Math.random() - 0.5) * jy;
    arr[idx * 3 + 2] = (Math.random() - 0.5) * 0.35;
    idx++;
  };

  // N particles along a straight bone shaft
  const bone = (x0: number, y0: number, x1: number, y1: number, n: number, w = 0.11) => {
    const dx = x1 - x0, dy = y1 - y0, len = Math.sqrt(dx*dx + dy*dy) || 1;
    const nx = -dy / len, ny = dx / len; // perpendicular
    for (let i = 0; i < n && idx < N; i++) {
      const t = n > 1 ? i / (n - 1) : 0;
      const r = (Math.random() - 0.5) * w;
      pt(x0 + dx*t + nx*r, y0 + dy*t + ny*r, 0, 0);
    }
  };

  // Elliptical arc from angle a0 to a1 (radians), N particles
  const arc = (cx: number, cy: number, rx: number, ry: number, a0: number, a1: number, n: number, w = 0.09) => {
    for (let i = 0; i < n && idx < N; i++) {
      const t = n > 1 ? i / (n - 1) : 0;
      const a = a0 + (a1 - a0) * t;
      const x = cx + rx * Math.cos(a);
      const y = cy + ry * Math.sin(a);
      const r = (Math.random() - 0.5) * w;
      pt(x + r * Math.cos(a), y + r * Math.sin(a), 0, 0);
    }
  };

  // Dense filled oval (for vertebral bodies etc.)
  const oval = (cx: number, cy: number, rx: number, ry: number, n: number) => {
    for (let i = 0; i < n && idx < N; i++) {
      const theta = Math.random() * Math.PI * 2;
      const r = Math.sqrt(Math.random());           // uniform disc
      pt(cx + r * rx * Math.cos(theta), cy + r * ry * Math.sin(theta), 0, 0);
    }
  };

  // ── SKULL ──────────────────────────────────────────────────────────────────
  // Cranium dome
  arc(0, 11.4, 1.76, 1.96, Math.PI * 0.08, Math.PI * 0.92, 200, 0.13);
  // Cranium base / temporal bone
  arc(0, 10.6, 1.82, 1.30, -Math.PI*0.06, Math.PI * 1.06, 70, 0.11);
  // Eye sockets (orbits)
  arc(-0.72, 10.45, 0.54, 0.40, 0, Math.PI * 2, 52, 0.08);
  arc( 0.72, 10.45, 0.54, 0.40, 0, Math.PI * 2, 52, 0.08);
  // Nasal cavity
  arc(0, 9.86, 0.30, 0.40, 0, Math.PI * 2, 30, 0.07);
  // Nasal bridge
  bone(0, 10.25, 0, 9.62, 14, 0.06);
  // Zygomatic arch (cheekbones)
  arc(-1.85, 10.15, 0.58, 0.26, -Math.PI*0.35, Math.PI*0.52, 32, 0.07);
  arc( 1.85, 10.15, 0.58, 0.26,  Math.PI*0.48, Math.PI*1.35, 32, 0.07);
  // Mandible (jaw arc)
  arc(0, 9.05, 1.28, 0.58, Math.PI * 0.06, Math.PI * 0.94, 90, 0.11);
  // Mandible ramus (vertical sides)
  bone(-1.24, 9.05, -1.58, 9.95, 28, 0.08);
  bone( 1.24, 9.05,  1.58, 9.95, 28, 0.08);
  // Teeth row suggestion
  bone(-0.88, 9.24, 0.88, 9.24, 22, 0.05);

  // ── VERTEBRAL COLUMN ───────────────────────────────────────────────────────
  // Cervical C1–C7 (small, narrow)
  for (let i = 0; i < 7 && idx < N; i++) {
    const y = 9.0 - i * 0.18;
    const xOff = Math.sin(i * 0.35) * 0.06;
    arc(xOff, y, 0.30, 0.10, 0, Math.PI * 2, 16, 0.06);
    pt(xOff + 0.38, y, 0.06, 0.04);
  }
  // Thoracic T1–T12
  for (let i = 0; i < 12 && idx < N; i++) {
    const y = 7.75 - i * 0.44;
    const xOff = -Math.sin(i * 0.22) * 0.05;
    arc(xOff, y, 0.40, 0.12, 0, Math.PI * 2, 18, 0.07);
    pt(xOff + 0.50, y, 0.06, 0.04);
    pt(xOff + 0.50, y + 0.10, 0.06, 0.03);
  }
  // Lumbar L1–L5 (larger bodies)
  for (let i = 0; i < 5 && idx < N; i++) {
    const y = 2.5 - i * 0.35;
    oval(0, y, 0.52, 0.14, 14);
    pt(0.60, y, 0.06, 0.04);
  }
  // Sacrum (fused triangular plate)
  for (let i = 0; i < 5 && idx < N; i++) {
    const y = 0.80 - i * 0.30;
    const hw = 0.54 - i * 0.08;
    bone(-hw, y, hw, y, Math.max(8, 12 - i*2), 0.07);
  }

  // ── RIBCAGE ────────────────────────────────────────────────────────────────
  // Sternum (manubrium + body)
  bone(0, 7.65, 0, 3.50, 60, 0.07);
  // Manubrium (top of sternum, slightly wider)
  arc(0, 7.7, 0.38, 0.20, Math.PI*0.1, Math.PI*0.9, 16, 0.08);

  // 12 rib pairs: arc from spine → outward → down to sternum
  // [spineY, maxReach, sternumY, arcDrop]
  const ribs: [number, number, number, number][] = [
    [7.55, 2.55, 7.45, 0.22],
    [7.10, 3.05, 7.05, 0.30],
    [6.65, 3.45, 6.65, 0.38],
    [6.20, 3.72, 6.20, 0.44],
    [5.75, 3.92, 5.75, 0.48],
    [5.30, 4.05, 5.30, 0.50],
    [4.85, 4.10, 4.85, 0.52],
    [4.40, 4.05, 4.40, 0.50],
    [3.95, 3.85, 3.95, 0.46],
    [3.50, 3.60,  0,   0   ],  // floating ribs (no sternum attachment)
    [3.05, 3.25,  0,   0   ],
    [2.60, 2.80,  0,   0   ],
  ];

  for (const [ys, xr, yster, drop] of ribs) {
    for (const side of [-1, 1]) {
      if (yster > 0) {
        // Rib arc: parametric curve from spine to sternum
        const nPts = 36;
        for (let i = 0; i < nPts && idx < N; i++) {
          const t = i / (nPts - 1);
          // x: goes out to xr at peak (t=0.5) then back to sternum edge
          const x = side * xr * Math.sin(t * Math.PI);
          // y: starts at ys, dips slightly at peak, rises to yster
          const y = ys + (yster - ys) * t + Math.sin(t * Math.PI) * drop;
          pt(x, y, 0.09, 0.05);
        }
      } else {
        // Floating rib — curves out and down
        bone(side * 0.25, ys, side * xr, ys - 0.35, 22, 0.08);
      }
    }
  }

  // ── CLAVICLES ─────────────────────────────────────────────────────────────
  for (const side of [-1, 1]) {
    const n = 38;
    for (let i = 0; i < n && idx < N; i++) {
      const t = i / (n - 1);
      const x = side * (0.18 + t * 3.22);
      const y = 7.92 - t * 0.42 + Math.sin(t * Math.PI) * 0.14;
      pt(x, y, 0.07, 0.05);
    }
  }

  // ── SCAPULAE (shoulder blades) ─────────────────────────────────────────────
  for (const side of [-1, 1]) {
    bone(side * 1.80, 7.42, side * 3.85, 7.00, 26, 0.09); // spine of scapula
    bone(side * 1.80, 7.42, side * 2.05, 5.15, 28, 0.08); // medial border
    bone(side * 3.85, 7.00, side * 2.40, 5.15, 24, 0.08); // lateral border
    arc(side * 3.78, 7.02, 0.32, 0.36, 0, Math.PI * 2, 20, 0.08); // glenoid
  }

  // ── PELVIS ─────────────────────────────────────────────────────────────────
  // Iliac crests (top rim — wide outward flare)
  for (const side of [-1, 1]) {
    arc(side * 1.85, -0.35, 2.05, 0.98, Math.PI*0.50, Math.PI*1.00, 55, 0.11);
    // Iliac blade fill (3 offset arcs)
    for (let r = 1; r <= 4; r++) {
      const s = 0.78 + r * 0.06;
      arc(side * 1.85, -0.35, 2.05*s, 0.98*s, Math.PI*0.52, Math.PI*0.98, 12, 0.09);
    }
    // Ischium (lower pelvis sides)
    arc(side * 1.80, -2.05, 0.80, 0.65, Math.PI*1.0, Math.PI*1.7, 28, 0.09);
  }
  // Pubic arch (bottom bridge)
  arc(0, -2.62, 1.58, 0.62, Math.PI*0.05, Math.PI*0.95, 58, 0.09);
  // Obturator foramen (two oval holes in pelvis)
  for (const side of [-1, 1]) {
    arc(side * 1.12, -1.88, 0.72, 0.50, 0, Math.PI * 2, 38, 0.08);
  }
  // Acetabulum (hip sockets)
  for (const side of [-1, 1]) {
    arc(side * 2.42, -2.38, 0.46, 0.46, 0, Math.PI * 2, 24, 0.08);
  }

  // ── LEGS ───────────────────────────────────────────────────────────────────
  for (const side of [-1, 1]) {
    const sx = side;
    // Femoral head (ball)
    arc(sx * 2.42, -2.50, 0.40, 0.40, 0, Math.PI * 2, 22, 0.08);
    // Femoral neck
    bone(sx * 2.42, -2.50, sx * 2.05, -3.22, 16, 0.10);
    // Greater trochanter
    arc(sx * 2.55, -3.10, 0.22, 0.18, Math.PI*0.4, Math.PI*1.6, 12, 0.07);
    // Femur shaft (angled inward to knee)
    bone(sx * 2.05, -3.22, sx * 1.58, -8.05, 110, 0.11);
    // Femoral condyles
    arc(sx * 1.48, -8.10, 0.32, 0.24, Math.PI*0.8, Math.PI*2.2, 18, 0.08);
    arc(sx * 1.68, -8.10, 0.32, 0.24, -Math.PI*0.2, Math.PI*1.0, 18, 0.08);
    // Patella
    arc(sx * 1.58, -8.18, 0.30, 0.34, 0, Math.PI * 2, 22, 0.08);
    // Tibia (main, medial)
    bone(sx * 1.44, -8.42, sx * 1.26, -13.05, 96, 0.11);
    // Tibial plateau (wider top)
    bone(sx * 1.08, -8.42, sx * 1.80, -8.42, 20, 0.07);
    // Fibula (thinner, lateral)
    bone(sx * 1.82, -8.62, sx * 1.62, -12.85, 72, 0.07);
    // Medial + lateral malleolus (ankle bumps)
    arc(sx * 1.22, -13.05, 0.20, 0.18, 0, Math.PI * 2, 12, 0.07);
    arc(sx * 1.64, -12.88, 0.20, 0.18, 0, Math.PI * 2, 12, 0.07);
    // Calcaneus (heel bone)
    arc(sx * 0.95, -13.28, 0.58, 0.26, Math.PI*0.85, Math.PI*2.15, 26, 0.09);
    // Metatarsals (5 rays from midfoot to toes)
    for (let toe = 0; toe < 5; toe++) {
      const fx = sx * (0.55 + toe * 0.28 * sx);
      bone(sx * 1.05, -13.26, fx, -13.68, 14, 0.06);
    }
    // Proximal phalanges (toe bases)
    for (let toe = 0; toe < 5; toe++) {
      const fx = sx * (0.55 + toe * 0.28 * sx);
      bone(fx, -13.68, fx, -13.92, 6, 0.06);
    }
  }

  // ── ARMS ───────────────────────────────────────────────────────────────────
  for (const side of [-1, 1]) {
    const sx = side;
    // Humeral head (ball at shoulder)
    arc(sx * 3.72, 7.02, 0.40, 0.40, 0, Math.PI * 2, 22, 0.09);
    // Humerus shaft
    bone(sx * 3.72, 6.88, sx * 5.22, 2.18, 92, 0.11);
    // Humeral epicondyles (elbow bumps)
    arc(sx * 5.15, 2.12, 0.28, 0.22, 0, Math.PI * 2, 16, 0.08);
    arc(sx * 5.30, 2.18, 0.22, 0.18, 0, Math.PI * 2, 14, 0.07);
    // Olecranon (bony elbow point)
    arc(sx * 4.92, 2.38, 0.18, 0.14, Math.PI*0.4, Math.PI*1.6, 12, 0.07);
    // Radius (lateral forearm)
    bone(sx * 5.22, 2.08, sx * 5.62, -1.82, 68, 0.10);
    // Ulna (medial forearm, slightly offset)
    bone(sx * 5.02, 2.30, sx * 5.38, -1.92, 64, 0.08);
    // Wrist carpal cluster
    for (let c = 0; c < 8 && idx < N; c++) {
      const wx = sx * (5.28 + (c % 4) * 0.20);
      const wy = -1.90 - Math.floor(c / 4) * 0.22;
      arc(wx, wy, 0.10, 0.10, 0, Math.PI * 2, 8, 0.06);
    }
    // Metacarpals (5 hand bones)
    for (let f = 0; f < 5 && idx < N; f++) {
      const mx = sx * (5.10 + f * 0.22 * sx);
      bone(mx, -2.10, mx, -2.90, 16, 0.07);
    }
    // Proximal phalanges
    for (let f = 0; f < 5 && idx < N; f++) {
      const mx = sx * (5.10 + f * 0.22 * sx);
      bone(mx, -2.92, mx, -3.32, 10, 0.06);
    }
  }

  // Minimal ambient scatter — just enough to not look empty
  while (idx < N) {
    arr[idx * 3]     = (Math.random() - 0.5) * 14;
    arr[idx * 3 + 1] = (Math.random() - 0.5) * 30;
    arr[idx * 3 + 2] = (Math.random() - 0.5) * 2;
    idx++;
  }
  return arr;
}

/** Abstract cardiovascular structure — branching topology, not a literal heart */
function genCardiovascular(): Float32Array {
  const arr = new Float32Array(N * 3);
  let idx = 0;

  // Dense central mass (myocardium region)
  idx = fillCluster(arr, idx, -1.5, 1.0, 0, 3.5, 2.8, 2.0, 700);

  // Aortic arc — sweeping upward and over
  const arcN = 500;
  for (let i = 0; i < arcN && idx < N; i++, idx++) {
    const t = (i / arcN) * Math.PI * 1.2 - 0.3;
    const r = 5.5 + gauss() * 0.5;
    arr[idx * 3]     = r * Math.cos(t) - 1 + gauss() * 0.4;
    arr[idx * 3 + 1] = r * Math.sin(t) + 1  + gauss() * 0.4;
    arr[idx * 3 + 2] = gauss() * 1.0;
  }

  // Pulmonary artery branches — fork left and right
  const branchData = [
    { startAngle: Math.PI * 0.55, sweep: 0.7, side:  1 },
    { startAngle: Math.PI * 0.55, sweep: 0.7, side: -1 },
  ];
  for (const b of branchData) {
    const bN = 280;
    for (let i = 0; i < bN && idx < N; i++, idx++) {
      const t = b.startAngle + (i / bN) * b.sweep;
      const r = 4.0 + (i / bN) * 3.0 + gauss() * 0.5;
      arr[idx * 3]     = b.side * Math.abs(Math.cos(t)) * r - 1 + gauss() * 0.5;
      arr[idx * 3 + 1] = Math.sin(t) * r * 0.5 + 2 + gauss() * 0.5;
      arr[idx * 3 + 2] = gauss() * 0.9;
    }
  }

  // Venous network — descending branches with narrowing
  for (let branch = 0; branch < 6 && idx < N; branch++) {
    const bN = 180;
    const angle = -0.4 + branch * 0.22;
    const startY = -1.5;
    for (let i = 0; i < bN && idx < N; i++, idx++) {
      const t = i / bN;
      arr[idx * 3]     = Math.sin(angle) * t * 8 + gauss() * (0.8 - t * 0.4);
      arr[idx * 3 + 1] = startY - t * 7  + gauss() * (0.6 - t * 0.3);
      arr[idx * 3 + 2] = gauss() * (1.0 - t * 0.5);
    }
  }

  // Sparse field
  while (idx < N) {
    arr[idx * 3]     = (Math.random() - 0.5) * 24;
    arr[idx * 3 + 1] = (Math.random() - 0.5) * 22;
    arr[idx * 3 + 2] = (Math.random() - 0.5) * 6;
    idx++;
  }
  return arr;
}

function genSplit(): Float32Array {
  const arr = new Float32Array(N * 3);
  const half = Math.floor(N / 2);

  for (let i = 0; i < half; i++) {
    const phi   = Math.acos(2 * Math.random() - 1);
    const theta = Math.random() * Math.PI * 2;
    const r = 5 + Math.random() * 4.5;
    arr[i * 3]     = -16 + r * Math.sin(phi) * Math.cos(theta) * 0.9;
    arr[i * 3 + 1] =       r * Math.sin(phi) * Math.sin(theta) * 0.9;
    arr[i * 3 + 2] =       r * Math.cos(phi) * 0.7;
  }
  for (let i = half; i < N; i++) {
    const j      = i - half;
    const rings  = 18;
    const perRing = Math.floor((N - half) / rings);
    const ring   = Math.floor(j / perRing);
    const inRing = j % perRing;
    const angle  = (inRing / perRing) * Math.PI * 2;
    const r = 1.5 + ring * 0.55;
    arr[i * 3]     = 16 + r * Math.cos(angle) + gauss() * 0.3;
    arr[i * 3 + 1] =      r * Math.sin(angle) * 0.8 + gauss() * 0.3;
    arr[i * 3 + 2] = (ring - rings / 2) * 0.5 + gauss() * 0.2;
  }
  return arr;
}

function genMerged(): Float32Array {
  const arr = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const phi   = Math.acos(2 * Math.random() - 1);
    const theta = Math.random() * Math.PI * 2;
    const r = 7 + Math.random() * 3.5 + gauss() * 0.6;
    arr[i * 3]     = r * Math.sin(phi) * Math.cos(theta) * 1.5;
    arr[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.6;
    arr[i * 3 + 2] = r * Math.cos(phi) * 0.9;
  }
  return arr;
}

function genGrid(): Float32Array {
  const arr = new Float32Array(N * 3);
  const cols = 90;
  const rows = Math.ceil(N / cols);
  for (let i = 0; i < N; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const noise = Math.sin(col * 0.4) * Math.cos(row * 0.3);
    arr[i * 3]     = (col / cols - 0.5) * 52;
    arr[i * 3 + 1] = (row / rows - 0.5) * 28;
    arr[i * 3 + 2] = noise * 4 + gauss() * 0.3;
  }
  return arr;
}

function genStreams(): Float32Array {
  const arr = new Float32Array(N * 3);
  const numStreams = 28;
  const perStream  = Math.floor(N / numStreams);
  for (let i = 0; i < N; i++) {
    const s = Math.min(Math.floor(i / perStream), numStreams - 1);
    const u = (i % perStream) / perStream;
    const amp = 1.2 + (s % 3) * 0.6;
    arr[i * 3]     = (u - 0.5) * 56;
    arr[i * 3 + 1] = (s / numStreams - 0.5) * 32 + Math.sin(u * Math.PI * 3 + s) * amp;
    arr[i * 3 + 2] = Math.cos(u * Math.PI * 2 + s * 0.5) * 1.2 + gauss() * 0.2;
  }
  return arr;
}

function genExpanded(): Float32Array {
  const arr = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const r     = 20 + Math.random() * 30;
    const phi   = Math.acos(2 * Math.random() - 1);
    const theta = Math.random() * Math.PI * 2;
    arr[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
    arr[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.7;
    arr[i * 3 + 2] = r * Math.cos(phi) * 0.75;
  }
  return arr;
}

function genCollapsed(): Float32Array {
  const arr = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const r     = Math.random() * 0.35;
    const phi   = Math.acos(2 * Math.random() - 1);
    const theta = Math.random() * Math.PI * 2;
    arr[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
    arr[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    arr[i * 3 + 2] = r * Math.cos(phi);
  }
  return arr;
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  scrollProgressRef: MutableRefObject<number>;
}

export default function ParticleScene({ scrollProgressRef }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mouseRef     = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(52, window.innerWidth / window.innerHeight, 0.1, 600);
    camera.position.set(0, 0, 45);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0b0d10, 1);
    container.appendChild(renderer.domElement);

    // ── particle attributes ───────────────────────────────────────────────────
    const positions = new Float32Array(N * 3);
    const colors    = new Float32Array(N * 3);
    const sizes     = new Float32Array(N);

    const palette = [
      { r: 0.36, g: 0.42, b: 0.48 },   // dim gray-blue   50%
      { r: 0.50, g: 0.57, b: 0.65 },   // medium gray-blue 22%
      { r: 0.72, g: 0.71, b: 0.68 },   // off-white        12%
      { r: 0.42, g: 0.72, b: 0.67 },   // muted mint        9%
      { r: 0.53, g: 0.49, b: 0.76 },   // muted violet      7%
    ];
    const palW = [0.50, 0.22, 0.12, 0.09, 0.07];

    for (let i = 0; i < N; i++) {
      let rnd = Math.random(), acc = 0, ci = 0;
      for (let w = 0; w < palW.length; w++) {
        acc += palW[w];
        if (rnd < acc) { ci = w; break; }
      }
      colors[i * 3]     = palette[ci].r;
      colors[i * 3 + 1] = palette[ci].g;
      colors[i * 3 + 2] = palette[ci].b;

      const tier = Math.random();
      if (tier < 0.60)      sizes[i] = 0.35 + Math.random() * 0.55;
      else if (tier < 0.90) sizes[i] = 1.1  + Math.random() * 0.9;
      else                  sizes[i] = 2.2  + Math.random() * 1.8;
    }

    // ── 10 phases ─────────────────────────────────────────────────────────────
    const phases = [
      genScattered(),        // 0  hero
      genFlowing(),          // 1  data
      genHumanSilhouette(),  // 2  human form
      genCardiovascular(),   // 3  biological structure
      genSplit(),            // 4  computation pathways
      genMerged(),           // 5  convergence
      genGrid(),             // 6  benchmarks
      genStreams(),           // 7  explainability
      genExpanded(),         // 8  research (pull back)
      genCollapsed(),        // 9  final (collapse)
    ];

    for (let i = 0; i < N * 3; i++) positions[i] = phases[0][i];

    // ── geometry + material ───────────────────────────────────────────────────
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aColor',   new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('aSize',    new THREE.BufferAttribute(sizes, 1));

    const vertexShader = /* glsl */ `
      attribute float aSize;
      attribute vec3  aColor;
      varying   vec3  vColor;
      varying   float vAlpha;
      uniform   float uTime;
      uniform   float uScroll;

      void main() {
        vColor = aColor;

        float breath = sin(uTime * 0.5 + position.x * 0.18 + position.z * 0.12) * 0.08 + 1.0;

        vec3 pos = position;
        float driftAmt = max(0.0, 1.0 - uScroll * 0.5);
        pos.x += sin(uTime * 0.22 + position.y * 0.15) * 0.3 * driftAmt;
        pos.y += cos(uTime * 0.18 + position.z * 0.12) * 0.22 * driftAmt;

        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        float dist = -mvPosition.z;
        float nearFade = smoothstep(1.0, 4.0, dist);
        float farFade  = smoothstep(95.0, 40.0, dist);
        vAlpha = nearFade * farFade;

        gl_PointSize  = aSize * breath * (240.0 / dist);
        gl_Position   = projectionMatrix * mvPosition;
      }
    `;

    const fragmentShader = /* glsl */ `
      varying vec3  vColor;
      varying float vAlpha;

      void main() {
        vec2  uv = gl_PointCoord - 0.5;
        float r  = length(uv) * 2.0;
        if (r > 1.0) discard;
        float alpha = (1.0 - smoothstep(0.55, 1.0, r)) * vAlpha * 0.88;
        gl_FragColor = vec4(vColor, alpha);
      }
    `;

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: { uTime: { value: 0 }, uScroll: { value: 0 } },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    // ── camera waypoints (one per phase) ─────────────────────────────────────
    // X offset biases the particle mass toward the side opposite the text:
    //   positive X → particles lean right  (text is LEFT-anchored)
    //   negative X → particles lean left   (text is RIGHT-anchored)
    const camPositions = [
      new THREE.Vector3( 14,  0,  45),  // 0 hero:         text left  → mass right
      new THREE.Vector3( -8, 10,  22),  // 1 data:         text right → mass left
      new THREE.Vector3( 10,  0,  46),  // 2 human:        text bottom-left → mass right, far enough to see full skeleton
      new THREE.Vector3( 14,  2,  26),  // 3 cardiovascular: text left → mass right
      new THREE.Vector3( 14, 18,  34),  // 4 split:        text left  → mass right
      new THREE.Vector3( -8,  2,  22),  // 5 merged:       text right → mass left
      new THREE.Vector3( -8,-10,  23),  // 6 grid:         text right → mass left
      new THREE.Vector3(  0,  2,  18),  // 7 streams:      text center → centered
      new THREE.Vector3( 14,  6,  58),  // 8 research:     text left  → mass right
      new THREE.Vector3(  0,  0,   8),  // 9 final:        centered
    ];

    const camLookAts = camPositions.map(() => new THREE.Vector3(0, 0, 0));
    // Look slightly right of center for phases where mass is pushed right
    camLookAts[0].set( 6, 0, 0);
    camLookAts[2].set( 4, 0, 0);  // look at chest-height center of figure
    camLookAts[3].set( 6, 0, 0);
    camLookAts[4].set( 6, 0, 0);
    camLookAts[5].set(-4, 0, 0);
    camLookAts[6].set(-4, 0, 0);
    camLookAts[8].set( 4, 0, 0);

    // ── animation loop ────────────────────────────────────────────────────────
    const targetBuf  = new Float32Array(N * 3);
    const camPos     = new THREE.Vector3(0, 0, 45);
    const camLook    = new THREE.Vector3(0, 0, 0);
    const tCamPos    = new THREE.Vector3();
    const tCamLook   = new THREE.Vector3();
    const maxPhase   = phases.length - 1;

    for (let i = 0; i < N * 3; i++) targetBuf[i] = phases[0][i];

    let animFrame: number;
    let time = 0, smoothScroll = 0;

    const animate = () => {
      animFrame = requestAnimationFrame(animate);
      time += 0.012;
      material.uniforms.uTime.value   = time;

      const raw = scrollProgressRef.current;
      smoothScroll += (raw - smoothScroll) * 0.045;

      const sc   = Math.max(0, Math.min(maxPhase, smoothScroll));
      const pA   = Math.min(Math.floor(sc), maxPhase - 1);
      const pB   = pA + 1;
      const t    = sc - pA;
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

      material.uniforms.uScroll.value = sc;

      // Lerp particle targets between phases
      const posA = phases[pA], posB = phases[pB];
      for (let i = 0; i < N * 3; i++) {
        targetBuf[i] = posA[i] + (posB[i] - posA[i]) * ease;
      }

      const speed = 0.02 + Math.abs(raw - smoothScroll) * 0.25;
      const mx = mouseRef.current.x;
      const my = mouseRef.current.y;

      for (let i = 0; i < N; i++) {
        const ix = i * 3, iy = ix + 1, iz = ix + 2;
        positions[ix] += (targetBuf[ix] - positions[ix]) * speed;
        positions[iy] += (targetBuf[iy] - positions[iy]) * speed;
        positions[iz] += (targetBuf[iz] - positions[iz]) * speed;

        const dx = positions[ix] - mx * 12;
        const dy = positions[iy] - my *  8;
        const d2 = dx * dx + dy * dy;
        if (d2 < 25) {
          const f = (1 - Math.sqrt(d2) / 5) * 0.05;
          positions[ix] += dx * f;
          positions[iy] += dy * f;
        }
      }
      geometry.attributes.position.needsUpdate = true;

      // Camera
      const cA = camPositions[pA];
      const cB = camPositions[Math.min(pB, camPositions.length - 1)];
      tCamPos.lerpVectors(cA, cB, ease);
      tCamPos.x += mx * 1.0;  // reduced: don't drag mass back over text
      tCamPos.y += my * 1.2;
      camPos.lerp(tCamPos, 0.038);
      camera.position.copy(camPos);

      const lA = camLookAts[pA];
      const lB = camLookAts[Math.min(pB, camLookAts.length - 1)];
      tCamLook.lerpVectors(lA, lB, ease);
      tCamLook.x += mx * 0.3;
      tCamLook.y += my * 0.3;
      camLook.lerp(tCamLook, 0.04);
      camera.lookAt(camLook);

      renderer.render(scene, camera);
    };

    animate();

    let smoothMx = 0, smoothMy = 0;
    const handleMouse = (e: MouseEvent) => {
      const tx = (e.clientX / window.innerWidth  - 0.5) * 2;
      const ty = (e.clientY / window.innerHeight - 0.5) * -2;
      smoothMx += (tx - smoothMx) * 0.06;
      smoothMy += (ty - smoothMy) * 0.06;
      mouseRef.current.x = smoothMx;
      mouseRef.current.y = smoothMy;
    };

    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('mousemove', handleMouse);
    window.addEventListener('resize',    onResize);

    return () => {
      cancelAnimationFrame(animFrame);
      window.removeEventListener('mousemove', handleMouse);
      window.removeEventListener('resize',    onResize);
      renderer.dispose();
      geometry.dispose();
      material.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [scrollProgressRef]);

  return (
    <div
      ref={containerRef}
      style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none' }}
    />
  );
}
