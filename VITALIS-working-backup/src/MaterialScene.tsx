import { useEffect, useRef, MutableRefObject } from 'react';
import * as THREE from 'three';

interface Props {
  scrollProgressRef: MutableRefObject<number>;
}

export default function MaterialScene({ scrollProgressRef }: Props) {
  const mountRef  = useRef<HTMLDivElement>(null);
  const mouseRef  = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25));
    renderer.setClearColor(0x0b0d10, 1);
    el.appendChild(renderer.domElement);

    const scene  = new THREE.Scene();
    const ortho  = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // ── Uniforms ──────────────────────────────────────────────────────────────
    const uniforms = {
      uTime:       { value: 0 },
      uScroll:     { value: 0 },
      uMouse:      { value: new THREE.Vector2() },
      uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
      uCamPos:     { value: new THREE.Vector3(0, 0, 6.5) },
      uCamTarget:  { value: new THREE.Vector3(0, 0, 0)   },
    };

    // ── Shaders ───────────────────────────────────────────────────────────────
    const vert = /* glsl */`void main(){ gl_Position = vec4(position.xy, 0.0, 1.0); }`;

    const frag = /* glsl */`
      precision highp float;

      uniform float uTime;
      uniform float uScroll;
      uniform vec2  uMouse;
      uniform vec2  uResolution;
      uniform vec3  uCamPos;
      uniform vec3  uCamTarget;

      // ── SDF primitives ─────────────────────────────────────────────────────

      float sdSphere(vec3 p, float r){ return length(p) - r; }

      float sdCapsule(vec3 p, vec3 a, vec3 b, float r){
        vec3 pa = p - a, ba = b - a;
        float h = clamp(dot(pa,ba)/dot(ba,ba), 0.0, 1.0);
        return length(pa - ba*h) - r;
      }

      float sdTorus(vec3 p, vec2 t){
        return length(vec2(length(p.xz)-t.x, p.y)) - t.y;
      }

      float sdBox(vec3 p, vec3 b){
        vec3 q = abs(p) - b;
        return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
      }

      float smin(float a, float b, float k){
        float h = clamp(0.5+0.5*(b-a)/k, 0.0, 1.0);
        return mix(b,a,h) - k*h*(1.0-h);
      }

      // ── Noise ──────────────────────────────────────────────────────────────

      float hash(vec3 p){
        p = fract(p*vec3(0.1031,0.1030,0.0973));
        p += dot(p,p.yxz+33.33);
        return fract((p.x+p.y)*p.z);
      }

      float n3(vec3 p){
        vec3 i=floor(p), f=fract(p);
        f=f*f*(3.0-2.0*f);
        return mix(
          mix(mix(hash(i),          hash(i+vec3(1,0,0)),f.x),
              mix(hash(i+vec3(0,1,0)), hash(i+vec3(1,1,0)),f.x),f.y),
          mix(mix(hash(i+vec3(0,0,1)), hash(i+vec3(1,0,1)),f.x),
              mix(hash(i+vec3(0,1,1)), hash(i+vec3(1,1,1)),f.x),f.y), f.z);
      }

      float fbm(vec3 p){
        return 0.500*n3(p) + 0.250*n3(p*2.03) + 0.125*n3(p*4.11);
      }

      // ── Phase SDFs (0‑9) ───────────────────────────────────────────────────

      // 0 — RAW DATA: amorphous blob
      float sdf0(vec3 p){
        float d = length(p) - 2.10;
        d += 0.48*sin(p.x*2.7+uTime*0.52)*sin(p.y*2.4+uTime*0.40)*sin(p.z*3.1+uTime*0.36);
        d -= 0.14*fbm(p*1.1 + uTime*0.07);
        return d;
      }

      // Ellipsoid SDF helper (non-uniform scaling trick)
      float sdEllipsoid(vec3 p, vec3 r){
        float k0 = length(p/r);
        float k1 = length(p/(r*r));
        return k0*(k0-1.0)/k1;
      }

      // 1 — HUMAN FORM: anatomically segmented body
      float sdf1(vec3 p){
        // Blend radii: tighter joins for fine detail, looser for large masses
        float kL = 0.20;  // loose (torso masses)
        float kM = 0.13;  // medium (limb-to-body)
        float kS = 0.08;  // tight (forearm/calf joints)

        // ── Head: oval (slightly taller than wide, flatter front-back) ────────
        float head = sdEllipsoid(p - vec3(0.0, 2.92, 0.0), vec3(0.60, 0.72, 0.56));

        // ── Neck: narrow column ────────────────────────────────────────────────
        float neck = sdCapsule(p, vec3(0.0, 2.32, 0.0), vec3(0.0, 2.58, 0.0), 0.19);

        // ── Torso: chest (broad) → waist (narrow) → hips (flare) ─────────────
        float chest = sdCapsule(p, vec3(0.0, 1.25, 0.0), vec3(0.0, 2.10, 0.0), 0.74);
        float waist = sdCapsule(p, vec3(0.0, 0.30, 0.0), vec3(0.0, 1.30, 0.0), 0.58);
        float hips  = sdCapsule(p, vec3(0.0,-0.58, 0.0), vec3(0.0, 0.35, 0.0), 0.76);

        // ── Shoulder caps (deltoids) ──────────────────────────────────────────
        float ldelt = sdEllipsoid(p - vec3(-0.96, 1.88, 0.0), vec3(0.34, 0.28, 0.28));
        float rdelt = sdEllipsoid(p - vec3( 0.96, 1.88, 0.0), vec3(0.34, 0.28, 0.28));

        // ── Upper arms ────────────────────────────────────────────────────────
        float lua = sdCapsule(p, vec3(-0.96, 1.75, 0.0), vec3(-1.50, 0.62, 0.04), 0.21);
        float rua = sdCapsule(p, vec3( 0.96, 1.75, 0.0), vec3( 1.50, 0.62, 0.04), 0.21);

        // ── Forearms (angle slightly outward below elbow) ─────────────────────
        float lfa = sdCapsule(p, vec3(-1.50, 0.62, 0.04), vec3(-1.72,-0.42, 0.08), 0.17);
        float rfa = sdCapsule(p, vec3( 1.50, 0.62, 0.04), vec3( 1.72,-0.42, 0.08), 0.17);

        // ── Hands ─────────────────────────────────────────────────────────────
        float lhand = sdEllipsoid(p - vec3(-1.78,-0.54, 0.08), vec3(0.19, 0.22, 0.10));
        float rhand = sdEllipsoid(p - vec3( 1.78,-0.54, 0.08), vec3(0.19, 0.22, 0.10));

        // ── Thighs ────────────────────────────────────────────────────────────
        float lth = sdCapsule(p, vec3(-0.40,-0.58, 0.0), vec3(-0.34,-1.92, 0.02), 0.36);
        float rth = sdCapsule(p, vec3( 0.40,-0.58, 0.0), vec3( 0.34,-1.92, 0.02), 0.36);

        // ── Calves (narrower below knee, slight forward taper) ────────────────
        float lca = sdCapsule(p, vec3(-0.34,-1.92, 0.02), vec3(-0.29,-3.10, 0.06), 0.24);
        float rca = sdCapsule(p, vec3( 0.34,-1.92, 0.02), vec3( 0.29,-3.10, 0.06), 0.24);

        // ── Ankles + feet (flat ellipsoid extending forward) ──────────────────
        float lank = sdCapsule(p, vec3(-0.29,-3.10, 0.06), vec3(-0.27,-3.30, 0.08), 0.15);
        float rank = sdCapsule(p, vec3( 0.29,-3.10, 0.06), vec3( 0.27,-3.30, 0.08), 0.15);
        float lfoot = sdEllipsoid(p - vec3(-0.18,-3.34, 0.24), vec3(0.16, 0.12, 0.28));
        float rfoot = sdEllipsoid(p - vec3( 0.18,-3.34, 0.24), vec3(0.16, 0.12, 0.28));

        // ── Smooth union assembly ─────────────────────────────────────────────
        float d = smin(head,  neck,  kM);
        d = smin(d, chest,  kL);
        d = smin(d, waist,  kL);
        d = smin(d, hips,   kL);
        d = smin(d, ldelt,  kM);
        d = smin(d, rdelt,  kM);
        d = smin(d, lua,    kM);
        d = smin(d, rua,    kM);
        d = smin(d, lfa,    kS);
        d = smin(d, rfa,    kS);
        d = smin(d, lhand,  kS);
        d = smin(d, rhand,  kS);
        d = smin(d, lth,    kM);
        d = smin(d, rth,    kM);
        d = smin(d, lca,    kS);
        d = smin(d, rca,    kS);
        d = smin(d, lank,   kS);
        d = smin(d, rank,   kS);
        d = smin(d, lfoot,  kS);
        d = smin(d, rfoot,  kS);

        // Fine surface variation — skin texture, very low amplitude
        d += 0.012*sin(p.x*11.0+uTime*0.18)*sin(p.y*10.5+uTime*0.14)*sin(p.z*9.0+uTime*0.10);
        return d;
      }

      // 2 — CARDIOVASCULAR: heart with branching vessels
      float sdf2(vec3 p){
        // Cardiac chambers — two overlapping lobes (right + left ventricle-like)
        float right = sdEllipsoid(p - vec3(-0.38, 0.0, 0.0), vec3(0.78, 0.95, 0.62));
        float left  = sdEllipsoid(p - vec3( 0.32, 0.08, 0.0), vec3(0.72, 0.90, 0.58));
        float core  = smin(right, left, 0.32);
        // Apex (bottom point of heart)
        float apex  = sdEllipsoid(p - vec3(-0.04, -1.10, 0.0), vec3(0.32, 0.44, 0.30));
        core = smin(core, apex, 0.28);
        // Surface pulsation — subtle rhythmic deformation
        core += 0.10*sin(p.x*5.2+uTime*1.20)*sin(p.y*4.8+uTime*0.95);
        core += 0.06*sin(p.x*9.0+uTime*1.60)*sin(p.z*8.5+uTime*1.40);

        // Aortic arch — main outflow tract rising then arching
        float aorta = sdCapsule(p, vec3( 0.18, 0.82, 0.0), vec3( 0.18, 1.70, 0.0), 0.18);
        float arch1 = sdCapsule(p, vec3( 0.18, 1.70, 0.0), vec3( 0.60, 1.92, 0.0), 0.16);
        float arch2 = sdCapsule(p, vec3( 0.60, 1.92, 0.0), vec3( 0.85, 1.80, 0.0), 0.15);
        float arch3 = sdCapsule(p, vec3( 0.85, 1.80, 0.0), vec3( 0.90, 1.30, 0.0), 0.13);
        // Pulmonary trunk (left side)
        float pulm  = sdCapsule(p, vec3(-0.18, 0.88, 0.0), vec3(-0.50, 1.55, 0.0), 0.16);
        float pulm1 = sdCapsule(p, vec3(-0.50, 1.55, 0.0), vec3(-1.10, 1.70, 0.0), 0.12);
        float pulm2 = sdCapsule(p, vec3(-0.50, 1.55, 0.0), vec3(-0.90, 1.18, 0.0), 0.10);
        // Coronary arteries — branching over surface
        float cor1  = sdCapsule(p, vec3( 0.10, 0.18, 0.62), vec3( 0.80, -0.50, 0.52), 0.07);
        float cor2  = sdCapsule(p, vec3( 0.80,-0.50, 0.52), vec3( 0.65,-1.05, 0.35), 0.055);
        float cor3  = sdCapsule(p, vec3(-0.10, 0.15, 0.62), vec3(-0.70,-0.45, 0.50), 0.065);
        float cor4  = sdCapsule(p, vec3(-0.70,-0.45, 0.50), vec3(-0.55,-0.95, 0.30), 0.048);
        // Inferior vena cava
        float ivc   = sdCapsule(p, vec3(-0.45,-0.80, 0.0), vec3(-0.50,-2.10, 0.0), 0.10);

        float d = smin(core, aorta, 0.18);
        d = smin(d, arch1, 0.14); d = smin(d, arch2, 0.12); d = smin(d, arch3, 0.10);
        d = smin(d, pulm,  0.16); d = smin(d, pulm1, 0.12); d = smin(d, pulm2, 0.10);
        d = min(d, cor1); d = min(d, cor2); d = min(d, cor3); d = min(d, cor4);
        d = smin(d, ivc, 0.12);
        return d;
      }

      // 3 — BREAST CANCER: microscopic cancer cell cluster
      float sdf3(vec3 p){
        // Dominant irregular cancer cell — lumpy, non-spherical
        float main_cell = sdSphere(p - vec3(0.0, 0.15, 0.0), 0.90);
        main_cell += 0.32*sin(p.x*4.8+uTime*0.30)*sin(p.y*3.9+uTime*0.22)*sin(p.z*4.3+uTime*0.18);
        main_cell += 0.14*sin(p.x*8.5+uTime*0.45)*sin(p.z*7.8+uTime*0.38);
        main_cell -= 0.10*fbm(p*2.4 + uTime*0.04);

        // Surrounding cells — varied sizes, irregular surfaces
        float c1 = sdSphere(p - vec3( 1.22,  0.60,  0.28), 0.44);
        c1 += 0.16*sin(p.x*5.5+uTime*0.40)*sin(p.y*4.8+uTime*0.32);

        float c2 = sdSphere(p - vec3(-1.12, -0.42,  0.58), 0.40);
        c2 += 0.14*sin(p.x*6.0+uTime*0.36)*sin(p.z*5.2+uTime*0.28);

        float c3 = sdSphere(p - vec3( 0.58, -1.08, -0.42), 0.36);
        c3 += 0.12*sin(p.y*6.4+uTime*0.44)*sin(p.z*5.8+uTime*0.30);

        float c4 = sdSphere(p - vec3(-0.62,  0.92, -0.65), 0.30);
        c4 += 0.10*sin(p.x*7.2+uTime*0.38)*sin(p.y*6.5+uTime*0.28);

        float c5 = sdSphere(p - vec3( 1.55, -0.68,  0.22), 0.26);
        float c6 = sdSphere(p - vec3(-1.38,  0.40, -0.42), 0.22);

        // Membrane filaments connecting cells (thin capsules)
        float m1 = sdCapsule(p, vec3(0.84, 0.15, 0.0), vec3(1.22, 0.60, 0.28), 0.055);
        float m2 = sdCapsule(p, vec3(-0.82, 0.15, 0.0), vec3(-1.12, -0.42, 0.58), 0.048);
        float m3 = sdCapsule(p, vec3(0.0, -0.72, 0.0), vec3(0.58, -1.08, -0.42), 0.045);
        float m4 = sdCapsule(p, vec3(-0.60, 0.15, 0.0), vec3(-0.62, 0.92, -0.65), 0.040);

        float d = smin(main_cell, c1, 0.14);
        d = smin(d, c2, 0.13);
        d = smin(d, c3, 0.12);
        d = smin(d, c4, 0.11);
        d = smin(d, c5, 0.09);
        d = smin(d, c6, 0.08);
        d = min(d, m1);
        d = min(d, m2);
        d = min(d, m3);
        d = min(d, m4);
        return d;
      }

      // 4 — COMPUTATION: two pathways (structured slab + deformed sphere)
      float sdf4(vec3 p){
        // Classical: ordered geometric slab + sphere cap — crisp
        vec3 lp = p - vec3(-1.60, 0.0, 0.0);
        float classical = sdBox(lp, vec3(0.62, 1.38, 0.28));
        classical = smin(classical, sdSphere(p - vec3(-1.60, 1.52, 0.0), 0.40), 0.28);

        // Quantum: deformed organic blob — more complex topology
        vec3 qp = p - vec3(1.60, 0.0, 0.0);
        float quantum = sdSphere(qp, 0.82);
        quantum += 0.34*sin(qp.x*4.2+uTime*0.88)*sin(qp.y*3.6+uTime*0.72)*sin(qp.z*4.8+uTime*0.65);
        quantum -= 0.12*fbm(qp*2.1 + uTime*0.10);

        return min(classical, quantum);
      }

      // 5 — CONVERGENCE: two forms merging into one
      float sdf5(vec3 p){
        float a = sdSphere(p - vec3(-0.80, 0.0, 0.0), 1.08);
        float b = sdSphere(p - vec3( 0.80, 0.0, 0.0), 1.08);
        float d = smin(a, b, 0.60);
        d += 0.18*sin(p.x*1.9+uTime*0.48)*sin(p.y*1.7+uTime*0.38);
        return d;
      }

      // 6 — BENCHMARK: data surface — torus + radial spokes
      float sdf6(vec3 p){
        float disc = sdTorus(p, vec2(1.82, 0.20));
        disc -= 0.07*n3(p*4.5 + uTime*0.15);
        // 6 data spokes for the 6 metrics
        float spokes = 99.0;
        for(int i=0;i<6;i++){
          float a = float(i)*(3.14159/3.0) + uTime*0.04;
          vec3 dir = vec3(cos(a), 0.0, sin(a));
          spokes = min(spokes, length(p - dir*clamp(dot(p,dir),0.0,1.85)) - 0.045);
        }
        // Center hub
        float hub = sdSphere(p, 0.22);
        return smin(min(disc, spokes), hub, 0.10);
      }

      // 7 — EXPLAINABILITY: core + orbiting signal nodes
      float sdf7(vec3 p){
        float core = sdSphere(p, 0.82);
        float t1 = uTime*0.38, t2 = uTime*0.52, t3 = uTime*0.28;
        float s1 = sdSphere(p - vec3(cos(t1)*1.58, sin(t1)*0.48, sin(t1)*0.98), 0.26);
        float s2 = sdSphere(p - vec3(cos(t2+2.09)*1.38, sin(t2+2.09)*0.78, cos(t2+2.09)*0.88), 0.20);
        float s3 = sdSphere(p - vec3(cos(t3+4.19)*1.18, 0.88, sin(t3+4.19)*1.25), 0.16);
        // Connective filaments
        float c1 = sdCapsule(p, vec3(0,0,0), vec3(cos(t1)*1.58,sin(t1)*0.48,sin(t1)*0.98), 0.03);
        float c2 = sdCapsule(p, vec3(0,0,0), vec3(cos(t2+2.09)*1.38,sin(t2+2.09)*0.78,cos(t2+2.09)*0.88), 0.03);
        float k = 0.16;
        float d = smin(core, s1, k);
        d = smin(d, s2, k); d = smin(d, s3, k);
        d = min(d, c1); d = min(d, c2);
        return d;
      }

      // 8 — RESEARCH: expanded sparse constellation of small volumes
      float sdf8(vec3 p){
        float d = 99.0;
        d = min(d, sdSphere(p - vec3( 2.50,  0.95, -0.90), 0.48));
        d = min(d, sdSphere(p - vec3(-2.20,  0.45,  0.45), 0.38));
        d = min(d, sdSphere(p - vec3( 0.45, -1.95,  0.78), 0.32));
        d = min(d, sdSphere(p - vec3(-0.95,  2.18, -0.48), 0.40));
        d = min(d, sdSphere(p - vec3( 1.78, -1.45,  1.18), 0.28));
        d = min(d, sdCapsule(p, vec3(2.5,0.95,-0.9),  vec3(-2.2,0.45,0.45),  0.04));
        d = min(d, sdCapsule(p, vec3(-2.2,0.45,0.45), vec3(0.45,-1.95,0.78), 0.04));
        d = min(d, sdCapsule(p, vec3(0.45,-1.95,0.78),vec3(1.78,-1.45,1.18), 0.04));
        return d;
      }

      // 9 — FINAL: collapsing to a point
      float sdf9(vec3 p){
        float pulse = 0.035*sin(uTime*2.20);
        return sdSphere(p, 0.06 + pulse);
      }

      // ── Morphing scene SDF ─────────────────────────────────────────────────

      float scene(vec3 p){
        // Loose bounding check — cheap early exit
        if(length(p) > 5.5) return length(p) - 5.0;

        float sc  = clamp(uScroll, 0.0, 8.999);
        float ph  = floor(sc);
        float t   = fract(sc);
        float e   = t<0.5 ? 2.0*t*t : 1.0-2.0*(1.0-t)*(1.0-t); // ease-in-out

        float d0, d1;
        if(ph < 0.5){      d0=sdf0(p); d1=sdf1(p);
        } else if(ph<1.5){ d0=sdf1(p); d1=sdf2(p);
        } else if(ph<2.5){ d0=sdf2(p); d1=sdf3(p);
        } else if(ph<3.5){ d0=sdf3(p); d1=sdf4(p);
        } else if(ph<4.5){ d0=sdf4(p); d1=sdf5(p);
        } else if(ph<5.5){ d0=sdf5(p); d1=sdf6(p);
        } else if(ph<6.5){ d0=sdf6(p); d1=sdf7(p);
        } else if(ph<7.5){ d0=sdf7(p); d1=sdf8(p);
        } else {           d0=sdf8(p); d1=sdf9(p); }

        return mix(d0, d1, e);
      }

      // ── Normal (tetrahedron method — 4 evals) ─────────────────────────────

      vec3 calcN(vec3 p){
        const float h = 0.003;
        const vec2 k = vec2(1.0,-1.0);
        return normalize(
          k.xyy*scene(p+k.xyy*h) + k.yyx*scene(p+k.yyx*h) +
          k.yxy*scene(p+k.yxy*h) + k.xxx*scene(p+k.xxx*h));
      }

      // ── Soft AO (5 steps) ─────────────────────────────────────────────────

      float ao(vec3 p, vec3 n){
        float a=0.0, s=0.18;
        for(int i=1;i<=5;i++){
          float dist=float(i)*s;
          a += max(0.0, dist-scene(p+n*dist))/dist;
        }
        return clamp(1.0-a*0.30, 0.0, 1.0);
      }

      // ── Phase material color ───────────────────────────────────────────────

      vec3 matColor(){
        float sc = uScroll;
        vec3 cold   = vec3(0.24, 0.34, 0.48);   // 0 raw data: deep cool
        vec3 teal   = vec3(0.28, 0.60, 0.62);   // 1 human: teal/cyan
        vec3 coral  = vec3(0.72, 0.32, 0.26);   // 2 cardio: warm coral/red
        vec3 violet = vec3(0.52, 0.26, 0.72);   // 3 cancer cells: violet/magenta
        vec3 comp   = vec3(0.38, 0.50, 0.66);   // 4-6 computation: blue-mint
        vec3 dim    = vec3(0.22, 0.30, 0.40);   // 7-9 explain/research/final: dimming

        vec3 c = cold;
        // 0→1: cold to teal (raw data → human body)
        c = mix(c, teal,   smoothstep(0.0, 1.0, sc));
        // 1→2: teal to coral (human → cardiovascular)
        c = mix(c, coral,  smoothstep(1.2, 2.0, sc) * (1.0-smoothstep(2.7, 3.2, sc)));
        // 2→3: coral to violet (cardio → cancer cells)
        c = mix(c, violet, smoothstep(2.6, 3.2, sc) * (1.0-smoothstep(3.8, 4.3, sc)));
        // 3→4-6: violet to comp blue (cancer → computation/convergence/benchmark)
        c = mix(c, comp,   smoothstep(3.8, 4.8, sc) * (1.0-smoothstep(6.5, 7.5, sc)));
        // 6→9: comp to dim (explain, research, final)
        c = mix(c, dim,    smoothstep(6.5, 9.0, sc));
        return c;
      }

      // ── Raymarcher (returns hit dist + min SDF for glow) ──────────────────

      vec2 march(vec3 ro, vec3 rd){
        float t=0.08, minD=99.0;
        for(int i=0;i<72;i++){
          vec3 p = ro + rd*t;
          float d = scene(p);
          minD = min(minD, d);
          if(d < 0.003) return vec2(t, minD);
          if(t > 20.0) break;
          t += max(d, 0.008) * 0.88;
        }
        return vec2(-1.0, minD);
      }

      // ── Main ──────────────────────────────────────────────────────────────

      void main(){
        vec2 uv = (gl_FragCoord.xy/uResolution - 0.5) * vec2(uResolution.x/uResolution.y, 1.0);

        // Camera basis
        vec3 ro = uCamPos;
        vec3 cw = normalize(uCamTarget - ro);
        vec3 cu = normalize(cross(cw, vec3(0,1,0)));
        vec3 cv = cross(cu, cw);
        float fov = 1.55;
        vec3 rd = normalize(uv.x*cu + uv.y*cv + fov*cw);

        // Subtle mouse tilt
        rd = normalize(rd + uMouse.x*0.05*cu + uMouse.y*0.04*cv);

        vec3 bg = vec3(0.043, 0.051, 0.063);
        vec2 res = march(ro, rd);
        float hit = res.x, minD = res.y;

        vec3 col = bg;

        // Glow halo from near-misses
        float glow = exp(-max(0.0, minD)*5.5) * 0.30;

        if(hit > 0.0){
          vec3 p = ro + rd*hit;
          vec3 n = calcN(p);
          float occ = ao(p, n);

          // Key light — upper-left front
          vec3 ldir = normalize(vec3(-0.55, 0.80, 0.50));
          float diff = max(0.0, dot(n, ldir));
          float spec = pow(max(0.0, dot(reflect(-ldir,n), -rd)), 48.0);

          // Fresnel rim (glassy/translucent edge)
          float fr = pow(1.0-abs(dot(n,-rd)), 2.8);

          vec3 mc  = matColor();
          vec3 rim = mix(mc*2.0, vec3(0.86,0.84,0.80), 0.35);

          // Dark body, bright rim + subtle diffuse + specular highlight
          col  = bg * 0.35;
          col += mc  * 0.14 * diff;
          col += vec3(0.85,0.83,0.80) * spec * 0.55;
          col += rim * fr * 0.75;
          col *= occ;
          // Thin-film subsurface warmth on shadowed side
          col += mc * 0.07 * (1.0-diff);
        }

        // Additive glow halo
        col += matColor() * 1.35 * glow * glow;

        // Vignette
        col *= 1.0 - 0.52*dot(uv*0.75, uv*0.75);

        // Filmic tone + gamma
        col  = col/(col + vec3(0.52));
        col  = pow(max(col, 0.0), vec3(0.44));

        gl_FragColor = vec4(col, 1.0);
      }
    `;

    const mat  = new THREE.ShaderMaterial({ vertexShader: vert, fragmentShader: frag, uniforms });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
    scene.add(mesh);

    // ── Camera waypoints per phase ─────────────────────────────────────────
    // X is negative → material appears right of center (opposite text which is left-anchored)
    // X is positive → material appears left of center (for right-anchored text sections)
    const waypoints = [
      { pos: new THREE.Vector3(-1.20,  0.00,  6.8), tgt: new THREE.Vector3(0,  0.00, 0) }, // 0 data/hero
      { pos: new THREE.Vector3(-1.00, -0.20,  7.2), tgt: new THREE.Vector3(0, -0.35, 0) }, // 1 human — full body view
      { pos: new THREE.Vector3(-0.80,  0.18,  5.8), tgt: new THREE.Vector3(0,  0.10, 0) }, // 2 cardiovascular
      { pos: new THREE.Vector3( 0.90,  0.12,  5.4), tgt: new THREE.Vector3(0,  0.10, 0) }, // 3 cancer cells — right side
      { pos: new THREE.Vector3( 0.00,  0.00,  7.0), tgt: new THREE.Vector3(0,  0.00, 0) }, // 4 computation split
      { pos: new THREE.Vector3( 1.10,  0.00,  5.6), tgt: new THREE.Vector3(0,  0.00, 0) }, // 5 convergence
      { pos: new THREE.Vector3(-0.80, -0.55,  7.8), tgt: new THREE.Vector3(0, -0.20, 0) }, // 6 benchmark
      { pos: new THREE.Vector3(-0.70,  0.00,  5.5), tgt: new THREE.Vector3(0,  0.00, 0) }, // 7 explainability
      { pos: new THREE.Vector3(-0.50,  0.40, 11.5), tgt: new THREE.Vector3(0,  0.00, 0) }, // 8 research
      { pos: new THREE.Vector3( 0.00,  0.00,  3.0), tgt: new THREE.Vector3(0,  0.00, 0) }, // 9 future
    ];

    const camPos  = new THREE.Vector3(0, 0, 6.8);
    const camTgt  = new THREE.Vector3();
    const tPos    = new THREE.Vector3();
    const tTgt    = new THREE.Vector3();

    let raf: number;
    let time = 0, smoothScroll = 0;

    const animate = () => {
      raf = requestAnimationFrame(animate);
      time += 0.012;
      uniforms.uTime.value = time;

      const raw = scrollProgressRef.current;
      smoothScroll += (raw - smoothScroll) * 0.042;

      const max9  = waypoints.length - 1;
      const sc    = Math.max(0, Math.min(max9 - 0.001, smoothScroll));
      const pA    = Math.min(Math.floor(sc), max9 - 1);
      const pB    = pA + 1;
      const ft    = sc - pA;
      const ease  = ft < 0.5 ? 2*ft*ft : 1 - 2*(1-ft)*(1-ft);

      uniforms.uScroll.value = sc;

      // Camera lerp
      const wA = waypoints[pA], wB = waypoints[pB];
      const mx = mouseRef.current.x, my = mouseRef.current.y;
      tPos.lerpVectors(wA.pos, wB.pos, ease);
      tPos.x += mx * 0.12;
      tPos.y += my * 0.09;
      tTgt.lerpVectors(wA.tgt, wB.tgt, ease);

      camPos.lerp(tPos, 0.040);
      camTgt.lerp(tTgt, 0.042);
      uniforms.uCamPos.value.copy(camPos);
      uniforms.uCamTarget.value.copy(camTgt);
      uniforms.uMouse.value.set(mx, my);

      renderer.render(scene, ortho);
    };

    animate();

    let smx = 0, smy = 0;
    const onMouse = (e: MouseEvent) => {
      const tx = (e.clientX / window.innerWidth  - 0.5) * 2;
      const ty = (e.clientY / window.innerHeight - 0.5) * -2;
      smx += (tx - smx) * 0.06;
      smy += (ty - smy) * 0.06;
      mouseRef.current = { x: smx, y: smy };
    };

    const onResize = () => {
      renderer.setSize(window.innerWidth, window.innerHeight);
      uniforms.uResolution.value.set(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('mousemove', onMouse);
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('mousemove', onMouse);
      window.removeEventListener('resize', onResize);
      renderer.dispose();
      mat.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  }, [scrollProgressRef]);

  return (
    <div ref={mountRef} style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none' }} />
  );
}
