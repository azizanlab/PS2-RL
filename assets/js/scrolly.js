/* Scroll-driven build-up of Figure 1 from its Figma layers.

   The section is a tall track with a sticky stage. Scroll position maps to a
   continuous step coordinate f in [0, N-1]: integer steps are "settled" states,
   and the first TR fraction of each step interval animates the transition into
   the next step (camera move + staggered layer entrances). Everything is a pure
   function of f, so scrolling backwards simply rewinds. */
(() => {
  'use strict';

  const FIG = window.PS2_FIGURE;
  const section = document.getElementById('framework');
  const track = document.getElementById('scrolly-track');
  if (!FIG || !section || !track) return;

  const xs = (prefix, n) => Array.from({ length: n }, (_, i) => `${prefix}${i + 1}`);

  // `add`: layers revealed in this step, in reveal order. `focus`: layers the camera frames.
  const STEPS = [
    { add: ['bg', 'title'], focus: 'all' },
    { add: ['p1-panel', 'p1-title', 'obj1-box', 'obj1'], focus: ['p1-title', 'obj1-box'] },
    { add: ['base-panel', 'base-title', 'base-B', 'base-B-label', 'base-curve', 'base-dash', 'base-xstar-dot', 'base-xstar',
            'base-arrow-l', 'base-arrow-m', 'base-arrow-r', 'base-eq-B', 'base-eq-pi', 'base-eq-e'], focus: ['base-panel'] },
    { add: ['sa-panel', 'sa-title', 'sa-band', 'sa-q', 'sa-indicators'], focus: ['sa-title', 'sa-band'] },
    { add: ['sa-F', 'sa-S', 'sa-B', 'sa-F-label', 'sa-S-label', 'sa-B-label',
            'sa-fast', 'sa-x-fast1', 'sa-x-fast2', 'sa-star-fast', 'sa-leg-fast', 'sa-leg-fast-txt', 'sa-leg-fast-q',
            'sa-slow', 'sa-x-slow1', 'sa-x-slow2', 'sa-x-slow3', 'sa-x-slow4', 'sa-star-slow', 'sa-leg-slow', 'sa-leg-slow-txt', 'sa-leg-slow-q',
            'sa-unsafe', 'sa-leg-unsafe', 'sa-leg-unsafe-txt', 'sa-leg-unsafe-q'], focus: ['sa-F', 'sa-leg-fast-txt', 'sa-leg-slow-q'] },
    { add: ['learn-band', 'learn-title', 'learn-blob-l', 'learn-B-l', ...xs('learn-dot', 11),
            'learn-arrow1', 'learn-nn', 'learn-arrow2', 'learn-blob-r'], focus: ['learn-band'] },
    { add: ['bk-panel', 'bk-title', 'bk-eq', 'bk-hl-B', 'bk-base-txt', 'bk-base-arrow', 'bk-hl-SA', 'bk-nn', 'bk-nn-arrow', 'bk-sa-txt'],
      focus: ['bk-panel'] },
    { add: ['big-arrow', 'p2-panel', 'p2-title', 'obj2-box', 'obj2', 'maxj-box', 'maxj'], focus: ['p2-title', 'obj2-box', 'maxj-box'] },
    { add: ['cil-panel', 'cil-title', 'cil-eq-box', 'cil-eq', 'cil-leg-check', 'cil-leg-check-txt', 'cil-leg-safe', 'cil-leg-safe-txt',
            'cil-leg-nosafe', 'cil-leg-nosafe-txt'], focus: ['cil-title', 'cil-eq-box', 'cil-leg-nosafe-txt'] },
    { add: ['cil-S', 'cil-CT', 'cil-B', 'cil-B-label', 'cil-CT-label', 'cil-S-label', 'cil-traj',
            'cil-node1', 'cil-flow1', 'cil-node2', 'cil-node3', 'cil-flow23', 'cil-node4', 'cil-flow4',
            'cil-star', 'cil-escape', 'cil-escape-dot', 'cil-noflow'], focus: ['cil-S', 'cil-leg-nosafe'] },
    { add: ['loop-box', 'loop', 'loop-pi', 'cil-wedge'], focus: ['p2-panel'] },
    { add: [], focus: 'all' },
  ];

  // entrance effects; anything not listed fades up
  const FX = {
    'bg': 'fade', 'p1-panel': 'fade', 'p2-panel': 'fade', 'obj1-box': 'fade', 'obj1': 'wipe-r',
    'base-panel': 'fade', 'base-B': 'pop', 'base-curve': 'wipe-d', 'base-dash': 'wipe-r', 'base-xstar-dot': 'pop',
    'base-arrow-l': 'wipe-u', 'base-arrow-m': 'wipe-u', 'base-arrow-r': 'wipe-u',
    'sa-panel': 'fade', 'sa-band': 'fade', 'sa-F': 'fade', 'sa-S': 'fade', 'sa-B': 'pop',
    'sa-fast': 'wipe-d', 'sa-slow': 'wipe-d', 'sa-unsafe': 'wipe-u',
    'sa-leg-fast': 'wipe-r', 'sa-leg-slow': 'wipe-r', 'sa-leg-unsafe': 'wipe-r',
    'sa-star-fast': 'pop', 'sa-star-slow': 'pop',
    'learn-band': 'fade', 'learn-blob-l': 'fade', 'learn-B-l': 'pop', 'learn-arrow1': 'wipe-r', 'learn-arrow2': 'wipe-r',
    'learn-blob-r': 'pop', 'learn-nn': 'pop',
    'bk-panel': 'fade', 'bk-hl-B': 'fade', 'bk-hl-SA': 'fade', 'bk-nn': 'pop', 'bk-nn-arrow': 'wipe-u', 'bk-base-arrow': 'fade',
    'big-arrow': 'wipe-r', 'obj2-box': 'fade', 'maxj-box': 'fade',
    'cil-panel': 'fade', 'cil-eq-box': 'fade', 'cil-leg-check': 'wipe-r', 'cil-leg-safe': 'wipe-r', 'cil-leg-nosafe': 'wipe-r',
    'cil-S': 'fade', 'cil-CT': 'fade', 'cil-B': 'pop', 'cil-traj': 'wipe-r', 'cil-flow1': 'wipe-u', 'cil-flow23': 'wipe-u',
    'cil-flow4': 'wipe-u', 'cil-star': 'pop', 'cil-escape': 'wipe-r', 'cil-escape-dot': 'pop', 'cil-noflow': 'wipe-l',
    'loop-box': 'fade', 'loop': 'fade', 'cil-wedge': 'fade',
  };
  ['learn-dot', 'cil-node'].forEach((p) => xs(p, 11).forEach((id) => { FX[id] = 'pop'; }));
  ['sa-x-fast1', 'sa-x-fast2', 'sa-x-slow1', 'sa-x-slow2', 'sa-x-slow3', 'sa-x-slow4'].forEach((id) => { FX[id] = 'pop'; });

  const N = STEPS.length;
  const LEAD = 0.35;          // dwell on the first state before anything moves
  const TAIL = 0.65;          // dwell on the final state
  const TR = 0.6;             // fraction of each step interval spent transitioning
  const MAX_SCALE = 1.0;      // CSS px per 2x-export px
  const GHOST = 0.13;         // opacity of the whole-figure preview on the intro step

  const stage = document.getElementById('fig-viewport');
  const world = document.getElementById('fig-world');
  const capsBox = document.getElementById('captions');
  const rail = document.getElementById('rail');
  const staticSteps = [...document.querySelectorAll('#steps > li')];
  const motionOn = document.getElementById('motion-on');

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (staticSteps.length !== N) console.warn('scrolly: step count mismatch', staticSteps.length, N);

  if (reduce.matches) {
    if (motionOn) {
      motionOn.hidden = false;
      motionOn.addEventListener('click', () => { motionOn.hidden = true; enable(); });
    }
    return;
  }
  enable();

  function enable() {
    section.classList.remove('is-static');
    track.hidden = false;
    build();
  }

  function build() {
    const W = FIG.width, H = FIG.height;
    world.style.width = `${W}px`;
    world.style.height = `${H}px`;

    // ------------------------------------------------------------ layers
    const byId = {};
    FIG.layers.forEach((l) => {
      const img = new Image();
      img.alt = '';
      img.decoding = 'async';
      img.draggable = false;
      img.src = FIG.base + l.src;
      Object.assign(img.style, { left: `${l.x}px`, top: `${l.y}px`, width: `${l.w}px`, height: `${l.h}px` });
      world.appendChild(img);
      byId[l.id] = { ...l, el: img, step: -1, order: 0, count: 1, fx: FX[l.id] || 'rise', state: null };
    });
    STEPS.forEach((s, k) => {
      s.add.forEach((id, i) => {
        const L = byId[id];
        if (!L) { console.warn('scrolly: unknown layer', id); return; }
        L.step = k; L.order = i; L.count = s.add.length;
      });
    });
    const orphans = Object.values(byId).filter((L) => L.step < 0);
    orphans.forEach((L) => { L.step = N - 1; L.order = 0; L.count = 1; });

    // ------------------------------------------------------------ camera
    const pad = (r, k) => {
      const m = Math.max(28, k * Math.max(r[2], r[3]));
      return [r[0] - m, r[1] - m, r[2] + 2 * m, r[3] + 2 * m];
    };
    STEPS.forEach((s) => {
      if (s.focus === 'all') { s.rect = [0, 0, W, H]; return; }
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      s.focus.forEach((id) => {
        const L = byId[id];
        x0 = Math.min(x0, L.x); y0 = Math.min(y0, L.y);
        x1 = Math.max(x1, L.x + L.w); y1 = Math.max(y1, L.y + L.h);
      });
      s.rect = pad([x0, y0, x1 - x0, y1 - y0], 0.05);
    });

    // ---------------------------------------------------------- captions
    const caps = staticSteps.map((li) => {
      const d = document.createElement('div');
      d.className = 'cap';
      d.setAttribute('aria-hidden', 'true');
      d.innerHTML = li.innerHTML;
      capsBox.appendChild(d);
      return d;
    });

    // -------------------------------------------------------------- rail
    const dots = [];
    const groups = [{ cls: '', steps: [0] }, { cls: 'p1', label: 'Phase I', steps: [1, 2, 3, 4, 5, 6] },
                    { cls: 'p2', label: 'Phase II', steps: [7, 8, 9, 10] }, { cls: '', steps: [11] }];
    groups.forEach((g) => {
      const box = document.createElement('div');
      box.className = `rail-group ${g.cls}`;
      if (g.label) {
        const lab = document.createElement('span');
        lab.className = 'rail-label';
        lab.textContent = g.label;
        box.appendChild(lab);
      }
      g.steps.forEach((k) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'rail-dot';
        const t = staticSteps[k] && staticSteps[k].querySelector('h3');
        b.setAttribute('aria-label', `Step ${k + 1}: ${t ? t.textContent : ''}`);
        b.title = t ? t.textContent : '';
        b.addEventListener('click', () => goTo(k));
        dots[k] = b;
        box.appendChild(b);
      });
      rail.appendChild(box);
    });
    const end = document.createElement('div');
    end.className = 'rail-end';
    end.innerHTML = '<a class="rail-skip" href="#method">Skip ↓</a>';
    rail.appendChild(end);

    // ----------------------------------------------------------- geometry
    let stepLen = 600, trackTop = 0, vw = 1, vh = 1;
    function layout() {
      const sticky = track.querySelector('.scrolly-sticky');
      stepLen = Math.max(420, Math.round(window.innerHeight * 0.8));
      track.style.height = `${Math.round(sticky.offsetHeight + stepLen * (N - 1 + LEAD + TAIL))}px`;
      trackTop = track.getBoundingClientRect().top + window.scrollY;
      capsBox.style.height = '0px';
      const capH = Math.max(...caps.map((d) => d.scrollHeight));
      capsBox.style.height = `${capH + 2}px`;
      vw = stage.clientWidth;
      vh = stage.clientHeight;
      target = fFromScroll();
      cur = target;
      render(cur);
    }
    function fFromScroll() {
      const f = (window.scrollY - trackTop) / stepLen - LEAD;
      return Math.min(N - 1, Math.max(0, f));
    }
    function goTo(k) {
      const f = k === 0 ? -LEAD : k - 1 + TR + 0.25;
      window.scrollTo({ top: trackTop + (f + LEAD) * stepLen + 2, behavior: 'smooth' });
    }

    // ------------------------------------------------------------ render
    const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
    const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    const easeOut = (t) => 1 - Math.pow(1 - t, 3);

    function cameraAt(k, tau) {
      const a = STEPS[k].rect, b = STEPS[Math.min(N - 1, k + 1)].rect;
      const e = ease(clamp01(tau / 0.85));
      const cx = (a[0] + a[2] / 2) * (1 - e) + (b[0] + b[2] / 2) * e;
      const cy = (a[1] + a[3] / 2) * (1 - e) + (b[1] + b[3] / 2) * e;
      const w = Math.exp(Math.log(a[2]) * (1 - e) + Math.log(b[2]) * e);
      const h = Math.exp(Math.log(a[3]) * (1 - e) + Math.log(b[3]) * e);
      const s = Math.min(vw / w, vh / h, MAX_SCALE);
      const hw = vw / (2 * s), hh = vh / (2 * s);
      const ccx = 2 * hw >= W ? W / 2 : Math.min(W - hw, Math.max(hw, cx));
      const ccy = 2 * hh >= H ? H / 2 : Math.min(H - hh, Math.max(hh, cy));
      return [vw / 2 - ccx * s, vh / 2 - ccy * s, s];
    }

    function applyLayer(L, p, ghost) {
      // p: entrance progress 0..1; ghost: preview opacity for layers not yet revealed
      const st = L.el.style;
      if (p <= 0) {
        if (ghost > 0) {
          L.el.classList.remove('hide');
          st.opacity = ghost.toFixed(3);
          st.transform = '';
          st.clipPath = '';
          L.state = 'g';
        } else if (L.state !== 0) {
          L.el.classList.add('hide'); st.opacity = '0'; L.state = 0;
        }
        return;
      }
      if (p >= 1) {
        if (L.state !== 1) {
          L.el.classList.remove('hide');
          st.opacity = '';
          st.transform = '';
          st.clipPath = '';
          L.state = 1;
        }
        return;
      }
      L.state = 0.5;
      L.el.classList.remove('hide');
      const q = easeOut(p);
      const inv = ((1 - q) * 100).toFixed(2);
      switch (L.fx) {
        case 'fade': st.opacity = q; break;
        case 'pop': st.opacity = Math.min(1, q * 1.6); st.transform = `scale(${0.55 + 0.45 * q})`; break;
        case 'wipe-r': st.opacity = 1; st.clipPath = `inset(0 ${inv}% 0 0)`; break;
        case 'wipe-l': st.opacity = 1; st.clipPath = `inset(0 0 0 ${inv}%)`; break;
        case 'wipe-d': st.opacity = 1; st.clipPath = `inset(0 0 ${inv}% 0)`; break;
        case 'wipe-u': st.opacity = 1; st.clipPath = `inset(${inv}% 0 0 0)`; break;
        default: st.opacity = q; st.transform = `translateY(${(1 - q) * 14}px)`;
      }
    }

    let lastCap = -1, lastDot = -1;
    const layers = Object.values(byId);
    function render(f) {
      const k = Math.min(N - 1, Math.floor(f + 1e-6));
      const tau = clamp01((f - k) / TR);
      const [tx, ty, s] = k >= N - 1 ? cameraAt(N - 1, 0) : cameraAt(k, tau);
      world.style.transform = `translate3d(${tx.toFixed(2)}px, ${ty.toFixed(2)}px, 0) scale(${s.toFixed(5)})`;

      const active = k + 1;
      const ghost = k === 0 ? GHOST * (1 - ease(clamp01(tau / 0.55))) : 0;
      layers.forEach((L) => {
        if (L.step <= k) applyLayer(L, 1, 0);
        else if (L.step > active) applyLayer(L, 0, ghost);
        else {
          const n = Math.max(1, L.count - 1);
          const start = 0.12 + 0.5 * (L.order / n);
          applyLayer(L, (tau - start) / 0.32, ghost);
        }
      });

      const c = k >= N - 1 ? N - 1 : (tau > 0.18 ? active : k);
      if (c !== lastCap) {
        caps.forEach((d, i) => {
          const on = i === c;
          d.classList.toggle('on', on);
          d.style.opacity = on ? '1' : '0';
          d.style.transform = on ? 'translateY(0)' : 'translateY(8px)';
          d.setAttribute('aria-hidden', String(!on));
        });
        lastCap = c;
      }
      if (c !== lastDot) {
        dots.forEach((d, i) => {
          d.classList.toggle('active', i === c);
          d.classList.toggle('done', i < c);
          if (i === c) d.setAttribute('aria-current', 'step'); else d.removeAttribute('aria-current');
        });
        lastDot = c;
      }
    }
    caps.forEach((d) => { d.style.transition = 'opacity .35s ease, transform .35s ease'; });

    // ---------------------------------------------------- scroll + smoothing
    let target = 0, cur = 0, raf = 0;
    function tick() {
      raf = 0;
      const d = target - cur;
      cur = Math.abs(d) < 0.0008 ? target : cur + d * 0.2;
      render(cur);
      if (cur !== target) raf = requestAnimationFrame(tick);
    }
    function onScroll() {
      target = fFromScroll();
      if (!raf) raf = requestAnimationFrame(tick);
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    let rT = 0;
    window.addEventListener('resize', () => { clearTimeout(rT); rT = setTimeout(layout, 120); });
    if ('ResizeObserver' in window) new ResizeObserver(() => { vw = stage.clientWidth; vh = stage.clientHeight; render(cur); }).observe(stage);
    window.addEventListener('load', layout);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(layout);
    layout();
  }
})();
