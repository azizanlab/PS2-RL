/* Shared page behaviour: math rendering, nav state, video playback, and the
   small interactive pieces on the overview page. */
(() => {
  'use strict';

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const MACROS = {
    '\\cB': '\\mathcal{B}', '\\cS': '\\mathcal{S}', '\\cF': '\\mathcal{F}', '\\cX': '\\mathcal{X}',
    '\\cU': '\\mathcal{U}', '\\cC': '\\mathcal{C}', '\\cR': '\\mathcal{R}', '\\cH': '\\mathcal{H}',
    '\\cP': '\\mathcal{P}', '\\cD': '\\mathcal{D}', '\\cG': '\\mathcal{G}',
    '\\KB': '\\mathbb{K}_{\\mathrm{BCBF}}', '\\E': '\\mathbb{E}', '\\R': '\\mathbb{R}',
    '\\bfb': '\\mathbf{b}', '\\bff': '\\mathbf{f}', '\\bfc': '\\mathbf{c}', '\\one': '\\mathbf{1}',
    '\\SA': '\\mathrm{SA}', '\\CIL': '\\mathrm{CIL}', '\\BCBF': '\\mathrm{BCBF}', '\\PS': '\\mathrm{PS2}',
  };

  function renderMath(root) {
    if (typeof window.renderMathInElement !== 'function') return;
    window.renderMathInElement(root, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\(', right: '\\)', display: false },
      ],
      macros: MACROS,
      throwOnError: false,
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
    });
  }
  window.PS2 = { renderMath, reduceMotion };
  renderMath(document.body);

  // ------------------------------------------------------------- nav state
  const nav = document.getElementById('site-nav');
  if (nav) {
    const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 8);
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // ---------------------------------------------------------------- videos
  // Looping clips play only while on screen; with reduced motion they stay paused with controls.
  const videos = [...document.querySelectorAll('video.autoplay')];
  videos.forEach((v) => {
    v.muted = true;
    if (reduceMotion) v.setAttribute('controls', '');
    else v.removeAttribute('controls');
  });
  if (!reduceMotion && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        const v = e.target;
        if (e.isIntersecting) {
          if (v.preload !== 'auto') v.preload = 'auto';
          const p = v.play();
          if (p && p.catch) p.catch(() => v.setAttribute('controls', ''));
        } else {
          v.pause();
        }
      });
    }, { threshold: 0.25 });
    videos.forEach((v) => io.observe(v));
  }

  // ------------------------------------------------------- compare player
  const METHODS = [
    { key: 'abp', group: 'Provably safe', label: 'PS2<sub>ABP</sub>', file: 'quad_ps2_abp', color: '#9A4A0E', p: '1.39', safe: '100%', viol: '0.00' },
    { key: 'mps', group: 'Provably safe', label: 'MPS', file: 'quad_mps', color: '#6b7180', p: '1.99', safe: '100%', viol: '0.00' },
    { key: 'cbf', group: 'Provably safe', label: 'CBF-RL', file: 'quad_cbf_rl', color: '#6b7180', p: '2.29', safe: '99.8%', viol: '0.21' },
    { key: 'penhigh', group: 'Penalty', label: 'SAC-Pen<sub>high</sub>', file: 'quad_sac_pen_high', color: '#6b7180', p: '1.14', safe: '99.9%', viol: '0.004' },
    { key: 'penlow', group: 'Penalty', label: 'SAC-Pen<sub>low</sub>', file: 'quad_sac_pen_low', color: '#6b7180', p: '1.00', safe: '47.3%', viol: '0.62' },
    { key: 'lag', group: 'CMDP', label: 'SAC-Lag.', file: 'quad_sac_lag', color: '#6b7180', p: '1.94', safe: '58.6%', viol: '1.79' },
    { key: 'cpo', group: 'CMDP', label: 'CPO', file: 'quad_cpo', color: '#6b7180', p: '10.62', safe: '70.0%', viol: '2.41' },
  ];
  const picker = document.getElementById('method-picker');
  const cmpVideo = document.getElementById('compare-video');
  if (picker && cmpVideo) {
    const title = document.getElementById('compare-title');
    const metrics = document.getElementById('compare-metrics');
    const synced = [...document.querySelectorAll('#compare video.sync')];
    let lastGroup = null;
    METHODS.forEach((m) => {
      if (m.group !== lastGroup) {
        const g = document.createElement('span');
        g.className = 'grp';
        g.textContent = m.group;
        picker.appendChild(g);
        lastGroup = m.group;
      }
      const b = document.createElement('button');
      b.type = 'button';
      b.innerHTML = m.label;
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', () => select(m, true));
      m.button = b;
      picker.appendChild(b);
    });
    function select(m, play) {
      METHODS.forEach((o) => o.button.setAttribute('aria-pressed', String(o === m)));
      title.innerHTML = `<span class="swatch" style="background:${m.color}"></span>${m.label}`;
      const bad = (s) => (s !== '100%' ? ' class="bad"' : '');
      metrics.innerHTML = `<span>pos. RMSE <b>${m.p} m</b></span><span>safe episodes <b${bad(m.safe)}>${m.safe}</b></span>` +
        `<span>worst violation <b${m.viol !== '0.00' ? ' class="bad"' : ''}>${m.viol} m</b></span>`;
      cmpVideo.poster = `assets/video/${m.file}.webp`;
      cmpVideo.src = `assets/video/${m.file}.mp4`;
      cmpVideo.load();
      if (play && !reduceMotion) {
        synced.forEach((v) => { v.currentTime = 0; const p = v.play(); if (p && p.catch) p.catch(() => {}); });
      }
    }
    select(METHODS[0], false);
  }

  // ----------------------------------------------------------- beta widget
  const beta = document.getElementById('beta');
  if (beta) {
    const out = document.getElementById('beta-val');
    const bars = [...document.querySelectorAll('.beta-bar')];
    const vals = [...document.querySelectorAll('.beta-row .v')];
    const update = () => {
      const b = parseFloat(beta.value);
      out.textContent = b.toFixed(2);
      bars.forEach((bar, i) => {
        const n = bar.dataset.n;
        const q = n === 'inf' ? 0 : Math.pow(b, +n);
        bar.style.width = `${(q * 100).toFixed(1)}%`;
        vals[i].textContent = q === 0 ? '0' : q.toFixed(3);
      });
    };
    beta.addEventListener('input', update);
    update();
  }

  // ------------------------------------------------- recoverability chart
  const svg = document.getElementById('recov-svg');
  if (svg) {
    const rows = [
      ['Near ceiling', 35.5, 69.3, 1194],
      ['Bridge (below apex)', 47.3, 78.6, 1885],
      ['General trace', 77.7, 96.1, 2484],
      ['Capture shell', 84.4, 100.0, 598],
      ['Overall', 60.9, 85.9, 6161],
    ];
    const C_ABP = '#9A4A0E', C_SA = '#2266B0';
    const W = 760, labelW = 168, padR = 46, rowH = 40, top = 26;
    const H = top + rows.length * rowH + 4;
    const x = (v) => labelW + (v / 100) * (W - labelW - padR);
    const ns = 'http://www.w3.org/2000/svg';
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const el = (tag, attrs, text) => {
      const e = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
      if (text != null) e.textContent = text;
      svg.appendChild(e);
      return e;
    };
    [0, 25, 50, 75, 100].forEach((t) => {
      el('line', { x1: x(t), x2: x(t), y1: top - 6, y2: H - 4, stroke: '#e6e3da', 'stroke-width': 1 });
      el('text', { x: x(t), y: 12, 'text-anchor': 'middle', 'font-size': 11, fill: '#6b7180', 'font-family': 'Inter, sans-serif' }, `${t}%`);
    });
    const tip = document.createElement('div');
    tip.className = 'chart-tip';
    document.body.appendChild(tip);
    const showTip = (evt, html) => {
      tip.innerHTML = html;
      tip.style.opacity = '1';
      const pad = 14;
      let left = evt.clientX + pad, topY = evt.clientY + pad;
      const r = tip.getBoundingClientRect();
      if (left + r.width > window.innerWidth - 8) left = evt.clientX - r.width - pad;
      if (topY + r.height > window.innerHeight - 8) topY = evt.clientY - r.height - pad;
      tip.style.left = `${left}px`;
      tip.style.top = `${topY}px`;
    };
    const hideTip = () => { tip.style.opacity = '0'; };
    rows.forEach(([name, a, l, n], i) => {
      const cy = top + i * rowH + rowH / 2;
      const isTotal = name === 'Overall';
      if (isTotal) el('line', { x1: 0, x2: W, y1: cy - rowH / 2, y2: cy - rowH / 2, stroke: '#d6d1c4', 'stroke-width': 1 });
      el('text', { x: 0, y: cy + 4, 'font-size': 13, fill: '#3c4250', 'font-weight': isTotal ? 700 : 500, 'font-family': 'Inter, sans-serif' }, name);
      el('line', { x1: x(a), x2: x(l), y1: cy, y2: cy, stroke: '#c9c4b8', 'stroke-width': 2, 'stroke-linecap': 'round' });
      el('circle', { cx: x(a), cy, r: 6, fill: C_ABP, stroke: '#fff', 'stroke-width': 2 });
      el('circle', { cx: x(l), cy, r: 6, fill: C_SA, stroke: '#fff', 'stroke-width': 2 });
      el('text', { x: x(a) - 10, y: cy + 4, 'text-anchor': 'end', 'font-size': 12, fill: '#6b7180', 'font-family': 'JetBrains Mono, monospace' }, a.toFixed(1));
      el('text', { x: x(l) + 10, y: cy + 4, 'font-size': 12, fill: '#15181f', 'font-weight': 600, 'font-family': 'JetBrains Mono, monospace' }, l.toFixed(1));
      const hit = el('rect', { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: 'transparent' });
      const html = `<b>${name}</b> · ${n.toLocaleString()} states<br>Analytic: ${a.toFixed(1)}%<br>Learned: ${l.toFixed(1)}% <b>(+${(l - a).toFixed(1)} pts)</b>`;
      hit.addEventListener('mousemove', (e) => showTip(e, html));
      hit.addEventListener('mouseleave', hideTip);
    });
  }

  // ---------------------------------------------------------------- copy
  document.querySelectorAll('[data-copy]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const src = document.querySelector(btn.dataset.copy);
      if (!src) return;
      try {
        await navigator.clipboard.writeText(src.textContent);
        btn.textContent = 'Copied';
      } catch (e) {
        const range = document.createRange();
        range.selectNodeContents(src);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        btn.textContent = 'Press ⌘/Ctrl+C';
      }
      setTimeout(() => { btn.textContent = 'Copy'; }, 1800);
    });
  });
})();
