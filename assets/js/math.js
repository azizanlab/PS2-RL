/* Highlight the table-of-contents entry for the section in view. */
(() => {
  'use strict';
  const links = [...document.querySelectorAll('#toc a[href^="#"]')];
  const targets = links.map((a) => document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean);
  if (!targets.length || !('IntersectionObserver' in window)) return;
  const visible = new Map();
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => visible.set(e.target.id, e.isIntersecting ? e.boundingClientRect.top : null));
    let best = null;
    targets.forEach((t) => { if (visible.get(t.id) != null && best === null) best = t.id; });
    if (best) links.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === `#${best}`));
  }, { rootMargin: '-70px 0px -65% 0px' });
  targets.forEach((t) => io.observe(t));
})();
