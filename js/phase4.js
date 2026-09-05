/* =========================================================
   Phase 4A — photo integration: conservative occlusion and
   image-derived lighting. The existing photo is re-painted only
   where SegFormer identifies existing furniture, so it remains in
   front of newly placed WebGL furniture.
   ========================================================= */

(() => {
  'use strict';

  const App = window.App;
  const canvas = document.getElementById('occlusionStage');
  const button = document.getElementById('occlusionBtn');
  const context = canvas.getContext('2d');
  const state = { enabled: false, mask: null, alphaMask: null };

  function setStatus(message) { App.setStatus(message); }

  function buildAlphaMask(image) {
    const work = document.createElement('canvas');
    work.width = image.naturalWidth; work.height = image.naturalHeight;
    const workContext = work.getContext('2d', { willReadFrequently: true });
    workContext.drawImage(image, 0, 0);
    const pixels = workContext.getImageData(0, 0, work.width, work.height);
    // Server masks are black/white PNGs with opaque black pixels. Turn mask
    // luminance into alpha before using destination-in.
    for (let i = 0; i < pixels.data.length; i += 4) {
      const alpha = pixels.data[i];
      pixels.data[i] = 255; pixels.data[i + 1] = 255; pixels.data[i + 2] = 255;
      pixels.data[i + 3] = alpha;
    }
    workContext.putImageData(pixels, 0, 0);
    return work;
  }

  function resize() {
    const size = App.getCanvasSize();
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    canvas.width = Math.max(1, Math.round(size.width * dpr));
    canvas.height = Math.max(1, Math.round(size.height * dpr));
    canvas.style.width = `${size.width}px`; canvas.style.height = `${size.height}px`;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function visible() {
    return state.enabled && Boolean(state.alphaMask) && Boolean(window.Phase3?.isEnabled?.());
  }

  function render() {
    resize();
    const size = App.getCanvasSize();
    context.clearRect(0, 0, size.width, size.height);
    canvas.classList.toggle('active', visible());
    if (!visible() || !App.state.roomImage) return;
    const fit = App.roomFit(); if (!fit) return;
    context.save();
    context.drawImage(App.state.roomImage, fit.x, fit.y, fit.w, fit.h);
    context.globalCompositeOperation = 'destination-in';
    context.drawImage(state.alphaMask, fit.x, fit.y, fit.w, fit.h);
    context.restore();
  }

  function paintOcclusionForExport(destination, width, height) {
    if (!visible() || !App.state.roomImage || !state.alphaMask) return;
    const layer = document.createElement('canvas');
    layer.width = width; layer.height = height;
    const layerContext = layer.getContext('2d');
    layerContext.drawImage(App.state.roomImage, 0, 0, width, height);
    layerContext.globalCompositeOperation = 'destination-in';
    layerContext.drawImage(state.alphaMask, 0, 0, width, height);
    destination.drawImage(layer, 0, 0);
  }

  async function loadFurnitureMask(data) {
    const src = data?.masks?.furniture;
    state.mask = null; state.alphaMask = null;
    if (!src) { render(); return; }
    const image = new Image();
    await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = src; });
    state.mask = image; state.alphaMask = buildAlphaMask(image); render();
  }

  button.addEventListener('click', () => {
    if (!state.alphaMask) { setStatus('Analysez d’abord la pièce pour obtenir le masque d’occlusion'); return; }
    state.enabled = !state.enabled;
    button.classList.toggle('active', state.enabled);
    button.setAttribute('aria-pressed', String(state.enabled));
    button.textContent = state.enabled ? '◐ Occlusion IA : ON' : '◐ Occlusion IA : OFF';
    render();
    setStatus(state.enabled
      ? 'Occlusion IA active — les meubles détectés restent au premier plan'
      : 'Occlusion IA désactivée');
  });

  window.addEventListener('room-analysis-ready', event => {
    loadFurnitureMask(event.detail).catch(() => {
      state.alphaMask = null; render(); setStatus('Masque d’occlusion IA invalide');
    });
    window.Phase3?.applyLighting?.(event.detail?.lighting);
  });
  window.addEventListener('room-image-changed', () => { state.alphaMask = null; state.mask = null; render(); });
  window.addEventListener('phase3-mode-changed', () => render());
  window.addEventListener('resize', render);

  window.Phase4 = { state, render, paintOcclusionForExport };
  render();
})();
