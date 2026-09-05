/* =========================================================
   La Cigogne D'Ailleurs — Creative
   Core editor + Phase 2 placement constraints.
   ========================================================= */

(() => {
  "use strict";

  const canvas = document.getElementById("stage");
  const ctx = canvas.getContext("2d");
  const statusEl = document.getElementById("status");
  const emptyState = document.getElementById("emptyState");
  const inspectorBody = document.getElementById("inspectorBody");
  const catalogEl = document.getElementById("catalog");

  const PPM = 60;
  const state = {
    roomImage: null,
    roomFileName: "",
    roomObjectUrl: null,
    items: [],
    selectedId: null,
    zCounter: 1,
    drag: null,
    analysis: null,
    floorMaskImg: null,
    floorMaskPx: null,
    depthImg: null,
    showMask: false,
    showDepth: false,
    showFurnitureZones: false,
    eraserOn: false,
    cssWidth: 0,
    cssHeight: 0,
  };

  function setStatus(msg) { statusEl.textContent = msg; }
  function syncPhase3() { if (window.Phase3?.sync) window.Phase3.sync(); }

  function resize() {
    const r = canvas.parentElement.getBoundingClientRect();
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    canvas.style.width = `${r.width}px`;
    canvas.style.height = `${r.height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    state.cssWidth = r.width;
    state.cssHeight = r.height;
    draw();
  }

  function getCanvasSize() {
    return {
      width: state.cssWidth || canvas.getBoundingClientRect().width,
      height: state.cssHeight || canvas.getBoundingClientRect().height,
    };
  }

  function roomFit() {
    if (!state.roomImage) return null;
    const { width: cw, height: ch } = getCanvasSize();
    const s = Math.min(cw / state.roomImage.naturalWidth, ch / state.roomImage.naturalHeight);
    const w = state.roomImage.naturalWidth * s;
    const h = state.roomImage.naturalHeight * s;
    return { x: (cw - w) / 2, y: (ch - h) / 2, w, h, scale: s };
  }

  function getItem(catId) { return CATALOG.find(c => c.id === catId); }
  function getSelected() { return state.items.find(i => i.uid === state.selectedId) || null; }
  function nextUid() { return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`; }

  function addItem(catId) {
    const cat = getItem(catId);
    if (!cat) return;
    const { width, height } = getCanvasSize();
    const fit = roomFit();
    const x = fit ? fit.x + fit.w * 0.5 : width * 0.5;
    const y = fit ? fit.y + fit.h * 0.72 : height * 0.5;
    const item = {
      uid: nextUid(), catId, name: cat.name, w: cat.w, h: cat.h, d: cat.d,
      x, y, rot: 0, scale: 1, z: state.zCounter++,
    };
    state.items.push(item);
    select(item.uid);
    if (state.analysis) constrainToFloor(item, true);
    syncPhase3();
    setStatus(`${cat.name} ajouté`);
  }

  function select(uid) {
    state.selectedId = uid;
    const item = getSelected();
    if (item) item.z = state.zCounter++;
    renderInspector();
    draw();
    syncPhase3();
  }

  function deleteItem(uid) {
    const item = state.items.find(i => i.uid === uid);
    state.items = state.items.filter(i => i.uid !== uid);
    if (state.selectedId === uid) state.selectedId = null;
    renderInspector(); draw();
    syncPhase3();
    if (item) setStatus(`${item.name} supprimé`);
  }

  function duplicateItem(uid) {
    const src = state.items.find(i => i.uid === uid);
    if (!src) return;
    const copy = { ...src, uid: nextUid(), x: src.x + 30, y: src.y + 30, z: state.zCounter++ };
    state.items.push(copy);
    state.selectedId = copy.uid;
    renderInspector(); draw();
    syncPhase3();
    if (state.analysis) constrainToFloor(copy, true);
    setStatus(`${copy.name} dupliqué`);
  }

  function sampleFloorAtCanvas(x, y) {
    if (!state.floorMaskPx || !state.floorMaskImg) return false;
    const fit = roomFit();
    if (!fit) return false;
    const W = state.floorMaskImg.naturalWidth;
    const H = state.floorMaskImg.naturalHeight;
    const ix = Math.round(((x - fit.x) / Math.max(1, fit.w)) * (W - 1));
    const iy = Math.round(((y - fit.y) / Math.max(1, fit.h)) * (H - 1));
    if (ix < 0 || iy < 0 || ix >= W || iy >= H) return false;
    return state.floorMaskPx[(iy * W + ix) * 4] > 128;
  }

  function getSupportPoint(item) {
    const { w, h } = itemSize(item);
    const corners = [
      [-w / 2, -h / 2], [w / 2, -h / 2],
      [w / 2, h / 2], [-w / 2, h / 2],
    ];
    const cos = Math.cos(item.rot), sin = Math.sin(item.rot);
    let best = { x: item.x, y: item.y };
    for (const [lx, ly] of corners) {
      const x = item.x + lx * cos - ly * sin;
      const y = item.y + lx * sin + ly * cos;
      if (y > best.y) best = { x, y };
    }
    return best;
  }

  function clampToRoom(item) {
    const fit = roomFit();
    if (!fit) return;
    const { w, h } = itemSize(item);
    const pad = 2;
    item.x = Math.max(fit.x + pad + w / 2, Math.min(fit.x + fit.w - pad - w / 2, item.x));
    item.y = Math.max(fit.y + pad + h / 2, Math.min(fit.y + fit.h - pad - h / 2, item.y));
  }

  // Keep the support point on the segmented floor, but do NOT force every
  // object to the floor horizon. The user can still position it anywhere in
  // the visible floor region.
  function constrainToFloor(item, silent = false) {
    if (!item) return;
    clampToRoom(item);
    if (!state.floorMaskPx || !state.floorMaskImg || !state.analysis) return;

    let support = getSupportPoint(item);
    if (sampleFloorAtCanvas(support.x, support.y)) return;

    const fit = roomFit();
    const W = state.floorMaskImg.naturalWidth;
    const H = state.floorMaskImg.naturalHeight;
    const ix = Math.max(0, Math.min(W - 1, Math.round(((support.x - fit.x) / fit.w) * (W - 1))));
    const startY = Math.max(0, Math.min(H - 1, Math.round(((support.y - fit.y) / fit.h) * (H - 1))));

    let targetY = -1;
    // Prefer the nearest floor pixel. This handles both floating-above-floor
    // and small segmentation holes without teleporting the furniture.
    for (let radius = 0; radius <= Math.min(H, 180); radius++) {
      const candidates = [startY + radius, startY - radius];
      for (const iy of candidates) {
        if (iy < 0 || iy >= H) continue;
        if (state.floorMaskPx[(iy * W + ix) * 4] > 128) { targetY = iy; break; }
      }
      if (targetY >= 0) break;
    }

    if (targetY >= 0) {
      const targetCanvasY = fit.y + (targetY / Math.max(1, H - 1)) * fit.h;
      item.y += targetCanvasY - support.y;
      clampToRoom(item);
      if (!silent) setStatus(`${item.name} ancré sur le sol ✔`);
    }
  }

  function getPerspectiveFactor(item) {
    const a = state.analysis;
    if (!a) return 1;
    const fit = roomFit();
    if (!fit) return 1;

    // Prefer the server's floor-depth profile. Depth Anything V2 is a
    // relative-depth model, so use robust percentile normalization rather
    // than treating its values as meters.
    let depthFactor = null;
    if (Array.isArray(a.floor_depth) && a.floor_depth.length) {
      const imageY = Math.max(0, Math.min(a.height - 1,
        Math.round(((item.y - fit.y) / Math.max(1, fit.h)) * (a.height - 1))));
      const depth = Number(a.floor_depth[imageY]);
      const lo = Number(a.depth_floor_low);
      const hi = Number(a.depth_floor_high);
      if (Number.isFinite(depth) && Number.isFinite(lo) && Number.isFinite(hi) && hi > lo + 1e-4) {
        const t = Math.max(0, Math.min(1, (depth - lo) / (hi - lo)));
        const polarity = a.depth_near_is_high === false ? 1 - t : t;
        depthFactor = 0.62 + polarity * 0.68;
      }
    }

    // Geometric fallback: lower objects are visually closer in a normal
    // perspective room photograph.
    const geometricT = Math.max(0, Math.min(1, (item.y - fit.y) / Math.max(1, fit.h)));
    const geometricFactor = 0.62 + geometricT * 0.68;
    return Math.max(0.58, Math.min(1.32, depthFactor == null
      ? geometricFactor
      : depthFactor * 0.75 + geometricFactor * 0.25));
  }

  function itemSize(item) {
    const persp = getPerspectiveFactor(item);
    return { w: item.w * PPM * item.scale * persp, h: item.d * PPM * item.scale * persp };
  }

  function drawItem(item, targetCtx = ctx) {
    const { w, h } = itemSize(item);
    const img = CATALOG_IMGS[item.catId];
    targetCtx.save();
    targetCtx.translate(item.x, item.y);
    targetCtx.rotate(item.rot);

    if (item.catId !== "rug") {
      targetCtx.save();
      targetCtx.translate(6, 8);
      targetCtx.fillStyle = "rgba(0,0,0,0.28)";
      targetCtx.beginPath();
      targetCtx.ellipse(0, 0, w / 2, h / 2, 0, 0, Math.PI * 2);
      targetCtx.fill();
      targetCtx.restore();
    }

    if (img && img.complete && img.naturalWidth) {
      targetCtx.drawImage(img, -w / 2, -h / 2, w, h);
    } else {
      targetCtx.fillStyle = "#666";
      targetCtx.fillRect(-w / 2, -h / 2, w, h);
    }
    targetCtx.restore();
  }

  function drawHandles(item) {
    const { w, h } = itemSize(item);
    ctx.save();
    ctx.translate(item.x, item.y);
    ctx.rotate(item.rot);
    ctx.strokeStyle = "#d9a441";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(-w / 2, -h / 2, w, h);
    ctx.setLineDash([]);

    const ry = -h / 2 - 28;
    ctx.beginPath(); ctx.moveTo(0, -h / 2); ctx.lineTo(0, ry); ctx.stroke();
    ctx.beginPath(); ctx.arc(0, ry, 8, 0, Math.PI * 2); ctx.fillStyle = "#d9a441"; ctx.fill();
    ctx.beginPath(); ctx.arc(w / 2, h / 2, 8, 0, Math.PI * 2); ctx.fillStyle = "#7fd08c"; ctx.fill();
    ctx.restore();
  }

  function drawDebugOverlays() {
    const f = roomFit();
    if (!f || !state.analysis) return;

    if (state.showDepth && state.depthImg?.complete && state.depthImg.naturalWidth) {
      ctx.save();
      ctx.globalAlpha = 0.24;
      ctx.drawImage(state.depthImg, f.x, f.y, f.w, f.h);
      ctx.restore();
    }

    if (state.showMask && state.floorMaskImg?.complete && state.floorMaskImg.naturalWidth) {
      ctx.save();
      ctx.globalAlpha = 0.28;
      ctx.drawImage(state.floorMaskImg, f.x, f.y, f.w, f.h);
      ctx.restore();
    }

    if (state.analysis.floor_top_profile?.length) {
      const vals = state.analysis.floor_top_profile;
      ctx.save();
      ctx.strokeStyle = "rgba(217,164,65,.95)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      vals.forEach((v, i) => {
        const x = f.x + (i / Math.max(1, vals.length - 1)) * f.w;
        const y = f.y + (Number(v) / Math.max(1, state.analysis.height - 1)) * f.h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }

    if (state.showFurnitureZones && Array.isArray(state.analysis.scene?.furniture)) {
      const sx = f.w / Math.max(1, state.analysis.width);
      const sy = f.h / Math.max(1, state.analysis.height);
      ctx.save();
      ctx.lineWidth = 1.5;
      ctx.font = "11px Segoe UI, sans-serif";
      state.analysis.scene.furniture.forEach((r, idx) => {
        const x = f.x + r.x * sx, y = f.y + r.y * sy;
        const w = r.width * sx, h = r.height * sy;
        ctx.strokeStyle = "rgba(127,208,140,.95)";
        ctx.fillStyle = "rgba(127,208,140,.12)";
        ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = "rgba(10,12,18,.82)";
        const label = `${idx + 1}. ${r.label}`;
        const tw = ctx.measureText(label).width + 8;
        ctx.fillRect(x, Math.max(f.y, y - 17), tw, 17);
        ctx.fillStyle = "#dff4e3";
        ctx.fillText(label, x + 4, Math.max(f.y + 12, y - 5));
      });
      ctx.restore();
    }
  }

  function draw() {
    const { width, height } = getCanvasSize();
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0e1117"; ctx.fillRect(0, 0, width, height);
    if (state.roomImage) {
      const f = roomFit();
      ctx.drawImage(state.roomImage, f.x, f.y, f.w, f.h);
      ctx.fillStyle = "rgba(10,12,18,0.12)"; ctx.fillRect(f.x, f.y, f.w, f.h);
      drawDebugOverlays();
    }
    [...state.items].sort((a, b) => a.z - b.z).forEach(item => drawItem(item));
    const selected = getSelected();
    if (selected) drawHandles(selected);
  }

  function pointerPos(e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  function localPoint(item, p) {
    const cos = Math.cos(-item.rot), sin = Math.sin(-item.rot);
    return { x: (p.x - item.x) * cos - (p.y - item.y) * sin, y: (p.x - item.x) * sin + (p.y - item.y) * cos };
  }

  function hitHandles(item, p) {
    const { w, h } = itemSize(item);
    const q = localPoint(item, p);
    if (Math.hypot(q.x, q.y + h / 2 + 28) < 13) return "rotate";
    if (Math.hypot(q.x - w / 2, q.y - h / 2) < 13) return "scale";
    return null;
  }

  function hitItem(p) {
    const sorted = [...state.items].sort((a, b) => b.z - a.z);
    for (const item of sorted) {
      const { w, h } = itemSize(item);
      const q = localPoint(item, p);
      if (Math.abs(q.x) <= w / 2 && Math.abs(q.y) <= h / 2) return item;
    }
    return null;
  }

  function renderInspector() {
    const item = getSelected();
    if (!item) {
      inspectorBody.innerHTML = '<p class="hint">Sélectionnez un meuble sur le plan.</p>';
      return;
    }
    const deg = Math.round(item.rot * 180 / Math.PI);
    const persp = Math.round(getPerspectiveFactor(item) * 100);
    inspectorBody.innerHTML = `
      <div class="obj-name">${item.name}</div>
      <div class="obj-dims">${item.w} m × ${item.d} m</div>
      <div class="metric"><span>Perspective IA</span><strong>${persp}%</strong></div>
      <div class="field"><label>Rotation : <span class="val" id="rotVal">${deg}°</span></label>
        <input type="range" id="rotSlider" min="-180" max="180" value="${Math.max(-180, Math.min(180, deg))}" /></div>
      <div class="field"><label>Échelle : <span class="val" id="scaleVal">${Math.round(item.scale * 100)}%</span></label>
        <input type="range" id="scaleSlider" min="30" max="300" value="${Math.round(item.scale * 100)}" /></div>
      <div class="row"><button class="btn ghost" id="dupBtn" type="button">⧉ Dupliquer</button>
        <button class="btn ghost danger" id="delBtn" type="button">✕ Supprimer</button></div>
      <p class="hint">Jaune : pivoter · vert : redimensionner</p>`;

    document.getElementById("rotSlider").oninput = e => {
      item.rot = Number(e.target.value) * Math.PI / 180;
      document.getElementById("rotVal").textContent = `${e.target.value}°`;
      constrainToFloor(item, true); draw();
    };
    document.getElementById("scaleSlider").oninput = e => {
      item.scale = Number(e.target.value) / 100;
      document.getElementById("scaleVal").textContent = `${e.target.value}%`;
      constrainToFloor(item, true); draw();
    };
    document.getElementById("dupBtn").onclick = () => duplicateItem(item.uid);
    document.getElementById("delBtn").onclick = () => deleteItem(item.uid);
  }

  function resetAnalysis() {
    state.analysis = null; state.floorMaskImg = null; state.floorMaskPx = null; state.depthImg = null;
    state.showMask = false; state.showDepth = false; state.showFurnitureZones = false;
    const ids = ["maskLabel", "depthLabel", "zonesLabel"];
    ids.forEach(id => { const el = document.getElementById(id); if (el) el.style.display = "none"; });
    ["maskChk", "depthChk", "zonesChk"].forEach(id => { const el = document.getElementById(id); if (el) el.checked = false; });
  }

  document.getElementById("roomUpload").addEventListener("change", e => {
    const file = e.target.files?.[0]; if (!file) return;
    if (!file.type.startsWith("image/")) { setStatus("Veuillez sélectionner une image"); return; }
    if (state.roomObjectUrl) URL.revokeObjectURL(state.roomObjectUrl);
    state.roomObjectUrl = URL.createObjectURL(file); state.roomFileName = file.name; resetAnalysis();
    const img = new Image();
    img.onload = () => {
      state.roomImage = img; emptyState.style.display = "none";
      window.dispatchEvent(new CustomEvent("room-image-changed"));
      setStatus(`Pièce importée : ${file.name}`); draw();
    };
    img.onerror = () => setStatus("Impossible de lire cette image"); img.src = state.roomObjectUrl;
  });

  canvas.addEventListener("mousedown", e => {
    if (state.eraserOn) return;
    const p = pointerPos(e); const selected = getSelected();
    if (selected) {
      const handle = hitHandles(selected, p);
      if (handle) {
        state.drag = { mode: handle, uid: selected.uid, dist0: Math.max(1, Math.hypot(p.x - selected.x, p.y - selected.y)) };
        canvas.setPointerCapture?.(e.pointerId); return;
      }
    }
    const item = hitItem(p);
    if (item) { select(item.uid); state.drag = { mode: "move", uid: item.uid, dx: p.x - item.x, dy: p.y - item.y }; return; }
    state.selectedId = null; renderInspector(); draw();
  });

  canvas.addEventListener("mousemove", e => {
    if (!state.drag || state.eraserOn) return;
    const p = pointerPos(e); const item = state.items.find(i => i.uid === state.drag.uid); if (!item) return;
    if (state.drag.mode === "move") {
      item.x = p.x - state.drag.dx; item.y = p.y - state.drag.dy; clampToRoom(item);
    } else if (state.drag.mode === "rotate") {
      item.rot = Math.atan2(p.y - item.y, p.x - item.x) + Math.PI / 2; constrainToFloor(item, true);
    } else if (state.drag.mode === "scale") {
      const dist = Math.hypot(p.x - item.x, p.y - item.y);
      item.scale = Math.min(3, Math.max(0.3, item.scale * (dist / state.drag.dist0)));
      state.drag.dist0 = Math.max(1, dist); constrainToFloor(item, true);
    }
    draw(); if (state.drag.mode !== "move") renderInspector();
  });

  window.addEventListener("mouseup", () => {
    if (!state.drag) return;
    const mode = state.drag.mode; const item = state.items.find(i => i.uid === state.drag.uid); state.drag = null;
    if (mode === "move" && item) { constrainToFloor(item, false); renderInspector(); draw(); syncPhase3(); }
  });

  canvas.addEventListener("dblclick", e => { if (!state.eraserOn) { const item = hitItem(pointerPos(e)); if (item) deleteItem(item.uid); } });
  canvas.addEventListener("wheel", e => {
    if (state.eraserOn) return; const item = hitItem(pointerPos(e)); if (!item) return; e.preventDefault();
    item.rot += e.deltaY > 0 ? 0.05 : -0.05; constrainToFloor(item, true); select(item.uid);
    setStatus(`${item.name} : ${Math.round(item.rot * 180 / Math.PI)}°`);
  }, { passive: false });

  window.addEventListener("keydown", e => {
    const tag = document.activeElement?.tagName; if (tag === "INPUT" || tag === "TEXTAREA") return;
    if ((e.key === "Delete" || e.key === "Backspace") && state.selectedId) { e.preventDefault(); deleteItem(state.selectedId); }
  });

  document.getElementById("exportBtn").addEventListener("click", () => {
    if (window.Phase3?.isEnabled?.()) {
      window.Phase3.exportPNG();
      return;
    }
    const out = document.createElement("canvas");
    if (state.roomImage) { out.width = state.roomImage.naturalWidth; out.height = state.roomImage.naturalHeight; }
    else { const { width, height } = getCanvasSize(); out.width = Math.round(width); out.height = Math.round(height); }
    const octx = out.getContext("2d");
    if (state.roomImage) octx.drawImage(state.roomImage, 0, 0, out.width, out.height); else { octx.fillStyle = "#0e1117"; octx.fillRect(0, 0, out.width, out.height); }
    const fit = roomFit();
    if (state.roomImage && fit) {
      // Convert the stage's letterboxed coordinates back to the source photo.
      const factor = out.width / Math.max(1, fit.w);
      octx.save(); octx.scale(factor, factor); octx.translate(-fit.x, -fit.y);
      [...state.items].sort((a, b) => a.z - b.z).forEach(item => drawItem(item, octx));
      octx.restore();
    } else {
      [...state.items].sort((a, b) => a.z - b.z).forEach(item => drawItem(item, octx));
    }
    const a = document.createElement("a"); a.download = "cigogne-design.png"; a.href = out.toDataURL("image/png"); a.click(); setStatus("Image exportée ✔");
  });

  document.getElementById("clearBtn").addEventListener("click", () => {
    state.items = []; state.selectedId = null; state.zCounter = 1; resetAnalysis(); renderInspector(); draw(); setStatus("Plan vidé");
  });

  CATALOG.forEach(item => {
    const btn = document.createElement("button"); btn.className = "catalog-item"; btn.type = "button";
    btn.innerHTML = `<img src="${svgUrl(item.make(item.color))}" alt="${item.name}" /><span class="meta"><span class="name">${item.name}</span><span class="dims">${item.w} × ${item.d} m</span></span>`;
    btn.onclick = () => addItem(item.id); catalogEl.appendChild(btn);
  });

  window.App = { PPM, state, canvas, ctx, setStatus, draw, roomFit, getCanvasSize, getSelected, hitItem, itemSize, resetAnalysis, getPerspectiveFactor, constrainToFloor, getSupportPoint };
  window.AppActions = { addItem, select, deleteItem, duplicateItem, renderInspector };
  window.roomFit = roomFit; window.setStatus = setStatus; window.getSelected = getSelected; window.hitItem = hitItem; window.itemSize = itemSize; window.draw = draw; window.PPM = PPM;

  resize(); renderInspector(); setStatus("Prêt — importez une photo ou placez un meuble");
})();
