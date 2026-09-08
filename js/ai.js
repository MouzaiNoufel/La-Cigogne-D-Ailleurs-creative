/* =========================================================
   Phase 2.5 — AI analysis + visual diagnostics.
   Depth Anything V2 + SegFormer through FastAPI.
   ========================================================= */

(() => {
  "use strict";

  const API = window.location.protocol === "file:"
    ? "http://localhost:8000"
    : `${window.location.protocol}//${window.location.hostname}:8000`;
  window.AI_API = API;

  const { state, canvas, setStatus, draw, roomFit, getSelected, itemSize, constrainToFloor } = window.App;
  const analyzeBtn = document.getElementById("analyzeBtn");
  const maskChk = document.getElementById("maskChk");
  const depthChk = document.getElementById("depthChk");
  const zonesChk = document.getElementById("zonesChk");
  const maskLabel = document.getElementById("maskLabel");
  const depthLabel = document.getElementById("depthLabel");
  const zonesLabel = document.getElementById("zonesLabel");
  const aiState = document.getElementById("aiState");

  async function fileFromRoomImage() {
    if (!state.roomImage) throw new Error("Aucune pièce chargée");
    const tmp = document.createElement("canvas");
    tmp.width = state.roomImage.naturalWidth; tmp.height = state.roomImage.naturalHeight;
    tmp.getContext("2d").drawImage(state.roomImage, 0, 0);
    return new Promise((resolve, reject) => {
      tmp.toBlob(blob => blob ? resolve(blob) : reject(new Error("Conversion image impossible")), "image/jpeg", 0.92);
    });
  }

  async function checkHealth() {
    let resp;
    try {
      resp = await fetch(`${API}/health`, { cache: "no-store" });
    } catch {
      const err = new Error("Le serveur IA est inaccessible. Vérifiez Uvicorn sur http://localhost:8000.");
      err.code = "OFFLINE";
      throw err;
    }
    if (!resp.ok) {
      const err = new Error(`Le serveur IA répond avec HTTP ${resp.status}.`);
      err.code = "HEALTH_HTTP";
      throw err;
    }
    return resp.json();
  }

  function setDebugLabels(visible) {
    [maskLabel, depthLabel, zonesLabel].forEach(el => { el.style.display = visible ? "inline-flex" : "none"; });
  }

  function loadImage(src, errorMessage) {
    return new Promise((resolve, reject) => {
      if (!src) return resolve(null);
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(errorMessage));
      img.src = src;
    });
  }

  async function loadAnalysisImages(data) {
    state.floorMaskImg = await loadImage(data.masks?.floor, "Masque du sol invalide");
    state.depthImg = await loadImage(data.depth, "Carte de profondeur invalide");

    if (state.floorMaskImg) {
      const c = document.createElement("canvas");
      c.width = state.floorMaskImg.naturalWidth; c.height = state.floorMaskImg.naturalHeight;
      const cx = c.getContext("2d", { willReadFrequently: true });
      cx.drawImage(state.floorMaskImg, 0, 0);
      state.floorMaskPx = cx.getImageData(0, 0, c.width, c.height).data;
    } else {
      state.floorMaskPx = null;
    }
  }

  async function updateAnalysis(data) {
    state.analysis = data;
    window.roomAnalysis = data;
    setDebugLabels(true);
    await loadAnalysisImages(data);
    state.showMask = maskChk.checked;
    state.showDepth = depthChk.checked;
    state.showFurnitureZones = zonesChk.checked;
    draw();

    // Reconcile existing furniture with the new floor mask. Do not move an
    // item unnecessarily if its support point is already valid.
    state.items.forEach(item => constrainToFloor(item, true));
    if (window.Phase3?.sync) window.Phase3.sync();
  }

  analyzeBtn.addEventListener("click", async () => {
    if (!state.roomImage) { setStatus("Importez d'abord une photo de pièce"); return; }
    analyzeBtn.disabled = true;
    aiState.textContent = "IA : connexion…";
    aiState.className = "ai-state loading";
    setStatus("Connexion au moteur IA…");

    try {
      const health = await checkHealth();
      setStatus(health.models_loaded ? "Modèles IA déjà chargés…" : "Chargement / analyse IA… profondeur + segmentation");
      const blob = await fileFromRoomImage();
      const fd = new FormData(); fd.append("file", blob, "room.jpg");

      let resp;
      try {
        resp = await fetch(`${API}/analyze`, { method: "POST", body: fd });
      } catch {
        const err = new Error("Connexion perdue pendant l'analyse IA.");
        err.code = "NETWORK"; throw err;
      }

      if (!resp.ok) {
        let detail = `HTTP ${resp.status}`;
        try { const err = await resp.json(); detail = err.detail || detail; } catch {}
        const err = new Error(`Le moteur IA a échoué : ${detail}`);
        err.code = "ANALYZE_HTTP";
        throw err;
      }

      const data = await resp.json();
      await updateAnalysis(data);
      const furnitureCount = data.scene?.furniture_count ?? 0;
      const hasFloor = Boolean(data.masks?.floor);
      const floorSource = data.floor_source === "segformer" ? "segmenté" : "approximé";
      setStatus(`Analyse terminée ✔ · sol ${hasFloor ? floorSource : "non détecté"} · ${furnitureCount} zone(s) meuble`);
      aiState.textContent = "IA : prête ✔";
      aiState.className = "ai-state ok";
    } catch (err) {
      console.error(err);
      if (err.code === "OFFLINE" || err.code === "HEALTH_HTTP" || err.code === "NETWORK") {
        setStatus(`Erreur connexion IA : ${err.message}`);
        aiState.textContent = "IA : hors ligne";
        aiState.className = "ai-state error";
      } else {
        setStatus(`Erreur analyse IA : ${err.message}`);
        aiState.textContent = "IA : erreur";
        aiState.className = "ai-state error";
      }
    } finally {
      analyzeBtn.disabled = false;
    }
  });

  maskChk.addEventListener("change", () => { state.showMask = maskChk.checked; draw(); });
  depthChk.addEventListener("change", () => { state.showDepth = depthChk.checked; draw(); });
  zonesChk.addEventListener("change", () => { state.showFurnitureZones = zonesChk.checked; draw(); });

  // Keyboard shortcuts for diagnostics: F = floor, D = depth, Z = zones.
  window.addEventListener("keydown", e => {
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (!state.analysis) return;
    if (e.key.toLowerCase() === "f") { maskChk.checked = !maskChk.checked; maskChk.dispatchEvent(new Event("change")); }
    if (e.key.toLowerCase() === "d") { depthChk.checked = !depthChk.checked; depthChk.dispatchEvent(new Event("change")); }
    if (e.key.toLowerCase() === "z") { zonesChk.checked = !zonesChk.checked; zonesChk.dispatchEvent(new Event("change")); }
  });

  window.addEventListener("resize", () => draw());
})();
