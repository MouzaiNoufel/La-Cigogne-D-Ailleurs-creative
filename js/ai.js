/* =========================================================
   Phase 2 — Room analysis
   Depth Anything V2 + SegFormer through FastAPI.
   No function redefinition: this module extends App safely.
   ========================================================= */

(() => {
  "use strict";

  const API = window.location.protocol === "file:"
    ? "http://localhost:8000"
    : `${window.location.protocol}//${window.location.hostname}:8000`;
  window.AI_API = API;

  const { state, canvas, setStatus, draw, roomFit, getCanvasSize, getSelected, itemSize } = window.App;
  const analyzeBtn = document.getElementById("analyzeBtn");
  const maskChk = document.getElementById("maskChk");
  const maskLabel = document.getElementById("maskLabel");

  async function fileFromRoomImage() {
    if (!state.roomImage) throw new Error("Aucune pièce chargée");
    const tmp = document.createElement("canvas");
    tmp.width = state.roomImage.naturalWidth;
    tmp.height = state.roomImage.naturalHeight;
    tmp.getContext("2d").drawImage(state.roomImage, 0, 0);
    return new Promise((resolve, reject) => {
      tmp.toBlob(blob => blob ? resolve(blob) : reject(new Error("Conversion image impossible")), "image/jpeg", 0.92);
    });
  }

  async function checkHealth() {
    const resp = await fetch(`${API}/health`, { cache: "no-store" });
    if (!resp.ok) throw new Error(`Serveur IA HTTP ${resp.status}`);
    return resp.json();
  }

  function loadFloorMask(data) {
    if (!data.masks?.floor) {
      state.floorMaskImg = null;
      state.floorMaskPx = null;
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        state.floorMaskImg = img;
        const c = document.createElement("canvas");
        c.width = img.naturalWidth;
        c.height = img.naturalHeight;
        const cx = c.getContext("2d", { willReadFrequently: true });
        cx.drawImage(img, 0, 0);
        state.floorMaskPx = cx.getImageData(0, 0, c.width, c.height).data;
        resolve();
      };
      img.onerror = () => reject(new Error("Masque du sol invalide"));
      img.src = data.masks.floor;
    });
  }

  function updateAnalysis(data) {
    state.analysis = data;
    window.roomAnalysis = data;
    maskLabel.style.display = "inline-flex";
    state.showMask = maskChk.checked;

    return loadFloorMask(data).then(() => {
      draw();
    });
  }

  analyzeBtn.addEventListener("click", async () => {
    if (!state.roomImage) {
      setStatus("Importez d'abord une photo de pièce");
      return;
    }

    analyzeBtn.disabled = true;
    setStatus("Connexion au moteur IA…");

    try {
      await checkHealth();
      setStatus("Analyse IA en cours… profondeur + segmentation");
      const blob = await fileFromRoomImage();
      const fd = new FormData();
      fd.append("file", blob, "room.jpg");

      const resp = await fetch(`${API}/analyze`, { method: "POST", body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      await updateAnalysis(data);

      const furnitureCount = data.scene?.furniture_count ?? 0;
      const hasFloor = Boolean(data.masks?.floor);
      setStatus(`Analyse terminée ✔ · sol ${hasFloor ? "détecté" : "approximé"} · ${furnitureCount} zone(s) meuble`);
    } catch (err) {
      console.error(err);
      if (err instanceof TypeError) {
        setStatus(`Serveur IA inaccessible — lancez FastAPI sur ${API}`);
      } else {
        setStatus(`Erreur IA : ${err.message}`);
      }
    } finally {
      analyzeBtn.disabled = false;
    }
  });

  maskChk.addEventListener("change", () => {
    state.showMask = maskChk.checked;
    draw();
  });

  function findFloorContactY(item) {
    if (!state.floorMaskImg || !state.floorMaskPx) return null;
    const fit = roomFit();
    if (!fit) return null;
    const W = state.floorMaskImg.naturalWidth;
    const H = state.floorMaskImg.naturalHeight;
    const { width: cw, height: ch } = getCanvasSize();
    const mx = Math.max(0, Math.min(W - 1, Math.round(((item.x - fit.x) / fit.w) * (W - 1))));
    if (!Number.isFinite(mx)) return null;

    // Find the highest run of floor pixels around the item's column, then
    // use its lower boundary as the visual floor contact line.
    const samples = [];
    for (let dx = -8; dx <= 8; dx++) {
      const x = Math.max(0, Math.min(W - 1, mx + dx));
      let first = -1;
      for (let y = 0; y < H; y++) {
        const alpha = state.floorMaskPx[(y * W + x) * 4];
        if (alpha > 128) { first = y; break; }
      }
      if (first >= 0) samples.push(first);
    }
    if (!samples.length) return null;

    // Robust median of the local floor boundary.
    samples.sort((a, b) => a - b);
    const firstY = samples[Math.floor(samples.length / 2)];
    return fit.y + (firstY / Math.max(1, H - 1)) * fit.h;
  }

  window.applyFloorAnchor = item => {
    if (!item || !state.analysis) return;
    const floorY = findFloorContactY(item);
    if (floorY == null) return;
    const size = itemSize(item);
    item.y = floorY - size.h / 2;
    draw();
    setStatus(`${item.name} ancré au sol ✔`);
  };

  window.addEventListener("resize", () => draw());
})();
