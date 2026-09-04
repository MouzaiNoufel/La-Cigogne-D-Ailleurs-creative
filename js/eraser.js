/* =========================================================
   Phase 2 optional tool — furniture removal via LaMa.
   ========================================================= */

(() => {
  "use strict";

  const { state, canvas, setStatus, draw, roomFit, hitItem } = window.App;
  const API = window.AI_API;
  const eraserBtn = document.getElementById("eraserBtn");

  eraserBtn.addEventListener("click", () => {
    state.eraserOn = !state.eraserOn;
    eraserBtn.textContent = state.eraserOn ? "🧹 Gomme : ON" : "🧹 Gomme : OFF";
    eraserBtn.classList.toggle("active", state.eraserOn);
    canvas.style.cursor = state.eraserOn ? "crosshair" : "default";
    setStatus(state.eraserOn
      ? "Gomme active — cliquez sur un meuble réel de la photo"
      : "Gomme désactivée");
  });

  canvas.addEventListener("click", async e => {
    if (!state.eraserOn || !state.roomImage) return;
    if (hitItem({ x: e.clientX - canvas.getBoundingClientRect().left, y: e.clientY - canvas.getBoundingClientRect().top })) return;

    const f = roomFit();
    if (!f) return;
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left;
    const py = e.clientY - r.top;
    const ix = Math.round((px - f.x) / f.scale);
    const iy = Math.round((py - f.y) / f.scale);
    if (ix < 0 || iy < 0 || ix >= state.roomImage.naturalWidth || iy >= state.roomImage.naturalHeight) return;

    eraserBtn.disabled = true;
    setStatus("Effacement IA en cours… (LaMa)");
    try {
      const tmp = document.createElement("canvas");
      tmp.width = state.roomImage.naturalWidth;
      tmp.height = state.roomImage.naturalHeight;
      tmp.getContext("2d").drawImage(state.roomImage, 0, 0);
      const blob = await new Promise((resolve, reject) => {
        tmp.toBlob(b => b ? resolve(b) : reject(new Error("Conversion image impossible")), "image/jpeg", 0.92);
      });

      const fd = new FormData();
      fd.append("file", blob, "room.jpg");
      fd.append("x", String(ix));
      fd.append("y", String(iy));

      const resp = await fetch(`${API}/remove`, { method: "POST", body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();

      const img = new Image();
      img.onload = () => {
        if (state.roomObjectUrl) URL.revokeObjectURL(state.roomObjectUrl);
        state.roomImage = img;
        state.roomFileName = "room-cleaned.jpg";
        state.analysis = null;
        state.floorMaskImg = null;
        state.floorMaskPx = null;
        document.getElementById("maskLabel").style.display = "none";
        document.getElementById("maskChk").checked = false;
        draw();
        setStatus("Meuble effacé ✔ — relancez l'analyse de la pièce");
      };
      img.src = data.image;
    } catch (err) {
      console.error(err);
      setStatus(`Échec de l'effacement : ${err.message}`);
    } finally {
      eraserBtn.disabled = false;
    }
  });
})();
