/* =========================================================
   Phase 4 — AI Scene Editing / Smart Replace
   SegFormer point selection + LaMa inpainting + undo.
   ========================================================= */

(() => {
  "use strict";

  const { state, canvas, setStatus, draw, roomFit, hitItem } = window.App;
  const API = window.AI_API;
  const eraserBtn = document.getElementById("eraserBtn");
  const undoBtn = document.getElementById("undoAiBtn");

  let history = [];
  let selectedMask = null;
  let selectedLabel = "";
  let busy = false;

  const overlay = document.createElement("canvas");
  overlay.id = "aiMaskOverlay";
  overlay.style.cssText = "position:absolute;inset:0;width:100%;height:100%;z-index:7;pointer-events:none;display:none";
  canvas.parentElement.appendChild(overlay);
  const octx = overlay.getContext("2d");

  function syncOverlaySize() {
    const r = canvas.parentElement.getBoundingClientRect();
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    overlay.width = Math.max(1, Math.round(r.width * dpr));
    overlay.height = Math.max(1, Math.round(r.height * dpr));
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawMask();
  }

  function drawMask() {
    const w = canvas.parentElement.clientWidth, h = canvas.parentElement.clientHeight;
    octx.clearRect(0, 0, w, h);
    if (!selectedMask || !state.roomImage) { overlay.style.display = "none"; return; }
    const fit = roomFit();
    if (!fit) return;
    overlay.style.display = "block";

    // IMPORTANT: the backend returns a grayscale L mask. In a browser an
    // opaque black PNG still has alpha=255 everywhere, so using RGBA alpha
    // directly makes the whole room look selected. Convert luminance -> alpha.
    const mw = selectedMask.naturalWidth || selectedMask.width;
    const mh = selectedMask.naturalHeight || selectedMask.height;
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = mw; maskCanvas.height = mh;
    const mc = maskCanvas.getContext("2d", { willReadFrequently: true });
    mc.drawImage(selectedMask, 0, 0);
    const md = mc.getImageData(0, 0, mw, mh);
    const overlayCanvas = document.createElement("canvas");
    overlayCanvas.width = mw; overlayCanvas.height = mh;
    const oc = overlayCanvas.getContext("2d");
    const od = oc.createImageData(mw, mh);
    const threshold = 100;
    for (let i = 0; i < md.data.length; i += 4) {
      const v = md.data[i];
      const a = v >= threshold ? 105 : 0;
      od.data[i] = 235; od.data[i + 1] = 72; od.data[i + 2] = 72; od.data[i + 3] = a;
    }
    oc.putImageData(od, 0, 0);
    octx.drawImage(overlayCanvas, fit.x, fit.y, fit.w, fit.h);

    // Draw a real contour from the grayscale mask, not from the PNG alpha.
    const step = Math.max(1, Math.round(Math.max(mw, mh) / 850));
    octx.save();
    octx.strokeStyle = "rgba(255,92,92,.98)";
    octx.lineWidth = 2.2;
    octx.setLineDash([6, 4]);
    octx.beginPath();
    for (let y = step; y < mh - step; y += step) {
      for (let x = step; x < mw - step; x += step) {
        const i = (y * mw + x) * 4;
        if (md.data[i] < threshold) continue;
        const edge =
          md.data[((y-step)*mw+x)*4] < threshold ||
          md.data[((y+step)*mw+x)*4] < threshold ||
          md.data[(y*mw+x-step)*4] < threshold ||
          md.data[(y*mw+x+step)*4] < threshold;
        if (edge) {
          const sx = fit.x + x / mw * fit.w;
          const sy = fit.y + y / mh * fit.h;
          octx.rect(sx, sy, Math.max(1, step / mw * fit.w), Math.max(1, step / mh * fit.h));
        }
      }
    }
    octx.stroke();
    octx.restore();
  }


  function roomBlob() {
    if (!state.roomImage) throw new Error("Aucune pièce chargée");
    const tmp = document.createElement("canvas");
    tmp.width = state.roomImage.naturalWidth; tmp.height = state.roomImage.naturalHeight;
    tmp.getContext("2d").drawImage(state.roomImage, 0, 0);
    return new Promise((resolve,reject)=>tmp.toBlob(b=>b?resolve(b):reject(new Error("Conversion image impossible")),"image/jpeg",.94));
  }

  function pushHistory() {
    if (!state.roomImage) return;
    const c=document.createElement("canvas"); c.width=state.roomImage.naturalWidth; c.height=state.roomImage.naturalHeight;
    c.getContext("2d").drawImage(state.roomImage,0,0);
    history.push(c.toDataURL("image/jpeg",.92));
    if(history.length>12) history.shift();
    undoBtn.disabled=false;
  }

  function loadRoom(src, statusText) {
    const img=new Image();
    img.onload=()=>{
      if(state.roomObjectUrl) URL.revokeObjectURL(state.roomObjectUrl);
      state.roomObjectUrl=null;
      state.roomImage=img;
      state.roomFileName="room-edited.jpg";
      state.analysis=null; state.floorMaskImg=null; state.floorMaskPx=null; state.depthImg=null;
      ["maskLabel","depthLabel","zonesLabel"].forEach(id=>{const el=document.getElementById(id);if(el)el.style.display="none";});
      ["maskChk","depthChk","zonesChk"].forEach(id=>{const el=document.getElementById(id);if(el)el.checked=false;});
      selectedMask=null; state.aiMaskImg=null; drawMask(); draw();
      if(window.Phase3?.sync) window.Phase3.sync();
      setStatus(statusText);
    };
    img.onerror=()=>setStatus("Impossible de restaurer l'image"); img.src=src;
  }

  function clearSelection() {
    selectedMask=null; selectedLabel=""; state.aiMaskImg=null; drawMask();
    eraserBtn.textContent="✨ Remplacer IA";
    eraserBtn.classList.remove("active");
    state.eraserOn=false;
    const threeStage=document.getElementById("threeStage");
    if(threeStage) threeStage.style.pointerEvents="auto";
  }

  eraserBtn.addEventListener("click", async () => {
    if(busy) return;
    if(selectedMask){
      busy=true; eraserBtn.disabled=true; undoBtn.disabled=true;
      try{
        pushHistory();
        setStatus(`Effacement IA de « ${selectedLabel} » en cours…`);
        const imageBlob=await roomBlob();
        const maskCanvas=document.createElement("canvas"); maskCanvas.width=selectedMask.naturalWidth; maskCanvas.height=selectedMask.naturalHeight;
        maskCanvas.getContext("2d").drawImage(selectedMask,0,0);
        const maskBlob=await new Promise((resolve,reject)=>maskCanvas.toBlob(b=>b?resolve(b):reject(new Error("Masque invalide")),"image/png"));
        const fd=new FormData(); fd.append("file",imageBlob,"room.jpg"); fd.append("mask",maskBlob,"mask.png");
        const resp=await fetch(`${API}/inpaint`,{method:"POST",body:fd});
        if(!resp.ok){const e=await resp.json().catch(()=>({}));throw new Error(e.detail||`HTTP ${resp.status}`);}
        const data=await resp.json();
        loadRoom(data.image,`« ${selectedLabel} » effacé ✔ — ajoutez maintenant votre nouveau meuble`);
        clearSelection();
      }catch(err){console.error(err);setStatus(`Échec de la suppression IA : ${err.message}`);undoBtn.disabled=history.length===0;}
      finally{busy=false;eraserBtn.disabled=false;}
      return;
    }
    state.eraserOn=!state.eraserOn;
    eraserBtn.textContent=state.eraserOn?"✨ Sélection IA : cliquez un meuble":"✨ Remplacer IA";
    eraserBtn.classList.toggle("active",state.eraserOn);
    canvas.style.cursor=state.eraserOn?"crosshair":"default";
    const threeStage=document.getElementById("threeStage");
    if(threeStage) threeStage.style.pointerEvents=state.eraserOn?"none":"auto";
    setStatus(state.eraserOn?"Sélection IA — cliquez le meuble réel à remplacer":"Édition IA désactivée");
  });

  canvas.addEventListener("click",async e=>{
    if(!state.eraserOn||busy||!state.roomImage) return;
    if(hitItem({x:e.clientX-canvas.getBoundingClientRect().left,y:e.clientY-canvas.getBoundingClientRect().top})) return;
    const f=roomFit(); if(!f) return;
    const r=canvas.getBoundingClientRect();
    const ix=Math.round(((e.clientX-r.left-f.x)/f.w)*(state.roomImage.naturalWidth-1));
    const iy=Math.round(((e.clientY-r.top-f.y)/f.h)*(state.roomImage.naturalHeight-1));
    if(ix<0||iy<0||ix>=state.roomImage.naturalWidth||iy>=state.roomImage.naturalHeight)return;
    busy=true; eraserBtn.disabled=true; setStatus("IA : identification du meuble…");
    try{
      const blob=await roomBlob(); const fd=new FormData(); fd.append("file",blob,"room.jpg"); fd.append("x",String(ix)); fd.append("y",String(iy));
      const resp=await fetch(`${API}/select-mask`,{method:"POST",body:fd});
      if(!resp.ok){const e=await resp.json().catch(()=>({}));throw new Error(e.detail||`HTTP ${resp.status}`);}
      const data=await resp.json();
      const img=new Image(); img.onload=()=>{selectedMask=img;selectedLabel=data.label||"meuble";state.aiMaskImg=img;drawMask();eraserBtn.textContent=`✨ Confirmer suppression « ${selectedLabel} »`;setStatus(`Sélection IA ✔ — « ${selectedLabel} » · zone rouge = objet ciblé · cliquez à nouveau pour confirmer`);}; img.src=data.mask;
    }catch(err){console.error(err);setStatus(`Sélection IA impossible : ${err.message}`);}finally{busy=false;eraserBtn.disabled=false;}
  });

  undoBtn.addEventListener("click",()=>{
    if(!history.length||busy)return;
    const src=history.pop(); loadRoom(src,"Dernière modification IA annulée ✔");
    undoBtn.disabled=history.length===0;
  });

  window.addEventListener("keydown",e=>{
    if(e.key==="Escape" && (state.eraserOn||selectedMask)){clearSelection();canvas.style.cursor="default";setStatus("Sélection IA annulée");}
    if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="z"&&history.length){e.preventDefault();undoBtn.click();}
  });
  window.addEventListener("resize",syncOverlaySize);
  syncOverlaySize();
})();
