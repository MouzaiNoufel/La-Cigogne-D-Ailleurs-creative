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
    octx.save();
    octx.globalAlpha = .38;
    octx.drawImage(selectedMask, fit.x, fit.y, fit.w, fit.h);
    octx.globalCompositeOperation = "source-in";
    octx.fillStyle = "rgba(217,164,65,.9)";
    octx.fillRect(fit.x, fit.y, fit.w, fit.h);
    octx.restore();

    octx.save();
    octx.globalAlpha = .95;
    octx.strokeStyle = "rgba(255,220,120,.95)";
    octx.lineWidth = 2;
    octx.setLineDash([7,5]);
    const mw = selectedMask.naturalWidth, mh = selectedMask.naturalHeight;
    // Outline by sampling the mask onto a small temporary canvas.
    const tmp = document.createElement("canvas"); tmp.width = mw; tmp.height = mh;
    const tc = tmp.getContext("2d"); tc.drawImage(selectedMask,0,0);
    const px = tc.getImageData(0,0,mw,mh).data;
    const step = Math.max(1, Math.round(Math.max(mw,mh)/700));
    octx.beginPath();
    for (let y=step; y<mh-step; y+=step) for (let x=step; x<mw-step; x+=step) {
      const i=(y*mw+x)*4; if(px[i+3]<80) continue;
      const edge = px[((y-step)*mw+x)*4+3]<80 || px[((y+step)*mw+x)*4+3]<80 || px[(y*mw+x-step)*4+3]<80 || px[(y*mw+x+step)*4+3]<80;
      if(edge){ const sx=fit.x+x/mw*fit.w, sy=fit.y+y/mh*fit.h; octx.rect(sx,sy,Math.max(1,step/mw*fit.w),Math.max(1,step/mh*fit.h)); }
    }
    octx.stroke(); octx.restore();
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
      }catch(err){console.error(err);setStatus(`Échec de l'effacement IA : ${err.message}`);undoBtn.disabled=history.length===0;}
      finally{busy=false;eraserBtn.disabled=false;}
      return;
    }
    state.eraserOn=!state.eraserOn;
    eraserBtn.textContent=state.eraserOn?"✨ Sélection IA : cliquez un meuble":"✨ Remplacer IA";
    eraserBtn.classList.toggle("active",state.eraserOn);
    canvas.style.cursor=state.eraserOn?"crosshair":"default";
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
      const img=new Image(); img.onload=()=>{selectedMask=img;selectedLabel=data.label||"meuble";state.aiMaskImg=img;drawMask();eraserBtn.textContent=`✨ Effacer « ${selectedLabel} »`;setStatus(`Sélection IA ✔ — « ${selectedLabel} » · cliquez à nouveau pour effacer`);}; img.src=data.mask;
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
