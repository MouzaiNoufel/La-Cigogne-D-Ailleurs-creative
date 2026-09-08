/* =========================================================
   Phase 3E — Photorealistic compositing / final render
   Uses the Phase 3D renderer as the physically-based foreground
   and performs a high-resolution photographic composite:
   - room color/exposure sampling
   - foreground edge feathering
   - shadow/contact reinforcement
   - local contrast / saturation harmonization
   - subtle film grain + vignette
   - 2K/4K final export
   - non-destructive preview controls
   ========================================================= */
(() => {
  'use strict';
  const App = window.App;
  const P3 = window.Phase3;
  if (!App) return;

  const stageWrap = document.querySelector('.stage-wrap');
  const inspectorBody = document.getElementById('inspectorBody');
  const state = {
    enabled: false,
    quality: '2k',
    exposure: 1.00,
    saturation: 0.96,
    contrast: 1.02,
    warmth: 0.00,
    edge: 0.65,
    shadow: 0.34,
    grain: 0.025,
    vignette: 0.06,
    preview: false,
    estimated: false
  };

  function setStatus(msg) { App.setStatus?.(msg); }
  function room() { return App.state.roomImage; }
  function threeCanvas() { return document.getElementById('threeStage'); }
  function roomFit() { return App.roomFit?.(); }
  function naturalSize() {
    const img = room();
    return img && img.naturalWidth ? {w: img.naturalWidth, h: img.naturalHeight} : App.getCanvasSize();
  }

  function estimatePhoto() {
    const img = room();
    if (!img) return;
    const w = Math.min(480, img.naturalWidth || 480);
    const h = Math.max(1, Math.round((img.naturalHeight || 320) * w / (img.naturalWidth || w)));
    const c = document.createElement('canvas'); c.width=w; c.height=h;
    const x=c.getContext('2d',{willReadFrequently:true}); x.drawImage(img,0,0,w,h);
    const d=x.getImageData(0,0,w,h).data;
    let r=0,g=0,b=0,n=0,lum=0;
    for(let i=0;i<d.length;i+=16){r+=d[i];g+=d[i+1];b+=d[i+2];lum+=.2126*d[i]+.7152*d[i+1]+.0722*d[i+2];n++;}
    if(!n)return;
    const L=lum/n/255, rb=(r/n)/(Math.max(1,b/n));
    state.exposure=Math.max(.84,Math.min(1.18,.96+(0.52-L)*.42));
    state.saturation=Math.max(.90,Math.min(1.08,.96+(rb-1)*.12));
    state.warmth=Math.max(-.12,Math.min(.12,(rb-1)*.22));
    state.estimated=true;
  }

  function drawRoom(ctx,w,h) {
    const img=room();
    if(img) ctx.drawImage(img,0,0,w,h);
    else {ctx.fillStyle='#0e1117';ctx.fillRect(0,0,w,h);}
  }

  function getForegroundLayer(w,h) {
    const src=threeCanvas();
    if(!src || !src.width || !src.height) return null;
    const c=document.createElement('canvas'); c.width=w; c.height=h;
    const x=c.getContext('2d');
    // The Three.js canvas is already the photo-matched, occlusion-aware
    // foreground. Scale it to the native room-photo resolution.
    x.drawImage(src,0,0,w,h);
    return c;
  }

  function featherAlpha(canvas, amount) {
    if(amount<=0) return canvas;
    // A tiny blur of the alpha only; preserve the PBR RGB detail.
    const w=canvas.width,h=canvas.height;
    const src=canvas.getContext('2d').getImageData(0,0,w,h);
    const a=new Uint8ClampedArray(w*h);
    for(let i=0,p=0;i<src.data.length;i+=4,p++)a[p]=src.data[i+3];
    const radius=Math.max(1,Math.round(amount*2.5));
    const out=new Uint8ClampedArray(a);
    for(let y=0;y<h;y++){
      for(let x=0;x<w;x++){
        let sum=0,cnt=0;
        for(let yy=Math.max(0,y-radius);yy<=Math.min(h-1,y+radius);yy+=Math.max(1,Math.floor(radius/2))){
          for(let xx=Math.max(0,x-radius);xx<=Math.min(w-1,x+radius);xx+=Math.max(1,Math.floor(radius/2))){
            sum+=a[yy*w+xx];cnt++;
          }
        }
        out[y*w+x]=sum/Math.max(1,cnt);
      }
    }
    const d=src.data;
    for(let p=0,i=0;p<out.length;p++,i+=4)d[i+3]=Math.max(0,Math.min(255,out[p]));
    canvas.getContext('2d').putImageData(src,0,0);
    return canvas;
  }

  function applyForegroundGrade(layer, exposure, sat, contrast, warmth) {
    const ctx=layer.getContext('2d');
    const w=layer.width,h=layer.height;
    const img=ctx.getImageData(0,0,w,h), d=img.data;
    const c=contrast, e=exposure;
    for(let i=0;i<d.length;i+=4){
      if(d[i+3]<2)continue;
      let r=d[i]/255,g=d[i+1]/255,b=d[i+2]/255;
      r=Math.max(0,Math.min(1,((r-.5)*c+.5)*e));
      g=Math.max(0,Math.min(1,((g-.5)*c+.5)*e));
      b=Math.max(0,Math.min(1,((b-.5)*c+.5)*e));
      const y=.2126*r+.7152*g+.0722*b;
      r=y+(r-y)*sat;g=y+(g-y)*sat;b=y+(b-y)*sat;
      if(warmth>0){r+=warmth*.06;b-=warmth*.035;}else{b+=(-warmth)*.045;r-=(-warmth)*.025;}
      d[i]=Math.max(0,Math.min(255,r*255));d[i+1]=Math.max(0,Math.min(255,g*255));d[i+2]=Math.max(0,Math.min(255,b*255));
    }
    ctx.putImageData(img,0,0);
  }

  function addSoftContact(ctx,w,h) {
    // Only a subtle global floor integration pass. Actual per-object contact
    // cards and physically based shadows are produced by Phase 3D.
    const fit=roomFit();
    if(!fit)return;
    const f=fit;
    const g=ctx.createRadialGradient(w*.5,h*.88,1,w*.5,h*.88,Math.max(w,h)*.30);
    g.addColorStop(0,'rgba(0,0,0,0.035)');g.addColorStop(1,'rgba(0,0,0,0)');
    ctx.save();ctx.globalCompositeOperation='multiply';ctx.fillStyle=g;ctx.fillRect(f.x,f.y,f.w,f.h);ctx.restore();
  }

  function gradeRoom(ctx,w,h) {
    if(Math.abs(state.exposure-1)<.001 && Math.abs(state.saturation-1)<.001 && Math.abs(state.contrast-1)<.001 && Math.abs(state.warmth)<.001)return;
    const img=ctx.getImageData(0,0,w,h),d=img.data;
    for(let i=0;i<d.length;i+=4){
      let r=d[i]/255,g=d[i+1]/255,b=d[i+2]/255;
      r=Math.max(0,Math.min(1,((r-.5)*state.contrast+.5)*state.exposure));
      g=Math.max(0,Math.min(1,((g-.5)*state.contrast+.5)*state.exposure));
      b=Math.max(0,Math.min(1,((b-.5)*state.contrast+.5)*state.exposure));
      const y=.2126*r+.7152*g+.0722*b;
      r=y+(r-y)*state.saturation;g=y+(g-y)*state.saturation;b=y+(b-y)*state.saturation;
      if(state.warmth>0){r+=state.warmth*.025;b-=state.warmth*.015;}else{b+=(-state.warmth)*.02;r-=(-state.warmth)*.012;}
      d[i]=Math.max(0,Math.min(255,r*255));d[i+1]=Math.max(0,Math.min(255,g*255));d[i+2]=Math.max(0,Math.min(255,b*255));
    }
    ctx.putImageData(img,0,0);
  }

  function finish(ctx,w,h) {
    if(state.grain>0){
      const c=document.createElement('canvas');c.width=Math.min(900,w);c.height=Math.min(900,h);
      const x=c.getContext('2d'),im=x.createImageData(c.width,c.height),d=im.data;
      for(let i=0;i<d.length;i+=4){const n=(Math.random()-.5)*255*state.grain;d[i]=128+n;d[i+1]=128+n;d[i+2]=128+n;d[i+3]=255;}
      x.putImageData(im,0,0);ctx.save();ctx.globalAlpha=.045;ctx.globalCompositeOperation='soft-light';ctx.drawImage(c,0,0,w,h);ctx.restore();
    }
    if(state.vignette>0){
      const g=ctx.createRadialGradient(w*.5,h*.5,Math.min(w,h)*.28,w*.5,h*.5,Math.max(w,h)*.72);
      g.addColorStop(.55,'rgba(0,0,0,0)');g.addColorStop(1,`rgba(0,0,0,${state.vignette})`);
      ctx.save();ctx.fillStyle=g;ctx.fillRect(0,0,w,h);ctx.restore();
    }
  }

  function renderFinal(showStatus=true) {
    if(!room()){setStatus('Importez une photo de pièce avant le rendu final');return null;}
    const native=naturalSize();
    const max=state.quality==='4k'?4096:state.quality==='native'?Math.max(native.w,native.h):2560;
    const scale=Math.min(1,max/Math.max(native.w,native.h));
    const w=Math.max(1,Math.round(native.w*scale)),h=Math.max(1,Math.round(native.h*scale));
    const out=document.createElement('canvas');out.width=w;out.height=h;
    const ctx=out.getContext('2d');
    drawRoom(ctx,w,h);
    gradeRoom(ctx,w,h);
    let fg=getForegroundLayer(w,h);
    if(fg){
      featherAlpha(fg,state.edge);
      applyForegroundGrade(fg,state.exposure,state.saturation,state.contrast,state.warmth);
      ctx.save();
      ctx.globalAlpha=1;
      ctx.drawImage(fg,0,0,w,h);
      ctx.restore();
    }
    if(state.shadow>0)addSoftContact(ctx,w,h);
    finish(ctx,w,h);
    if(showStatus)setStatus(`Rendu photoréaliste ${state.quality.toUpperCase()} prêt ✔`);
    return out;
  }

  function preview() {
    const out=renderFinal(false); if(!out)return;
    const existing=document.getElementById('p3ePreview');
    if(existing)existing.remove();
    const overlay=document.createElement('div');overlay.id='p3ePreview';
    Object.assign(overlay.style,{position:'absolute',inset:'12px',zIndex:'30',background:'#0e1117ee',display:'flex',alignItems:'center',justifyContent:'center',padding:'18px',cursor:'zoom-out'});
    const img=document.createElement('img');img.src=out.toDataURL('image/jpeg',.94);
    Object.assign(img.style,{maxWidth:'100%',maxHeight:'100%',objectFit:'contain',boxShadow:'0 18px 50px rgba(0,0,0,.45)',borderRadius:'8px'});
    overlay.appendChild(img);overlay.onclick=()=>overlay.remove();stageWrap.appendChild(overlay);
  }

  function download() {
    const out=renderFinal(true);if(!out)return;
    const a=document.createElement('a');a.download=`cigogne-photorealistic-${state.quality}.png`;a.href=out.toDataURL('image/png');a.click();
  }

  function renderInspector() {
    const old=document.querySelector('.phase3e-card');if(old)old.remove();
    const html=`<div class="phase3-card phase3e-card">
      <div class="phase3-title">✨ PHOTOREALISTIC COMPOSITOR · 3E</div>
      <div class="phase3-row"><span>Pipeline</span><strong>${state.estimated?'photo harmonisée ✔':'prête'}</strong></div>
      <div class="phase3-segment">
        <button id="p3ePreview" class="btn ghost">👁 Prévisualiser</button>
        <button id="p3eExport" class="btn">📸 Rendu final</button>
      </div>
      <div class="phase3-divider"></div>
      <div class="phase3-subtitle">🎨 PHOTO MATCH</div>
      <label class="p3-check"><input id="p3eAuto" type="checkbox" ${state.estimated?'checked':''}> Harmonisation automatique</label>
      <div class="field"><label>Exposition <span id="p3eExpVal">${state.exposure.toFixed(2)}</span></label><input id="p3eExp" type="range" min=".82" max="1.20" step=".01" value="${state.exposure}"></div>
      <div class="field"><label>Saturation <span id="p3eSatVal">${state.saturation.toFixed(2)}</span></label><input id="p3eSat" type="range" min=".85" max="1.12" step=".01" value="${state.saturation}"></div>
      <div class="field"><label>Contraste <span id="p3eConVal">${state.contrast.toFixed(2)}</span></label><input id="p3eCon" type="range" min=".88" max="1.16" step=".01" value="${state.contrast}"></div>
      <div class="field"><label>Température <span id="p3eWarmVal">${state.warmth.toFixed(2)}</span></label><input id="p3eWarm" type="range" min="-.12" max=".12" step=".01" value="${state.warmth}"></div>
      <div class="phase3-divider"></div>
      <div class="phase3-subtitle">🪶 INTÉGRATION</div>
      <div class="field"><label>Bords doux <span id="p3eEdgeVal">${Math.round(state.edge*100)}%</span></label><input id="p3eEdge" type="range" min="0" max="1" step=".05" value="${state.edge}"></div>
      <div class="field"><label>Contact / ombres <span id="p3eShadowVal">${Math.round(state.shadow*100)}%</span></label><input id="p3eShadow" type="range" min="0" max="1" step=".05" value="${state.shadow}"></div>
      <div class="field"><label>Grain photo <span id="p3eGrainVal">${Math.round(state.grain*100)}%</span></label><input id="p3eGrain" type="range" min="0" max=".08" step=".005" value="${state.grain}"></div>
      <div class="field"><label>Vignette <span id="p3eVigVal">${Math.round(state.vignette*100)}%</span></label><input id="p3eVig" type="range" min="0" max=".16" step=".01" value="${state.vignette}"></div>
      <div class="field"><label>Qualité export</label><select id="p3eQuality" style="width:100%;background:var(--bg-3);color:var(--text);border:1px solid var(--border);border-radius:7px;padding:8px"><option value="2k" ${state.quality==='2k'?'selected':''}>2K — rapide</option><option value="native" ${state.quality==='native'?'selected':''}>Native — résolution photo</option><option value="4k" ${state.quality==='4k'?'selected':''}>4K — max 4096 px</option></select></div>
      <p class="hint">Le rendu combine le fond photographique, le rendu PBR/ombres de Phase 3D, l'occlusion profondeur IA et une passe de harmonisation photographique. Ce module ne prétend pas générer de nouveaux pixels avec un modèle diffusion.</p>
    </div>`;
    inspectorBody.insertAdjacentHTML('beforeend',html);
    const q=id=>document.getElementById(id);
    if(state.estimated)q('p3eAuto').checked=true;
    q('p3eAuto').onchange=e=>{if(e.target.checked){estimatePhoto();renderInspector();}else state.estimated=false;};
    const bind=(id,key,fmt)=>q(id).oninput=e=>{state[key]=+e.target.value;q(id+'Val').textContent=fmt(state[key]);};
    bind('p3eExp','exposure',v=>v.toFixed(2));bind('p3eSat','saturation',v=>v.toFixed(2));bind('p3eCon','contrast',v=>v.toFixed(2));bind('p3eWarm','warmth',v=>v.toFixed(2));
    bind('p3eEdge','edge',v=>Math.round(v*100)+'%');bind('p3eShadow','shadow',v=>Math.round(v*100)+'%');bind('p3eGrain','grain',v=>Math.round(v*100)+'%');bind('p3eVig','vignette',v=>Math.round(v*100)+'%');
    q('p3eQuality').onchange=e=>state.quality=e.target.value;
    q('p3ePreview').onclick=preview;q('p3eExport').onclick=download;
  }

  function init() {
    if(room())estimatePhoto();
    // Render the 3E inspector only when a room exists or when 3D is active.
    const observer=new MutationObserver(()=>{if(room() && !document.querySelector('.phase3e-card'))renderInspector();});
    observer.observe(inspectorBody,{childList:true});
    if(room())renderInspector();
    window.Phase3E={state,render:renderFinal,preview,export:download,refresh:renderInspector};
  }
  init();
})();
