/* =========================================================
   Phase 3D — Room reconstruction proxy + lighting match
   Adds:
   - AI depth texture compositing against the 3D depth buffer
   - floor-calibrated depth -> approximate metric mapping
   - adjustable occlusion threshold / strength
   - foreground furniture can hide newly placed 3D objects
   - transform edits persist back to the 2D editor
   - keeps Phase 3C PBR/photo-match workflow intact
   - room proxy planes from floor/depth calibration
   - automatic photographic lighting estimation
   - color-temperature / exposure matching
   - per-object contact shadow cards
   ========================================================= */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

(() => {
  'use strict';

  const App = window.App;
  const AppActions = window.AppActions;
  if (!App || !AppActions) return;

  const stageWrap = document.querySelector('.stage-wrap');
  const canvas = document.getElementById('threeStage');
  const modeBtn = document.getElementById('threeModeBtn');
  const glbInput = document.getElementById('glbUpload');
  const glbBtn = document.getElementById('glbBtn');
  const inspectorBody = document.getElementById('inspectorBody');
  const status = App.setStatus;

  const state = {
    enabled: false,
    scene: new THREE.Scene(),
    camera: null,
    renderer: null,
    orbit: null,
    transform: null,
    raycaster: new THREE.Raycaster(),
    pointer: new THREE.Vector2(),
    floor: null,
    shadow: null,
    groups: new Map(),
    assets: new Map(),
    loader: new GLTFLoader(),
    environment: null,
    calibration: { height: 2.7, fov: 52, depth: 6.5, targetY: 1.05, pitch: 0.23, auto: true },
    mode: 'photo',
    selectedUid: null,
    roomDepthTexture: null,
    roomDepthCanvas: null,
    roomDepthCtx: null,
    roomDepthNearIsHigh: true,
    depthFit: { a: 10, b: 0.6, ready: false },
    occlusion: { enabled: true, strength: 0.82, threshold: 0.18 },
    lighting: { auto: true, temperature: 0.50, exposure: 1.05, key: 2.8, ambient: 1.35, fill: 0.65, estimated: false },
    roomProxy: { enabled: true, wallDepth: 7.5, width: 12, height: 5.0 },
    keyLight: null, ambientLight: null, fillLight: null,
    proxyGroup: null, contactTexture: null,
    target: null,
    postScene: null,
    postCamera: null,
    postMaterial: null,
    postQuad: null,
    depthTexture: null,
    last: performance.now(),
  };

  function size() { return { w: Math.max(1, stageWrap.clientWidth), h: Math.max(1, stageWrap.clientHeight) }; }
  function catalogItem(item) { return (window.CATALOG || []).find(x => x.id === item.catId) || null; }
  function color(item) { return new THREE.Color(catalogItem(item)?.color || '#9aa4b2'); }
  function material(c, roughness = .62, metalness = .04) { return new THREE.MeshStandardMaterial({ color: c, roughness, metalness }); }
  function box(w, h, d, c, y = h / 2, rough = .68) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material(c, rough));
    m.position.y = y; m.castShadow = true; m.receiveShadow = true; return m;
  }

  function procedural(catId, item) {
    const root = new THREE.Group(), c = color(item), w = item.w, d = item.d;
    if (catId === 'sofa') {
      root.add(box(w,.28,d,c,.18)); root.add(box(w,.68,.20,c.clone().offsetHSL(0,-.03,-.08),.62));
      for (const x of [-w/2+.08,w/2-.08]) { const a=box(.16,.70,d,c.clone().offsetHSL(0,-.03,-.10),.52); a.position.x=x; root.add(a); }
      for (const x of [-w*.24,0,w*.24]) { const q=box(w*.20,.13,.48,new THREE.Color('#d6c7ba'),.48,.88); q.position.set(x,.48,.08); root.add(q); }
    } else if (catId === 'armchair') {
      root.add(box(w,.28,d,c,.18)); root.add(box(w-.16,.52,.16,c,.55));
      for (const x of [-w/2+.08,w/2-.08]) { const a=box(.15,.62,d,c.clone().offsetHSL(0,-.03,-.10),.47); a.position.x=x; root.add(a); }
      root.add(box(w*.58,.12,d*.55,new THREE.Color('#d8c9df'),.48));
    } else if (catId === 'bed') {
      root.add(box(w,.30,d,c.clone().offsetHSL(0,-.04,-.08),.15)); const h=box(w,.55,.18,c.clone().offsetHSL(0,-.03,-.12),.58); h.position.z=-d/2+.09; root.add(h);
      const mattress=box(w*.92,.10,d*.74,new THREE.Color('#eee9e0'),.37); mattress.position.z=.14; root.add(mattress);
      for (const x of [-w*.25,w*.25]) { const p=box(w*.22,.09,.40,new THREE.Color('#f8f6f1'),.46,.9); p.position.set(x,.46,-d/2+.43); root.add(p); }
    } else if (catId === 'table') {
      root.add(box(w,.09,d,c,.79,.55)); for (const sx of [-1,1]) for (const sz of [-1,1]) { const l=box(.065,.76,.065,c.clone().offsetHSL(0,-.04,-.13),.38,.55); l.position.x=sx*(w/2-.09); l.position.z=sz*(d/2-.09); root.add(l); }
    } else if (catId === 'chair') {
      root.add(box(w,.10,d,c,.48)); const back=box(w,.52,.08,c,.74); back.position.z=-d/2+.04; root.add(back);
      for (const sx of [-1,1]) for (const sz of [-1,1]) { const l=box(.045,.45,.045,c.clone().offsetHSL(0,-.04,-.12),.225,.55); l.position.x=sx*(w/2-.06); l.position.z=sz*(d/2-.06); root.add(l); }
    } else if (catId === 'lamp') {
      const pole=new THREE.Mesh(new THREE.CylinderGeometry(.022,.028,1.45,20),material(c,.38,.10)); pole.position.y=.73; pole.castShadow=true; root.add(pole);
      const shade=new THREE.Mesh(new THREE.ConeGeometry(.23,.30,32,1,true),material(c,.52,.02)); shade.position.y=1.43; shade.castShadow=true; root.add(shade);
      const base=new THREE.Mesh(new THREE.CylinderGeometry(.15,.17,.07,28),material(c.clone().offsetHSL(0,-.05,-.15),.35,.12)); base.position.y=.035; root.add(base);
    } else if (catId === 'plant') {
      const pot=new THREE.Mesh(new THREE.CylinderGeometry(.15,.20,.25,24),material(new THREE.Color('#8a5a33'),.8)); pot.position.y=.125; pot.castShadow=true; root.add(pot);
      for(let i=0;i<9;i++){const leaf=new THREE.Mesh(new THREE.SphereGeometry(.13,16,10),material(c,.9)); leaf.scale.set(1,.65,.48); leaf.position.set(Math.sin(i*1.8)*.15,.38+(i%4)*.10,Math.cos(i*1.8)*.15); leaf.rotation.z=i%2?-.3:.3; root.add(leaf);}
    } else if (catId === 'rug') {
      const rug=new THREE.Mesh(new THREE.BoxGeometry(w,.035,d),new THREE.MeshStandardMaterial({color:c,roughness:.96})); rug.position.y=.018; rug.receiveShadow=true; root.add(rug);
      const border=new THREE.Mesh(new THREE.BoxGeometry(w*.86,.012,d*.74),new THREE.MeshStandardMaterial({color:c.clone().offsetHSL(0,-.08,.10),roughness:1})); border.position.y=.04; border.receiveShadow=true; root.add(border);
    } else if (catId === 'tvstand') {
      root.add(box(w,.55,d,c,.275)); root.add(box(w*.85,.04,d+.03,c.clone().offsetHSL(0,-.04,.12),.56,.48));
      for(const x of [-w*.25,w*.25]){const door=box(w*.19,.35,.015,c.clone().offsetHSL(0,-.02,.10),.29,.58); door.position.x=x; door.position.z=d/2+.008; root.add(door);}
    } else root.add(box(w,.5,d,c,.25));
    return root;
  }

  function cloneAsset(scene) {
    const root=scene.clone(true); root.traverse(o=>{if(!o.isMesh)return;o.castShadow=true;o.receiveShadow=true;if(o.material?.map)o.material.map.colorSpace=THREE.SRGBColorSpace;}); return root;
  }
  function modelDimensions(root){const b=new THREE.Box3().setFromObject(root),s=new THREE.Vector3();b.getSize(s);return{box:b,x:Math.max(.0001,s.x),y:Math.max(.0001,s.y),z:Math.max(.0001,s.z)};}
  function normalizeAsset(root,item){
    const d=modelDimensions(root), tw=Math.max(.08,item.w), td=Math.max(.08,item.d), direct=Math.max(tw/d.x,td/d.z), swapped=Math.max(tw/d.z,td/d.x);
    const useSwap=Math.abs(tw-d.z*swapped)+Math.abs(td-d.x*swapped)<Math.abs(tw-d.x*direct)+Math.abs(td-d.z*direct);
    if(useSwap)root.rotation.y=Math.PI/2; const fd=useSwap?{x:d.z,z:d.x}:{x:d.x,z:d.z}; root.scale.setScalar((tw/Math.max(.0001,fd.x)+td/Math.max(.0001,fd.z))/2);
    const after=new THREE.Box3().setFromObject(root),center=new THREE.Vector3();after.getCenter(center);root.position.sub(center);const floorBox=new THREE.Box3().setFromObject(root);root.position.y-=floorBox.min.y;
  }
  function createGroup(item){
    const g=new THREE.Group();g.userData.uid=item.uid;g.userData.item=item;const asset=state.assets.get(item.catId);const model=asset?cloneAsset(asset):procedural(item.catId,item);if(asset)normalizeAsset(model,item);g.add(model);g.traverse(o=>{if(o.isMesh){o.castShadow=true;o.receiveShadow=true;}});addContactCard(g,item);state.scene.add(g);state.groups.set(item.uid,g);return g;
  }
  function disposeObject(obj){obj.traverse(o=>{if(o.geometry)o.geometry.dispose();if(o.material){const ms=Array.isArray(o.material)?o.material:[o.material];ms.forEach(m=>m.dispose());}});state.scene.remove(obj);}
  function rect(){return canvas.getBoundingClientRect();}

  function screenToFloor(clientX,clientY){
    const r=rect();state.pointer.set(((clientX-r.left)/r.width)*2-1,-((clientY-r.top)/r.height)*2+1);state.raycaster.setFromCamera(state.pointer,state.camera);const ray=state.raycaster.ray;if(Math.abs(ray.direction.y)<1e-6)return null;const t=-ray.origin.y/ray.direction.y;return t>0?ray.origin.clone().addScaledVector(ray.direction,t):null;
  }
  function support(item){return App.getSupportPoint(item);}
  function positionFrom2D(item,g){
    const fit=App.roomFit();if(!fit||!App.state.roomImage)return;const sp=support(item),r=rect(),cs=App.getCanvasSize(),sx=r.width/cs.width,sy=r.height/cs.height,p=screenToFloor(r.left+sp.x*sx,r.top+sp.y*sy);if(p)g.position.set(p.x,0,p.z);g.rotation.y=-item.rot;g.scale.setScalar(Math.max(.55,item.scale*App.getPerspectiveFactor(item)));
  }
  function syncAll(){
    if(!state.enabled)return;
    if(state.lighting.auto && !state.lighting.estimated) { estimateRoomLighting(); applyLighting(); }
    updateRoomProxy();const live=new Set(App.state.items.map(i=>i.uid));for(const[uid,g]of state.groups)if(!live.has(uid)){disposeObject(g);state.groups.delete(uid);}for(const item of App.state.items){const g=state.groups.get(item.uid)||createGroup(item);if(!g.userData.edited3D)positionFrom2D(item,g);g.visible=true;}refreshSelection();updateDepthCalibration();
  }
  function resize(){const{w,h}=size(),dpr=Math.min(2,window.devicePixelRatio||1);state.renderer.setPixelRatio(dpr);state.renderer.setSize(w,h,false);state.camera.aspect=w/h;state.camera.updateProjectionMatrix();if(state.target)state.target.setSize(Math.round(w*dpr),Math.round(h*dpr));}

  function derivePitchFromFloor(){
    const a=App.state.analysis;if(!state.calibration.auto||!a||!Array.isArray(a.floor_top_profile)||!a.floor_top_profile.length)return;const H=a.height||App.state.roomImage?.naturalHeight||1000,median=a.floor_top_y||(a.floor_top_profile.reduce((x,y)=>x+y,0)/a.floor_top_profile.length),fovRad=THREE.MathUtils.degToRad(state.calibration.fov),vAngle=(.5-median/H)*fovRad;state.calibration.pitch=Math.max(-.15,Math.min(.9,vAngle));const h=state.calibration.height;state.calibration.depth=Math.max(2.2,Math.min(14,(h-state.calibration.targetY)/Math.tan(Math.max(.08,state.calibration.pitch))));
  }
  function updateCamera(){
    derivePitchFromFloor();const c=state.calibration;state.camera.fov=c.fov;state.camera.position.set(0,c.height,c.depth);const lookZ=c.depth-(c.height-c.targetY)/Math.tan(Math.max(.08,c.pitch));state.camera.lookAt(0,c.targetY,Number.isFinite(lookZ)?lookZ:0);state.camera.updateProjectionMatrix();if(state.orbit){state.orbit.target.set(0,c.targetY,0);state.orbit.update();}
  }

  // Fit the AI's relative depth values to the 3D scene using pixels that the
  // segmentation model says belong to the floor. This makes occlusion useful
  // without pretending Depth Anything is metric depth.
  function updateDepthCalibration(){
    const a=App.state.analysis, dImg=App.state.depthImg, mask=App.state.floorMaskPx, mImg=App.state.floorMaskImg;
    if(!a||!dImg||!mask||!mImg||!state.camera)return;
    state.roomDepthNearIsHigh=Boolean(a.depth_near_is_high);
    const W=dImg.naturalWidth,H=dImg.naturalHeight,mW=mImg.naturalWidth,mH=mImg.naturalHeight;
    const data=document.createElement('canvas');data.width=W;data.height=H;const c=data.getContext('2d',{willReadFrequently:true});c.drawImage(dImg,0,0,W,H);const dp=c.getImageData(0,0,W,H).data;
    const step=Math.max(10,Math.floor(Math.sqrt(W*H/2600))), samples=[];
    for(let y=Math.floor(H*.12);y<H;y+=step){
      for(let x=step/2;x<W;x+=step){
        const mx=Math.round(x/(W-1)*(mW-1)),my=Math.round(y/(H-1)*(mH-1));
        if(mask[(my*mW+mx)*4]<128)continue;
        const raw=dp[(y*W+Math.round(x))*4]/255, near=state.roomDepthNearIsHigh?raw:1-raw;
        const r=rect(), cs=App.getCanvasSize(),fit=App.roomFit();if(!fit)continue;
        const cx=r.left+(fit.x+(x/(W-1))*fit.w)*(r.width/cs.width), cy=r.top+(fit.y+(y/(H-1))*fit.h)*(r.height/cs.height);
        const p=screenToFloor(cx,cy);if(!p||!Number.isFinite(p.z)||p.z<.2||p.z>30)continue;
        samples.push([near,Math.abs(p.z)]);
      }
    }
    if(samples.length<40)return;
    // Robust-ish linear fit after trimming extreme residual candidates.
    let sx=0,sy=0,sxx=0,sxy=0;for(const[q,z]of samples){sx+=q;sy+=z;sxx+=q*q;sxy+=q*z;}const n=samples.length,den=n*sxx-sx*sx;if(Math.abs(den)<1e-7)return;let A=(n*sxy-sx*sy)/den,B=(sy-A*sx)/n;A=Math.max(.5,Math.min(30,A));B=Math.max(.1,Math.min(8,B));state.depthFit={a:A,b:B,ready:true};
  }

  function refreshSelection(){
    const selected=App.getSelected();state.selectedUid=selected?.uid||null;for(const[uid,g]of state.groups){g.traverse(o=>{if(!o.isMesh||!o.material)return;const ms=Array.isArray(o.material)?o.material:[o.material];ms.forEach(m=>{if(!m.userData)m.userData={};if(!m.userData.p3Base)m.userData.p3Base=m.emissive?m.emissive.clone():new THREE.Color(0);if(m.emissive)m.emissive.copy(m.userData.p3Base).lerp(new THREE.Color('#d9a441'),uid===state.selectedUid?.12:0);});});}if(state.transform){const g=selected?state.groups.get(selected.uid):null;state.transform.detach();if(g&&state.enabled&&state.mode==='photo')state.transform.attach(g);}
  }
  function luminance(r,g,b){ return (0.2126*r + 0.7152*g + 0.0722*b) / 255; }
  function estimateRoomLighting(){
    const img=App.state.roomImage;
    if(!img || !state.lighting.auto) return;
    const w=Math.min(360,img.naturalWidth||360), h=Math.max(1,Math.round((img.naturalHeight||240)*w/(img.naturalWidth||w)));
    const c=document.createElement('canvas'); c.width=w; c.height=h; const ctx=c.getContext('2d',{willReadFrequently:true});
    ctx.drawImage(img,0,0,w,h); const data=ctx.getImageData(0,0,w,h).data;
    let sum=0, wx=0, wy=0, red=0, blue=0, count=0;
    const step=Math.max(3,Math.floor(Math.min(w,h)/70));
    for(let y=0;y<h;y+=step){
      for(let x=0;x<w;x+=step){
        const i=(y*w+x)*4, r=data[i],g=data[i+1],b=data[i+2], lum=luminance(r,g,b);
        const weight=Math.pow(Math.max(0,lum-.48),1.7);
        sum+=lum; red+=r; blue+=b; count++;
        if(weight>0){ wx+=x*weight; wy+=y*weight; }
      }
    }
    if(!count)return;
    const mean=sum/count, cx=wx/(Math.max(1,wx?sum*0+1:1));
    // Recompute the bright centroid with a bounded denominator so a dark room
    // still produces a stable neutral key direction.
    let bright=0,bx=0,by=0;
    for(let y=0;y<h;y+=step){ for(let x=0;x<w;x+=step){ const i=(y*w+x)*4; const lum=luminance(data[i],data[i+1],data[i+2]); const q=Math.pow(Math.max(0,lum-.50),1.9); bright+=q; bx+=x*q; by+=y*q; }}
    const px=bright>1e-5?bx/bright:w*.5, py=bright>1e-5?by/bright:h*.35;
    const nx=(px/(w-1))-.5, ny=(py/(h-1))-.5;
    const rb=(red/Math.max(1,count))/(blue/Math.max(1,count));
    // 0 = cool, 1 = warm. Clamp to avoid extreme casts from JPEG noise.
    state.lighting.temperature=THREE.MathUtils.clamp(.5 + (rb-1)*.95,.15,.85);
    state.lighting.ambient=THREE.MathUtils.clamp(.72 + mean*1.15,.72,1.65);
    state.lighting.key=THREE.MathUtils.clamp(1.8 + mean*2.6,1.8,4.2);
    state.lighting.fill=THREE.MathUtils.clamp(.35 + mean*.65,.35,1.0);
    state.lighting.estimated=true;
    state.lighting._brightX=nx; state.lighting._brightY=ny;
  }
  function kelvinColor(t){
    // UI temperature is normalized 0..1. Map to 3000K..7000K.
    const k=3000+t*4000, temp=k/100, r=k<=6600?255:329.698727*Math.pow(temp-60,-0.1332047592), g=k<=6600?99.470802586*Math.log(temp)-161.119568166:288.1221695283*Math.pow(temp-60,-0.0755148492), b=k>=6600?255:k<=1900?0:138.5177312231*Math.log(temp-10)-305.0447927307;
    return new THREE.Color(Math.max(0,Math.min(255,r))/255,Math.max(0,Math.min(255,g))/255,Math.max(0,Math.min(255,b))/255);
  }
  function setupRoomProxy(){
    state.proxyGroup=new THREE.Group(); state.proxyGroup.name='AI Room Proxy'; state.scene.add(state.proxyGroup);
    const floorMat=new THREE.ShadowMaterial({opacity:.26});
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(state.roomProxy.width,18),floorMat); floor.rotation.x=-Math.PI/2; floor.position.y=.004; floor.receiveShadow=true; floor.name='floor-shadow-catcher'; state.proxyGroup.add(floor); state.floor=floor;
    const wallMat=new THREE.ShadowMaterial({opacity:.12});
    const wall=new THREE.Mesh(new THREE.PlaneGeometry(state.roomProxy.width,state.roomProxy.height),wallMat); wall.position.set(0,state.roomProxy.height*.5,-state.roomProxy.wallDepth); wall.receiveShadow=true; wall.name='back-wall-shadow-catcher'; state.proxyGroup.add(wall);
    const left=new THREE.Mesh(new THREE.PlaneGeometry(18,state.roomProxy.height),wallMat.clone()); left.rotation.y=Math.PI/2; left.position.set(-state.roomProxy.width*.5,state.roomProxy.height*.5,-state.roomProxy.wallDepth*.35); left.receiveShadow=true; left.name='left-wall-shadow-catcher'; state.proxyGroup.add(left);
    const right=new THREE.Mesh(new THREE.PlaneGeometry(18,state.roomProxy.height),wallMat.clone()); right.rotation.y=-Math.PI/2; right.position.set(state.roomProxy.width*.5,state.roomProxy.height*.5,-state.roomProxy.wallDepth*.35); right.receiveShadow=true; right.name='right-wall-shadow-catcher'; state.proxyGroup.add(right);
  }
  function updateRoomProxy(){
    if(!state.proxyGroup)return;
    state.proxyGroup.visible=state.roomProxy.enabled;
    const children=state.proxyGroup.children;
    const floor=children[0], wall=children[1], left=children[2], right=children[3];
    floor.scale.x=state.roomProxy.width/12;
    wall.position.z=-state.roomProxy.wallDepth; left.position.x=-state.roomProxy.width*.5; right.position.x=state.roomProxy.width*.5;
    left.position.z=right.position.z=-state.roomProxy.wallDepth*.35;
  }
  function contactTexture(){
    if(state.contactTexture)return state.contactTexture;
    const c=document.createElement('canvas'); c.width=128;c.height=64;const x=c.getContext('2d');
    const g=x.createRadialGradient(64,32,2,64,32,60);g.addColorStop(0,'rgba(0,0,0,.48)');g.addColorStop(.45,'rgba(0,0,0,.20)');g.addColorStop(1,'rgba(0,0,0,0)');x.fillStyle=g;x.fillRect(0,0,128,64);
    state.contactTexture=new THREE.CanvasTexture(c);state.contactTexture.colorSpace=THREE.SRGBColorSpace;return state.contactTexture;
  }
  function addContactCard(g,item){
    const m=new THREE.SpriteMaterial({map:contactTexture(),transparent:true,opacity:.46,depthWrite:false,depthTest:true});
    const s=new THREE.Sprite(m);s.scale.set(Math.max(.18,item.w*.92),Math.max(.10,item.d*.42),1);s.position.set(0,.008,.03);s.renderOrder=-1;g.add(s);g.userData.contact=s;
  }
  function applyLighting(){
    if(!state.keyLight)return;
    const t=state.lighting.temperature, warm=kelvinColor(t), dir=state.lighting._brightX||-.18;
    state.keyLight.color.copy(warm); state.keyLight.intensity=state.lighting.key; state.keyLight.position.set(THREE.MathUtils.clamp(dir*8,-6,6),6.5,4.5);
    state.ambientLight.intensity=state.lighting.ambient; state.fillLight.intensity=state.lighting.fill; state.renderer.toneMappingExposure=state.lighting.exposure;
  }
  function setupLighting(){
    const pmrem=new THREE.PMREMGenerator(state.renderer),env=new RoomEnvironment();state.environment=pmrem.fromScene(env,.04).texture;state.scene.environment=state.environment;env.dispose();pmrem.dispose();
    state.ambientLight=new THREE.HemisphereLight(0xffffff,0x6b6b6b,1.35);state.scene.add(state.ambientLight);
    state.keyLight=new THREE.DirectionalLight(0xfff5e8,2.8);state.keyLight.position.set(-3.5,6,4);state.keyLight.castShadow=true;state.keyLight.shadow.mapSize.set(2048,2048);state.keyLight.shadow.camera.near=.1;state.keyLight.shadow.camera.far=24;state.keyLight.shadow.camera.left=-8;state.keyLight.shadow.camera.right=8;state.keyLight.shadow.camera.top=8;state.keyLight.shadow.camera.bottom=-8;state.keyLight.shadow.bias=-.0005;state.scene.add(state.keyLight);
    state.fillLight=new THREE.DirectionalLight(0xcfe1ff,.65);state.fillLight.position.set(4,3,-2);state.scene.add(state.fillLight);
    estimateRoomLighting();applyLighting();
  }
  function makeShadowCatcher(){const geo=new THREE.PlaneGeometry(18,18),mat=new THREE.ShadowMaterial({opacity:.24});state.shadow=new THREE.Mesh(geo,mat);state.shadow.rotation.x=-Math.PI/2;state.shadow.position.y=.002;state.shadow.receiveShadow=true;state.shadow.renderOrder=-2;state.scene.add(state.shadow);}

  function setupPostProcess(){
    const dpr=Math.min(2,window.devicePixelRatio||1),{w,h}=size();
    state.target=new THREE.WebGLRenderTarget(Math.round(w*dpr),Math.round(h*dpr),{minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,format:THREE.RGBAFormat,depthBuffer:true,stencilBuffer:false});
    state.depthTexture=new THREE.DepthTexture(Math.round(w*dpr),Math.round(h*dpr),THREE.UnsignedIntType);state.target.depthTexture=state.depthTexture;state.target.depthTexture.minFilter=THREE.NearestFilter;state.target.depthTexture.magFilter=THREE.NearestFilter;
    state.postScene=new THREE.Scene();state.postCamera=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
    state.postMaterial=new THREE.ShaderMaterial({transparent:true,depthTest:false,depthWrite:false,uniforms:{uColor:{value:state.target.texture},uDepth:{value:state.target.depthTexture},uRoomDepth:{value:null},uRoomRect:{value:new THREE.Vector4(0,0,1,1)},uNear:{value:.05},uFar:{value:100},uFitA:{value:10},uFitB:{value:.6},uThreshold:{value:.18},uStrength:{value:.82},uEnabled:{value:1}},vertexShader:`varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}`,fragmentShader:`precision highp float;varying vec2 vUv;uniform sampler2D uColor;uniform sampler2D uDepth;uniform sampler2D uRoomDepth;uniform vec4 uRoomRect;uniform float uNear,uFar,uFitA,uFitB,uThreshold,uStrength;uniform int uEnabled;float linearDepth(float z){float ndc=z*2.0-1.0;return (2.0*uNear*uFar)/(uFar+uNear-ndc*(uFar-uNear));}void main(){vec4 col=texture2D(uColor,vUv);if(col.a<0.002){discard;}if(uEnabled==1&&uRoomDepth!=null){vec2 uv=(vUv-uRoomRect.xy)/uRoomRect.zw;if(uv.x<0.0||uv.x>1.0||uv.y<0.0||uv.y>1.0){gl_FragColor=col;return;}float raw=texture2D(uRoomDepth,uv).r;float near=raw;float roomZ=uFitA*near+uFitB;float objZ=linearDepth(texture2D(uDepth,vUv).r);float delta=roomZ-objZ;float occl=smoothstep(-uThreshold,uThreshold,delta);if(delta< -uThreshold){col.a*=1.0-uStrength;}else if(delta<uThreshold){col.a*=mix(1.0,0.0,uStrength*(1.0-occl));}}gl_FragColor=col;}`});
    // GLSL ES does not allow comparing a sampler to null. Replace the check
    // with a feature flag only; sampler is always bound after initialization.
    state.postMaterial.fragmentShader=state.postMaterial.fragmentShader.replace('&&uRoomDepth!=null','');state.postMaterial.needsUpdate=true;
    state.postQuad=new THREE.Mesh(new THREE.PlaneGeometry(2,2),state.postMaterial);state.postScene.add(state.postQuad);
  }

  function updatePostUniforms(){
    if(!state.postMaterial)return;const u=state.postMaterial.uniforms;u.uRoomDepth.value=state.roomDepthTexture||state.target.texture;const fit=App.roomFit(),cs=App.getCanvasSize();u.uRoomRect.value=fit?new THREE.Vector4(fit.x/cs.width,fit.y/cs.height,fit.w/cs.width,fit.h/cs.height):new THREE.Vector4(0,0,1,1);u.uFitA.value=state.depthFit.a;u.uFitB.value=state.depthFit.b;u.uThreshold.value=state.occlusion.threshold;u.uStrength.value=state.occlusion.enabled?state.occlusion.strength:0;u.uEnabled.value=state.occlusion.enabled&&state.depthFit.ready?1:0;u.uNear.value=state.camera.near;u.uFar.value=state.camera.far;
  }

  function loadRoomDepth(){
    if(!App.state.depthImg)return;const img=App.state.depthImg,W=img.naturalWidth,H=img.naturalHeight;state.roomDepthCanvas=document.createElement('canvas');state.roomDepthCanvas.width=W;state.roomDepthCanvas.height=H;state.roomDepthCtx=state.roomDepthCanvas.getContext('2d');state.roomDepthCtx.drawImage(img,0,0);state.roomDepthTexture=new THREE.CanvasTexture(state.roomDepthCanvas);state.roomDepthTexture.colorSpace=THREE.NoColorSpace;state.roomDepthTexture.minFilter=THREE.LinearFilter;state.roomDepthTexture.magFilter=THREE.LinearFilter;updateDepthCalibration();updatePostUniforms();
  }

  function setupTransform(){
    state.transform=new TransformControls(state.camera,canvas);state.transform.setMode('translate');state.transform.setSpace('world');state.transform.setSize(.75);
    state.transform.addEventListener('dragging-changed',e=>{if(state.orbit)state.orbit.enabled=!e.value&&state.mode==='orbit';});
    state.transform.addEventListener('objectChange',()=>{const g=state.transform.object;if(!g?.userData?.uid)return;g.userData.edited3D=true;});
    state.transform.addEventListener('mouseUp',()=>{const g=state.transform.object;if(g)commitGroupTo2D(g);});
    state.scene.add(state.transform.getHelper());
  }
  function hit3D(clientX,clientY){const r=rect();state.pointer.set(((clientX-r.left)/r.width)*2-1,-((clientY-r.top)/r.height)*2+1);state.raycaster.setFromCamera(state.pointer,state.camera);for(const hit of state.raycaster.intersectObjects([...state.groups.values()],true)){let o=hit.object;while(o&&!o.userData.uid)o=o.parent;if(o?.userData.uid)return App.state.items.find(i=>i.uid===o.userData.uid)||null;}return null;}
  function commitGroupTo2D(g){
    const item=App.state.items.find(i=>i.uid===g?.userData?.uid);if(!item)return;const r=rect(),p=g.position.clone();p.y=.01;const projected=p.project(state.camera),px=r.left+(projected.x+1)*.5*r.width,py=r.top+(1-projected.y)*.5*r.height,fit=App.roomFit();if(!fit)return;const cs=App.getCanvasSize();item.x=fit.x+((px-r.left)/r.width)*cs.width;item.y=fit.y+((py-r.top)/r.height)*cs.height;item.rot=-g.rotation.y;const base=Math.max(.55,App.getPerspectiveFactor(item));item.scale=Math.max(.15,Math.min(4,g.scale.x/base));App.constrainToFloor(item,true);App.draw();g.userData.edited3D=false;status('Modification 3D enregistrée ✔');}

  function renderInspector3D(){
    if(!state.enabled)return;const item=App.getSelected();const old=document.querySelector('.phase3-card');if(old)old.remove();
    inspectorBody.insertAdjacentHTML('beforeend',`<div class="phase3-card phase3d-card"><div class="phase3-title">✨ ROOM RECONSTRUCTION · 3D</div><div class="phase3-segment"><button id="p3Photo" class="btn ghost ${state.mode==='photo'?'active':''}">Photo Match</button><button id="p3Orbit" class="btn ghost ${state.mode==='orbit'?'active':''}">Vue 3D</button></div><div class="phase3-row"><span>Scene</span><strong>AI room proxy · 4 surfaces</strong></div><div class="phase3-row"><span>Occlusion</span><strong>${state.occlusion.enabled&&state.depthFit.ready?'AI depth · actif':'IA depth · calibration…'}</strong></div>${item?`<div class="phase3-row"><span>Objet</span><strong>${item.name}</strong></div>`:'<p class="hint">Sélectionnez un meuble.</p>'}<div class="phase3-divider"></div><div class="phase3-subtitle">📐 CAMÉRA</div><div class="field"><label>Hauteur caméra <span id="p3HVal">${state.calibration.height.toFixed(2)} m</span></label><input id="p3H" type="range" min="1.6" max="4.2" step="0.05" value="${state.calibration.height}"></div><div class="field"><label>FOV <span id="p3FVal">${state.calibration.fov}°</span></label><input id="p3F" type="range" min="35" max="70" step="1" value="${state.calibration.fov}"></div><label class="p3-check"><input id="p3Auto" type="checkbox" ${state.calibration.auto?'checked':''}> Calibration automatique depuis le sol</label><div class="phase3-row"><span>Profondeur proxy</span><strong>${state.calibration.depth.toFixed(2)} m</strong></div><div class="phase3-divider"></div><div class="phase3-subtitle">💡 LIGHT MATCH</div><div class="phase3-row"><span>Analyse photo</span><strong>${state.lighting.estimated?'détectée ✔':'manuelle'}</strong></div><label class="p3-check"><input id="p3LightAuto" type="checkbox" ${state.lighting.auto?'checked':''}> Lumière estimée depuis la photo</label><div class="field"><label>Température <span id="p3TempVal">${Math.round(3000+state.lighting.temperature*4000)} K</span></label><input id="p3Temp" type="range" min="0" max="1" step="0.02" value="${state.lighting.temperature}"></div><div class="field"><label>Exposition <span id="p3EVal">${state.lighting.exposure.toFixed(2)}</span></label><input id="p3E" type="range" min="0.70" max="1.45" step="0.01" value="${state.lighting.exposure}"></div><div class="field"><label>Lumière clé <span id="p3KVal">${state.lighting.key.toFixed(2)}</span></label><input id="p3K" type="range" min="0.8" max="4.5" step="0.1" value="${state.lighting.key}"></div><div class="phase3-divider"></div><div class="phase3-subtitle">🧱 ROOM PROXY</div><label class="p3-check"><input id="p3Proxy" type="checkbox" ${state.roomProxy.enabled?'checked':''}> Surfaces de pièce pour les ombres</label><div class="field"><label>Profondeur mur <span id="p3WVal">${state.roomProxy.wallDepth.toFixed(1)} m</span></label><input id="p3W" type="range" min="3" max="14" step="0.1" value="${state.roomProxy.wallDepth}"></div><div class="field"><label>Largeur pièce <span id="p3RWVal">${state.roomProxy.width.toFixed(1)} m</span></label><input id="p3RW" type="range" min="5" max="18" step="0.1" value="${state.roomProxy.width}"></div><div class="phase3-divider"></div><div class="phase3-subtitle">🧠 DEPTH COMPOSITOR</div><div class="field"><label>Force occlusion <span id="p3OVal">${Math.round(state.occlusion.strength*100)}%</span></label><input id="p3O" type="range" min="0" max="1" step="0.05" value="${state.occlusion.strength}"></div><div class="field"><label>Tolérance profondeur <span id="p3TVal">${state.occlusion.threshold.toFixed(2)}</span></label><input id="p3T" type="range" min="0.04" max="0.45" step="0.01" value="${state.occlusion.threshold}"></div><label class="p3-check"><input id="p3Occl" type="checkbox" ${state.occlusion.enabled?'checked':''}> Occlusion par profondeur IA</label><div class="phase3-actions"><button id="p3Move" class="btn ghost">↔ Déplacer</button><button id="p3Rotate" class="btn ghost">↻ Tourner</button><button id="p3Scale" class="btn ghost">⤢ Échelle</button></div><p class="hint">La pièce est représentée par un proxy géométrique calibré sur le sol et la profondeur IA. Le moteur estime aussi une direction, température et exposition de lumière depuis la photo. Ce n'est pas encore un scan 3D métrique.</p></div>`);
    const q=id=>document.getElementById(id);
    q('p3Photo').onclick=()=>setMode('photo');q('p3Orbit').onclick=()=>setMode('orbit');
    q('p3H').oninput=e=>{state.calibration.height=+e.target.value;q('p3HVal').textContent=`${state.calibration.height.toFixed(2)} m`;updateCamera();syncAll();};
    q('p3F').oninput=e=>{state.calibration.fov=+e.target.value;q('p3FVal').textContent=`${state.calibration.fov}°`;updateCamera();syncAll();};
    q('p3Auto').onchange=e=>{state.calibration.auto=e.target.checked;updateCamera();syncAll();};
    q('p3LightAuto').onchange=e=>{state.lighting.auto=e.target.checked;if(state.lighting.auto){estimateRoomLighting();applyLighting();}renderInspector3D();};
    q('p3Temp').oninput=e=>{state.lighting.temperature=+e.target.value;state.lighting.auto=false;applyLighting();q('p3TempVal').textContent=`${Math.round(3000+state.lighting.temperature*4000)} K`;};
    q('p3E').oninput=e=>{state.lighting.exposure=+e.target.value;applyLighting();q('p3EVal').textContent=state.lighting.exposure.toFixed(2);};
    q('p3K').oninput=e=>{state.lighting.key=+e.target.value;state.lighting.auto=false;applyLighting();q('p3KVal').textContent=state.lighting.key.toFixed(2);};
    q('p3Proxy').onchange=e=>{state.roomProxy.enabled=e.target.checked;updateRoomProxy();};
    q('p3W').oninput=e=>{state.roomProxy.wallDepth=+e.target.value;q('p3WVal').textContent=`${state.roomProxy.wallDepth.toFixed(1)} m`;updateRoomProxy();};
    q('p3RW').oninput=e=>{state.roomProxy.width=+e.target.value;q('p3RWVal').textContent=`${state.roomProxy.width.toFixed(1)} m`;updateRoomProxy();};
    q('p3O').oninput=e=>{state.occlusion.strength=+e.target.value;q('p3OVal').textContent=`${Math.round(state.occlusion.strength*100)}%`;updatePostUniforms();};
    q('p3T').oninput=e=>{state.occlusion.threshold=+e.target.value;q('p3TVal').textContent=state.occlusion.threshold.toFixed(2);updatePostUniforms();};
    q('p3Occl').onchange=e=>{state.occlusion.enabled=e.target.checked;updatePostUniforms();renderInspector3D();};
    q('p3Move').onclick=()=>setTransform('translate');q('p3Rotate').onclick=()=>setTransform('rotate');q('p3Scale').onclick=()=>setTransform('scale');
  }
  function setTransform(mode){if(!state.transform)return;state.transform.setMode(mode);state.transform.setSize(mode==='scale'?.68:.82);refreshSelection();status(`Gizmo 3D : ${mode==='translate'?'déplacement':mode==='rotate'?'rotation':'échelle'}`);}
  function setMode(mode){state.mode=mode;if(state.orbit)state.orbit.enabled=mode==='orbit';if(state.transform)state.transform.enabled=mode==='photo';if(mode==='photo'){updateCamera();syncAll();status('Photo Match actif — caméra verrouillée sur la perspective de la pièce');}else status('Vue 3D libre — inspection du modèle');renderInspector3D();refreshSelection();}
  function setEnabled(enabled){if(enabled&&!App.state.roomImage){status('Importez d’abord une pièce pour utiliser la 3D');return;}state.enabled=enabled;canvas.classList.toggle('active',enabled);modeBtn.classList.toggle('active',enabled);modeBtn.textContent=enabled?'✨ 3D · ROOM MATCH':'🧊 Mode 3D';if(enabled){updateCamera();if(App.state.depthImg&&!state.roomDepthTexture)loadRoomDepth();syncAll();setMode('photo');renderInspector3D();}else{state.transform?.detach();status('Mode 2D actif');}}

  function init(){
    const{w,h}=size();state.camera=new THREE.PerspectiveCamera(state.calibration.fov,w/h,.05,100);state.camera.position.set(0,state.calibration.height,state.calibration.depth);
    state.renderer=new THREE.WebGLRenderer({canvas,alpha:true,antialias:true,preserveDrawingBuffer:true,powerPreference:'high-performance'});state.renderer.outputColorSpace=THREE.SRGBColorSpace;state.renderer.toneMapping=THREE.ACESFilmicToneMapping;state.renderer.toneMappingExposure=1.05;state.renderer.shadowMap.enabled=true;state.renderer.shadowMap.type=THREE.PCFSoftShadowMap;setupLighting();setupRoomProxy();
    state.orbit=new OrbitControls(state.camera,canvas);state.orbit.enableDamping=true;state.orbit.dampingFactor=.08;state.orbit.enabled=false;state.orbit.target.set(0,state.calibration.targetY,0);setupTransform();setupPostProcess();resize();window.addEventListener('resize',resize);requestAnimationFrame(loop);
  }
  function loop(){requestAnimationFrame(loop);if(!state.enabled){state.renderer.setRenderTarget(null);state.renderer.clear();return;}state.orbit.update();updatePostUniforms();state.renderer.setRenderTarget(state.target);state.renderer.clear(true,true,true);state.renderer.render(state.scene,state.camera);state.renderer.setRenderTarget(null);state.renderer.clear(true,true,true);state.renderer.render(state.postScene,state.postCamera);}

  modeBtn.addEventListener('click',()=>setEnabled(!state.enabled));
  glbBtn.addEventListener('click',()=>{if(!App.getSelected()){status('Sélectionnez d’abord un meuble, puis importez son GLB');return;}glbInput.click();});
  glbInput.addEventListener('change',async e=>{const file=e.target.files?.[0];if(!file)return;const item=App.getSelected();if(!item)return;if(!/\.glb$/i.test(file.name)&&!/\.gltf$/i.test(file.name)){status('Format accepté : .glb ou .gltf');return;}status(`Chargement du modèle réaliste : ${file.name}…`);try{let gltf;if(/\.glb$/i.test(file.name))gltf=await state.loader.parseAsync(await file.arrayBuffer(),'');else{const url=URL.createObjectURL(file);try{gltf=await state.loader.loadAsync(url);}finally{URL.revokeObjectURL(url);}}state.assets.set(item.catId,gltf.scene);const old=state.groups.get(item.uid);if(old)disposeObject(old);const group=createGroup(item);positionFrom2D(item,group);if(!state.enabled)setEnabled(true);else refreshSelection();status(`${file.name} chargé ✔ — PBR + room match`);}catch(err){console.error('[Phase3D] GLB error',err);status(`Échec du chargement 3D : ${err.message||err}`);}finally{glbInput.value='';}});
  canvas.addEventListener('pointerdown',e=>{if(!state.enabled||state.mode!=='photo'||state.transform?.dragging)return;const item=hit3D(e.clientX,e.clientY);if(item){AppActions.select(item.uid);refreshSelection();}});
  canvas.addEventListener('dblclick',e=>{if(!state.enabled)return;const item=hit3D(e.clientX,e.clientY);if(item)AppActions.select(item.uid);});

  window.Phase3={state,enable:()=>setEnabled(true),disable:()=>setEnabled(false),sync:()=>{if(App.state.depthImg&&!state.roomDepthTexture)loadRoomDepth();syncAll();},loadSelectedGLB:()=>glbInput.click(),setMode};
  const originalDraw=App.draw;App.draw=function patchedDraw(){originalDraw();if(state.enabled)syncAll();};
  init();console.info('[Phase3D] Room reconstruction proxy + lighting match initialized.');
})();
