/* =========================================================
   Phase 3A — Three.js 3D furniture layer
   Hybrid photo + 3D editor. Uses GLB/glTF when supplied and
   procedural furniture as a zero-asset fallback.
   ========================================================= */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

(() => {
  'use strict';

  const App = window.App;
  if (!App) { console.error('[Phase3] window.App is unavailable. Check script order.'); return; }

  const stageWrap = document.querySelector('.stage-wrap');
  const threeCanvas = document.getElementById('threeStage');
  const modeBtn = document.getElementById('threeModeBtn');
  const glbInput = document.getElementById('glbUpload');
  const glbBtn = document.getElementById('glbBtn');
  const inspectorBody = document.getElementById('inspectorBody');
  const status = App.setStatus;

  const state3d = {
    enabled: false,
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    raycaster: new THREE.Raycaster(),
    pointer: new THREE.Vector2(),
    floor: null,
    groups: new Map(),
    glbLibrary: new Map(),
    loader: new GLTFLoader(),
    calibration: { height: 3.0, fov: 50, depth: 7.5, targetY: 1.1 },
    drag: null,
    lastTime: performance.now()
  };

  function getSize() {
    return { w: Math.max(1, stageWrap.clientWidth), h: Math.max(1, stageWrap.clientHeight) };
  }

  function hexFor(item) {
    const c = (window.CATALOG || []).find(x => x.id === item.catId)?.color || '#9aa4b2';
    return new THREE.Color(c);
  }

  function makeMaterial(color, roughness = .7) {
    return new THREE.MeshStandardMaterial({ color, roughness, metalness: .04 });
  }

  function box(w, h, d, color, y = h / 2) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), makeMaterial(color));
    m.position.y = y;
    return m;
  }

  function createProcedural(catId, item) {
    const root = new THREE.Group();
    const c = hexFor(item);
    const w = item.w, d = item.d;

    if (catId === 'sofa') {
      root.add(box(w, .28, d, c, .18));
      root.add(box(w, .72, .20, c.clone().offsetHSL(0, -.04, -.08), .64));
      root.add(box(.16, .72, d, c.clone().offsetHSL(0, -.04, -.08), .54));
      root.add(box(.16, .72, d, c.clone().offsetHSL(0, -.04, -.08), .54)).position.x = w / 2 - .08;
    } else if (catId === 'armchair') {
      root.add(box(w, .28, d, c, .18));
      root.add(box(w - .12, .55, .18, c, .57));
      root.add(box(.16, .62, d, c.clone().offsetHSL(0, -.04, -.08), .48)).position.x = -w / 2 + .08;
      root.add(box(.16, .62, d, c.clone().offsetHSL(0, -.04, -.08), .48)).position.x = w / 2 - .08;
    } else if (catId === 'bed') {
      root.add(box(w, .30, d, c.clone().offsetHSL(0, -.05, -.1), .15));
      root.add(box(w, .16, .55, new THREE.Color('#f1eee7'), .38)).position.z = -d / 2 + .35;
      root.add(box(w - .12, .10, d - .65, c, .33)).position.z = .18;
    } else if (catId === 'table') {
      root.add(box(w, .10, d, c, .78));
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
        const leg = box(.07, .75, .07, c.clone().offsetHSL(0, -.03, -.1), .38);
        leg.position.x = sx * (w / 2 - .09); leg.position.z = sz * (d / 2 - .09); root.add(leg);
      }
    } else if (catId === 'chair') {
      root.add(box(w, .10, d, c, .48));
      const back = box(w, .52, .08, c, .74); back.position.z = -d / 2 + .04; root.add(back);
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
        const leg = box(.045, .45, .045, c.clone().offsetHSL(0, -.03, -.1), .225);
        leg.position.x = sx * (w / 2 - .06); leg.position.z = sz * (d / 2 - .06); root.add(leg);
      }
    } else if (catId === 'lamp') {
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(.025, .025, 1.45, 16), makeMaterial(c)); pole.position.y = .73; root.add(pole);
      const shade = new THREE.Mesh(new THREE.ConeGeometry(.22, .28, 24, 1, true), makeMaterial(c)); shade.position.y = 1.42; root.add(shade);
      const base = new THREE.Mesh(new THREE.CylinderGeometry(.14, .16, .06, 24), makeMaterial(c.clone().offsetHSL(0, -.05, -.12))); base.position.y = .03; root.add(base);
    } else if (catId === 'plant') {
      const pot = new THREE.Mesh(new THREE.CylinderGeometry(.16, .20, .24, 20), makeMaterial(new THREE.Color('#8a5a33'))); pot.position.y = .12; root.add(pot);
      for (let i = 0; i < 7; i++) {
        const leaf = new THREE.Mesh(new THREE.SphereGeometry(.13, 12, 8), makeMaterial(c));
        leaf.scale.set(1, .7, .45); leaf.position.set(Math.sin(i * 1.8) * .13, .35 + (i % 3) * .12, Math.cos(i * 1.8) * .13); root.add(leaf);
      }
    } else if (catId === 'rug') {
      const mat = new THREE.MeshStandardMaterial({ color: c, roughness: .95, side: THREE.DoubleSide });
      const rug = new THREE.Mesh(new THREE.BoxGeometry(w, .035, d), mat); rug.position.y = .018; root.add(rug);
      const border = new THREE.Mesh(new THREE.BoxGeometry(w * .82, .012, d * .72), new THREE.MeshStandardMaterial({ color: c.clone().offsetHSL(0, -.08, .10), roughness: 1 })); border.position.y = .04; root.add(border);
    } else if (catId === 'tvstand') {
      root.add(box(w, .55, d, c, .275));
      root.add(box(w * .85, .04, d + .03, c.clone().offsetHSL(0, -.04, .12), .56));
    } else {
      root.add(box(w, .5, d, c, .25));
    }
    root.userData.baseHeight = catId === 'lamp' ? 0 : (catId === 'rug' ? .018 : .02);
    return root;
  }

  function cloneGLB(scene) {
    const clone = scene.clone(true);
    clone.traverse(obj => {
      if (obj.isMesh) {
        obj.castShadow = true;
        obj.receiveShadow = true;
      }
    });
    return clone;
  }

  function normalizeModel(root, item) {
    const box3 = new THREE.Box3().setFromObject(root);
    const size = new THREE.Vector3(); box3.getSize(size);
    const center = new THREE.Vector3(); box3.getCenter(center);
    root.position.sub(center);
    const maxDim = Math.max(size.x, size.y, size.z, .001);
    const target = Math.max(item.w, item.d, .25);
    root.scale.setScalar(target / maxDim);
    const after = new THREE.Box3().setFromObject(root);
    const minY = after.min.y;
    root.position.y -= minY;
  }

  function createGroup(item) {
    const group = new THREE.Group();
    group.userData.uid = item.uid;
    group.userData.catId = item.catId;
    group.userData.item = item;
    const asset = state3d.glbLibrary.get(item.catId);
    const model = asset ? cloneGLB(asset) : createProcedural(item.catId, item);
    if (asset) normalizeModel(model, item);
    group.add(model);
    group.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    state3d.scene.add(group);
    state3d.groups.set(item.uid, group);
    return group;
  }

  function disposeObject(obj) {
    obj.traverse(o => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach(m => m.dispose());
      }
    });
    state3d.scene.remove(obj);
  }

  function screenToFloor(clientX, clientY) {
    const rect = threeCanvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((clientY - rect.top) / rect.height) * 2 + 1;
    state3d.pointer.set(x, y);
    state3d.raycaster.setFromCamera(state3d.pointer, state3d.camera);
    const hit = new THREE.Vector3();
    const ray = state3d.raycaster.ray;
    if (Math.abs(ray.direction.y) < 1e-5) return null;
    const t = -ray.origin.y / ray.direction.y;
    if (t < 0) return null;
    return ray.origin.clone().addScaledVector(ray.direction, t);
  }

  function supportScreen(item) {
    const sp = App.getSupportPoint(item);
    return { x: sp.x, y: sp.y };
  }

  function positionFrom2D(item, group) {
    const f = App.roomFit();
    if (!f || !App.state.roomImage) return;
    const sp = supportScreen(item);
    const rect = threeCanvas.getBoundingClientRect();
    const scaleX = rect.width / App.getCanvasSize().width;
    const scaleY = rect.height / App.getCanvasSize().height;
    const p = screenToFloor(rect.left + sp.x * scaleX, rect.top + sp.y * scaleY);
    if (p) {
      group.position.x = p.x;
      group.position.z = p.z;
      group.position.y = 0;
    }
    group.rotation.y = -item.rot;
    const perspective = App.getPerspectiveFactor(item);
    const base = Math.max(.25, item.scale * perspective);
    group.scale.setScalar(base);
  }

  function syncAll() {
    if (!state3d.enabled) return;
    const live = new Set(App.state.items.map(i => i.uid));
    for (const [uid, group] of state3d.groups) {
      if (!live.has(uid)) { disposeObject(group); state3d.groups.delete(uid); }
    }
    for (const item of App.state.items) {
      const group = state3d.groups.get(item.uid) || createGroup(item);
      positionFrom2D(item, group);
      group.visible = true;
      group.renderOrder = item.z;
    }
    refreshSelectionVisual();
  }

  function resizeRenderer() {
    const { w, h } = getSize();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    state3d.renderer.setPixelRatio(dpr);
    state3d.renderer.setSize(w, h, false);
    state3d.camera.aspect = w / h;
    state3d.camera.updateProjectionMatrix();
  }

  function init() {
    state3d.scene = new THREE.Scene();
    state3d.camera = new THREE.PerspectiveCamera(state3d.calibration.fov, 1, .01, 100);
    state3d.camera.position.set(0, state3d.calibration.height, state3d.calibration.depth);
    state3d.camera.lookAt(0, state3d.calibration.targetY, 0);

    state3d.renderer = new THREE.WebGLRenderer({ canvas: threeCanvas, alpha: true, antialias: true, preserveDrawingBuffer: true });
    state3d.renderer.outputColorSpace = THREE.SRGBColorSpace;
    state3d.renderer.shadowMap.enabled = true;
    state3d.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    const hemi = new THREE.HemisphereLight(0xffffff, 0x3a3a3a, 2.0); state3d.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 3.0); key.position.set(3, 6, 4); key.castShadow = true; state3d.scene.add(key);
    const fill = new THREE.DirectionalLight(0xbfd5ff, 1.0); fill.position.set(-4, 3, 2); state3d.scene.add(fill);

    const floorMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });
    state3d.floor = new THREE.Mesh(new THREE.PlaneGeometry(50, 50), floorMat);
    state3d.floor.rotation.x = -Math.PI / 2;
    state3d.floor.userData.phase3Floor = true;
    state3d.scene.add(state3d.floor);

    state3d.controls = new OrbitControls(state3d.camera, threeCanvas);
    state3d.controls.enabled = false;
    state3d.controls.enableDamping = true;
    state3d.controls.target.set(0, state3d.calibration.targetY, 0);

    resizeRenderer();
    window.addEventListener('resize', resizeRenderer);
    requestAnimationFrame(loop);
  }

  function loop(t) {
    const dt = Math.min(.05, (t - state3d.lastTime) / 1000); state3d.lastTime = t;
    if (state3d.enabled) {
      state3d.controls.update();
      state3d.renderer.render(state3d.scene, state3d.camera);
    } else state3d.renderer.clear();
    requestAnimationFrame(loop);
  }

  function setEnabled(enabled) {
    state3d.enabled = enabled;
    threeCanvas.classList.toggle('active', enabled);
    modeBtn.classList.toggle('active', enabled);
    modeBtn.textContent = enabled ? '🧊 Mode 3D : ON' : '🧊 Mode 3D';
    if (enabled) {
      if (!App.state.roomImage) { setEnabled(false); status('Importez d’abord une pièce pour utiliser la 3D'); return; }
      syncAll();
      status('Mode 3D actif — les meubles sont rendus en profondeur');
      renderInspector3D();
    } else {
      status('Mode 2D actif');
      renderInspector3D();
    }
  }

  function selectedGroup() {
    const item = App.getSelected(); return item ? state3d.groups.get(item.uid) : null;
  }

  function refreshSelectionVisual() {
    const selected = App.getSelected();
    for (const [uid, group] of state3d.groups) {
      const active = selected?.uid === uid;
      group.traverse(o => {
        if (!o.isMesh || !o.material) return;
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach(m => {
          if (!m.userData) m.userData = {};
          if (!m.userData.phase3BaseEmissive && m.emissive) m.userData.phase3BaseEmissive = m.emissive.clone();
          if (m.emissive) m.emissive.copy(m.userData.phase3BaseEmissive).lerp(new THREE.Color('#d9a441'), active ? .22 : 0);
        });
      });
    }
  }

  function renderInspector3D() {
    if (!state3d.enabled) return;
    const item = App.getSelected();
    const deg = item ? Math.round(item.rot * 180 / Math.PI) : 0;
    inspectorBody.insertAdjacentHTML('beforeend', `
      <div class="phase3-card">
        <div class="phase3-title">🧊 Rendu 3D</div>
        <div class="phase3-row"><span>État</span><strong>GLB + Three.js</strong></div>
        ${item ? `<div class="phase3-row"><span>Objet</span><strong>${item.name}</strong></div>` : '<p class="hint">Sélectionnez un meuble pour ses contrôles 3D.</p>'}
        <div class="field"><label>Hauteur caméra <span id="camHVal">${state3d.calibration.height.toFixed(1)} m</span></label><input id="camH" type="range" min="1.5" max="5.5" step="0.1" value="${state3d.calibration.height}"></div>
        <div class="field"><label>FOV <span id="camFovVal">${state3d.calibration.fov}°</span></label><input id="camFov" type="range" min="30" max="75" step="1" value="${state3d.calibration.fov}"></div>
        <div class="field"><label>Profondeur scène <span id="camDVal">${state3d.calibration.depth.toFixed(1)} m</span></label><input id="camD" type="range" min="4" max="14" step="0.1" value="${state3d.calibration.depth}"></div>
        <p class="hint">Réglez ces trois paramètres si le mobilier 3D ne colle pas parfaitement à la perspective de la photo.</p>
      </div>`);
    const bind = (id, fn) => { const el = document.getElementById(id); if (el) el.oninput = fn; };
    bind('camH', e => { state3d.calibration.height = +e.target.value; document.getElementById('camHVal').textContent = `${(+e.target.value).toFixed(1)} m`; updateCamera(); syncAll(); });
    bind('camFov', e => { state3d.calibration.fov = +e.target.value; document.getElementById('camFovVal').textContent = `${e.target.value}°`; updateCamera(); syncAll(); });
    bind('camD', e => { state3d.calibration.depth = +e.target.value; document.getElementById('camDVal').textContent = `${(+e.target.value).toFixed(1)} m`; updateCamera(); syncAll(); });
  }

  function updateCamera() {
    state3d.camera.position.set(0, state3d.calibration.height, state3d.calibration.depth);
    state3d.camera.fov = state3d.calibration.fov;
    state3d.camera.updateProjectionMatrix();
    state3d.controls.target.set(0, state3d.calibration.targetY, 0);
    state3d.camera.lookAt(state3d.controls.target);
  }

  function hit3D(clientX, clientY) {
    const rect = threeCanvas.getBoundingClientRect();
    state3d.pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    state3d.pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    state3d.raycaster.setFromCamera(state3d.pointer, state3d.camera);
    const objects = [...state3d.groups.values()];
    const hits = state3d.raycaster.intersectObjects(objects, true);
    for (const hit of hits) {
      let o = hit.object;
      while (o && !o.userData.uid) o = o.parent;
      if (o?.userData.uid) return App.state.items.find(i => i.uid === o.userData.uid) || null;
    }
    return null;
  }

  threeCanvas.addEventListener('pointerdown', e => {
    if (!state3d.enabled) return;
    const item = hit3D(e.clientX, e.clientY);
    if (!item) return;
    AppActions.select(item.uid);
    refreshSelectionVisual();
    state3d.drag = { uid: item.uid, lastX: e.clientX, lastY: e.clientY };
    threeCanvas.setPointerCapture?.(e.pointerId);
    e.preventDefault();
  });

  threeCanvas.addEventListener('pointermove', e => {
    if (!state3d.enabled || !state3d.drag) return;
    const item = App.state.items.find(i => i.uid === state3d.drag.uid); if (!item) return;
    const p = screenToFloor(e.clientX, e.clientY); if (!p) return;
    const group = state3d.groups.get(item.uid); if (!group) return;
    group.position.x = p.x; group.position.z = p.z;
    const rect = threeCanvas.getBoundingClientRect();
    const sp = supportScreen(item);
    const canvasX = ((e.clientX - rect.left) / rect.width) * App.getCanvasSize().width;
    const canvasY = ((e.clientY - rect.top) / rect.height) * App.getCanvasSize().height;
    item.x = canvasX;
    item.y = canvasY;
    App.constrainToFloor(item, true);
    positionFrom2D(item, group);
    App.draw();
    state3d.drag.lastX = e.clientX; state3d.drag.lastY = e.clientY;
  });

  threeCanvas.addEventListener('pointerup', e => { state3d.drag = null; threeCanvas.releasePointerCapture?.(e.pointerId); });
  threeCanvas.addEventListener('dblclick', e => { const item = hit3D(e.clientX, e.clientY); if (item) AppActions.deleteItem(item.uid); });

  modeBtn.addEventListener('click', () => setEnabled(!state3d.enabled));
  glbBtn.addEventListener('click', () => {
    if (!App.getSelected()) { status('Sélectionnez d’abord un meuble, puis importez son GLB'); return; }
    glbInput.click();
  });

  glbInput.addEventListener('change', async e => {
    const file = e.target.files?.[0]; if (!file) return;
    const item = App.getSelected(); if (!item) return;
    if (!/\.glb$/i.test(file.name) && !/\.gltf$/i.test(file.name)) { status('Format accepté : .glb ou .gltf'); return; }
    status(`Chargement 3D : ${file.name}…`);
    try {
      let gltf;
      if (/\.glb$/i.test(file.name)) {
        // Parse the selected file directly. This avoids blob-URL path issues and
        // is the most reliable path for self-contained GLB assets.
        const buffer = await file.arrayBuffer();
        gltf = await state3d.loader.parseAsync(buffer, '');
      } else {
        // A standalone .gltf can be loaded when it has no external .bin/textures.
        const url = URL.createObjectURL(file);
        try {
          gltf = await state3d.loader.loadAsync(url);
        } finally {
          URL.revokeObjectURL(url);
        }
      }
      state3d.glbLibrary.set(item.catId, gltf.scene);
      const old = state3d.groups.get(item.uid); if (old) disposeObject(old);
      const group = createGroup(item); positionFrom2D(item, group);
      if (!state3d.enabled) setEnabled(true);
      status(`${file.name} chargé ✔ — modèle 3D associé à ${item.name}`);
    } catch (err) {
      console.error(err); status(`Échec du chargement 3D : ${err.message || err}`);
    } finally { glbInput.value = ''; }
  });

  window.Phase3 = {
    state: state3d,
    enable: () => setEnabled(true),
    disable: () => setEnabled(false),
    sync: syncAll,
    loadSelectedGLB: () => glbInput.click()
  };

  // App changes should propagate to the 3D layer.
  const originalDraw = App.draw;
  App.draw = function patchedDraw() { originalDraw(); if (state3d.enabled) syncAll(); };

  init();
  console.info('[Phase3] Three.js layer initialized successfully.');
})();
