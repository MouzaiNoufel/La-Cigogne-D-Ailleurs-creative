/* =========================================================
   Phase 3B — Photo-matched 3D compositing
   Goal: move from a basic 3D overlay to a modern photo editor:
   - locked photo-match camera
   - automatic perspective calibration from the AI floor profile
   - PBR GLB rendering + studio environment
   - real transform gizmo
   - contact shadow catcher
   - depth/furniture occlusion pass
   - dimension-aware GLB normalization
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
  if (!App || !AppActions) {
    console.error('[Phase3B] Core app unavailable.');
    return;
  }

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
    pmrem: null,
    environment: null,
    calibration: {
      height: 2.7,
      fov: 52,
      depth: 6.5,
      targetY: 1.05,
      pitch: 0.23,
      auto: true,
    },
    mode: 'photo',
    selectedUid: null,
    drag: null,
    roomTexture: null,
    occluder: null,
    occluderDepth: null,
    occluderMask: null,
    last: performance.now(),
  };

  function size() {
    return { w: Math.max(1, stageWrap.clientWidth), h: Math.max(1, stageWrap.clientHeight) };
  }

  function catalogItem(item) {
    return (window.CATALOG || []).find(x => x.id === item.catId) || null;
  }

  function color(item) {
    return new THREE.Color(catalogItem(item)?.color || '#9aa4b2');
  }

  function material(c, roughness = .62, metalness = .04) {
    return new THREE.MeshStandardMaterial({ color: c, roughness, metalness });
  }

  function box(w, h, d, c, y = h / 2, rough = .68) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material(c, rough));
    m.position.y = y;
    m.castShadow = true;
    m.receiveShadow = true;
    return m;
  }

  function procedural(catId, item) {
    const root = new THREE.Group();
    const c = color(item);
    const w = item.w, d = item.d;

    if (catId === 'sofa') {
      root.add(box(w, .28, d, c, .18));
      root.add(box(w, .68, .20, c.clone().offsetHSL(0, -.03, -.08), .62));
      for (const x of [-w / 2 + .08, w / 2 - .08]) {
        const arm = box(.16, .70, d, c.clone().offsetHSL(0, -.03, -.10), .52); arm.position.x = x; root.add(arm);
      }
      for (const x of [-w * .24, 0, w * .24]) {
        const cushion = box(w * .20, .13, .48, new THREE.Color('#d6c7ba'), .48, .88);
        cushion.position.set(x, .48, .08); root.add(cushion);
      }
    } else if (catId === 'armchair') {
      root.add(box(w, .28, d, c, .18));
      root.add(box(w - .16, .52, .16, c, .55));
      for (const x of [-w / 2 + .08, w / 2 - .08]) {
        const arm = box(.15, .62, d, c.clone().offsetHSL(0, -.03, -.10), .47); arm.position.x = x; root.add(arm);
      }
      const cushion = box(w * .58, .12, d * .55, new THREE.Color('#d8c9df'), .48); root.add(cushion);
    } else if (catId === 'bed') {
      root.add(box(w, .30, d, c.clone().offsetHSL(0, -.04, -.08), .15));
      root.add(box(w, .55, .18, c.clone().offsetHSL(0, -.03, -.12), .58)).position.z = -d / 2 + .09;
      root.add(box(w * .92, .10, d * .74, new THREE.Color('#eee9e0'), .37)).position.z = .14;
      for (const x of [-w * .25, w * .25]) {
        const pillow = box(w * .22, .09, .40, new THREE.Color('#f8f6f1'), .46, .9);
        pillow.position.set(x, .46, -d / 2 + .43); root.add(pillow);
      }
    } else if (catId === 'table') {
      root.add(box(w, .09, d, c, .79, .55));
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
        const leg = box(.065, .76, .065, c.clone().offsetHSL(0, -.04, -.13), .38, .55);
        leg.position.x = sx * (w / 2 - .09); leg.position.z = sz * (d / 2 - .09); root.add(leg);
      }
    } else if (catId === 'chair') {
      root.add(box(w, .10, d, c, .48));
      const back = box(w, .52, .08, c, .74); back.position.z = -d / 2 + .04; root.add(back);
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
        const leg = box(.045, .45, .045, c.clone().offsetHSL(0, -.04, -.12), .225, .55);
        leg.position.x = sx * (w / 2 - .06); leg.position.z = sz * (d / 2 - .06); root.add(leg);
      }
    } else if (catId === 'lamp') {
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(.022, .028, 1.45, 20), material(c, .38, .10)); pole.position.y = .73; pole.castShadow = true; root.add(pole);
      const shade = new THREE.Mesh(new THREE.ConeGeometry(.23, .30, 32, 1, true), material(c, .52, .02)); shade.position.y = 1.43; shade.castShadow = true; root.add(shade);
      const base = new THREE.Mesh(new THREE.CylinderGeometry(.15, .17, .07, 28), material(c.clone().offsetHSL(0, -.05, -.15), .35, .12)); base.position.y = .035; root.add(base);
    } else if (catId === 'plant') {
      const pot = new THREE.Mesh(new THREE.CylinderGeometry(.15, .20, .25, 24), material(new THREE.Color('#8a5a33'), .8)); pot.position.y = .125; pot.castShadow = true; root.add(pot);
      for (let i = 0; i < 9; i++) {
        const leaf = new THREE.Mesh(new THREE.SphereGeometry(.13, 16, 10), material(c, .9));
        leaf.scale.set(1.0, .65, .48);
        leaf.position.set(Math.sin(i * 1.8) * .15, .38 + (i % 4) * .10, Math.cos(i * 1.8) * .15);
        leaf.rotation.z = (i % 2 ? -.3 : .3); root.add(leaf);
      }
    } else if (catId === 'rug') {
      const rug = new THREE.Mesh(new THREE.BoxGeometry(w, .035, d), new THREE.MeshStandardMaterial({ color: c, roughness: .96 }));
      rug.position.y = .018; rug.receiveShadow = true; root.add(rug);
      const border = new THREE.Mesh(new THREE.BoxGeometry(w * .86, .012, d * .74), new THREE.MeshStandardMaterial({ color: c.clone().offsetHSL(0, -.08, .10), roughness: 1 }));
      border.position.y = .04; border.receiveShadow = true; root.add(border);
    } else if (catId === 'tvstand') {
      root.add(box(w, .55, d, c, .275));
      root.add(box(w * .85, .04, d + .03, c.clone().offsetHSL(0, -.04, .12), .56, .48));
      for (const x of [-w * .25, w * .25]) {
        const door = box(w * .19, .35, .015, c.clone().offsetHSL(0, -.02, .10), .29, .58); door.position.x = x; door.position.z = d / 2 + .008; root.add(door);
      }
    } else root.add(box(w, .5, d, c, .25));

    return root;
  }

  function cloneAsset(scene) {
    const root = scene.clone(true);
    root.traverse(o => {
      if (!o.isMesh) return;
      o.castShadow = true; o.receiveShadow = true;
      if (o.material?.map) o.material.map.colorSpace = THREE.SRGBColorSpace;
    });
    return root;
  }

  function modelDimensions(root) {
    const b = new THREE.Box3().setFromObject(root);
    const s = new THREE.Vector3(); b.getSize(s);
    return { box: b, x: Math.max(.0001, s.x), y: Math.max(.0001, s.y), z: Math.max(.0001, s.z) };
  }

  function normalizeAsset(root, item) {
    const dims = modelDimensions(root);
    // GLB assets often arrive in centimetres, inches, or arbitrary units.
    // Match the two horizontal dimensions while preserving aspect ratio.
    const targetW = Math.max(.08, item.w);
    const targetD = Math.max(.08, item.d);
    const direct = Math.max(targetW / dims.x, targetD / dims.z);
    const swapped = Math.max(targetW / dims.z, targetD / dims.x);
    const useSwap = Math.abs(targetW - dims.z * swapped) + Math.abs(targetD - dims.x * swapped)
      < Math.abs(targetW - dims.x * direct) + Math.abs(targetD - dims.z * direct);
    if (useSwap) root.rotation.y = Math.PI / 2;
    const finalDims = useSwap ? { x: dims.z, z: dims.x } : { x: dims.x, z: dims.z };
    const sx = targetW / Math.max(.0001, finalDims.x);
    const sz = targetD / Math.max(.0001, finalDims.z);
    const scale = (sx + sz) / 2;
    root.scale.setScalar(scale);
    const after = new THREE.Box3().setFromObject(root);
    const center = new THREE.Vector3(); after.getCenter(center);
    root.position.sub(center);
    const floorBox = new THREE.Box3().setFromObject(root);
    root.position.y -= floorBox.min.y;
  }

  function createGroup(item) {
    const group = new THREE.Group();
    group.userData.uid = item.uid;
    group.userData.item = item;
    const asset = state.assets.get(item.catId);
    const model = asset ? cloneAsset(asset) : procedural(item.catId, item);
    if (asset) normalizeAsset(model, item);
    group.add(model);
    group.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    state.scene.add(group);
    state.groups.set(item.uid, group);
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
    state.scene.remove(obj);
  }

  function rect() { return canvas.getBoundingClientRect(); }

  function screenToFloor(clientX, clientY) {
    const r = rect();
    state.pointer.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    state.raycaster.setFromCamera(state.pointer, state.camera);
    const ray = state.raycaster.ray;
    if (Math.abs(ray.direction.y) < 1e-6) return null;
    const t = -ray.origin.y / ray.direction.y;
    return t > 0 ? ray.origin.clone().addScaledVector(ray.direction, t) : null;
  }

  function support(item) { return App.getSupportPoint(item); }

  function positionFrom2D(item, group) {
    const fit = App.roomFit();
    if (!fit || !App.state.roomImage) return;
    const sp = support(item);
    const r = rect();
    const sx = r.width / App.getCanvasSize().width;
    const sy = r.height / App.getCanvasSize().height;
    const px = r.left + sp.x * sx;
    const py = r.top + sp.y * sy;
    const p = screenToFloor(px, py);
    if (p) {
      group.position.set(p.x, 0, p.z);
    }
    group.rotation.y = -item.rot;
    const perspective = App.getPerspectiveFactor(item);
    const base = Math.max(.55, item.scale * perspective);
    group.scale.setScalar(base);
  }

  function syncAll() {
    if (!state.enabled) return;
    const live = new Set(App.state.items.map(i => i.uid));
    for (const [uid, group] of state.groups) {
      if (!live.has(uid)) { disposeObject(group); state.groups.delete(uid); }
    }
    for (const item of App.state.items) {
      const group = state.groups.get(item.uid) || createGroup(item);
      positionFrom2D(item, group);
      group.visible = true;
    }
    refreshSelection();
  }

  function resize() {
    const { w, h } = size();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    state.renderer.setPixelRatio(dpr);
    state.renderer.setSize(w, h, false);
    state.camera.aspect = w / h;
    state.camera.updateProjectionMatrix();
  }

  function derivePitchFromFloor() {
    const a = App.state.analysis;
    if (!state.calibration.auto || !a || !Array.isArray(a.floor_top_profile) || !a.floor_top_profile.length) return;
    const H = a.height || App.state.roomImage?.naturalHeight || 1000;
    const median = a.floor_top_y || (a.floor_top_profile.reduce((x, y) => x + y, 0) / a.floor_top_profile.length);
    const fovRad = THREE.MathUtils.degToRad(state.calibration.fov);
    const vAngle = (0.5 - median / H) * fovRad;
    state.calibration.pitch = Math.max(-.15, Math.min(.9, vAngle));
    const h = state.calibration.height;
    state.calibration.depth = Math.max(2.2, Math.min(14, (h - state.calibration.targetY) / Math.tan(Math.max(.08, state.calibration.pitch))));
  }

  function updateCamera() {
    derivePitchFromFloor();
    const c = state.calibration;
    state.camera.fov = c.fov;
    state.camera.position.set(0, c.height, c.depth);
    const lookZ = c.depth - (c.height - c.targetY) / Math.tan(Math.max(.08, c.pitch));
    state.camera.lookAt(0, c.targetY, Number.isFinite(lookZ) ? lookZ : 0);
    state.camera.updateProjectionMatrix();
    if (state.orbit) {
      state.orbit.target.set(0, c.targetY, 0);
      state.orbit.update();
    }
  }

  function refreshSelection() {
    const selected = App.getSelected();
    state.selectedUid = selected?.uid || null;
    for (const [uid, group] of state.groups) {
      group.traverse(o => {
        if (!o.isMesh || !o.material) return;
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach(m => {
          if (!m.userData) m.userData = {};
          if (!m.userData.p3Base) m.userData.p3Base = m.emissive ? m.emissive.clone() : new THREE.Color(0);
          if (m.emissive) m.emissive.copy(m.userData.p3Base).lerp(new THREE.Color('#d9a441'), uid === state.selectedUid ? .12 : 0);
        });
      });
    }
    if (state.transform) {
      const g = selected ? state.groups.get(selected.uid) : null;
      state.transform.detach();
      if (g && state.enabled && state.mode === 'photo') state.transform.attach(g);
    }
  }

  function setupLighting() {
    const pmrem = new THREE.PMREMGenerator(state.renderer);
    const envScene = new RoomEnvironment();
    state.environment = pmrem.fromScene(envScene, 0.04).texture;
    state.scene.environment = state.environment;
    envScene.dispose(); pmrem.dispose();

    state.scene.add(new THREE.HemisphereLight(0xffffff, 0x6b6b6b, 1.35));
    const key = new THREE.DirectionalLight(0xfff5e8, 2.8);
    key.position.set(-3.5, 6, 4); key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = .1; key.shadow.camera.far = 20;
    key.shadow.camera.left = -8; key.shadow.camera.right = 8;
    key.shadow.camera.top = 8; key.shadow.camera.bottom = -8;
    state.scene.add(key);
    const fill = new THREE.DirectionalLight(0xcfe1ff, .65); fill.position.set(4, 3, -2); state.scene.add(fill);
  }

  function makeShadowCatcher() {
    const geo = new THREE.PlaneGeometry(18, 18);
    const mat = new THREE.ShadowMaterial({ opacity: .24 });
    state.shadow = new THREE.Mesh(geo, mat);
    state.shadow.rotation.x = -Math.PI / 2;
    state.shadow.position.y = .002;
    state.shadow.receiveShadow = true;
    state.shadow.renderOrder = -2;
    state.scene.add(state.shadow);
  }

  function setupTransform() {
    state.transform = new TransformControls(state.camera, canvas);
    state.transform.setMode('translate');
    state.transform.setSpace('world');
    state.transform.setSize(.75);
    state.transform.addEventListener('dragging-changed', e => {
      if (state.orbit) state.orbit.enabled = !e.value && state.mode === 'orbit';
    });
    state.transform.addEventListener('objectChange', () => {
      const group = state.transform.object;
      if (!group?.userData?.uid) return;
      const item = App.state.items.find(i => i.uid === group.userData.uid);
      if (!item) return;
      item.x = item.x; // 2D state is reconciled on drag end.
    });
    state.scene.add(state.transform.getHelper());
  }

  function hit3D(clientX, clientY) {
    const r = rect();
    state.pointer.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    state.raycaster.setFromCamera(state.pointer, state.camera);
    const hits = state.raycaster.intersectObjects([...state.groups.values()], true);
    for (const hit of hits) {
      let o = hit.object;
      while (o && !o.userData.uid) o = o.parent;
      if (o?.userData.uid) return App.state.items.find(i => i.uid === o.userData.uid) || null;
    }
    return null;
  }

  function commitGroupTo2D(group) {
    const item = App.state.items.find(i => i.uid === group?.userData?.uid);
    if (!item) return;
    const r = rect();
    const p = group.position.clone();
    p.y = .01;
    const projected = p.clone().project(state.camera);
    const px = r.left + (projected.x + 1) * .5 * r.width;
    const py = r.top + (1 - projected.y) * .5 * r.height;
    const fit = App.roomFit();
    if (!fit) return;
    const cs = App.getCanvasSize();
    item.x = fit.x + ((px - r.left) / r.width) * cs.width;
    item.y = fit.y + ((py - r.top) / r.height) * cs.height;
    App.constrainToFloor(item, true);
    App.draw();
  }

  function renderInspector3D() {
    if (!state.enabled) return;
    const item = App.getSelected();
    inspectorBody.insertAdjacentHTML('beforeend', `
      <div class="phase3-card phase3b-card">
        <div class="phase3-title">✨ PHOTO MATCH · 3D</div>
        <div class="phase3-segment">
          <button id="p3Photo" class="btn ghost ${state.mode === 'photo' ? 'active' : ''}">Photo Match</button>
          <button id="p3Orbit" class="btn ghost ${state.mode === 'orbit' ? 'active' : ''}">Vue 3D</button>
        </div>
        <div class="phase3-row"><span>Rendu</span><strong>Physically Based</strong></div>
        <div class="phase3-row"><span>Occlusion</span><strong>${App.state.analysis?.masks?.furniture ? 'AI depth' : '—'}</strong></div>
        ${item ? `<div class="phase3-row"><span>Objet</span><strong>${item.name}</strong></div>` : '<p class="hint">Sélectionnez un meuble.</p>'}
        <div class="field"><label>Hauteur caméra <span id="p3HVal">${state.calibration.height.toFixed(2)} m</span></label><input id="p3H" type="range" min="1.6" max="4.2" step="0.05" value="${state.calibration.height}"></div>
        <div class="field"><label>FOV <span id="p3FVal">${state.calibration.fov}°</span></label><input id="p3F" type="range" min="35" max="70" step="1" value="${state.calibration.fov}"></div>
        <label class="p3-check"><input id="p3Auto" type="checkbox" ${state.calibration.auto ? 'checked' : ''}> Calibration automatique depuis le sol</label>
        <div class="phase3-row"><span>Profondeur</span><strong>${state.calibration.depth.toFixed(2)} m</strong></div>
        <div class="phase3-actions">
          <button id="p3Move" class="btn ghost">↔ Déplacer</button>
          <button id="p3Rotate" class="btn ghost">↻ Tourner</button>
          <button id="p3Scale" class="btn ghost">⤢ Échelle</button>
        </div>
        <p class="hint">En Photo Match, les axes 3D sont projetés dans la photo. Utilisez les gizmos pour une manipulation précise.</p>
      </div>`);

    const q = id => document.getElementById(id);
    q('p3Photo').onclick = () => setMode('photo');
    q('p3Orbit').onclick = () => setMode('orbit');
    q('p3H').oninput = e => { state.calibration.height = +e.target.value; q('p3HVal').textContent = `${state.calibration.height.toFixed(2)} m`; updateCamera(); syncAll(); };
    q('p3F').oninput = e => { state.calibration.fov = +e.target.value; q('p3FVal').textContent = `${state.calibration.fov}°`; updateCamera(); syncAll(); };
    q('p3Auto').onchange = e => { state.calibration.auto = e.target.checked; updateCamera(); syncAll(); };
    q('p3Move').onclick = () => setTransform('translate');
    q('p3Rotate').onclick = () => setTransform('rotate');
    q('p3Scale').onclick = () => setTransform('scale');
  }

  function setTransform(mode) {
    if (!state.transform) return;
    state.transform.setMode(mode);
    state.transform.setSize(mode === 'scale' ? .68 : .82);
    refreshSelection();
    status(`Gizmo 3D : ${mode === 'translate' ? 'déplacement' : mode === 'rotate' ? 'rotation' : 'échelle'}`);
  }

  function setMode(mode) {
    state.mode = mode;
    if (state.orbit) state.orbit.enabled = mode === 'orbit';
    if (state.transform) state.transform.enabled = mode === 'photo';
    if (mode === 'photo') {
      updateCamera(); syncAll(); status('Photo Match actif — caméra verrouillée sur la perspective de la pièce');
    } else {
      status('Vue 3D libre — inspection du modèle');
    }
    renderInspector3D(); refreshSelection();
  }

  function setEnabled(enabled) {
    if (enabled && !App.state.roomImage) { status('Importez d’abord une pièce pour utiliser la 3D'); return; }
    state.enabled = enabled;
    canvas.classList.toggle('active', enabled);
    modeBtn.classList.toggle('active', enabled);
    modeBtn.textContent = enabled ? '🧊 3D · PHOTO MATCH' : '🧊 Mode 3D';
    if (enabled) {
      updateCamera(); syncAll(); setMode('photo'); renderInspector3D();
    } else {
      state.transform?.detach(); status('Mode 2D actif');
    }
  }

  function init() {
    const { w, h } = size();
    state.camera = new THREE.PerspectiveCamera(state.calibration.fov, w / h, .05, 100);
    state.camera.position.set(0, state.calibration.height, state.calibration.depth);

    state.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    state.renderer.outputColorSpace = THREE.SRGBColorSpace;
    state.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    state.renderer.toneMappingExposure = 1.05;
    state.renderer.shadowMap.enabled = true;
    state.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    setupLighting();
    makeShadowCatcher();

    state.floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }));
    state.floor.rotation.x = -Math.PI / 2;
    state.scene.add(state.floor);

    state.orbit = new OrbitControls(state.camera, canvas);
    state.orbit.enableDamping = true;
    state.orbit.dampingFactor = .08;
    state.orbit.enabled = false;
    state.orbit.target.set(0, state.calibration.targetY, 0);

    setupTransform();
    resize();
    window.addEventListener('resize', resize);
    requestAnimationFrame(loop);
  }

  function loop(t) {
    state.last = t;
    if (state.enabled) {
      state.orbit.update();
      state.renderer.render(state.scene, state.camera);
    } else state.renderer.clear();
    requestAnimationFrame(loop);
  }

  modeBtn.addEventListener('click', () => setEnabled(!state.enabled));

  glbBtn.addEventListener('click', () => {
    if (!App.getSelected()) { status('Sélectionnez d’abord un meuble, puis importez son GLB'); return; }
    glbInput.click();
  });

  glbInput.addEventListener('change', async e => {
    const file = e.target.files?.[0];
    if (!file) return;
    const item = App.getSelected();
    if (!item) return;
    if (!/\.glb$/i.test(file.name) && !/\.gltf$/i.test(file.name)) { status('Format accepté : .glb ou .gltf'); return; }
    status(`Chargement du modèle réaliste : ${file.name}…`);
    try {
      let gltf;
      if (/\.glb$/i.test(file.name)) gltf = await state.loader.parseAsync(await file.arrayBuffer(), '');
      else {
        const url = URL.createObjectURL(file);
        try { gltf = await state.loader.loadAsync(url); } finally { URL.revokeObjectURL(url); }
      }
      state.assets.set(item.catId, gltf.scene);
      const old = state.groups.get(item.uid);
      if (old) disposeObject(old);
      const group = createGroup(item);
      positionFrom2D(item, group);
      if (!state.enabled) setEnabled(true);
      else refreshSelection();
      status(`${file.name} chargé ✔ — rendu PBR + dimensions normalisées`);
    } catch (err) {
      console.error('[Phase3B] GLB error', err);
      status(`Échec du chargement 3D : ${err.message || err}`);
    } finally { glbInput.value = ''; }
  });

  canvas.addEventListener('pointerdown', e => {
    if (!state.enabled || state.mode !== 'photo' || state.transform?.dragging) return;
    const item = hit3D(e.clientX, e.clientY);
    if (!item) return;
    AppActions.select(item.uid);
    refreshSelection();
  });

  canvas.addEventListener('dblclick', e => {
    if (!state.enabled) return;
    const item = hit3D(e.clientX, e.clientY);
    if (item) AppActions.select(item.uid);
  });

  window.Phase3 = {
    state,
    enable: () => setEnabled(true),
    disable: () => setEnabled(false),
    sync: syncAll,
    loadSelectedGLB: () => glbInput.click(),
    setMode,
  };

  const originalDraw = App.draw;
  App.draw = function patchedDraw() { originalDraw(); if (state.enabled) syncAll(); };

  init();
  console.info('[Phase3B] Photo-matched 3D compositor initialized.');
})();
