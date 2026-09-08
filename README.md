# La Cigogne D'Ailleurs — Creative Room Visualizer

## Phase 2.5 — Robust visual placement

This build keeps the Phase 1/2 editor and strengthens the room-analysis layer before Phase 3.

### What was improved
- Clean separation between editor and AI extension.
- Floor mask cleanup with morphology and lower-boundary connected-component filtering.
- Smoothed floor-boundary profile across the image.
- Furniture movement is constrained to the room and anchored by the object's support point instead of snapping everything to the floor horizon.
- Rotated furniture uses its true bottom-most transformed corner for floor contact.
- Perspective scaling uses relative depth normalization plus a geometric fallback. Depth Anything V2 is a relative-depth model, not a metric camera measurement.
- Existing furniture is rechecked against the floor mask after analysis.
- Diagnostic overlays: floor mask, depth map, furniture zones, and floor boundary.
- Keyboard diagnostics: F = floor, D = depth, Z = zones.
- AI status messages distinguish server-offline errors from `/analyze` HTTP/model errors.
- `/` backend route now gives a small service summary instead of `Not Found`.
- `/health` reports CPU/CUDA and GPU information.

## Run

### Frontend
From the project root:

```bash
python -m http.server 5500
```

Open `http://localhost:5500`.

### Backend
In another terminal:

```bash
cd server
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check `http://localhost:8000/health`.

## Phase 2 pipeline

```text
Room photo
  ↓
Resize for inference
  ├── Depth Anything V2 → relative depth profile
  └── SegFormer ADE20K → floor / wall / furniture masks
  ↓
Floor-mask cleanup + floor boundary profile
  ↓
Perspective-aware 2D furniture sizing
  ↓
Floor support-point constraint
  ↓
Diagnostics + editor
```

## Phase 3C — 3D foundation

This build adds a hybrid Three.js 3D layer over the existing room photo. Three.js loads GLB/glTF through `GLTFLoader`; the catalog also has procedural 3D fallbacks so the phase can be tested without downloading furniture assets. Three.js recommends glTF/GLB for runtime 3D delivery, and `GLTFLoader` supports glTF 2.0.

### Phase 3C features
- `Mode 3D` overlays real-time WebGL furniture on the room photo.
- Existing Phase 2.5 furniture positions are projected onto a virtual floor plane.
- Procedural 3D furniture is generated for every catalog item.
- `Importer GLB` associates a `.glb`/`.gltf` model with the selected catalog type.
- Camera calibration: height, FOV and scene depth.
- 3D objects can be selected and moved directly in the 3D viewport.
- 2D and 3D layers remain available; Phase 2.5 is preserved.

The camera is intentionally a calibration layer rather than claiming metric reconstruction: Depth Anything V2 supplies relative depth, while the 3D camera still needs scene calibration.

Three.js references: https://threejs.org/docs/pages/GLTFLoader.html and https://threejs.org/manual/en/loading-3d-models.html.

### Next Phase 3C
- Per-object 3D transform gizmos.
- Real floor-plane calibration from the detected floor profile.
- GLB asset metadata and automatic dimension normalization.
- Occlusion-aware compositing.
- Better photo/3D lighting matching.
- Optional React Three Fiber migration once the 3D interaction model is stable.


## Phase 3C — Photo-matched 3D

Phase 3C upgrades the 3D layer from a basic overlay to a photo-matched compositor. It adds a locked perspective camera, automatic camera pitch/depth estimation from the Phase 2.5 floor profile, PBR/tone-mapped rendering, studio environment lighting, soft contact shadows, GLB dimension normalization, OrbitControls inspection mode, and TransformControls for precise 3D manipulation.

The intended workflow is: analyze room → enter 3D Photo Match → select furniture → import GLB → adjust camera only when necessary → use the gizmo for precise placement.

This is still not a full scanned-room reconstruction. Accurate per-pixel occlusion and full room geometry remain a later step.


## Phase 3C — Depth-aware compositing

Phase 3C keeps the Phase 3B PBR/photo-match workflow and adds a screen-space depth compositor. The AI depth map from Depth Anything V2 is fitted against the detected floor geometry, then compared with the Three.js depth buffer. This allows foreground room surfaces to hide parts of newly placed 3D furniture instead of treating the model as a flat overlay.

Important: Depth Anything V2 is relative depth, so this is an approximate occlusion system rather than a metrically reconstructed room. The inspector exposes occlusion strength and depth tolerance for ambiguous regions. 2D state is also synchronized after TransformControls edits.

### Phase 3C stack
- Three.js 0.185.1
- GLTFLoader / OrbitControls / TransformControls
- PBR materials + ACES tone mapping + RoomEnvironment
- AI depth texture compositing
- floor-calibrated relative-depth fit
- adjustable occlusion threshold
- persistent 2D/3D transform synchronization

### Next major step
Phase 3D should move beyond approximate screen-space occlusion toward explicit room geometry: wall/floor planes, furniture/background masks, camera calibration from vanishing points, and eventually a reconstructed scene representation.


## Phase 3D — Room reconstruction proxy + lighting match

Phase 3D moves the compositor toward a reconstructed room representation without claiming a metric scan. It adds a lightweight four-surface room proxy (floor + three walls), photographic lighting estimation, and contact-shadow treatment while preserving the Phase 3C depth compositor.

### Phase 3D features
- Four-surface room proxy used as a shadow-receiving reconstruction scaffold.
- Camera/depth calibration remains driven by the detected floor profile and relative depth.
- Automatic light estimation from the room photograph: dominant bright-region direction, approximate color temperature, ambient level, and key-light intensity.
- Manual temperature, exposure and key-light controls for correction.
- Per-object contact shadow cards to improve floor contact at small scales.
- AI depth occlusion remains active in the final screen-space compositor.

Important limitation: the room proxy is an approximation. It is not a LiDAR scan, dense 3D reconstruction, or metrically guaranteed wall mesh. The next stage should use stronger geometric inference (vanishing points / planes / furniture masks) and eventually a learned or scanned scene representation.

### Phase 3D stack
- Three.js 0.185.1
- GLTFLoader / OrbitControls / TransformControls / RoomEnvironment
- PBR materials + ACES tone mapping
- AI relative-depth compositor
- Room proxy shadow surfaces
- Image-based lighting estimation and manual calibration controls
