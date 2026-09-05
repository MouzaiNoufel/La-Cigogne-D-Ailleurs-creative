# La Cigogne D'Ailleurs — Creative Room Visualizer

## Phase 4A — photo integration and safe AI input

Phase 4A adds the first photo-integration layer while keeping the project
independent of a future hosting or AI-provider choice.

- **Occlusion IA:** after room analysis, use the new header toggle to draw
  detected existing furniture over the WebGL layer. This is conservative:
  it protects foreground furniture, but it cannot yet decide which detected
  object is behind a new item.
- **Photo-derived lighting:** `/analyze` returns low-cost color, brightness,
  and bright-area hints. The 3D scene applies them to its ambient, key, and
  fill lights. These are visual hints, not physical light reconstruction.
- **Export parity:** when Occlusion IA is active, the 3D PNG export includes
  the same front-layer mask as the editor preview.
- **Input safeguards:** the API limits image uploads to 15 MB, rejects
  non-image MIME types, and limits decoded image size before AI inference.

### Phase 4A workflow

1. Import a room image, then click **Analyser la pièce**.
2. Add a product and activate **Mode 3D**.
3. Turn on **Occlusion IA** only when an existing detected object should stay
   visibly in front of the new product. Turn it off when the mask hides too
   much; segmentation remains probabilistic.
4. Fine-tune the Phase 3B camera controls, then export the final PNG.

### Deferred work

Per-pixel depth ordering, automatic physical light reconstruction, generative
refinement, saved projects, product synchronization, and production hosting
remain later phases because they require product, infrastructure, and provider
decisions.

## Phase 3B — calibrated 3D placement

This package extends the Phase 3A prototype without changing the product or
deployment architecture. It adds:

- Three.js TransformControls for per-object move, rotation, and scale.
- A manual floor-horizon calibration plus a **Calibrer depuis le sol IA**
  shortcut based on the detected floor boundary. This is visual calibration,
  not a claim of metric room reconstruction.
- GLB asset metadata: filename, original bounding size, and catalog W/H/D
  normalization. The catalog now declares a height for every item.
- Transparent contact shadows over the room photo.
- A 3D PNG export that composites the WebGL scene over the original photo at
  the original image resolution. The legacy 2D export also now handles
  letterboxing correctly.

### Phase 3B workflow

1. Import a room photo and run **Analyser la pièce**.
2. Add a catalog item, switch on **Mode 3D**, then select the item.
3. Optionally choose **Calibrer depuis le sol IA** and fine tune the camera.
4. Use **Déplacer**, **Tourner**, or **Échelle** then drag the appropriate
   Three.js handle. Prefer a self-contained `.glb` when importing a product.
5. Use **Exporter PNG** while Mode 3D is enabled to save the composed result.

### Not yet included

Depth-aware occlusion of a 3D object by existing furniture, server-side saved
projects, WooCommerce synchronization, and AI photorealistic refinement are
deliberately left for later phases. They require the team’s integration and
hosting decisions.

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

## Phase 3A — 3D foundation

This build adds a hybrid Three.js 3D layer over the existing room photo. Three.js loads GLB/glTF through `GLTFLoader`; the catalog also has procedural 3D fallbacks so the phase can be tested without downloading furniture assets. Three.js recommends glTF/GLB for runtime 3D delivery, and `GLTFLoader` supports glTF 2.0.

### Phase 3A features
- `Mode 3D` overlays real-time WebGL furniture on the room photo.
- Existing Phase 2.5 furniture positions are projected onto a virtual floor plane.
- Procedural 3D furniture is generated for every catalog item.
- `Importer GLB` associates a `.glb`/`.gltf` model with the selected catalog type.
- Camera calibration: height, FOV and scene depth.
- 3D objects can be selected and moved directly in the 3D viewport.
- 2D and 3D layers remain available; Phase 2.5 is preserved.

The camera is intentionally a calibration layer rather than claiming metric reconstruction: Depth Anything V2 supplies relative depth, while the 3D camera still needs scene calibration.

Three.js references: https://threejs.org/docs/pages/GLTFLoader.html and https://threejs.org/manual/en/loading-3d-models.html.

### Next Phase 3B
- Per-object 3D transform gizmos.
- Real floor-plane calibration from the detected floor profile.
- GLB asset metadata and automatic dimension normalization.
- Occlusion-aware compositing.
- Better photo/3D lighting matching.
- Optional React Three Fiber migration once the 3D interaction model is stable.
