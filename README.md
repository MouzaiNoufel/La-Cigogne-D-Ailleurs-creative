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

## Phase 3

Phase 3 remains controlled 3D placement with Three.js / React Three Fiber and GLB/glTF furniture assets. The current depth map is still relative depth, so this Phase 2.5 system is a robust 2D placement aid rather than calibrated metric 3D reconstruction.
