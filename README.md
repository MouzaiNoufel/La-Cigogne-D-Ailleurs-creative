# La Cigogne D'Ailleurs — Creative Room Visualizer

AI-assisted interior room visualizer: import a room photo, place catalog furniture, analyze the room with computer vision, and remove existing furniture with LaMa.

## Current Phase 1 + Phase 2

### Frontend
- Room image upload
- Furniture catalog with real-world dimensions
- Add / move / rotate / scale furniture
- Duplicate / delete
- PNG export
- AI room analysis button
- Floor segmentation overlay
- Relative depth-based perspective scaling
- Automatic floor anchoring after furniture movement
- Optional furniture removal tool

### AI backend
- Depth Anything V2 Small for monocular depth
- SegFormer ADE20K for semantic segmentation
- Floor / wall / window / door / furniture masks
- Scene metadata and furniture bounding boxes
- LaMa inpainting for furniture removal
- CUDA automatically used when available
- Large input images are resized for inference to control VRAM usage

## Project structure

```text
index.html
css/
  style.css
js/
  furniture-data.js   # catalog + generated SVG previews
  app.js              # core editor and canvas interactions
  ai.js               # Phase 2 AI integration
  eraser.js           # furniture removal integration
server/
  main.py             # FastAPI AI service
  requirements.txt
```

## Important script order

The browser must load the catalog before the editor, and the editor before AI extensions:

```html
<script src="js/furniture-data.js"></script>
<script src="js/app.js"></script>
<script src="js/ai.js"></script>
<script src="js/eraser.js"></script>
```

The previous version loaded `app.js` in the `<head>` before the DOM existed, loaded `eraser.js` before its dependencies, and loaded `app.js` twice. That structure has been removed.

## Run the frontend

From the project root:

```bash
python -m http.server 5500
```

Open:

```text
http://localhost:5500
```

## Run the AI server

In another terminal:

```bash
cd server
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check:

```text
http://localhost:8000/health
```

The first `/analyze` request downloads the pretrained models and may take a while.

## Phase 2 pipeline

```text
Room photo
    ↓
Resize for inference
    ↓
Depth Anything V2 ──────→ relative depth profile
    ↓
SegFormer ADE20K ───────→ floor / wall / window / door / furniture masks
    ↓
Scene metadata
    ↓
Perspective correction + floor anchoring
    ↓
2D furniture visualization
```

Furniture removal uses:

```text
Click furniture
    ↓
SegFormer mask selection
    ↓
Mask dilation / feathering
    ↓
LaMa inpainting
    ↓
Cleaned room image
    ↓
Run analysis again
```

## Next major phase

The next architectural step is Phase 3: controlled 3D placement with Three.js / React Three Fiber and GLB/glTF furniture assets. The current Phase 2 depth is relative, not metric 3D reconstruction, so it should be treated as a placement aid rather than a calibrated physical camera model.
