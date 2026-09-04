"""
La Cigogne D'Ailleurs — Creative · Phase 2 AI server

Endpoints
---------
GET  /health
POST /analyze  -> depth + semantic masks + scene metadata
POST /remove   -> furniture selection + LaMa inpainting

Run:
    cd server
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
"""

import base64
import io
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Cigogne AI - Room Analysis", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_ANALYSIS_SIDE = 1280

# Lazy-loaded models. This keeps /health fast and avoids loading GPU memory
# until the first actual AI request.
depth_pipe = None
seg_pipe = None
lama = None

ADE_STRUCT = {"floor", "wall", "windowpane", "door"}
ADE_FURNITURE = {
    "bed", "sofa", "chair", "table", "desk", "cabinet",
    "shelving", "lamp", "rug", "chest of drawers", "countertop",
    "armchair", "ottoman", "stool", "bench", "wardrobe",
}


def get_pipes():
    """Load Depth Anything V2 + SegFormer once, using CUDA when available."""
    global depth_pipe, seg_pipe
    if depth_pipe is not None and seg_pipe is not None:
        return depth_pipe, seg_pipe

    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading AI models (device={'CUDA' if device == 0 else 'CPU'})...")

    depth_pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    seg_pipe = pipeline(
        "image-segmentation",
        model="nvidia/segformer-b3-finetuned-ade-512-512",
        device=device,
    )
    print("AI models ready.")
    return depth_pipe, seg_pipe


def get_lama():
    global lama
    if lama is None:
        from simple_lama_inpainting import LaMa
        print("Loading LaMa...")
        lama = LaMa()
        print("LaMa ready.")
    return lama


def to_data_url(img: Image.Image, max_side: int | None = None) -> str:
    """Convert a PIL image to a compact PNG data URL."""
    out = img.convert("RGBA")
    if max_side:
        scale = min(1.0, max_side / max(out.size))
        if scale < 1:
            out = out.resize((max(1, int(out.width * scale)), max(1, int(out.height * scale))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def read_image(file: UploadFile) -> Image.Image:
    try:
        raw = file.file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        if image.width < 64 or image.height < 64:
            raise ValueError("image too small")
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image invalide: {exc}") from exc


def resize_for_analysis(image: Image.Image) -> Image.Image:
    scale = min(1.0, MAX_ANALYSIS_SIDE / max(image.size))
    if scale >= 1:
        return image
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def mask_array(mask, size: Tuple[int, int], threshold: int = 128) -> np.ndarray:
    """Normalize a Transformers segmentation mask to a boolean HxW array."""
    if isinstance(mask, Image.Image):
        gray = mask.convert("L").resize(size, Image.Resampling.NEAREST)
        arr = np.asarray(gray)
    else:
        arr = np.asarray(mask)
        if arr.ndim > 2:
            arr = arr.squeeze()
        if arr.shape[::-1] != size:
            gray = Image.fromarray(arr.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
            arr = np.asarray(gray)
    return arr > threshold


def smooth_profile(values: np.ndarray) -> np.ndarray:
    """Fill missing rows and smooth the 1D floor-depth signal."""
    values = values.astype(np.float32)
    valid = np.flatnonzero(values > 0)
    if valid.size == 0:
        return values
    if valid.size == 1:
        values[:] = values[valid[0]]
        return values

    values = np.interp(np.arange(len(values)), valid, values[valid]).astype(np.float32)
    # Small moving average without requiring another dependency.
    kernel = np.ones(15, dtype=np.float32) / 15.0
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def floor_profile(depth: np.ndarray, floor_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    H, W = depth.shape
    profile = np.zeros(H, dtype=np.float32)
    for y in range(H):
        row = depth[y][floor_mask[y]]
        if row.size:
            profile[y] = float(np.median(row))

    profile = smooth_profile(profile)

    # Estimate the first visible floor boundary from columns that actually
    # contain floor. This is more stable than simply using the first non-zero
    # row of the aggregated mask.
    boundaries: List[int] = []
    for x in range(W):
        ys = np.flatnonzero(floor_mask[:, x])
        if ys.size:
            boundaries.append(int(ys[0]))
    if boundaries:
        floor_top = int(np.median(boundaries))
    else:
        floor_top = int(H * 0.60)

    return profile, floor_top


def aggregate_masks(seg_results, original_size: Tuple[int, int], analysis_size: Tuple[int, int]):
    """Aggregate ADE20K classes and retain useful per-region metadata."""
    W, H = original_size
    aw, ah = analysis_size
    masks: Dict[str, np.ndarray] = {}
    regions: List[dict] = []

    for result in seg_results:
        label = str(result.get("label", "")).lower().strip()
        if label not in ADE_STRUCT and label not in ADE_FURNITURE:
            continue

        arr_small = mask_array(result["mask"], (aw, ah))
        if not arr_small.any():
            continue

        # Keep the model-resolution mask for calculations; resize once to the
        # original image for API output and client-side overlays.
        arr = np.asarray(
            Image.fromarray((arr_small * 255).astype(np.uint8))
            .resize((W, H), Image.Resampling.NEAREST)
        ) > 128

        key = "furniture" if label in ADE_FURNITURE else label
        masks[key] = masks[key] | arr if key in masks else arr

        if label in ADE_FURNITURE:
            ys, xs = np.where(arr)
            if len(xs):
                regions.append({
                    "label": label,
                    "x": int(xs.min()),
                    "y": int(ys.min()),
                    "width": int(xs.max() - xs.min() + 1),
                    "height": int(ys.max() - ys.min() + 1),
                    "area": int(arr.sum()),
                })

    # SegFormer can miss the floor in difficult images. Use a conservative
    # lower-region prior rather than returning no floor at all.
    if "floor" not in masks:
        fallback = np.zeros((H, W), dtype=bool)
        fallback[int(H * 0.60):, :] = True
        masks["floor"] = fallback
        floor_source = "fallback"
    else:
        floor_source = "segformer"

    return masks, regions, floor_source


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": app.version,
        "models_loaded": depth_pipe is not None and seg_pipe is not None,
        "lama_loaded": lama is not None,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image = read_image(file)
    W, H = image.size
    analysis_image = resize_for_analysis(image)
    aw, ah = analysis_image.size
    dp, sp = get_pipes()

    # 1) Monocular depth. Normalize to [0, 1] for relative perspective use.
    depth_result = dp(analysis_image)
    depth_pil = depth_result["depth"].convert("L").resize((aw, ah), Image.Resampling.BILINEAR)
    depth = np.asarray(depth_pil, dtype=np.float32) / 255.0

    # 2) Semantic segmentation.
    seg_results = sp(analysis_image)
    masks, regions, floor_source = aggregate_masks(seg_results, (W, H), (aw, ah))

    floor_original = masks["floor"]
    floor_for_depth = np.asarray(
        Image.fromarray((floor_original * 255).astype(np.uint8))
        .resize((aw, ah), Image.Resampling.NEAREST)
    ) > 128

    profile_small, floor_top_small = floor_profile(depth, floor_for_depth)
    profile = np.interp(
        np.linspace(0, ah - 1, H),
        np.arange(ah),
        profile_small,
    ).astype(np.float32)

    # Reference depth is the near-camera part of the floor. Quantile is more
    # robust than blindly taking the final image row.
    floor_values = depth[floor_for_depth]
    reference_depth = float(np.quantile(floor_values, 0.90)) if floor_values.size else float(np.max(depth))
    reference_depth = max(reference_depth, 0.05)
    floor_top_y = int(round(floor_top_small / max(1, ah - 1) * max(1, H - 1)))

    depth_vis = Image.fromarray(np.clip(depth * 255, 0, 255).astype(np.uint8), mode="L")
    out_masks = {
        name: to_data_url(Image.fromarray((arr * 255).astype(np.uint8)), max_side=1600)
        for name, arr in masks.items()
    }

    return JSONResponse({
        "version": 2,
        "width": W,
        "height": H,
        "analysis_width": aw,
        "analysis_height": ah,
        "depth": to_data_url(depth_vis, max_side=1600),
        "masks": out_masks,
        "floor_depth": profile.tolist(),
        "reference_depth": reference_depth,
        "floor_top_y": floor_top_y,
        "floor_source": floor_source,
        "scene": {
            "furniture_count": len(regions),
            "furniture": regions,
            "has_floor": "floor" in masks,
            "has_walls": "wall" in masks,
            "has_windows": "windowpane" in masks,
            "has_doors": "door" in masks,
        },
    })


@app.post("/remove")
async def remove(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
):
    """Find the furniture region containing (x, y), then inpaint it with LaMa."""
    image = read_image(file)
    W, H = image.size
    x = int(np.clip(x, 0, W - 1))
    y = int(np.clip(y, 0, H - 1))
    _, sp = get_pipes()

    analysis_image = resize_for_analysis(image)
    aw, ah = analysis_image.size
    sx = aw / W
    sy = ah / H
    ax = int(round(x * sx))
    ay = int(round(y * sy))

    seg_results = sp(analysis_image)
    candidates = []
    for result in seg_results:
        label = str(result.get("label", "")).lower().strip()
        if label not in ADE_FURNITURE:
            continue
        arr_small = mask_array(result["mask"], (aw, ah))
        area = int(arr_small.sum())
        if area < max(40, int(aw * ah * 0.0002)):
            continue
        y0, y1 = max(0, ay - 5), min(ah, ay + 6)
        x0, x1 = max(0, ax - 5), min(aw, ax + 6)
        if arr_small[y0:y1, x0:x1].any():
            candidates.append((area, label, arr_small))

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="Aucun meuble détecté à cet endroit — cliquez au centre du meuble.",
        )

    # Prefer the smallest containing region. This prevents a large generic
    # furniture mask from winning over a more precise overlapping region.
    _, selected_label, selected_small = min(candidates, key=lambda item: item[0])

    mask_original = np.asarray(
        Image.fromarray((selected_small * 255).astype(np.uint8))
        .resize((W, H), Image.Resampling.NEAREST)
    )

    # Dilate + feather slightly so LaMa gets clean object boundaries.
    import cv2
    mask_u8 = (mask_original > 128).astype(np.uint8) * 255
    kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=2)
    dilated = cv2.GaussianBlur(dilated, (5, 5), 0)
    final_mask = Image.fromarray(dilated).convert("L")

    result = get_lama()(image, final_mask)
    if not isinstance(result, Image.Image):
        result = Image.fromarray(np.asarray(result).astype(np.uint8))

    return JSONResponse({
        "image": to_data_url(result),
        "label": selected_label,
    })
