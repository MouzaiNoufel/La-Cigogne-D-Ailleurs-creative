from __future__ import annotations

import base64
import io
from typing import Dict, List, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import pipeline

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

MAX_ANALYSIS_SIDE = 1280
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
SEG_MODEL = "nvidia/segformer-b3-finetuned-ade-512-512"

# ADE20K labels used by the visualizer. Keep aliases broad because model
# label strings can vary slightly across Transformers versions.
ADE_STRUCT = {"floor", "wall", "windowpane", "door"}
ADE_FURNITURE = {
    "bed", "bedclothes", "chair", "sofa", "table", "desk", "cabinet",
    "chest of drawers", "counter", "bench", "shelf", "shelves", "ottoman",
    "armchair", "seat", "stool", "bookcase", "wardrobe", "coffee table",
    "dining table", "pillow", "lamp", "television", "monitor", "plant",
}

app = FastAPI(title="La Cigogne D'Ailleurs AI", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

depth_pipe = None
seg_pipe = None
lama = None


def get_device():
    return 0 if torch.cuda.is_available() else -1


def get_pipes():
    global depth_pipe, seg_pipe
    if depth_pipe is None or seg_pipe is None:
        device = get_device()
        device_name = "CUDA" if device >= 0 else "CPU"
        print(f"Loading AI models (device={device_name})...")
        depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device=device)
        seg_pipe = pipeline("image-segmentation", model=SEG_MODEL, device=device)
        print("AI models ready.")
    return depth_pipe, seg_pipe


def get_lama():
    global lama
    if lama is None:
        print("Loading LaMa inpainting model...")
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()
        print("LaMa ready.")
    return lama


def to_data_url(img: Image.Image, max_side: int | None = None) -> str:
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


def percentile_pair(values: np.ndarray, low_q=0.10, high_q=0.90):
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if values.size < 10:
        return 0.0, 1.0
    lo, hi = np.quantile(values, [low_q, high_q])
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    return float(lo), float(hi)


def clean_floor_mask(mask: np.ndarray) -> np.ndarray:
    """Close small holes and retain the useful lower connected floor area."""
    if cv2 is None or not mask.any():
        return mask
    u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((7, 7), np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # Prefer components touching the lower image boundary; furniture masks
    # should not accidentally become the floor.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    if n > 1:
        bottom_labels = set(labels[-1, :].tolist()) - {0}
        if bottom_labels:
            keep = np.isin(labels, list(bottom_labels))
            u8 = np.where(keep, 255, 0).astype(np.uint8)
    return u8 > 128


def smooth_profile(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    valid = np.flatnonzero(values > 0)
    if valid.size == 0:
        return values
    if valid.size == 1:
        values[:] = values[valid[0]]
        return values
    values = np.interp(np.arange(len(values)), valid, values[valid]).astype(np.float32)
    if cv2 is not None:
        return cv2.GaussianBlur(values.reshape(1, -1), (0, 0), 6).reshape(-1).astype(np.float32)
    kernel = np.ones(15, dtype=np.float32) / 15.0
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def floor_profile(depth: np.ndarray, floor_mask: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    H, W = depth.shape
    profile = np.zeros(H, dtype=np.float32)
    for y in range(H):
        row = depth[y][floor_mask[y]]
        if row.size:
            profile[y] = float(np.median(row))
    profile = smooth_profile(profile)

    tops = np.full(W, np.nan, dtype=np.float32)
    for x in range(W):
        ys = np.flatnonzero(floor_mask[:, x])
        if ys.size:
            tops[x] = ys[0]
    valid = np.flatnonzero(np.isfinite(tops))
    if valid.size:
        tops = np.interp(np.arange(W), valid, tops[valid]).astype(np.float32)
        if cv2 is not None:
            tops = cv2.GaussianBlur(tops.reshape(1, -1), (0, 0), max(2, W / 160)).reshape(-1).astype(np.float32)
        floor_top = int(np.median(tops[valid]))
    else:
        floor_top = int(H * 0.60)
        tops[:] = floor_top
    return profile, floor_top, tops


def aggregate_masks(seg_results, original_size: Tuple[int, int], analysis_size: Tuple[int, int]):
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
        arr = np.asarray(
            Image.fromarray((arr_small * 255).astype(np.uint8)).resize((W, H), Image.Resampling.NEAREST)
        ) > 128
        key = "furniture" if label in ADE_FURNITURE else label
        masks[key] = masks[key] | arr if key in masks else arr
        if label in ADE_FURNITURE:
            ys, xs = np.where(arr)
            if len(xs):
                regions.append({
                    "label": label,
                    "x": int(xs.min()), "y": int(ys.min()),
                    "width": int(xs.max() - xs.min() + 1),
                    "height": int(ys.max() - ys.min() + 1),
                    "area": int(arr.sum()),
                })

    if "floor" in masks:
        masks["floor"] = clean_floor_mask(masks["floor"])
        floor_source = "segformer"
    else:
        fallback = np.zeros((H, W), dtype=bool)
        fallback[int(H * 0.62):, :] = True
        masks["floor"] = fallback
        floor_source = "fallback"
    return masks, regions, floor_source


@app.get("/")
def root():
    return {"ok": True, "service": "La Cigogne D'Ailleurs AI", "version": app.version, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": app.version,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
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

    depth_result = dp(analysis_image)
    depth_pil = depth_result["depth"].convert("L").resize((aw, ah), Image.Resampling.BILINEAR)
    depth = np.asarray(depth_pil, dtype=np.float32) / 255.0

    seg_results = sp(analysis_image)
    masks, regions, floor_source = aggregate_masks(seg_results, (W, H), (aw, ah))
    floor_original = masks["floor"]
    floor_for_depth = np.asarray(
        Image.fromarray((floor_original * 255).astype(np.uint8)).resize((aw, ah), Image.Resampling.NEAREST)
    ) > 128

    profile_small, floor_top_small, floor_top_profile_small = floor_profile(depth, floor_for_depth)
    profile = np.interp(np.linspace(0, ah - 1, H), np.arange(ah), profile_small).astype(np.float32)
    floor_top_profile = np.interp(
        np.linspace(0, aw - 1, W), np.arange(aw), floor_top_profile_small
    ).astype(np.float32)

    floor_values = depth[floor_for_depth]
    depth_low, depth_high = percentile_pair(floor_values)
    # Depth Anything V2 is relative depth. Infer its polarity from the
    # expected indoor perspective trend on the detected floor: pixels lower
    # in the image should normally represent the nearer part of the floor.
    y_idx = np.arange(ah, dtype=np.float32)
    valid_rows = (y_idx >= floor_top_small) & (profile_small > 0)
    if int(valid_rows.sum()) >= 8:
        corr = np.corrcoef(y_idx[valid_rows], profile_small[valid_rows])[0, 1]
        near_is_high = bool(np.isfinite(corr) and corr >= 0)
    else:
        near_is_high = True

    floor_top_y = int(round(floor_top_small / max(1, ah - 1) * max(1, H - 1)))
    depth_vis = Image.fromarray(np.clip(depth * 255, 0, 255).astype(np.uint8), mode="L")
    out_masks = {name: to_data_url(Image.fromarray((arr * 255).astype(np.uint8)), max_side=1600) for name, arr in masks.items()}

    # Downsample the floor boundary for a lightweight frontend debug overlay.
    sample_n = min(240, W)
    sample_x = np.linspace(0, W - 1, sample_n).astype(int)
    top_profile_out = floor_top_profile[sample_x].round(1).tolist()

    return JSONResponse({
        "version": 3.0,
        "width": W, "height": H,
        "analysis_width": aw, "analysis_height": ah,
        "depth": to_data_url(depth_vis, max_side=1600),
        "masks": out_masks,
        "floor_depth": profile.tolist(),
        "reference_depth": float(depth_high),
        "depth_floor_low": float(depth_low),
        "depth_floor_high": float(depth_high),
        "depth_near_is_high": bool(near_is_high),
        "floor_top_y": floor_top_y,
        "floor_top_profile": top_profile_out,
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


async def _segment_furniture_at_point(image: Image.Image, x: int, y: int):
    """Return a clean full-resolution furniture mask and semantic label for a click."""
    W, H = image.size
    x = int(np.clip(x, 0, W - 1)); y = int(np.clip(y, 0, H - 1))
    _, sp = get_pipes()
    analysis_image = resize_for_analysis(image)
    aw, ah = analysis_image.size
    sx = aw / W; sy = ah / H
    ax = int(round(x * sx)); ay = int(round(y * sy))

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
        # A small click neighborhood is much more forgiving than a single pixel.
        y0, y1 = max(0, ay - 8), min(ah, ay + 9)
        x0, x1 = max(0, ax - 8), min(aw, ax + 9)
        if arr_small[y0:y1, x0:x1].any():
            candidates.append((area, label, arr_small))

    if not candidates:
        raise HTTPException(status_code=404, detail="Aucun meuble détecté à cet endroit — cliquez au centre du meuble.")

    # Prefer the smallest matching mask: a chair/pillow should not lose to a
    # huge surrounding class when both contain the click.
    _, selected_label, selected_small = min(candidates, key=lambda item: item[0])
    mask_original = np.asarray(
        Image.fromarray((selected_small * 255).astype(np.uint8)).resize((W, H), Image.Resampling.NEAREST)
    )
    if cv2 is not None:
        mask_u8 = (mask_original > 128).astype(np.uint8) * 255
        # Close tiny holes, then expand only slightly so LaMa gets a useful
        # context ring without deleting large parts of neighboring furniture.
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        mask_u8 = cv2.dilate(mask_u8, np.ones((5, 5), np.uint8), iterations=1)
        final_mask = Image.fromarray(mask_u8).convert("L")
    else:
        final_mask = Image.fromarray((mask_original > 128).astype(np.uint8) * 255).convert("L")
    return image, selected_label, final_mask


@app.post("/select-mask")
async def select_mask(file: UploadFile = File(...), x: int = Form(...), y: int = Form(...)):
    """Phase 4 smart-selection endpoint used by the frontend preview step."""
    image = read_image(file)
    _, label, mask = await _segment_furniture_at_point(image, x, y)
    return JSONResponse({"mask": to_data_url(mask, max_side=1600), "label": label})


@app.post("/inpaint")
async def inpaint(file: UploadFile = File(...), mask: UploadFile = File(...)):
    """Remove a user-confirmed furniture mask with LaMa."""
    image = read_image(file)
    try:
        raw = await mask.read()
        mask_img = Image.open(io.BytesIO(raw)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Masque invalide: {exc}") from exc
    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size, Image.Resampling.BILINEAR)
    arr = np.asarray(mask_img, dtype=np.uint8)
    if cv2 is not None:
        # Make the transition softer and protect against a hard pasted seam.
        arr = cv2.GaussianBlur(arr, (0, 0), 1.25)
    mask_img = Image.fromarray(arr).convert("L")
    result = get_lama()(image, mask_img)
    if not isinstance(result, Image.Image):
        result = Image.fromarray(np.asarray(result).astype(np.uint8))
    return JSONResponse({"image": to_data_url(result), "label": "meuble"})


@app.post("/remove")
async def remove(file: UploadFile = File(...), x: int = Form(...), y: int = Form(...)):
    image = read_image(file)
    W, H = image.size
    x = int(np.clip(x, 0, W - 1)); y = int(np.clip(y, 0, H - 1))
    _, sp = get_pipes()
    analysis_image = resize_for_analysis(image)
    aw, ah = analysis_image.size
    sx = aw / W; sy = ah / H
    ax = int(round(x * sx)); ay = int(round(y * sy))

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
        raise HTTPException(status_code=404, detail="Aucun meuble détecté à cet endroit — cliquez au centre du meuble.")

    _, selected_label, selected_small = min(candidates, key=lambda item: item[0])
    mask_original = np.asarray(
        Image.fromarray((selected_small * 255).astype(np.uint8)).resize((W, H), Image.Resampling.NEAREST)
    )
    if cv2 is not None:
        mask_u8 = (mask_original > 128).astype(np.uint8) * 255
        dilated = cv2.dilate(mask_u8, np.ones((9, 9), np.uint8), iterations=2)
        dilated = cv2.GaussianBlur(dilated, (5, 5), 0)
        final_mask = Image.fromarray(dilated).convert("L")
    else:
        final_mask = Image.fromarray((mask_original > 128).astype(np.uint8) * 255).convert("L")

    result = get_lama()(image, final_mask)
    if not isinstance(result, Image.Image):
        result = Image.fromarray(np.asarray(result).astype(np.uint8))
    return JSONResponse({"image": to_data_url(result), "label": selected_label})
