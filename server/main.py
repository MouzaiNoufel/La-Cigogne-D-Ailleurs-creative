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

# Phase 5.2.16 REMOVE-ONLY quality gate. Generative replacement engines are
# intentionally disabled for the Remove AI workflow. A candidate must clear
# both numbers to be considered usable. furniture_penalty alone is not sufficient: a warped/
# ghosted candidate that SegFormer does not recognise as furniture-shaped can
# still have a very low furniture_penalty while looking clearly wrong. These
# starting values are a reasoned guess, not a calibrated measurement — read
# the "seam=" numbers this build prints for results you judge good vs. bad on
# your own machine and tighten/loosen MAX_ACCEPTABLE_SEAM to match.
LAMA_CLEAN_FURNITURE_PENALTY = 0.30
MAX_ACCEPTABLE_SEAM = 45.0
# Phase 5.2.16: generative replacement engines intentionally disabled in Remove AI.
# The product requirement is REMOVE, never replacement furniture.

# ADE20K labels used by the visualizer. Keep aliases broad because model
# label strings can vary slightly across Transformers versions.
ADE_STRUCT = {"floor", "wall", "windowpane", "door"}
ADE_FURNITURE = {
    "bed", "bedclothes", "chair", "sofa", "table", "desk", "cabinet",
    "chest of drawers", "counter", "bench", "shelf", "shelves", "ottoman",
    "armchair", "seat", "stool", "bookcase", "wardrobe", "coffee table",
    "dining table", "pillow", "lamp", "television", "monitor", "plant",
}

app = FastAPI(title="La Cigogne D'Ailleurs AI", version="5.2.28")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

depth_pipe = None
seg_pipe = None
lama = None
rorem_pipe = None
rorem_failed = False
smarteraser_pipe = None
smarteraser_failed = False
diffusion_pipe = None
diffusion_failed = False
powerpaint_pipe = None
powerpaint_failed = False
sam_model = None
sam_processor = None
sam_failed = False


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



def get_sam():
    """Load point-prompted SAM lazily.

    SAM is intentionally lazy because the normal room-analysis path does not
    need it. If SAM cannot be loaded, the caller falls back to an OpenCV
    GrabCut refinement instead of returning a 500 to the browser.
    """
    global sam_model, sam_processor, sam_failed
    if sam_model is not None and sam_processor is not None:
        return sam_model, sam_processor
    if sam_failed:
        return None, None
    try:
        from transformers import SamModel, SamProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM point-refiner (device={device})...")
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
        sam_model = sam_model.to(device)
        sam_model.eval()
        print("SAM point-refiner ready.")
        return sam_model, sam_processor
    except Exception as exc:
        sam_failed = True
        print(f"WARNING: SAM unavailable, using GrabCut fallback: {exc}")
        return None, None


def _connected_component_at_point(seed: np.ndarray, x: int, y: int) -> np.ndarray:
    """Return only the semantic component containing the user click."""
    if not seed.any():
        return seed.astype(bool)
    if cv2 is None:
        return seed.astype(bool)
    u8 = (seed.astype(np.uint8) * 255)
    # Close tiny holes but never grow the semantic region aggressively.
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    if n <= 1:
        return seed.astype(bool)
    yy = int(np.clip(y, 0, seed.shape[0] - 1)); xx = int(np.clip(x, 0, seed.shape[1] - 1))
    lab = int(labels[yy, xx])
    if lab > 0:
        return labels == lab
    # If the semantic prediction misses the exact click by a few pixels,
    # choose the nearest non-empty component rather than the largest one.
    ys, xs = np.where(labels > 0)
    if xs.size:
        k = int(np.argmin((xs - xx) ** 2 + (ys - yy) ** 2))
        return labels == labels[ys[k], xs[k]]
    return seed.astype(bool)


def _semantic_box(seed: np.ndarray, x: int, y: int, image_size: Tuple[int, int]):
    H, W = seed.shape
    component = _connected_component_at_point(seed, x, y)
    ys, xs = np.where(component)
    if xs.size < 20:
        # Small local fallback around the click.
        half = int(max(48, min(W, H) * 0.10))
        x0, x1 = max(0, x-half), min(W-1, x+half)
        y0, y1 = max(0, y-half), min(H-1, y+half)
        return component, (x0, y0, x1, y1)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    # Expand enough to include furniture edges/legs while keeping the prompt
    # local. This is deliberately much tighter than a whole-scene semantic mask.
    bw, bh = x1-x0+1, y1-y0+1
    pad_x = int(np.clip(round(bw * 0.14), 8, 120))
    pad_y = int(np.clip(round(bh * 0.14), 8, 120))
    return component, (max(0, x0-pad_x), max(0, y0-pad_y), min(W-1, x1+pad_x), min(H-1, y1+pad_y))


def _sample_positive_points(component: np.ndarray, x: int, y: int, max_points: int = 7):
    """Create a compact set of positive prompts inside the semantic component."""
    H, W = component.shape
    ys, xs = np.where(component)
    pts = [(float(x), float(y))]
    if xs.size == 0:
        return pts
    # Prefer interior pixels (distance transform) so prompts do not sit on edges.
    if cv2 is not None:
        dist = cv2.distanceTransform((component.astype(np.uint8)), cv2.DIST_L2, 5)
        flat = np.argsort(dist.ravel())[::-1]
        seen = {(int(x), int(y))}
        for idx in flat[:max(300, max_points*50)]:
            yy, xx = np.unravel_index(int(idx), dist.shape)
            if dist[yy, xx] < 3:
                break
            key = (int(xx), int(yy))
            if key in seen:
                continue
            if ((xx-x)**2 + (yy-y)**2) > max(W,H)**2 * 0.08:
                continue
            seen.add(key); pts.append((float(xx), float(yy)))
            if len(pts) >= max_points:
                break
    if len(pts) < 3:
        # Deterministic quantile samples across the semantic component.
        for q in (0.25, 0.5, 0.75):
            pts.append((float(np.quantile(xs, q)), float(np.quantile(ys, q))))
            if len(pts) >= max_points:
                break
    # Deduplicate after rounding.
    out=[]; seen=set()
    for px, py in pts:
        k=(int(round(px)), int(round(py)))
        if k not in seen:
            seen.add(k); out.append([float(k[0]), float(k[1])])
    return out[:max_points]


def _grabcut_refine(image: Image.Image, x: int, y: int, seed: np.ndarray, box=None):
    """Local GrabCut fallback constrained by the semantic object component."""
    if cv2 is None:
        return Image.fromarray((seed.astype(np.uint8) * 255), mode="L")
    rgb = np.asarray(image.convert("RGB"))
    H, W = seed.shape
    component, sb = _semantic_box(seed, x, y, image.size)
    if box is None:
        box = sb
    bx0, by0, bx1, by1 = map(int, box)
    crop = rgb[by0:by1+1, bx0:bx1+1]
    local_seed = component[by0:by1+1, bx0:bx1+1]
    if local_seed.sum() < 20:
        return Image.fromarray((component.astype(np.uint8) * 255), mode="L")

    gc = np.full(local_seed.shape, cv2.GC_PR_BGD, np.uint8)
    gc[local_seed] = cv2.GC_PR_FGD
    core = cv2.erode((local_seed.astype(np.uint8) * 255), np.ones((5,5), np.uint8), iterations=1)
    gc[core > 0] = cv2.GC_FGD
    # A narrow frame is definite background; do not mark the whole crop bg.
    frame = np.zeros(local_seed.shape, np.uint8)
    frame[:3,:] = frame[-3:,:] = frame[:,:3] = frame[:,-3:] = 1
    gc[frame > 0] = cv2.GC_BGD
    py = int(np.clip(y-by0, 0, gc.shape[0]-1)); px = int(np.clip(x-bx0, 0, gc.shape[1]-1))
    gc[py,px] = cv2.GC_FGD
    bgd=np.zeros((1,65),np.float64); fgd=np.zeros((1,65),np.float64)
    try:
        cv2.grabCut(crop, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        out=((gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD)).astype(np.uint8)
    except Exception:
        out=local_seed.astype(np.uint8)
    full=np.zeros((H,W),np.uint8); full[by0:by1+1,bx0:bx1+1]=out
    # Never return disconnected scene-wide islands.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(full, 8)
    if n > 1:
        lab=int(labels[int(np.clip(y,0,H-1)),int(np.clip(x,0,W-1))])
        if lab > 0:
            full=(labels==lab).astype(np.uint8)
    return Image.fromarray(full*255, mode="L")


def refine_mask_with_sam(image: Image.Image, x: int, y: int, seed: np.ndarray):
    """Segment the *clicked physical object*, not the whole semantic region.

    Important design rule: SegFormer is context/label evidence only. It must
    never be used as a multi-positive prompt for SAM because ADE20K can merge
    adjacent furniture (for example a bed + nightstand). SAM is therefore
    prompted primarily by the user's click, with a few nearby positive points
    and negative points just outside the local object area.
    """
    model, processor = get_sam()
    component, (bx0, by0, bx1, by1) = _semantic_box(seed, x, y, image.size)
    if model is None or processor is None:
        return _grabcut_refine(image, x, y, component, (bx0, by0, bx1, by1))

    H, W = component.shape
    try:
        # Do NOT feed the whole semantic component as positive points. If the
        # semantic model merged two touching objects, that would explicitly
        # tell SAM to merge them too. Use the user's click as the anchor and
        # only add close interior positives.
        positives = [(float(x), float(y))]
        if cv2 is not None and component.any():
            dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
            # Interior points must be close to the click. This prevents a bed
            # click from adding a nightstand several hundred pixels away.
            radius = max(28.0, min(180.0, 0.22 * max(bx1 - bx0 + 1, by1 - by0 + 1)))
            yy0 = max(0, int(y - radius)); yy1 = min(H, int(y + radius + 1))
            xx0 = max(0, int(x - radius)); xx1 = min(W, int(x + radius + 1))
            local = dist[yy0:yy1, xx0:xx1]
            flat = np.argsort(local.ravel())[::-1]
            for idx in flat[:500]:
                ly, lx = np.unravel_index(int(idx), local.shape)
                py, px = yy0 + ly, xx0 + lx
                if local[ly, lx] < 4:
                    break
                if (px - x) ** 2 + (py - y) ** 2 > radius ** 2:
                    continue
                if all((px - int(px0)) ** 2 + (py - int(py0)) ** 2 > 18 ** 2 for px0, py0 in positives):
                    positives.append((float(px), float(py)))
                if len(positives) >= 4:
                    break

        # Negative points outside the semantic component help SAM separate a
        # neighboring object when the prompt is close to its boundary.
        negatives = []
        ring = max(12, int(min(W, H) * 0.018))
        candidates_neg = [
            (x-ring, y), (x+ring, y), (x, y-ring), (x, y+ring),
            (x-ring, y-ring), (x+ring, y-ring),
            (x-ring, y+ring), (x+ring, y+ring),
        ]
        for px, py in candidates_neg:
            px = int(np.clip(px, 0, W - 1)); py = int(np.clip(py, 0, H - 1))
            if not component[py, px] and all((px-a)**2 + (py-b)**2 > 10**2 for a,b in negatives):
                negatives.append((float(px), float(py)))
            if len(negatives) >= 4:
                break

        points = positives + negatives
        labels = [1] * len(positives) + [0] * len(negatives)

        # Point-only prompting is intentional. A coarse semantic bounding box
        # can force SAM to return adjacent furniture as one object.
        inputs = processor(
            images=image,
            input_points=[points],
            input_labels=[labels],
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )
        candidates = masks[0]
        scores = np.asarray(outputs.iou_scores.detach().cpu().numpy()[0]).reshape(-1)
        seed_f = component.astype(bool)
        seed_area = max(1.0, float(seed_f.sum()))
        image_area = float(H * W)
        best = None
        best_score = -1e9

        for i in range(int(candidates.shape[0])):
            m = np.asarray(candidates[i].numpy()).astype(bool)
            if m.shape != (H, W):
                continue
            if not m[int(np.clip(y, 0, H-1)), int(np.clip(x, 0, W-1))]:
                continue
            area = float(m.sum())
            if area < 80 or area > image_area * 0.35:
                continue

            inter = float((m & seed_f).sum())
            union = float((m | seed_f).sum())
            iou_seed = inter / max(1.0, union)
            precision = inter / max(1.0, area)
            area_ratio = area / image_area
            # Penalize masks that are much larger than the local semantic
            # object prior. This is the key guard against bed+nightstand.
            size_ratio = area / seed_area
            size_penalty = max(0.0, size_ratio - 1.35) + max(0.0, 0.45 - size_ratio) * 0.35
            point_hit = sum(
                1 for px, py in positives
                if m[int(np.clip(round(py), 0, H-1)), int(np.clip(round(px), 0, W-1))]
            ) / max(1, len(positives))
            negative_hit = sum(
                1 for px, py in negatives
                if m[int(np.clip(round(py), 0, H-1)), int(np.clip(round(px), 0, W-1))]
            ) / max(1, len(negatives))
            sam_score = float(scores[i]) if i < len(scores) else 0.0

            score = (
                2.8 * point_hit
                + 1.4 * (1.0 - negative_hit)
                + 1.1 * sam_score
                + 0.9 * iou_seed
                + 0.5 * precision
                - 1.4 * size_penalty
                - 0.25 * area_ratio
            )
            if score > best_score:
                best_score = score
                best = m

        if best is not None:
            if cv2 is not None:
                u8 = (best.astype(np.uint8) * 255)
                # Keep the clicked connected component only.
                n, labels_cc, _, _ = cv2.connectedComponentsWithStats(u8, 8)
                lab = int(labels_cc[int(np.clip(y, 0, H-1)), int(np.clip(x, 0, W-1))])
                if lab > 0:
                    best = labels_cc == lab

                # Semantic guardrail: the SegFormer component is context, not
                # the final mask, but it is valuable for rejecting a nearby
                # second object. Allow SAM to extend beyond the semantic seed
                # only by a controlled edge band (for legs/soft contours).
                seed_band = cv2.dilate(
                    component.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
                    iterations=1,
                ) > 0
                constrained = best & seed_band
                # If the constraint would remove too much of the SAM result,
                # keep the SAM mask; otherwise use the constrained mask. This
                # is what prevents bed + nightstand from becoming one object
                # when SegFormer has a separate bed component.
                if constrained.sum() >= max(80, int(best.sum() * 0.68)):
                    best = constrained

                # Small edge cleanup only. No large dilation.
                best = cv2.morphologyEx(
                    (best.astype(np.uint8) * 255),
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                ) > 127
            return Image.fromarray((best.astype(np.uint8) * 255), mode="L")
    except Exception as exc:
        print(f"WARNING: SAM point refinement failed, using GrabCut fallback: {exc}")

    return _grabcut_refine(image, x, y, component, (bx0, by0, bx1, by1))



def get_smarteraser_inpainter():
    """SmartEraser is unavailable through the generic diffusers API in this build.

    The released checkpoint requires SmartEraser's custom
    StableDiffusionInpaintRegionPipeline from the official repository.
    We explicitly skip it rather than loading weights and failing later.
    """
    global smarteraser_failed
    smarteraser_failed = True
    print("SmartEraser skipped: official custom pipeline is not installed; using RORem.")
    return None


def get_rorem_inpainter():
    """Load LetsThink/RORem without requesting a nonexistent weight variant."""
    global rorem_pipe, rorem_failed
    if rorem_pipe is not None:
        return rorem_pipe
    if rorem_failed:
        return None
    try:
        import torch as _torch
        from diffusers import AutoPipelineForInpainting
        if not _torch.cuda.is_available():
            print("RORem skipped: CUDA is required for client-demo quality.")
            return None

        print("Loading RORem object-removal model (LetsThink/RORem) with native checkpoint weights...")
        last_error = None
        pipe = None

        # The current LetsThink/RORem model card exposes F16 safetensors, but
        # not a separately named `fp16` variant. Requesting variant='fp16'
        # makes Diffusers fail before inference starts.
        for dtype_name, dtype in (("float16", _torch.float16), ("bfloat16", _torch.bfloat16)):
            try:
                print(f"Trying RORem dtype={dtype_name} (no variant override)...")
                pipe = AutoPipelineForInpainting.from_pretrained(
                    "LetsThink/RORem",
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
                print(f"RORem checkpoint loaded with dtype={dtype_name}.")
                break
            except Exception as exc:
                last_error = exc
                print(f"RORem dtype={dtype_name} failed: {exc}")
                pipe = None

        if pipe is None:
            raise RuntimeError(f"RORem checkpoint could not be loaded: {last_error}")

        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        rorem_pipe = pipe
        print("RORem object-removal model READY (native checkpoint weights).")
        return rorem_pipe
    except Exception as exc:
        rorem_failed = True
        print(f"WARNING: RORem unavailable: {exc}")
        return None

def _resize_by_short_side(image: Image.Image, mask: Image.Image, short_side: int = 512):
    """Resize image and mask like the official RORem inference code."""
    w, h = image.size
    scale = float(short_side) / max(1, min(w, h))
    nw = max(64, int(round(w * scale)))
    nh = max(64, int(round(h * scale)))
    nw = max(64, (nw // 8) * 8)
    nh = max(64, (nh // 8) * 8)
    return (image.resize((nw, nh), Image.Resampling.BICUBIC),
            mask.resize((nw, nh), Image.Resampling.NEAREST))


def _dilate_binary_mask(mask: Image.Image, pixels: int = 8) -> Image.Image:
    """Small halo for object/contact-shadow edges."""
    if cv2 is None or pixels <= 0:
        return mask.convert("L")
    arr = np.asarray(mask.convert("L"), np.uint8)
    k = max(3, int(pixels) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.dilate((arr > 127).astype(np.uint8), kernel, 1) * 255
    return Image.fromarray(out.astype(np.uint8), "L")


def _square_object_crop(image: Image.Image, object_mask: np.ndarray, context_ratio: float = 0.24):
    """Build an object-centred square crop for RORem's native 512x512 regime.

    The previous implementation kept the whole room width for a large bed.
    That made the bed occupy only a small part of the 512px latent and forced
    the base RORem checkpoint to solve a nearly 1024x512 scene. The official
    RORem release states that the base checkpoint is optimal at 512x512.
    A square crop keeps the selected object large while retaining enough wall,
    rug and floor context to reconstruct the room instead of another object.
    """
    H, W = object_mask.shape
    ys, xs = np.where(object_mask)
    if xs.size == 0:
        return image.convert("RGB"), object_mask.copy(), (0, 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    base = max(bw, bh)
    side = int(round(base * (1.0 + 2.0 * context_ratio)))
    side = max(base + 64, side)
    side = min(side, W, H)
    side = max(256, side)

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    cx0 = int(round(cx - side / 2.0))
    cy0 = int(round(cy - side / 2.0))
    cx0 = max(0, min(cx0, W - side))
    cy0 = max(0, min(cy0, H - side))
    cx1, cy1 = cx0 + side, cy0 + side
    return image.crop((cx0, cy0, cx1, cy1)), object_mask[cy0:cy1, cx0:cx1], (cx0, cy0)


def _rorem_remove_candidate(pipe, image: Image.Image, object_mask: np.ndarray, seed: int):
    """Run RORem with a square object-centred 512x512 working image."""
    import torch as _torch
    crop, crop_mask, origin = _square_object_crop(image, object_mask, context_ratio=0.24)
    base_mask = Image.fromarray((crop_mask.astype(np.uint8) * 255), "L")

    # RORem needs a little more context around large furniture so that its
    # generated region does not preserve a bed/sofa silhouette. Keep the
    # user's SAM mask untouched outside the generation crop, and adapt the
    # internal dilation to object size: larger holes get a slightly wider
    # erase halo, while small furniture keeps the tighter 30px setting.
    mask_area_ratio = float(crop_mask.mean()) if crop_mask.size else 0.0
    if mask_area_ratio >= 0.20:
        dilation_px = 42
    elif mask_area_ratio >= 0.10:
        dilation_px = 38
    elif mask_area_ratio >= 0.04:
        dilation_px = 34
    else:
        dilation_px = 30
    work_mask = _dilate_binary_mask(base_mask, pixels=dilation_px)
    work = crop.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    work_mask = work_mask.resize((512, 512), Image.Resampling.NEAREST)

    # Keep the official quality wording, but add an explicit REMOVE-ONLY scene
    # instruction. This is deliberately phrased as reconstruction of the existing
    # room, not generation of a replacement object.
    prompt = (
        "4K, high quality, masterpiece, highly detailed, sharp focus, professional, "
        "photorealistic, realistic, seamless empty background, clean continuous wall, "
        "continuous floor and rug, preserve the room architecture, lighting and perspective, "
        "remove the selected object completely, no replacement object"
    )
    negative = (
        "low quality, worst, bad proportions, blurry, deformed, disfigured, unclear background, "
        "furniture, bed, sofa, couch, table, chair, cabinet, object, replacement object, ghost object"
    )
    generator = _torch.Generator(device="cpu").manual_seed(int(seed))
    with _torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            height=512,
            width=512,
            image=work,
            mask_image=work_mask,
            guidance_scale=1.0,
            num_inference_steps=50,
            strength=0.99,
            generator=generator,
        ).images[0].convert("RGB")

    restored = out.resize(crop.size, Image.Resampling.LANCZOS)
    placed = _place_crop(image, restored, origin)
    halo_full = np.zeros_like(object_mask, bool)
    hm = np.asarray(work_mask.resize(crop.size, Image.Resampling.NEAREST), np.uint8) > 127
    oy, ox = origin[1], origin[0]
    hh, ww = crop.size[1], crop.size[0]
    halo_full[oy:oy+hh, ox:ox+ww] = hm
    candidate = _composite_inside_mask(
        image, placed, Image.fromarray((halo_full.astype(np.uint8) * 255), "L")
    )
    return candidate, halo_full


def _target_change_score(original: Image.Image, candidate: Image.Image, mask: np.ndarray) -> float:
    """Measure how much the candidate actually changed the selected object.

    Returns 0..1. This deliberately evaluates ONLY the original confirmed
    object mask, not the dilated context halo used internally by RORem.
    A removal candidate that barely changes the bed must never pass merely
    because the surrounding seam looks good.
    """
    src = np.asarray(original.convert("RGB"), dtype=np.float32)
    out = np.asarray(candidate.convert("RGB").resize(original.size, Image.Resampling.LANCZOS), dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if m.shape != src.shape[:2]:
        m = np.asarray(Image.fromarray((m.astype(np.uint8) * 255), "L").resize(original.size, Image.Resampling.NEAREST), dtype=np.uint8) > 127
    if not m.any():
        return 0.0
    delta = np.abs(src - out).mean(axis=2) / 255.0
    mean_change = float(np.mean(delta[m]))
    changed_fraction = float(np.mean(delta[m] > (12.0 / 255.0)))
    # Mean color change catches large residual ghosts; changed-pixel fraction
    # catches the case where only a few details were altered.
    return float(np.clip(0.70 * mean_change + 0.30 * changed_fraction, 0.0, 1.0))


def _rorem_quality_candidate(original: Image.Image, candidate: Image.Image, mask: np.ndarray):
    """Return a score that heavily penalizes surviving furniture."""
    score, seam, fp = _candidate_quality(original, candidate, Image.fromarray((mask.astype(np.uint8) * 255), "L"))
    # RORem is a removal model; for this product, visible furniture inside the
    # selected hole is much worse than a small boundary-color mismatch.
    total = float(seam + 420.0 * fp)
    return total, seam, fp


def get_powerpaint_inpainter():
    """Deprecated in Phase 5.2.16: replacement/generative backends are not used by Remove AI."""
    return None


def _powerpaint_remove(*args, **kwargs):
    raise RuntimeError("PowerPaint is disabled for the remove-only workflow")

def get_diffusion_inpainter():
    """Load the actual diffusion inpainting backend, or return None with a clear diagnostic."""
    global diffusion_pipe, diffusion_failed
    if diffusion_pipe is not None:
        return diffusion_pipe
    if diffusion_failed:
        return None
    try:
        import torch as _torch
        from diffusers import StableDiffusionInpaintPipeline
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        dtype = _torch.float16 if device == "cuda" else _torch.float32
        print(f"Loading REQUIRED generative inpainting model (device={device})...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        else:
            pipe.to(device)
        diffusion_pipe = pipe
        print("Generative inpainting model READY.")
        return diffusion_pipe
    except Exception as exc:
        diffusion_failed = True
        print(f"ERROR: generative inpainting unavailable: {exc}")
        print("Install/update requirements with: pip install -r server\\requirements.txt")
        return None


def _seam_score(original_crop: Image.Image, candidate: Image.Image, mask_crop: Image.Image) -> float:
    """Robust boundary continuity score. Lower is better.

    This deliberately never uses a 1e6 sentinel for a normal thin/irregular
    mask. It samples visible pixels just outside the hole and compares them to
    generated pixels immediately inside the hole. A very large value is used
    only for an actual shape/size error.
    """
    if cv2 is None:
        return 0.0
    a = np.asarray(mask_crop.convert("L"), np.uint8) > 127
    src = np.asarray(original_crop.convert("RGB"), np.uint8)
    gen = np.asarray(candidate.convert("RGB"), np.uint8)
    if src.shape != gen.shape or src.shape[:2] != a.shape:
        return 1e6
    if not a.any():
        return 1e6

    # One-pixel contour inside and a short visible band outside.
    er = cv2.erode(a.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1).astype(bool)
    inner = a & ~er
    dil = cv2.dilate(a.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), 1).astype(bool)
    outer = dil & ~a
    if not inner.any() or not outer.any():
        # Boundary-touching masks can have no exterior context in the crop.
        # Use a valid low-weight interior texture statistic instead of 1e6.
        inner = a
        if not outer.any():
            return float(np.std(cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY)[inner])) * 0.15

    src_g = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gen_g = cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY).astype(np.float32)
    src_blur = cv2.GaussianBlur(src_g,(0,0),2.0)
    gen_blur = cv2.GaussianBlur(gen_g,(0,0),2.0)

    # Compare robust medians and local gradients.
    lum_gap = abs(float(np.median(gen_g[inner])) - float(np.median(src_g[outer])))
    tex_gap = abs(float(np.std(gen_blur[inner])) - float(np.std(src_blur[outer])))

    sx=cv2.Sobel(src_g,cv2.CV_32F,1,0,ksize=3); sy=cv2.Sobel(src_g,cv2.CV_32F,0,1,ksize=3)
    gx=cv2.Sobel(gen_g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gen_g,cv2.CV_32F,0,1,ksize=3)
    sg=cv2.magnitude(sx,sy); gg=cv2.magnitude(gx,gy)
    grad_gap=abs(float(np.median(gg[inner]))-float(np.median(sg[outer])))

    # Excess edge energy is a strong signal for hallucinated furniture.
    deep = er & a
    if not deep.any(): deep=a
    edge_excess=max(0.0,float(np.mean(gg[deep]))-max(1.8*float(np.mean(sg[outer])),float(np.mean(sg[outer]))+5.0))
    return float(lum_gap + 0.45*tex_gap + 0.30*grad_gap + 0.70*edge_excess)

def _square_working_crop(crop: Image.Image, mask: np.ndarray, target: int = 512):
    """Return square 512x512 image/mask plus geometry needed to undo padding."""
    cw, ch = crop.size
    side = max(cw, ch)
    pad_left = (side - cw) // 2
    pad_right = side - cw - pad_left
    pad_top = (side - ch) // 2
    pad_bottom = side - ch - pad_top
    crop_np = np.asarray(crop.convert("RGB"))
    if cv2 is not None:
        square_np = cv2.copyMakeBorder(
            crop_np, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )
    else:
        square = Image.new("RGB", (side, side))
        square.paste(crop, (pad_left, pad_top))
        square_np = np.asarray(square)
    square_mask = np.zeros((side, side), dtype=np.uint8)
    square_mask[pad_top:pad_top+ch, pad_left:pad_left+cw] = mask
    return (
        Image.fromarray(square_np).resize((target, target), Image.Resampling.LANCZOS),
        Image.fromarray(square_mask).resize((target, target), Image.Resampling.NEAREST),
        (side, pad_left, pad_top, cw, ch),
    )


def _restore_square(result_sq: Image.Image, geometry):
    side, pad_left, pad_top, cw, ch = geometry
    arr = np.asarray(result_sq.convert("RGB"))
    if cv2 is not None:
        arr = cv2.resize(arr, (side, side), interpolation=cv2.INTER_LANCZOS4)
    else:
        arr = np.asarray(Image.fromarray(arr).resize((side, side), Image.Resampling.LANCZOS))
    arr = arr[pad_top:pad_top+ch, pad_left:pad_left+cw]
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def _furniture_hallucination_score(candidate: Image.Image, target_mask: np.ndarray) -> float:
    """Estimate how much furniture the generated hole contains.

    SegFormer is already loaded for surface classification, so reuse it as a
    cheap semantic guardrail. The score is the fraction of the confirmed
    reconstruction mask that SegFormer labels as furniture. Lower is better.
    It is intentionally a penalty, not a hard classifier: a few boundary
    pixels can be mislabeled in a photograph.
    """
    try:
        _, sp = get_pipes()
        arr = np.asarray(target_mask, bool)
        if not arr.any():
            return 1.0
        results = sp(candidate.convert("RGB"))
        H, W = arr.shape
        furniture = np.zeros((H, W), bool)
        for r in results:
            label = str(r.get("label", "")).lower().strip()
            if label not in ADE_FURNITURE:
                continue
            # SegFormer can assign low-confidence furniture labels to large
            # texture regions (rug/wall/floor). For quality gating, only count
            # reasonably confident semantic detections.
            confidence = float(r.get("score", 1.0) or 1.0)
            if confidence < 0.65:
                continue
            rm = r.get("mask")
            if rm is None:
                continue
            m = mask_array(rm, (W, H))
            furniture |= np.asarray(m, bool)
        return float((furniture & arr).sum() / max(1, arr.sum()))
    except Exception as exc:
        # Never make reconstruction fail because the optional semantic
        # validation failed. A neutral score simply disables this penalty.
        print(f"WARNING: furniture hallucination check unavailable: {exc}")
        return 0.0


def _diffusion_surface_pass(pipe, crop: Image.Image, target_mask: np.ndarray,
                            prompt: str, negative: str, seed: int):
    """One tightly constrained diffusion pass for one room surface."""
    import torch as _torch
    work, work_mask, geometry = _square_working_crop(crop, target_mask, 512)
    generator = _torch.Generator(device="cpu").manual_seed(seed)
    with _torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=work,
            mask_image=work_mask,
            num_inference_steps=32,
            guidance_scale=5.5,
            strength=1.0,
            generator=generator,
        ).images[0].convert("RGB")
    if out.size != work.size:
        out = out.resize(work.size, Image.Resampling.LANCZOS)
    seam = _seam_score(work, out, work_mask)
    hallucination = _furniture_hallucination_score(out, np.asarray(work_mask.convert("L")) > 127)
    # A visible piece of furniture inside a surface reconstruction is much
    # worse than a small color/texture mismatch. Make it dominate candidate
    # selection while still allowing minor segmentation noise.
    score = float(seam + 180.0 * hallucination)
    return _restore_square(out, geometry), score, seam, hallucination


def _compose_candidate(original: Image.Image, generated: Image.Image, mask_img: Image.Image) -> Image.Image:
    """Composite a generated full-image candidate strictly inside the mask."""
    src=np.asarray(original.convert("RGB"),np.float32)
    gen=np.asarray(generated.convert("RGB").resize(original.size,Image.Resampling.LANCZOS),np.float32)
    m=np.asarray(mask_img.convert("L").resize(original.size,Image.Resampling.NEAREST),np.uint8)>127
    alpha=m.astype(np.float32)
    if cv2 is not None:
        # Only a very small feather; never leak into neighbouring furniture.
        alpha=cv2.GaussianBlur(alpha,(0,0),0.65)
        alpha*=m
    out=src*(1-alpha[...,None])+gen*alpha[...,None]
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8))


def _candidate_quality(original: Image.Image, candidate: Image.Image, mask_img: Image.Image):
    """Return quality tuple used for real candidate selection."""
    m=np.asarray(mask_img.convert("L"),np.uint8)>127
    ys,xs=np.where(m)
    if xs.size==0: return 1e6,1e6,1.0
    x0,x1=max(0,int(xs.min())-32),min(original.width,int(xs.max())+33)
    y0,y1=max(0,int(ys.min())-32),min(original.height,int(ys.max())+33)
    oc=original.crop((x0,y0,x1,y1)); cc=candidate.crop((x0,y0,x1,y1)); mc=mask_img.crop((x0,y0,x1,y1))
    seam=_seam_score(oc,cc,mc)
    # SegFormer needs actual visible wall/floor to tell a large reconstructed
    # region apart from real furniture. The tight seam crop above is mostly
    # hole with only a ~32px visible margin — for a large object that is an
    # out-of-distribution input for a scene segmentation model and measurably
    # biases it toward guessing "furniture" even on a clean fill (observed in
    # testing: two different backends and three different seeds all scored
    # furniture_penalty in the same 0.77-0.80 band on one large-hole removal,
    # which independent hallucinations would not do). Classify from a much
    # wider context crop instead; only pixels inside the confirmed mask are
    # ever counted, so this cannot start counting a real neighbouring object.
    wide_candidate, wide_mask_arr, _ = _local_object_crop(candidate, m, pad_ratio=1.4)
    furniture=_furniture_hallucination_score(wide_candidate, wide_mask_arr)
    return float(seam+260.0*furniture), float(seam), float(furniture)


def _local_object_crop(image: Image.Image, object_mask: np.ndarray, pad_ratio: float = 0.85):
    """Return a generous context crop around the object, preserving the full hole."""
    H, W = object_mask.shape
    ys, xs = np.where(object_mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    px = max(64, int(bw * pad_ratio))
    py = max(64, int(bh * pad_ratio))
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(W, x1 + px), min(H, y1 + py)
    return image.crop((cx0, cy0, cx1, cy1)), object_mask[cy0:cy1, cx0:cx1], (cx0, cy0)


def _place_crop(base: Image.Image, crop: Image.Image, origin):
    out = base.copy()
    out.paste(crop.convert("RGB"), origin)
    return out


def _semantic_furniture_shield(image: Image.Image, target_mask: np.ndarray) -> np.ndarray:
    """Find visible furniture that should NOT be used as donor context.

    Remove AI must reconstruct the hidden room, not copy neighboring furniture
    into the hole. We therefore make a temporary *context shield* around
    furniture outside the selected object. The shield is used only as input to
    LaMa; the original pixels are restored byte-for-byte in the final image.
    """
    if cv2 is None:
        return np.zeros_like(target_mask, bool)
    try:
        _, sp = get_pipes()
        H, W = target_mask.shape
        shield = np.zeros((H, W), bool)
        for r in sp(image.convert("RGB")):
            label = str(r.get("label", "")).lower().strip()
            if label not in ADE_FURNITURE:
                continue
            score = float(r.get("score", 1.0) or 1.0)
            if score < 0.50:
                continue
            rm = r.get("mask")
            if rm is None:
                continue
            m = mask_array(rm, (W, H))
            # Only shield visible furniture outside the selected object.
            m = np.asarray(m, bool) & ~target_mask
            if int(m.sum()) >= 180:
                shield |= m
        # Give the shield a small safety halo so LaMa does not pull furniture
        # edges into the reconstruction. Never change the user's actual mask.
        if shield.any():
            shield = cv2.dilate(
                shield.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            ) > 0
            shield &= ~target_mask
        return shield
    except Exception as exc:
        print(f"WARNING: furniture context shield unavailable: {exc}")
        return np.zeros_like(target_mask, bool)


def _build_remove_only_context(image: Image.Image, target_mask: np.ndarray) -> Image.Image:
    """Create a temporary furniture-neutral context for object removal.

    This is NOT the output image. It exists only to stop LaMa from seeing a
    neighboring sofa/table/bed and hallucinating that furniture into the hole.
    The final composite always comes from the original image outside target.
    """
    shield = _semantic_furniture_shield(image, target_mask)
    if not shield.any():
        return image.convert("RGB")
    context = _multiscale_cv_inpaint(image.convert("RGB"), shield, radius=3.0)
    print(f"Remove-only context shield: {int(shield.sum())}px of neighboring furniture hidden from donor context")
    return context


def _surface_donor_context(image: Image.Image, object_mask: np.ndarray,
                           surface_map: np.ndarray, surface_id: int) -> Image.Image:
    """Make a donor image dominated by the surface being reconstructed.

    The selected object is already absent from ``image`` when this helper is
    called.  We also neutralize clearly different surfaces inside the local
    object neighborhood.  This is a donor-context trick only: the final image
    is always composited over the untouched original outside the target mask.
    """
    if cv2 is None:
        return image.convert("RGB")
    src = np.asarray(image.convert("RGB"), np.uint8)
    shield = _semantic_furniture_shield(image, object_mask)
    keep = (surface_map == surface_id) & ~object_mask & ~shield
    # Start from a heavily blurred copy so non-target surfaces cannot provide
    # sharp furniture/edge donors.  Same-surface pixels remain untouched.
    blurred = cv2.GaussianBlur(src, (0, 0), 11.0)
    out = src.copy()
    out[~keep] = blurred[~keep]
    return Image.fromarray(out, "RGB")

def _planar_wall_fill(image: Image.Image, target: np.ndarray) -> Image.Image:
    """Deterministic wall reconstruction using visible same-row wall pixels.

    Walls in room photos are usually locally planar and slowly varying in
    color.  For this surface we should not ask an inpainting network to invent
    texture.  Interpolating between real wall pixels preserves the actual
    paint tone and avoids the soft brown/grey blob produced by large-hole
    LaMa/OpenCV passes.
    """
    if cv2 is None or not target.any():
        return image.convert("RGB")
    src=np.asarray(image.convert("RGB"),np.float32)
    out=src.copy(); H,W=target.shape
    wall_target=target.copy()
    # Lightly close tiny gaps so each row has a stable boundary.
    for y in range(H):
        xs=np.where(wall_target[y])[0]
        if xs.size<2: continue
        x0,x1=int(xs.min()),int(xs.max())
        # Real wall samples immediately outside the hole, plus a small robust
        # horizontal window to reduce lamp/object contamination.
        left=max(0,x0-18); right=min(W-1,x1+18)
        L=np.where(~wall_target[y,left:x0])[0]
        R=np.where(~wall_target[y,x1+1:right+1])[0]
        if L.size and R.size:
            lx=left+int(L[-1]); rx=x1+1+int(R[0])
            if rx>lx:
                lc=np.median(src[max(0,y-2):min(H,y+3),max(0,lx-3):min(W,lx+4)],axis=(0,1))
                rc=np.median(src[max(0,y-2):min(H,y+3),max(0,rx-3):min(W,rx+4)],axis=(0,1))
                span=rx-lx
                for x in range(x0,x1+1):
                    if wall_target[y,x]:
                        t=(x-lx)/span
                        # Preserve a gentle illumination gradient instead of
                        # producing a flat single-color rectangle.
                        out[y,x]=lc*(1-t)+rc*t
        elif L.size:
            lx=left+int(L[-1]); out[y,x0:x1+1]=src[y,lx]
        elif R.size:
            rx=x1+1+int(R[0]); out[y,x0:x1+1]=src[y,rx]
    # Small-radius NS pass only at the contour to blend one-pixel seams.
    contour=cv2.dilate(wall_target.astype(np.uint8),np.ones((3,3),np.uint8),1).astype(bool) & ~wall_target
    if contour.any():
        repaired=cv2.inpaint(np.clip(out,0,255).astype(np.uint8),(contour.astype(np.uint8)*255),2.0,cv2.INPAINT_NS)
        # Do not alter the interior interpolation with this pass.
        out[contour]=repaired[contour]
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8),'RGB')

def _smart_quality(original: Image.Image, candidate: Image.Image, mask: np.ndarray):
    full=Image.fromarray((mask.astype(np.uint8)*255),'L')
    score,seam,fp=_candidate_quality(original,candidate,full)
    change=_target_change_score(original,candidate,mask)
    # Reward actual erasure while penalising furniture/ghosts. The change term
    # is capped so a hallucinated replacement cannot win merely by changing more pixels.
    total=float(seam + 260.0*fp + 160.0*max(0.0, 0.18-change))
    return total,seam,fp,change


def _residual_structure_score(candidate: Image.Image, object_mask: np.ndarray) -> float:
    """Estimate whether a furniture-like structural pattern survived inside the hole.

    SegFormer catches semantic furniture, but it can miss a generated sofa/bed
    when the hallucination has low confidence. This second signal is deliberately
    lightweight: compare edge density inside the confirmed removal mask with a
    nearby context ring. A reconstructed wall/floor/rug should not suddenly have
    a much higher concentration of strong edges than its surroundings.

    Returns 0..1, where lower is better. It is a ranking signal, not a standalone
    detector, so textured rugs and wood grain do not get rejected by themselves.
    """
    try:
        if cv2 is None:
            return 0.0
        m = np.asarray(object_mask, bool)
        if not m.any():
            return 0.0
        gray = np.asarray(candidate.convert("L"), np.uint8)
        if gray.shape != m.shape:
            gray = np.asarray(candidate.convert("L").resize((m.shape[1], m.shape[0]), Image.Resampling.BILINEAR), np.uint8)
        edges = cv2.Canny(gray, 60, 140) > 0
        inner = float(edges[m].mean())
        ring = cv2.dilate(m.astype(np.uint8), np.ones((21, 21), np.uint8), iterations=1).astype(bool) & ~m
        if not ring.any():
            return 0.0
        outer = float(edges[ring].mean())
        excess = max(0.0, inner - max(0.055, outer * 1.35))
        return float(np.clip(excess / 0.20, 0.0, 1.0))
    except Exception as exc:
        print(f"WARNING: residual structure check unavailable: {exc}")
        return 0.0


def _score_rorem_candidate(src: Image.Image, cand: Image.Image, object_mask: np.ndarray):
    full_mask = Image.fromarray((object_mask.astype(np.uint8) * 255), "L")
    total, seam, fp = _candidate_quality(src, cand, full_mask)
    change = _target_change_score(src, cand, object_mask)
    residual = _residual_structure_score(cand, object_mask)
    # Semantic furniture is still the strongest signal. The structure term is
    # intentionally smaller: it should break ties in favour of a cleaner hole,
    # not reject naturally textured floors/rugs.
    total = float(
        seam
        + 300.0 * fp
        + 120.0 * residual
        + 160.0 * max(0.0, 0.22 - change)
    )
    return total, seam, fp, change, residual


def _remove_only_large_object(image: Image.Image, object_mask: np.ndarray) -> dict | None:
    """Large-object REMOVE-ONLY pipeline with conservative candidate ensemble.

    Phase 5.2.27 keeps the successful RORem-only architecture but improves
    selection: five fixed seeds are evaluated, the best candidate receives up
    to two residual-removal passes, and a lightweight structural-residual score
    helps prefer a genuinely empty reconstruction over a plausible-looking
    sofa/bed ghost. No replacement or generic LaMa fallback is used.
    """
    if not object_mask.any():
        return None
    src = image.convert("RGB")
    pipe = get_rorem_inpainter()
    if pipe is None:
        print("REMOVE-ONLY large-object removal rejected: RORem unavailable")
        return None

    candidates = []
    for seed in (2026, 7319, 11037, 17011, 23027):
        try:
            cand, effective_mask = _rorem_remove_candidate(pipe, src, object_mask, seed)
            total, seam, fp, change, residual = _score_rorem_candidate(src, cand, object_mask)
            print(
                f"RORem pass1 seed={seed}: score={total:.2f}, seam={seam:.2f}, "
                f"furniture_penalty={fp:.4f}, residual_structure={residual:.3f}, "
                f"target_change={change:.3f}"
            )
            candidates.append((total, seam, fp, change, residual, cand, seed, effective_mask))
        except Exception as exc:
            print(f"RORem pass1 seed={seed} failed: {exc}")

    if not candidates:
        return None

    # Prefer semantic cleanliness first, then total score. This prevents a tiny
    # seam improvement from winning over a visibly furniture-shaped hallucination.
    candidates.sort(key=lambda x: (x[2], x[4], x[0], x[1], -x[3]))
    best = candidates[0]
    _, _, _, _, _, best_img, best_seed, _ = best

    # Two independent residual passes give the selected best image a chance to
    # erase a sofa/bed ghost that survived the first removal. We still score every
    # refinement against the ORIGINAL room, so quality cannot drift indefinitely.
    for pass_index, offset in enumerate((44021, 58117), start=2):
        try:
            second_seed = offset + int(best_seed)
            refined, effective_mask2 = _rorem_remove_candidate(pipe, best_img, object_mask, second_seed)
            total2, seam2, fp2, change2, residual2 = _score_rorem_candidate(src, refined, object_mask)
            print(
                f"RORem pass{pass_index} seed={second_seed}: score={total2:.2f}, seam={seam2:.2f}, "
                f"furniture_penalty={fp2:.4f}, residual_structure={residual2:.3f}, "
                f"target_change={change2:.3f}"
            )
            # Accept a refinement only when semantic/structural cleanliness improves
            # or the overall score is meaningfully better. This protects already-good
            # removals from unnecessary texture drift.
            if (fp2 < best[2] - 0.008) or (residual2 < best[4] - 0.04) or (total2 < best[0] - 4.0):
                best = (total2, seam2, fp2, change2, residual2, refined, second_seed, effective_mask2)
                best_img = refined
                best_seed = second_seed
                print(f"RORem refinement selected: pass {pass_index}")
            else:
                print(f"RORem refinement rejected: pass {pass_index} did not improve enough")
        except Exception as exc:
            print(f"RORem pass{pass_index} failed: {exc}; keeping current best")

    total, seam, fp, change, residual, cand, seed, effective_mask = best
    # The structural score is intentionally not a hard gate by itself. The
    # existing semantic + seam gates remain the safety barrier against returning
    # an obvious bad reconstruction.
    if change >= 0.28 and fp <= 0.40 and seam <= 50.0:
        print(
            f"Selected REMOVE-ONLY backend=rorem-official-512-square-ensemble seed={seed}, "
            f"score={total:.2f}, seam={seam:.2f}, furniture_penalty={fp:.4f}, "
            f"residual_structure={residual:.3f}, target_change={change:.3f}, passed_gate=True"
        )
        return {
            "image": cand,
            "passed_gate": True,
            "backend": "rorem-official-512-square-ensemble",
            "seam": seam,
            "furniture_penalty": fp,
        }

    # Phase 5.2.28: reliability recovery band. A candidate that is otherwise
    # very clean can miss the strict seam gate by only a few points because the
    # selected object touches a high-contrast edge (rug border, wood edge, etc.).
    # Do not lower the normal gate globally. Instead allow a narrow, high-
    # confidence recovery band so the UI does not report a 503 for a candidate
    # that is already semantically clean and has genuinely erased the object.
    # This is intentionally much stricter on furniture/residual scores than the
    # normal gate.
    if (
        change >= 0.32
        and fp <= 0.20
        and residual <= 0.45
        and seam <= 58.0
    ):
        print(
            f"Selected REMOVE-ONLY recovery-band seed={seed}, score={total:.2f}, "
            f"seam={seam:.2f}, furniture_penalty={fp:.4f}, "
            f"residual_structure={residual:.3f}, target_change={change:.3f}, passed_gate=True"
        )
        return {
            "image": cand,
            "passed_gate": True,
            "backend": "rorem-official-512-square-recovery",
            "seam": seam,
            "furniture_penalty": fp,
        }

    print(
        f"REMOVE-ONLY large-object removal rejected: best score={total:.2f}, "
        f"target_change={change:.3f}, furniture_penalty={fp:.4f}, "
        f"residual_structure={residual:.3f}, seam={seam:.2f}"
    )
    return None

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
    """Select one physical object: semantic class for label + SAM silhouette."""
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
        y0, y1 = max(0, ay - 10), min(ah, ay + 11)
        x0, x1 = max(0, ax - 10), min(aw, ax + 11)
        if arr_small[y0:y1, x0:x1].any():
            candidates.append((area, label, arr_small))

    if not candidates:
        raise HTTPException(status_code=404, detail="Aucun meuble détecté à cet endroit — cliquez au centre du meuble.")

    _, selected_label, selected_small = min(candidates, key=lambda item: item[0])
    seed = np.asarray(
        Image.fromarray((selected_small * 255).astype(np.uint8)).resize((W, H), Image.Resampling.NEAREST)
    ) > 128
    # Restrict the semantic prior to the physical component containing the click.
    seed = _connected_component_at_point(seed, x, y)

    # SAM is responsible for the actual object silhouette; SegFormer only
    # provides class/context and a local prompt region.
    sam_mask = refine_mask_with_sam(image, x, y, seed)
    if sam_mask is not None:
        final_mask = sam_mask
    else:
        if cv2 is not None:
            u8 = cv2.morphologyEx((seed * 255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
            final_mask = Image.fromarray(u8).convert("L")
        else:
            final_mask = Image.fromarray((seed * 255).astype(np.uint8)).convert("L")
    return image, selected_label, final_mask


def prepare_inpaint_mask(mask_img: Image.Image, image_size: Tuple[int, int]) -> Image.Image:
    """Create a minimal context mask; never inflate a selection into a scene region."""
    if mask_img.size != image_size:
        # NEAREST, not BILINEAR: this is a binary object silhouette, not a
        # photo. Smooth interpolation manufactures grey edge pixels that then
        # get re-thresholded, rounding off real detail (e.g. bed legs) instead
        # of just anti-aliasing — and the same source mask already goes
        # through this resize twice (once on the frontend round-trip, once
        # here), so the effect compounds.
        mask_img=mask_img.resize(image_size,Image.Resampling.NEAREST)
    arr=np.asarray(mask_img,dtype=np.uint8)
    if cv2 is None:
        return Image.fromarray(arr).convert("L")
    binary=(arr>127).astype(np.uint8)*255
    scale=max(image_size)/1600.0
    radius=int(np.clip(round(1.6*scale),1,4))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel,iterations=1)
    binary=cv2.dilate(binary,kernel,iterations=1)
    return Image.fromarray(binary).convert("L")


def _surface_masks(image: Image.Image):
    """Get coarse wall/floor context only; never use it as the object mask."""
    try:
        _, sp=get_pipes()
        analysis=resize_for_analysis(image)
        aw,ah=analysis.size; W,H=image.size
        results=sp(analysis)
        surfaces={}
        wanted={"floor","wall","rug","carpet"}
        for r in results:
            label=str(r.get("label","")).lower().strip()
            if label not in wanted:
                continue
            a=mask_array(r["mask"],(aw,ah))
            up=np.asarray(Image.fromarray((a*255).astype(np.uint8)).resize((W,H),Image.Resampling.NEAREST))>128
            surfaces[label]=surfaces.get(label,False)|up
        return surfaces
    except Exception as exc:
        print(f"WARNING: surface context unavailable: {exc}")
        return {}


def _nearest_surface_map(mask: np.ndarray, surfaces: dict) -> np.ndarray:
    """Route the selected hole to the *actual* visible surface.

    Euclidean nearest-surface assignment is wrong for perspective rooms: a
    rug can be physically closer to a wall pixel than the wall itself.  We
    first estimate the wall/floor transition from the visible segmentation,
    then use the rug segmentation only below that transition.
    IDs: 1=floor, 2=wall, 3=rug/carpet.
    """
    H, W = mask.shape
    floor=np.asarray(surfaces.get("floor", np.zeros_like(mask)),bool)
    wall=np.asarray(surfaces.get("wall", np.zeros_like(mask)),bool)
    rug=np.asarray(surfaces.get("rug", np.zeros_like(mask)),bool) | np.asarray(surfaces.get("carpet", np.zeros_like(mask)),bool)
    labels=np.zeros((H,W),np.uint8)
    hole=mask.astype(bool)

    # Estimate the visible floor/wall boundary column-by-column.  Robust
    # quantiles stop isolated segmentation pixels from moving the boundary.
    boundary=np.full(W, int(H*0.58), dtype=np.float32)
    valid=[]
    for x in range(W):
        wy=np.where(wall[:,x])[0]
        fy=np.where(floor[:,x])[0]
        if wy.size and fy.size:
            valid.append((x, float(np.percentile(wy,90))))
        elif wy.size:
            valid.append((x, float(np.percentile(wy,90))))
    if valid:
        vx=np.array([v[0] for v in valid],np.float32); vy=np.array([v[1] for v in valid],np.float32)
        boundary=np.interp(np.arange(W,dtype=np.float32),vx,vy,left=float(vy[0]),right=float(vy[-1]))
        if cv2 is not None:
            boundary=cv2.GaussianBlur(boundary.reshape(1,-1),(0,0),max(3.0,W/180.0)).ravel()

    yy=np.indices((H,W))[0]
    # A small transition band is assigned using surface proximity.
    wall_zone=hole & (yy <= (boundary[None,:]-6))
    lower_zone=hole & (yy >= (boundary[None,:]+6))
    transition=hole & ~(wall_zone|lower_zone)
    labels[wall_zone]=2

    # Below the wall line: rug wins only when the pixel is genuinely near a
    # visible rug region. Otherwise it is floor. This prevents the rug from
    # being projected upward across the entire bed footprint.
    if cv2 is not None:
        def dist_to(m):
            if not m.any(): return np.full((H,W),1e6,np.float32)
            return cv2.distanceTransform((~m).astype(np.uint8),cv2.DIST_L2,5)
        dr=dist_to(rug); df=dist_to(floor); dw=dist_to(wall)
        rug_pick=lower_zone & rug.any() & (dr <= np.minimum(df*1.35, 170.0))
        labels[lower_zone]=1
        labels[rug_pick]=3
        # Transition pixels use the nearest plausible surface, but wall is
        # strongly preferred above the estimated boundary.
        dstack=np.stack([dw,df,dr],axis=0)
        idx=np.argmin(dstack,axis=0)
        transition_labels=np.where(idx==0,2,np.where(idx==2,3,1)).astype(np.uint8)
        labels[transition]=transition_labels[transition]
    else:
        labels[lower_zone]=1; labels[transition]=np.where(yy[transition]<H*0.58,2,1)

    labels[hole & (labels==0)] = np.where(yy[hole & (labels==0)] < H*0.58,2,1)
    return labels

def _add_contact_shadow_mask(image: Image.Image, object_mask: np.ndarray, surface_map: np.ndarray) -> np.ndarray:
    """Add only the likely contact shadow around the lower object edge.

    This deliberately avoids expanding the whole object mask. A narrow floor
    shadow band is detected from local luminance and is reconstructed together
    with the object so dark 'ghost furniture' does not remain.
    """
    if cv2 is None or not object_mask.any():
        return object_mask.copy()
    H, W = object_mask.shape
    ys, xs = np.where(object_mask)
    if xs.size < 20:
        return object_mask.copy()
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    band_h = int(np.clip((y1 - y0 + 1) * 0.10, 8, 36))
    dil = cv2.dilate(object_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), 1).astype(bool)
    ring = dil & (~object_mask) & np.isin(surface_map, (1, 3))
    # Only the lower part of the object's footprint can be contact shadow.
    lower = np.zeros_like(ring)
    lower[max(0, y1 - band_h):min(H, y1 + band_h + 1), max(0, x0 - 16):min(W, x1 + 17)] = True
    ring &= lower
    if not ring.any():
        return object_mask.copy()

    rgb = np.asarray(image.convert("RGB"), np.float32)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Compare each candidate to a local horizontal neighborhood. Shadows are
    # usually substantially darker than the surrounding carpet/floor.
    blur = cv2.GaussianBlur(gray, (0, 0), 7)
    darkness = blur - gray
    shadow = ring & (darkness > 9.0)
    # Keep this conservative: at most a small fraction of the original object
    # area is allowed to become shadow mask.
    max_extra = max(80, int(object_mask.sum() * 0.10))
    if int(shadow.sum()) > max_extra:
        vals = darkness[shadow]
        threshold = float(np.quantile(vals, 1.0 - max_extra / max(1, len(vals))))
        shadow = shadow & (darkness >= threshold)
    return object_mask | shadow


def _multiscale_cv_inpaint(image: Image.Image, mask: np.ndarray, radius: float = 3.0) -> Image.Image:
    """Stable multi-scale classical inpainting.

    Large furniture holes are difficult for single-pass Telea/NS because the
    algorithm has to propagate pixels across a very large missing region. We
    first solve the low-frequency structure at reduced resolution, then refine
    at the original resolution. This avoids the long vertical smear produced
    by direct row-copy texture synthesis.
    """
    src = np.asarray(image.convert("RGB"), np.uint8)
    if cv2 is None or not mask.any():
        return image
    H, W = src.shape[:2]
    # Work at up to 768px on the long side for the coarse structural pass.
    scale = min(1.0, 768.0 / max(H, W))
    cw, ch = max(32, int(round(W * scale))), max(32, int(round(H * scale)))
    small = cv2.resize(src, (cw, ch), interpolation=cv2.INTER_AREA)
    smask = cv2.resize((mask.astype(np.uint8) * 255), (cw, ch), interpolation=cv2.INTER_NEAREST)
    # A modest radius at coarse scale gives broad background continuity.
    coarse = cv2.inpaint(small, smask, max(2.0, min(6.0, radius * 0.75)), cv2.INPAINT_NS)
    coarse_up = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_CUBIC)

    # Use the coarse result only inside the hole, then make a local full-res
    # inpaint pass to restore edges and small texture transitions.
    work = src.copy()
    work[mask] = coarse_up[mask]
    fine = cv2.inpaint(work, (mask.astype(np.uint8) * 255), max(2.0, min(7.0, radius)), cv2.INPAINT_TELEA)

    # Prefer coarse reconstruction in the middle of a large hole and fine
    # reconstruction near the contour.
    dist = cv2.distanceTransform((mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)
    alpha = np.clip(dist / 28.0, 0.0, 1.0)[..., None]
    result = src.astype(np.float32)
    mixed = fine.astype(np.float32) * (1.0 - alpha) + coarse_up.astype(np.float32) * alpha
    result[mask] = mixed[mask]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _texture_residual_transfer(base: Image.Image, original: Image.Image,
                               target: np.ndarray, surface_map: np.ndarray) -> Image.Image:
    """Transfer only high-frequency texture from visible same-surface pixels.

    We deliberately do NOT copy RGB pixels directly. The low-frequency image
    comes from geometric/multiscale inpainting; only fine carpet/wood grain is
    borrowed from nearby visible pixels. A varying 2-D donor offset prevents
    the vertical bands seen in the previous implementation.
    """
    if cv2 is None or not target.any():
        return base
    src = np.asarray(original.convert("RGB"), np.float32)
    out = np.asarray(base.convert("RGB"), np.float32).copy()
    H, W = target.shape
    visible = (~target) & (surface_map == 1)
    if int(visible.sum()) < 500:
        return base

    # Prefer a broad visible floor/carpet donor below the object, with fallback
    # to the full visible surface. We use residual texture, not absolute color.
    ys, xs = np.where(target)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    candidates = np.where(visible & (np.indices((H, W))[0] >= max(y0, int(H * 0.42))))
    if len(candidates[0]) < 500:
        candidates = np.where(visible)
    if len(candidates[0]) < 500:
        return base

    cy = int(np.median(candidates[0])); cx = int(np.median(candidates[1]))
    # A donor texture window should remain local to the room surface.
    half_h = int(np.clip((y1 - y0 + 1) * 0.55, 24, 180))
    half_w = int(np.clip((x1 - x0 + 1) * 0.55, 32, 240))
    sy0, sy1 = max(0, cy - half_h), min(H, cy + half_h + 1)
    sx0, sx1 = max(0, cx - half_w), min(W, cx + half_w + 1)
    donor = src[sy0:sy1, sx0:sx1]
    if donor.shape[0] < 8 or donor.shape[1] < 8:
        return base

    # High-pass residual. Keep it weak; this is texture, not geometry/color.
    donor_blur = cv2.GaussianBlur(donor, (0, 0), 2.2)
    residual = donor - donor_blur
    # Estimate residual amplitude from visible donor pixels.
    residual = np.clip(residual, -22.0, 22.0)

    yy, xx = np.indices((H, W), dtype=np.float32)
    # Smoothly varying 2-D offsets; no fixed x/y row mapping.
    ox = (np.sin(yy * 0.021 + xx * 0.004) * 0.22 +
          np.sin(yy * 0.007 - xx * 0.013) * 0.13) * max(8, donor.shape[1] * 0.25)
    oy = (np.sin(xx * 0.017 - yy * 0.005) * 0.18 +
          np.sin(xx * 0.006 + yy * 0.011) * 0.10) * max(6, donor.shape[0] * 0.20)
    map_x = np.mod(xx - x0 + ox, donor.shape[1] - 1).astype(np.float32)
    map_y = np.mod(yy - y0 + oy, donor.shape[0] - 1).astype(np.float32)
    tex = np.empty_like(src)
    for c in range(3):
        tex[..., c] = cv2.remap(residual[..., c], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Normalize texture amplitude to the local visible surface statistics.
    local_std = float(np.std(residual))
    if not np.isfinite(local_std) or local_std < 0.5:
        return base
    gain = float(np.clip(0.55 / max(0.55, local_std), 0.35, 0.85))
    texture = tex * gain

    # Feather texture so the boundary remains controlled.
    feather = cv2.GaussianBlur(target.astype(np.float32), (0, 0), 2.0)
    a = np.clip(feather * 0.72, 0.0, 0.72)[..., None]
    out = out + texture * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _surface_guided_reconstruct(image: Image.Image, hard_mask: np.ndarray, surface_map: np.ndarray) -> Image.Image:
    """Reconstruct a large furniture hole as photographed room surfaces.

    This is intentionally non-generative: it cannot invent a new bed/table.
    Wall and floor/rug regions are reconstructed independently, using a
    coarse inpaint pass followed by a narrow full-resolution pass. For large
    holes this is a safer first choice than Stable Diffusion, whose latent
    prior can hallucinate replacement furniture even with a strong negative
    prompt.
    """
    if cv2 is None or not hard_mask.any():
        return image.convert("RGB")
    src=image.convert("RGB")
    result=src
    # Do the wall first, then floor/rug. Each pass sees the original visible
    # context and only receives pixels from the corresponding surface class.
    for sid in (2,3,1):
        target=hard_mask & (surface_map==sid)
        if not target.any():
            continue
        # Large smooth walls benefit from a broader coarse pass; floor/rug
        # needs a little more local texture retention.
        radius=6.0 if sid==2 else 5.0
        result=_multiscale_cv_inpaint(result,target,radius=radius)
        if sid in (1,3):
            result=_texture_residual_transfer(result,src,target,surface_map)
    unresolved=hard_mask & ~((surface_map==1)|(surface_map==2)|(surface_map==3))
    if unresolved.any():
        result=_multiscale_cv_inpaint(result,unresolved,radius=4.0)
    # Strictly restore the original outside the selected object.
    return _composite_inside_mask(src,result,Image.fromarray((hard_mask.astype(np.uint8)*255),"L"))

def _surface_texture_fill(image: Image.Image, hard_mask: np.ndarray, surface_map: np.ndarray) -> Image.Image:
    """Reconstruct room surfaces without direct RGB stretching.

    - Wall: multi-scale structural inpainting.
    - Floor/rug: multi-scale structure + weak high-frequency texture transfer.
    - All pixels outside the confirmed reconstruction region stay unchanged.
    """
    if cv2 is None or not hard_mask.any():
        return image
    src = image.convert("RGB")
    result = src

    # Reconstruct each surface independently. This prevents a floor texture
    # donor from bleeding into the wall or vice versa.
    for sid in (2, 1):
        target = hard_mask & (surface_map == sid)
        if not target.any():
            continue
        radius = 3.5 if sid == 2 else 4.5
        structural = _multiscale_cv_inpaint(result, target, radius=radius)
        if sid == 1:
            structural = _texture_residual_transfer(structural, src, target, surface_map)
        result = structural

    # If surface classification missed a small fraction of the mask, repair it
    # conservatively with the same multiscale method rather than leaving a hole.
    unresolved = hard_mask & ~((surface_map == 1) | (surface_map == 2))
    if unresolved.any():
        result = _multiscale_cv_inpaint(result, unresolved, radius=3.0)
    return result

def _composite_inside_mask(original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    """Guarantee pixels outside the confirmed mask are byte-for-byte preserved."""
    src=np.asarray(original.convert("RGB"))
    # LaMa/SimpleLama can return an image one pixel smaller/larger after its
    # internal padding/cropping. Always normalize generated + mask to the
    # original dimensions before broadcasting/blending.
    if generated.size != original.size:
        generated = generated.convert("RGB").resize(original.size, Image.Resampling.BICUBIC)
    gen=np.asarray(generated.convert("RGB"))
    mask = mask.convert("L")
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.Resampling.BILINEAR)
    m=np.asarray(mask,dtype=np.float32)/255.0
    # Only a 1px-ish feather is allowed at the confirmed contour.
    if cv2 is not None:
        m=cv2.GaussianBlur(m,(0,0),0.8)
    a=m[...,None]
    out=(src*(1-a)+gen*a).clip(0,255).astype(np.uint8)
    return Image.fromarray(out)


def hybrid_inpaint(image: Image.Image, mask_img: Image.Image, allow_below_gate: bool = False):
    """AI Remove 2.5: precise SAM mask + model-based reconstruction for large objects.

    Returns (composited_image, meta) where meta is
    {"passed_gate": bool, "backend": str|None, "seam": float|None, "furniture_penalty": float|None}.
    """
    hard_mask_img = prepare_inpaint_mask(mask_img, image.size)
    object_mask = np.asarray(hard_mask_img.convert("L")) > 127
    if not object_mask.any():
        raise HTTPException(status_code=422, detail="Masque vide")

    # Phase 5.2.12: log what was actually selected. Two separate small masks
    # (e.g. decor on each nightstand) versus one large mask (the bed itself)
    # look identical in the seam/furniture_penalty log lines above, but are
    # completely different bugs to chase. This makes it visible without
    # needing a screenshot each time.
    _ys, _xs = np.where(object_mask)
    print(
        f"Mask stats: {100*float(object_mask.mean()):.2f}% of frame "
        f"({int(object_mask.sum())}px), bbox=({int(_xs.min())},{int(_ys.min())})-"
        f"({int(_xs.max())},{int(_ys.max())}) in a {image.size[0]}x{image.size[1]} image"
    )
    if min(image.size) < 800:
        print(
            f"WARNING: source photo is only {image.size[0]}x{image.size[1]} "
            f"({image.size[0]*image.size[1]} total px). Large-object inpainting "
            f"on a hole this size (see area above) with this little real detail "
            f"to reconstruct from is inherently harder, independent of any "
            f"threshold or model choice. Consider testing with a higher-"
            f"resolution photo (1500px+ on the short side) before tuning "
            f"quality gates further."
        )

    area_ratio = float(object_mask.mean())
    # Large furniture is the difficult case. Do NOT silently fall back to the
    # broken deterministic texture-stretch path: either the generative model
    # runs, or the API tells the user exactly what is missing.
    if area_ratio >= 0.015:
        result = _remove_only_large_object(image, object_mask)
        if result is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Suppression de grande zone refusée : aucun moteur dédié "
                    "SmartEraser/RORem n'a produit une reconstruction de qualité acceptable. "
                    "Consultez le terminal pour le candidat et l'erreur exacte."
                ),
            )
        # Quality metrics are advisory only. A REMOVE-ONLY candidate is returned
        # to the editor so the user can see the real reconstruction and undo it
        # if necessary; never hide it behind a debug-only preview.
        final_mask = Image.fromarray((object_mask.astype(np.uint8)*255), "L")
        composited = _composite_inside_mask(image, result["image"], final_mask)
        meta = {"passed_gate": result["passed_gate"], "backend": result["backend"], "seam": result["seam"], "furniture_penalty": result["furniture_penalty"]}
        return composited, meta

    # Small objects can still use the stable LaMa path; this avoids paying the
    # diffusion cost for pillows, lamps, etc. There is no seam/furniture gate
    # here — small holes are what LaMa is reliably good at.
    try:
        lama_img = get_lama()(image, hard_mask_img)
        composited = _composite_inside_mask(image, lama_img, hard_mask_img)
        return composited, {"passed_gate": True, "backend": "lama-small-object", "seam": None, "furniture_penalty": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inpainting petit objet impossible: {exc}") from exc


@app.get("/inpaint-status")
async def inpaint_status():
    """Expose the REMOVE-ONLY engine status."""
    lama_available = False
    try:
        import simple_lama_inpainting  # noqa: F401
        lama_available = True
    except Exception:
        pass
    return JSONResponse({
        "phase": "5.2.28",
        "mode": "REMOVE-ONLY",
        "large_object_strategy": "RORem official 512-short-side dedicated object removal with ensemble, refinement, and narrow recovery band; no LaMa fallback is exposed for large-object Remove AI",
        "lama_package_available": lama_available,
        "generative_replacement_enabled": False,
        "surface_texture_fallback": "planar wall interpolation + local LaMa for rug/floor",
        "quality_gate": {
            "strict_max_acceptable_seam": 50.0,
            "recovery_max_acceptable_seam": 58.0,
            "recovery_max_furniture_penalty": 0.20,
            "recovery_min_target_change": 0.32,
            "recovery_max_residual_structure": 0.45,
            "max_furniture_penalty": LAMA_CLEAN_FURNITURE_PENALTY,
        },
    })


@app.post("/select-mask")
async def select_mask(file: UploadFile = File(...), x: int = Form(...), y: int = Form(...)):
    """Phase 4 smart-selection endpoint used by the frontend preview step."""
    image = read_image(file)
    _, label, mask = await _segment_furniture_at_point(image, x, y)
    arr=np.asarray(mask.convert("L"))>127
    ys,xs=np.where(arr)
    ratio=float(arr.mean())
    if xs.size < 80:
        raise HTTPException(status_code=422, detail="Masque trop petit — cliquez davantage au centre du meuble.")
    if ratio > 0.42:
        raise HTTPException(status_code=422, detail="Masque non fiable — le modèle a sélectionné une zone trop grande. Cliquez au centre du meuble.")
    bbox={"x":int(xs.min()),"y":int(ys.min()),"width":int(xs.max()-xs.min()+1),"height":int(ys.max()-ys.min()+1)}
    # No max_side cap here: unlike the /analyze debug overlays, this exact
    # mask is what the frontend redraws to a canvas and posts back for the
    # real /inpaint call (see eraser.js). Downscaling it here was throwing
    # away SAM's full-resolution boundary before reconstruction ever saw it,
    # for a single-channel silhouette that compresses to a tiny PNG anyway.
    return JSONResponse({"mask": to_data_url(mask), "label": label, "area_ratio": ratio, "bbox": bbox, "method": "SAM + semantic context"})


@app.post("/inpaint")
async def inpaint(file: UploadFile = File(...), mask: UploadFile = File(...), debug: bool = Form(False)):
    """Remove a user-confirmed furniture mask with LaMa."""
    image = read_image(file)
    try:
        raw = await mask.read()
        mask_img = Image.open(io.BytesIO(raw)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Masque invalide: {exc}") from exc
    result, meta = hybrid_inpaint(image, mask_img, allow_below_gate=debug)
    if not isinstance(result, Image.Image):
        result = Image.fromarray(np.asarray(result).astype(np.uint8))
    return JSONResponse({
        "image": to_data_url(result),
        "width": result.width,
        "height": result.height,
        "label": "meuble",
        "method": "ai-remove-only",
        "quality_gate_passed": meta["passed_gate"],
        "backend": meta["backend"],
        "seam": meta["seam"],
        "furniture_penalty": meta["furniture_penalty"],
    })


@app.post("/remove")
async def remove(file: UploadFile = File(...), x: int = Form(...), y: int = Form(...), debug: bool = Form(False)):
    """One-click backward-compatible remove endpoint using precise selection."""
    image = read_image(file)
    _, label, mask = await _segment_furniture_at_point(image, x, y)
    result, meta = hybrid_inpaint(image, mask, allow_below_gate=debug)
    return JSONResponse({
        "image": to_data_url(result),
        "width": result.width,
        "height": result.height,
        "label": label,
        "method": "ai-remove-only",
        "quality_gate_passed": meta["passed_gate"],
        "backend": meta["backend"],
        "seam": meta["seam"],
        "furniture_penalty": meta["furniture_penalty"],
    })
