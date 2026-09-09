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

app = FastAPI(title="La Cigogne D'Ailleurs AI", version="4.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

depth_pipe = None
seg_pipe = None
lama = None
diffusion_pipe = None
diffusion_failed = False
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
            "runwayml/stable-diffusion-inpainting",
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
    """Score a reconstruction by boundary continuity, not just mean color.

    A low-contrast hallucination (for example a newly generated dark piece of
    furniture) can have a deceptively good mean-color score. We therefore
    compare luminance + gradient continuity in a narrow ring around the
    confirmed mask and penalize excessive edge energy inside the generated
    region.
    """
    if cv2 is None:
        return 0.0
    a = np.asarray(mask_crop, np.uint8) > 127
    if not a.any():
        return 1e9
    src = np.asarray(original_crop.convert("RGB"), np.uint8)
    gen = np.asarray(candidate.convert("RGB"), np.uint8)

    dil = cv2.dilate(a.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
    ero = cv2.erode(a.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    ring = dil & (~a)
    inner = a & (~ero)
    if not ring.any() or not inner.any():
        return 1e9

    src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gen_gray = cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY).astype(np.float32)
    src_grad = cv2.Laplacian(src_gray, cv2.CV_32F)
    gen_grad = cv2.Laplacian(gen_gray, cv2.CV_32F)

    # Boundary luminance continuity.
    boundary_src = src_gray[ring].mean()
    boundary_gen = gen_gray[inner].mean()
    color_score = abs(boundary_src - boundary_gen)

    # Compare local gradient magnitudes. Large artificial edges inside the hole
    # are a strong signal that the model invented an object.
    grad_src = np.abs(src_grad[ring]).mean()
    grad_gen = np.abs(gen_grad[inner]).mean()
    edge_penalty = max(0.0, grad_gen - max(grad_src * 1.45, grad_src + 4.0))

    # Penalize very dark/bright generated interiors relative to the visible ring.
    ring_mean = float(src_gray[ring].mean())
    inner_mean = float(gen_gray[inner].mean())
    luminance_penalty = max(0.0, abs(inner_mean - ring_mean) - 28.0) * 0.45

    return float(color_score + edge_penalty * 1.8 + luminance_penalty)


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
    score = _seam_score(work, out, work_mask)
    return _restore_square(out, geometry), score


def _diffusion_inpaint_large_object(image: Image.Image, object_mask: np.ndarray) -> Image.Image | None:
    """Surface-aware generative reconstruction for substantial furniture masks.

    The old implementation sent the *entire furniture hole* to one generic
    "empty bedroom" prompt. Stable Diffusion then had enough contextual evidence
    to hallucinate another bed/table inside the hole. This version splits the
    confirmed mask into structural surfaces first and generates each surface
    independently: wall gets a wall-only prompt, floor/rug gets a floor-only
    prompt. The generated pixels are still composited strictly inside the
    confirmed mask.
    """
    if not object_mask.any():
        return None
    pipe = get_diffusion_inpainter()
    if pipe is None:
        return None
    try:
        src = image.convert("RGB")
        W, H = src.size
        surfaces = _surface_masks(src)
        surface_map = _nearest_surface_map(object_mask, surfaces)

        # If segmentation cannot establish surfaces, use a conservative
        # horizontal prior rather than asking diffusion to reconstruct the whole
        # room with an "empty bedroom" prompt.
        if not np.any(surface_map == 1) and not np.any(surface_map == 2):
            surface_map = np.where(np.indices(object_mask.shape)[0] >= int(H * 0.55), 1, 2).astype(np.uint8)

        result = src.copy()
        surface_specs = [
            (
                2,
                "continuous flat white painted interior wall, subtle plaster texture, same wall lighting, "
                "no furniture, no decoration, no object, no frame, no plant, seamless wall surface",
                "bed, mattress, pillow, sofa, chair, table, cabinet, nightstand, furniture, object, "
                "painting, picture frame, plant, lamp, person, room corner, doorway, dark blob",
                7103,
            ),
            (
                1,
                "continuous flat light gray carpet and pale wooden floor surface, same perspective and lighting, "
                "natural carpet fibers or subtle wood grain, seamless empty floor, no furniture, no object",
                "bed, mattress, pillow, sofa, chair, table, cabinet, nightstand, furniture, object, "
                "person, rug edge, wall, cabinet, dark blob, vertical object, duplicated furniture",
                9137,
            ),
        ]

        for sid, prompt, negative, seed in surface_specs:
            target = object_mask & (surface_map == sid)
            if not target.any():
                continue
            ys, xs = np.where(target)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            pad_x = int(np.clip(round(bw * 0.38), 80, 260))
            pad_y = int(np.clip(round(bh * 0.38), 80, 220))
            cx0, cy0 = max(0, x0-pad_x), max(0, y0-pad_y)
            cx1, cy1 = min(W, x1+pad_x+1), min(H, y1+pad_y+1)
            crop = src.crop((cx0, cy0, cx1, cy1))
            local_mask = target[cy0:cy1, cx0:cx1]

            # Two candidates are retained, but only for this surface. This makes
            # candidate selection meaningful and prevents a bed-shaped global
            # hallucination from winning because its average color is smooth.
            candidates = []
            for local_seed in (seed, seed + 17):
                out, score = _diffusion_surface_pass(pipe, crop, local_mask, prompt, negative, local_seed)
                candidates.append((score, out))
                print(f"Generative surface={sid} seed={local_seed}, seam_score={score:.2f}")
            generated_crop = min(candidates, key=lambda z: z[0])[1]

            # Blend softly only inside this surface's confirmed mask. A very
            # narrow feather hides the diffusion tile boundary without bleeding
            # into untouched furniture/wall/floor pixels.
            alpha = local_mask.astype(np.float32)
            if cv2 is not None:
                alpha = cv2.GaussianBlur(alpha, (0, 0), 0.85)
            # Feather only inward: never modify pixels outside the confirmed mask.
            alpha = np.clip(alpha, 0.0, 1.0) * local_mask.astype(np.float32)
            alpha = alpha[..., None]
            base = np.asarray(result.crop((cx0, cy0, cx1, cy1)), np.float32)
            gen = np.asarray(generated_crop, np.float32)
            blended = np.clip(base * (1.0-alpha) + gen * alpha, 0, 255).astype(np.uint8)
            result.paste(Image.fromarray(blended), (cx0, cy0))

        return result
    except Exception as exc:
        print(f"ERROR: generative inpainting failed: {exc}")
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
        mask_img=mask_img.resize(image_size,Image.Resampling.BILINEAR)
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
    """Assign each masked pixel to the nearest visible structural surface.

    1=floor/rug, 2=wall. The object mask itself is never replaced by the
    semantic surface masks; they are only used to decide *how* the background
    should be reconstructed.
    """
    H, W = mask.shape
    labels = np.zeros((H, W), np.uint8)
    floor = np.zeros_like(mask, bool)
    wall = np.zeros_like(mask, bool)
    for k, v in surfaces.items():
        if k in {"floor", "rug", "carpet"}:
            floor |= v
        elif k == "wall":
            wall |= v
    labels[floor] = 1
    labels[wall] = 2

    known = (labels > 0) & (~mask)
    if not known.any():
        labels[int(H * 0.60):, :] = 1
        labels[:int(H * 0.60), :] = 2
        return labels

    if cv2 is None:
        return labels

    # Nearest-label propagation, but only into the confirmed hole.
    inv = (~known).astype(np.uint8)
    _, inds = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, cv2.DIST_LABEL_PIXEL)
    ky, kx = np.where(known)
    vals = labels[ky, kx]
    idx = np.asarray(inds, np.int64) - 1
    propagated = np.zeros_like(labels)
    valid = (idx >= 0) & (idx < vals.size)
    propagated[valid] = vals[idx[valid]]
    propagated[known] = labels[known]
    # Preserve known semantic labels exactly.
    propagated[~mask & (labels > 0)] = labels[~mask & (labels > 0)]
    return propagated


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
    ring = dil & (~object_mask) & (surface_map == 1)
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


def hybrid_inpaint(image: Image.Image, mask_img: Image.Image) -> Image.Image:
    """AI Remove 2.4: precise mask + mandatory generative reconstruction for large objects."""
    hard_mask_img = prepare_inpaint_mask(mask_img, image.size)
    object_mask = np.asarray(hard_mask_img.convert("L")) > 127
    if not object_mask.any():
        raise HTTPException(status_code=422, detail="Masque vide")

    area_ratio = float(object_mask.mean())
    # Large furniture is the difficult case. Do NOT silently fall back to the
    # broken deterministic texture-stretch path: either the generative model
    # runs, or the API tells the user exactly what is missing.
    if area_ratio >= 0.015:
        generated = _diffusion_inpaint_large_object(image, object_mask)
        if generated is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Le moteur de reconstruction générative n'est pas disponible. "
                    "Installez les dépendances de server/requirements.txt puis redémarrez le serveur."
                ),
            )
        final_mask = Image.fromarray((object_mask.astype(np.uint8)*255), "L")
        return _composite_inside_mask(image, generated, final_mask)

    # Small objects can still use the stable LaMa path; this avoids paying the
    # diffusion cost for pillows, lamps, etc.
    try:
        lama_img = get_lama()(image, hard_mask_img)
        return _composite_inside_mask(image, lama_img, hard_mask_img)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inpainting petit objet impossible: {exc}") from exc


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
    return JSONResponse({"mask": to_data_url(mask, max_side=1600), "label": label, "area_ratio": ratio, "bbox": bbox, "method": "SAM + semantic context"})


@app.post("/inpaint")
async def inpaint(file: UploadFile = File(...), mask: UploadFile = File(...)):
    """Remove a user-confirmed furniture mask with LaMa."""
    image = read_image(file)
    try:
        raw = await mask.read()
        mask_img = Image.open(io.BytesIO(raw)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Masque invalide: {exc}") from exc
    result = hybrid_inpaint(image, mask_img)
    if not isinstance(result, Image.Image):
        result = Image.fromarray(np.asarray(result).astype(np.uint8))
    return JSONResponse({"image": to_data_url(result), "label": "meuble", "method": "generative-reconstruction" if float((np.asarray(mask_img.convert("L")) > 127).mean()) >= 0.015 else "lama-small-object"})


@app.post("/remove")
async def remove(file: UploadFile = File(...), x: int = Form(...), y: int = Form(...)):
    """One-click backward-compatible remove endpoint using precise selection."""
    image = read_image(file)
    _, label, mask = await _segment_furniture_at_point(image, x, y)
    result = hybrid_inpaint(image, mask)
    return JSONResponse({"image": to_data_url(result), "label": label, "method": "generative-reconstruction" if float((np.asarray(mask.convert("L")) > 127).mean()) >= 0.015 else "lama-small-object"})
