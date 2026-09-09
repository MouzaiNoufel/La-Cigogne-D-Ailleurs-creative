# La Cigogne D'Ailleurs — Phase 5.2 AI Remove 2.0

Professional Editor + **object-first AI removal and background reconstruction**.

## What changed in 5.2

### 1. Object-first selection
- SegFormer is used only to identify the furniture class and create a local semantic prior.
- The semantic mask is reduced to the connected component containing the user's click.
- SAM (`facebook/sam-vit-base`) receives a tight semantic bounding box plus multiple interior positive points.
- Candidate masks are validated by click coverage, semantic overlap, locality and area before being accepted.
- GrabCut remains a local fallback if SAM cannot load or fails.

### 2. Correct mask preview
- The frontend now treats the returned grayscale mask as **luminance**, not PNG alpha.
- The old bug that visually made the entire room appear selected is removed.
- The selected object is shown with a red transparent overlay and a contour before confirmation.
- The action button explicitly says **Confirmer suppression**.

### 3. Surface-constrained generative reconstruction

Large furniture removal no longer sends the entire furniture hole to a generic “empty bedroom” diffusion prompt. The confirmed mask is first split by visible structural surface (wall vs floor/rug), and each surface is reconstructed independently with a surface-specific prompt and negative prompt. This prevents the inpainting model from treating the surrounding bedroom context as permission to invent another bed, table, or cabinet. Candidate selection also penalizes artificial edge energy, not only average boundary color. Generated pixels remain constrained to the confirmed mask with an inward-only feather.

### 3. Safer inpainting
- The confirmed object mask is kept tight; only a very small context ring is added.
- The original pixels outside the confirmed mask are preserved during final compositing.
- Small holes use local OpenCV texture reconstruction before LaMa refinement.
- Floor/wall context is detected separately and is **never** allowed to become the object mask.
- Large semantic regions are rejected instead of silently deleting a scene-sized area.

### 4. Background reconstruction strategy

```text
click object
    -> semantic class/context
    -> clicked component
    -> SAM silhouette
    -> mask validation
    -> red preview
    -> surface context (floor / wall)
    -> local texture reconstruction
    -> LaMa semantic reconstruction
    -> restore original pixels outside mask
```

This remains single-image reconstruction. If a large object completely hides background information, no algorithm can recover the exact unseen pixels; the goal is to produce a plausible, spatially consistent reconstruction without destroying neighboring surfaces.

## Backend

Version: **4.1.0**

Start with:

```powershell
cd "C:\Users\pc\Desktop\La Cigogne D'Ailleurs creative\server"
uvicorn main:app --reload --port 8000
```

Frontend:

```powershell
cd "C:\Users\pc\Desktop\La Cigogne D'Ailleurs creative"
python -m http.server 5500
```


## Phase 5.2.4 — Generative Background Reconstruction

Large furniture removal now uses an optional local diffusion inpainting model (`runwayml/stable-diffusion-inpainting`) on a padded crop around the selected object. The generated result is composited strictly inside the confirmed object mask. If the model cannot load, the deterministic surface-aware fallback remains available.

For an RTX 3070 Ti 8 GB, the backend uses CPU offload and attention slicing when CUDA is available. The first generative removal downloads the model weights from Hugging Face.


## Phase 5.2.4 — Generative reconstruction enforcement
Large furniture removal now requires the diffusion inpainting backend instead of silently falling back to the old surface texture stretch. The backend logs the selected reconstruction method and tries two deterministic seeds, selecting the lower seam-discontinuity candidate. If diffusers/model loading is unavailable, `/inpaint` returns HTTP 503 with an actionable dependency message rather than producing a misleading low-quality result.
