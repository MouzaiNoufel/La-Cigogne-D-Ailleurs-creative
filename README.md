# La Cigogne D'Ailleurs — Phase 5.2.15

## Object-removal reconstruction overhaul

This build keeps the existing **SAM point-selection / mask refinement pipeline** and changes the large-object reconstruction policy.

### What was wrong in 5.2.14

The terminal showed:

- Surface-guided candidate: `score=234.35`
- LaMa: `score=204.10` / `175.89`
- Diffusion: `score=143.32` / `170.21` / `191.49`
- Yet the backend selected **surface-guided**.

That was the wrong policy. The surface-guided OpenCV reconstruction was producing the blurred/stretched furniture-shaped region visible in the screenshots. A higher numerical score did not mean a better visual result.

### What 5.2.15 changes

1. **Surface-guided reconstruction is diagnostic only. It can no longer win the final large-object result.**
2. **PowerPaint v2 object-removal** is now the first large-object backend.
   - Model: `Sanster/PowerPaint_v2`
   - Uses PowerPaint's dedicated object-removal task prompts (`P_ctxt` / `P_obj`).
   - Runs lazily and uses CPU offload on CUDA so it can fit more realistically on an 8 GB GPU.
3. **LaMa** remains the deterministic fallback.
4. Existing **Stable Diffusion 1.5 inpainting** remains the final generative fallback.
5. Candidate selection now prioritizes the task-specific object-removal engine rather than forcing a surface-fill result.
6. SegFormer furniture validation now ignores low-confidence semantic detections, reducing false furniture penalties on walls/rugs/floors.
7. The confirmed SAM mask is still the only region allowed to change in the final composite.
8. PowerPaint/LaMa/SD are released between attempts when possible to reduce VRAM pressure.

### Important expectation

No single-image inpainting model can recover the exact pixels that were physically hidden by a bed. The goal is a convincing photographic reconstruction of the room context. Large masks are inherently harder, and the user's current 1408×768 image has only 768 pixels on its short side. A higher-resolution original is still preferable.

### PowerPaint model download

The first large-object removal using PowerPaint will download its model from Hugging Face if it is not already cached. The project does **not** bundle the multi-GB model weights inside the ZIP.

PowerPaint is specifically documented by its authors as supporting object removal/context reconstruction and recommends a high guidance scale for suppressing undesired object generation.

### Start

Backend:
```powershell
cd "C:\Users\pc\Desktop\La Cigogne D'Ailleurs creative\server"
uvicorn main:app --reload --port 8000
```

Frontend:
```powershell
cd "C:\Users\pc\Desktop\La Cigogne D'Ailleurs creative"
python -m http.server 5500
```

Open:
`http://localhost:5500`

### Version

Backend: `4.8.0-phase5.2.15`


### Phase 5.2.28
- RORem REMOVE-ONLY ensemble + refinement retained.
- Added a narrow high-confidence recovery band to prevent false 503 rejections when seam is only slightly above the strict gate.
- Added safer frontend history handling: failed/dimension-mismatched AI removals no longer pollute undo history or clear the active selection.
