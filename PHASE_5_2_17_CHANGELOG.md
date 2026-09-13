# Phase 5.2.17 — Surface-Aware REMOVE-ONLY AI

## What was fixed

Phase 5.2.16 correctly removed all replacement/generative insertion paths, but a large selected object was still being reconstructed by LaMa as one giant mixed wall/floor/rug hole. On the bedroom test this could leave the bed-like silhouette or fail the quality gate.

Phase 5.2.17 changes the reconstruction strategy:

- The confirmed SAM object mask remains the only user-editable removal region.
- The selected object is removed from a temporary donor context before LaMa sees it.
- SegFormer routes the hole into **wall**, **rug/carpet**, **floor**, and residual regions.
- Each surface is reconstructed independently with LaMa.
- Neighboring furniture is blurred out of the sharp donor context so tables, nightstands, sofas, etc. are not copied into the hole.
- Rug/carpet is kept separate from wood floor so wood texture is not used to rebuild a rug.
- A narrow floor/rug contact-shadow cleanup is used only as a temporary reconstruction aid; the final visible edit is still composited strictly inside the user's confirmed object mask.
- A clean-context whole-object LaMa pass remains only as a fallback.
- **No PowerPaint, Stable Diffusion, or furniture replacement model is used by Remove AI.**

## Expected behavior

Click the bed → confirm removal → the bed disappears and the visible result reconstructs the photographed wall/rug/floor surfaces. The nightstands, dresser, lamps, doors, plants, shelves, etc. outside the selected mask remain unchanged.

## Backend

- FastAPI
- SAM ViT-B point refinement
- SegFormer B3 ADE20K scene/surface routing
- LaMa inpainting
- OpenCV multi-scale donor-context preparation

## Validation

`server/main.py` is syntax-checked with `py_compile` before packaging.
