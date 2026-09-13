# Phase 5.2.18 — True Remove-Only Reconstruction

## Why 5.2.17 was rejected
The previous surface-aware candidate used a large coarse OpenCV inpaint pass. On the bedroom benchmark this produced a smooth brown/grey blob instead of reconstructing the visible room surfaces.

## Fix
- Removed the coarse surface-LaMa/OpenCV candidate from the primary path.
- Wall pixels are reconstructed deterministically from real same-row neighboring wall pixels with local illumination interpolation.
- Rug and floor pixels are sent to separate tight local LaMa holes, so each receives real neighboring surface context.
- Surface routing now estimates the wall/floor transition before assigning rug/floor; rug is not allowed to propagate upward across the wall.
- The selected object and neighboring furniture are removed only from the temporary donor context.
- Final compositing restores every pixel outside the confirmed object mask.
- No Stable Diffusion, PowerPaint, or furniture-replacement model is used.
- Quality scores remain diagnostics; a non-passing score no longer forces a fake debug preview or 503. The user receives the actual remove-only result and can undo it.
