# Phase 5.2.16 — REMOVE-ONLY AI

## Critical correction
The previous 5.2.15 build was incorrect for this product requirement: it introduced a generative replacement path and could turn a selected bed into another bed/object.

## 5.2.16 behavior
- The current AI action is **REMOVE ONLY**.
- Clicking a furniture object selects its SAM mask, then `/inpaint` reconstructs only the hidden room/background.
- PowerPaint and Stable Diffusion replacement/insertion paths are disabled for this workflow.
- Large-object removal uses LaMa with a temporary semantic furniture shield so neighboring furniture is not used as donor context.
- The final image is always composited over the original image outside the confirmed SAM mask.
- No replacement furniture is intentionally generated.
- UI label changed from `Remplacer IA` to `Supprimer IA`.

## Test
Use the same bedroom image and click the bed. The expected behavior is: the bed disappears; the wall/floor/rug background is reconstructed; side furniture remains exactly where it was.
