# Phase 5.2.21 — Real object-removal upgrade

## Why 5.2.20 was rejected
The RORem candidates failed the removal-quality gate, so the previous build silently exposed a LaMa surface fallback. That fallback produced a recolored/reconstructed bed instead of a convincing empty background.

## Changes
- Added SmartEraser as the primary large-object Remove AI engine.
- SmartEraser uses masked-region guidance and is specifically designed to prevent regeneration of the removed object while preserving surrounding context.
- Uses the public Hugging Face SmartEraser checkpoint mirror `creative-graphic-design/smarteraser-checkpoints`.
- Three seeds are evaluated.
- Added a target-change metric so a candidate that barely changes the original object cannot pass.
- Added stricter semantic furniture + seam + target-change gating.
- RORem is now secondary, not primary.
- The old LaMa surface-split fallback is NOT exposed for large-object Remove AI anymore. A poor reconstruction now fails instead of pretending the object was removed.
- Final compositing remains mask-bounded, preserving the original image outside the removal region.

## Research basis
SmartEraser's masked-region guidance was designed specifically to address the classic mask-and-inpaint failure where the removed object is regenerated and the surrounding context is not preserved.
