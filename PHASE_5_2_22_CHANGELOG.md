# Phase 5.2.22 — RORem Official Inference Fix

## Why 5.2.21 failed

The SmartEraser checkpoint loaded its model metadata but the installed Diffusers package did not provide the custom `StableDiffusionInpaintRegionPipeline` required by the official SmartEraser implementation. It was therefore unavailable.

RORem was then executed with an incorrect square 1024×1024 adaptation. The official RORem implementation recommends `AutoPipelineForInpainting`, a 512px working resolution for the base checkpoint, `strength=0.99`, and `guidance_scale=1.0`; the project had deviated from that regime.

## Fixes

- SmartEraser is explicitly skipped unless its official custom pipeline is present.
- RORem now loads with `AutoPipelineForInpainting`.
- RORem uses a 512-pixel short-side working image with preserved aspect ratio instead of a forced square 1024 crop.
- RORem uses 50 inference steps, strength 0.99, guidance scale 1.0, and the generic quality prompt/negative prompt recommended by the official repository.
- The input mask receives only a small 8px removal halo to catch contact-shadow and edge residues.
- The output is resized back to the exact local crop and hard-composited only in the removal region; original pixels outside the removal region remain untouched.
- Three deterministic seeds are evaluated.
- Candidate selection now prioritizes actual target disappearance (`target_change`) before furniture hallucination and seam quality.
- Large-object Remove AI has no LaMa blur fallback. A weak dedicated-removal result is rejected instead of shown as success.

## Official reference

RORem's official repository states that its base model performs best at 512×512 and that the mixed-resolution checkpoint is intended for images larger than 512×512. This build first corrects the base RORem inference path to the official 512-short-side regime.
