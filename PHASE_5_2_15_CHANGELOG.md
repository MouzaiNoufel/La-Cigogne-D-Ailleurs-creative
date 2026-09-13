# Phase 5.2.15 — Object Removal Reconstruction Fix

## Root cause confirmed from 5.2.14 logs

The backend generated better numerical candidates than the final selected result, then deliberately chose the `surface-guided` candidate. That candidate is a classical multiscale OpenCV reconstruction and was the source of the visible stretched/blurred room region.

Example from the reported run:

- surface-guided: score 234.35
- LaMa tight: score 175.89
- diffusion seed 4217: score 143.32

The 5.2.15 policy therefore forbids surface-guided reconstruction from winning the final large-object result.

## New priority

1. PowerPaint v2 object removal
2. LaMa
3. Stable Diffusion 1.5 inpainting

The existing SAM selection/refinement is retained.

## VRAM policy

PowerPaint is lazy-loaded and CPU-offloaded on CUDA. If it fails or does not meet the boundary-quality gate, it is released before LaMa/SD are loaded.

## No model weights bundled

PowerPaint v2 is downloaded by Hugging Face on first use. The multi-GB weights are intentionally not placed inside the project ZIP.
