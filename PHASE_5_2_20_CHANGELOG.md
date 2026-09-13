# Phase 5.2.20 — Dedicated Object-Removal Upgrade

## Goal
Upgrade large-object AI removal from the previous LaMa-only ~2/10 visual result toward a client-demo-quality result without turning Remove AI into a furniture replacement tool.

## Main change
Added **RORem (Robust Object Remover)** as the primary large-object removal backend.

- Model: `LetsThink/RORem`
- Dedicated object-removal model, not a generic text-to-image replacement pipeline.
- Lazy-loaded only when a large removal is requested.
- Uses CUDA with model CPU offload to reduce VRAM pressure on 8GB GPUs.
- Two deterministic seeds are tested (`2026`, `7319`) and the best removal candidate is selected.
- Candidate scoring heavily penalizes surviving furniture inside the selected mask.
- Strict final compositing preserves the original photo outside the confirmed removal mask.

## Fallback
If RORem cannot load (network, model download, VRAM or dependency failure), the existing surface-routed LaMa pipeline remains available as a fallback.

## Why this change
The previous large-object path was repeatedly producing a brown/grey furniture-shaped blur. Generic LaMa was being asked to reconstruct a very large masked bed area, which is not a strong dedicated object-removal prior. The new primary backend is specifically trained for robust object removal.

## Product rule preserved
- Remove AI remains REMOVE-ONLY.
- No PowerPaint replacement mode.
- No generic Stable Diffusion furniture-generation mode.
- No automatic insertion of another bed/sofa/table.

## Validation
- `main.py` syntax checked with Python 3.
- ZIP packaged as a complete ready-to-test project.
