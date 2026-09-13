# Phase 5.2.25 — RORem reconstruction upgrade

## Goal
Continue the true **Remove AI** path without replacing the selected bed with another bed.

## Changes
- RORem now receives an **object-centred square crop** instead of the previous near-full-room wide crop.
- The working image is always **512×512**, matching the native/optimal regime documented by RORem for the base checkpoint.
- The selected bed therefore occupies substantially more of the model's latent resolution while retaining wall/rug/floor context.
- Increased the internal RORem mask dilation to approximately the documented 20px model-space range.
- Reverted to the official RORem CFG prompt/negative prompt rather than an over-specified furniture-negative prompt.
- Candidate selection now ranks by reconstruction quality first; target-change is used as a hard removal gate rather than a reason to prefer the most aggressively changed candidate.
- Tightened acceptance gates to reject obvious replacement/furniture hallucinations.
- No LaMa, Stable Diffusion generic replacement, blur fallback, or furniture substitution was added to Remove AI.

## Compatibility
- Existing installed `LetsThink/RORem` checkpoint is reused; no new model download is required if Phase 5.2.24 already loaded RORem successfully.
