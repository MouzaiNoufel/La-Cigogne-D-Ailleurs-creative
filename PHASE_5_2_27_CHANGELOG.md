# Phase 5.2.27 — REMOVE AI quality ensemble

## Goal
Improve the already-working RORem remove-only pipeline without reintroducing object replacement or blur fallbacks.

## Changes
- Kept RORem as the only large-object removal engine.
- Adaptive internal mask dilation: larger selected objects receive a wider generation halo; small objects keep a tighter halo.
- Increased first-pass candidate search from 3 to 5 deterministic seeds.
- Added a lightweight structural-residual score based on edge density inside the confirmed hole versus nearby context. It is used as a ranking signal to help reject sofa/bed-like ghost structures that semantic segmentation can miss.
- Candidate scoring now gives slightly more weight to semantic furniture hallucination and also includes the structural-residual signal.
- Added up to two conservative residual-removal passes on the best candidate. A refinement is accepted only when it materially improves semantic cleanliness, structural residual, or total quality score.
- Original-room quality gates remain: target change >= 0.28, furniture penalty <= 0.40, seam <= 50.
- No generic LaMa fallback, Stable Diffusion replacement, PowerPaint, or furniture insertion path is enabled for Remove AI.

## Expected behavior
- Better chance of removing large beds/sofas/tables completely.
- Better resistance to leaving a furniture-shaped ghost in the hole.
- Small-object removals should remain close to the successful 5.2.26 behavior.
- Inference may take longer because more RORem candidates are evaluated.

## Test order
1. Remove a large bed/sofa.
2. Remove a second object from the resulting image.
3. Remove a third object.
4. Verify the room dimensions remain unchanged after every operation.
