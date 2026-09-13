# Phase 5.2.24 — RORem quality-metric fix

## Fixed
- Fixed the `/inpaint` crash: `_target_change_score` was referenced but missing.
- The RORem candidates can now complete quality evaluation instead of every candidate failing with `NameError`.
- Target-change is now measured on the original confirmed SAM object mask, not the internally dilated RORem halo.
- Candidate quality still rejects furniture hallucination and bad seams; no LaMa large-object fallback is reintroduced.
- The selected candidate is hard-composited only inside the removal mask, preserving the original room outside it.

## Important
This release fixes the concrete runtime blocker shown in Phase 5.2.23. It does not claim 9/10 visual quality until the RORem output is visually tested on the bedroom image.
