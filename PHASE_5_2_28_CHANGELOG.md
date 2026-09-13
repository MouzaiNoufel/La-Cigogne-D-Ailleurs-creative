# Phase 5.2.28 — Remove AI reliability pass

## Why this phase
Phase 5.2.27 proved that sequential REMOVE-ONLY editing works, but some legitimate reconstructions were still returned as HTTP 503 because the strict seam gate was missed by only a few points.

Example observed case:
- target change: 0.435
- furniture penalty: 0.1447
- seam: 51.30
- strict seam gate: 50.0

This was a **quality-gate rejection**, not a RORem crash or resolution failure.

## Changes

### 1. Narrow high-confidence recovery band
The normal quality gate remains unchanged:
- target change >= 0.28
- furniture penalty <= 0.40
- seam <= 50

A candidate can additionally pass a narrow recovery band only when all are true:
- target change >= 0.32
- furniture penalty <= 0.20
- residual structure <= 0.45
- seam <= 58

This avoids accepting a visibly bad furniture hallucination merely to eliminate 503 responses.

### 2. Frontend failure safety
AI removal history is now committed **only after** the returned image passes dimension validation and is successfully loaded.

If the backend returns the wrong dimensions:
- the original room remains untouched
- the failed attempt is not added to undo history
- the selected object is not cleared unnecessarily

### 3. Status metadata
The `/inpaint-status` endpoint now reports Phase 5.2.28 and the strict/recovery thresholds.

## Architecture preserved
- REMOVE-ONLY
- RORem for large objects
- 512x512 object-centered crop
- candidate ensemble + refinement
- no furniture replacement model
- no generic LaMa fallback for large objects
