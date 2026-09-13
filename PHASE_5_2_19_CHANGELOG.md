# Phase 5.2.19 — REMOVE-ONLY crash fix

Critical fix: Phase 5.2.18 `/inpaint` returned HTTP 500 because `_remove_only_large_object()` attempted NumPy boolean indexing directly on a PIL Image. The temporary donor image is now kept as an RGB NumPy array during OpenCV neutralization, then converted back to PIL before LaMa. The neighboring-furniture shield uses the same safe representation.

This fixes the browser `Failed to fetch` symptom caused by the backend 500. The REMOVE-ONLY architecture remains unchanged: no Stable Diffusion or PowerPaint replacement path.
