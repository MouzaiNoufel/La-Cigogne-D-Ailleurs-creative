# Phase 5.2.23 — RORem Loader Repair

## Critical fix
5.2.22 never reached RORem inference because Diffusers was explicitly asked for `variant="fp16"`, while the current `LetsThink/RORem` model repository exposes native F16 safetensors without a separately named `fp16` variant.

The loader now:
- uses the official `AutoPipelineForInpainting` API;
- removes the invalid variant override;
- requests safetensors directly;
- tries float16 first and bfloat16 second;
- retains CPU offload and attention/VAE memory optimizations.

## Quality behavior
The large-object Remove AI path still has **no LaMa fallback**. A dedicated RORem candidate must pass the removal gate or the endpoint returns 503.
