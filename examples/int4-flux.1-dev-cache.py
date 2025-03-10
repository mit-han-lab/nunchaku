import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe

transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev", offload=True)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
)
pipeline.enable_sequential_cpu_offload()
apply_cache_on_pipe(pipeline, residual_diff_threshold=0.12)
image = pipeline(["A cat holding a sign that says hello world"], num_inference_steps=50).images[0]
image.save("flux.1-dev-int4.png")
