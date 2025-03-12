import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel

transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-shuttle-jaguar")
pipeline = FluxPipeline.from_pretrained(
    "shuttleai/shuttle-jaguar", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
image = pipeline(
    "A cat holding a sign that says hello world",
    width=1024,
    height=1024,
    num_inference_steps=4,
    guidance_scale=3.5,
    generator=torch.Generator().manual_seed(245),
).images[0]
image.save("shuttle-jaguar-int4.png")
