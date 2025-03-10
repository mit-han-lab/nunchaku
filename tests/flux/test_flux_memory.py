import pytest
import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel


@pytest.mark.parametrize(
    "use_qencoder,cpu_offload,memory_limit",
    [
        (False, False, 17),
        (False, True, 13),
        (True, False, 12),
        (True, True, 6),
    ],
)
def test_flux_schnell_memory(use_qencoder: bool, cpu_offload: bool, memory_limit: float):
    torch.cuda.reset_peak_memory_stats()
    pipeline_init_kwargs = {
        "transformer": NunchakuFluxTransformer2dModel.from_pretrained(
            "mit-han-lab/svdq-int4-flux.1-schnell", offload=cpu_offload
        )
    }
    if use_qencoder:
        text_encoder_2 = NunchakuT5EncoderModel.from_pretrained("mit-han-lab/svdq-flux.1-t5")
        pipeline_init_kwargs["text_encoder_2"] = text_encoder_2
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
    ).to("cuda")

    if cpu_offload:
        pipeline.enable_sequential_cpu_offload()

    pipeline(
        "A cat holding a sign that says hello world", width=1024, height=1024, num_inference_steps=50, guidance_scale=0
    )
    memory = torch.cuda.max_memory_reserved(0) / 1024**3
    assert memory < memory_limit
    del pipeline
    # release the gpu memory
    torch.cuda.empty_cache()
