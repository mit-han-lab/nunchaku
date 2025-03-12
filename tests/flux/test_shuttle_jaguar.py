import os

import pytest
import torch
from diffusers import FluxPipeline

from nunchaku import NunchakuFluxTransformer2dModel
from tests.data import get_dataset
from tests.flux.utils import run_pipeline
from tests.utils import already_generate, compute_lpips


@pytest.mark.parametrize(
    "precision,height,width,num_inference_steps,guidance_scale,use_qencoder,cpu_offload,max_dataset_size,expected_lpips",
    [
        ("int4", 1024, 1024, 4, 3.5, False, False, 16, 0.25),
        ("int4", 2048, 512, 4, 3.5, False, False, 16, 0.21),
    ],
)
def test_shuttle_jaguar(
    precision: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    use_qencoder: bool,
    cpu_offload: bool,
    max_dataset_size: int,
    expected_lpips: float,
):
    dataset = get_dataset(name="MJHQ", max_dataset_size=max_dataset_size)
    save_root = os.path.join("results", "shuttle-jaguar", f"w{width}h{height}t{num_inference_steps}g{guidance_scale}")

    save_dir_16bit = os.path.join(save_root, "bf16")
    if not already_generate(save_dir_16bit, max_dataset_size):
        pipeline = FluxPipeline.from_pretrained("shuttleai/shuttle-jaguar", torch_dtype=torch.bfloat16)
        pipeline = pipeline.to("cuda")

        run_pipeline(
            dataset,
            pipeline,
            save_dir=save_dir_16bit,
            forward_kwargs={
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )
        del pipeline
        # release the gpu memory
        torch.cuda.empty_cache()

    save_dir_4bit = os.path.join(
        save_root, f"{precision}-qencoder" if use_qencoder else f"{precision}" + ("-cpuoffload" if cpu_offload else "")
    )
    if not already_generate(save_dir_4bit, max_dataset_size):
        pipeline_init_kwargs = {}
        if precision == "int4":
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                "mit-han-lab/svdq-int4-shuttle-jaguar", offload=cpu_offload
            )
        else:
            assert precision == "fp4"
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                "mit-han-lab/svdq-fp4-shuttle-jaguar", precision="fp4", offload=cpu_offload
            )
        pipeline_init_kwargs["transformer"] = transformer
        if use_qencoder:
            raise NotImplementedError
            # text_encoder_2 = NunchakuT5EncoderModel.from_pretrained("mit-han-lab/svdq-flux.1-t5")
            # pipeline_init_kwargs["text_encoder_2"] = text_encoder_2
        pipeline = FluxPipeline.from_pretrained(
            "shuttleai/shuttle-jaguar", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
        )
        pipeline = pipeline.to("cuda")
        if cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        run_pipeline(
            dataset,
            pipeline,
            save_dir=save_dir_4bit,
            forward_kwargs={
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )
        del pipeline
        # release the gpu memory
        torch.cuda.empty_cache()
    lpips = compute_lpips(save_dir_16bit, save_dir_4bit)
    print(f"lpips: {lpips}")
    assert lpips < expected_lpips * 1.05
