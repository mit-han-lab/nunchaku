import pytest

from tests.flux.test_flux_dev import run_test_flux_dev


@pytest.mark.parametrize(
    "num_inference_steps,lora_name,lora_scale,cpu_offload,expected_lpips",
    [
        (25, "realism", 0.9, False, 0.16),
        (25, "ghibsky", 1, False, 0.16),
        (28, "anime", 1, False, 0.27),
        (24, "sketch", 1, False, 0.35),
        (28, "yarn", 1, False, 0.22),
        (25, "haunted_linework", 1, False, 0.34),
    ],
)
def test_flux_dev_loras(num_inference_steps, lora_name, lora_scale, cpu_offload, expected_lpips):
    run_test_flux_dev(
        precision="int4",
        height=1024,
        width=1024,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        use_qencoder=False,
        cpu_offload=cpu_offload,
        lora_name=lora_name,
        lora_scale=lora_scale,
        cache_threshold=0,
        max_dataset_size=8,
        expected_lpips=expected_lpips,
    )


def test_flux_dev_hypersd8_1080x1920():
    run_test_flux_dev(
        precision="int4",
        height=1080,
        width=1920,
        num_inference_steps=8,
        guidance_scale=3.5,
        use_qencoder=False,
        cpu_offload=False,
        lora_name="hypersd8",
        lora_scale=0.125,
        cache_threshold=0,
        max_dataset_size=8,
        expected_lpips=0.44,
    )
