import pytest

from .test_flux_dev import run_test_flux_dev


@pytest.mark.parametrize(
    "height,width,num_inference_steps,cache_threshold,lora_name,use_qencoder,cpu_offload,expected_lpips",
    [
        # (1024, 1024, 50, 0, None, False, False, 0.5),  # 13min20s 5min55s 0.19539418816566467
        # (1024, 1024, 50, 0.05, None, False, True, 0.5),  # 7min11s 0.21917256712913513
        # (1024, 1024, 50, 0.12, None, False, True, 0.5),  # 2min58s, 0.24101486802101135
        # (1024, 1024, 50, 0.2, None, False, True, 0.5),  # 2min23s, 0.3101634383201599
        # (1024, 1024, 50, 0.5, None, False, True, 0.5),  # 1min44s 0.6543852090835571
        # (1024, 1024, 30, 0, None, False, False, 0.5),  # 8min2s 3min40s 0.2141970843076706
        # (1024, 1024, 30, 0.05, None, False, True, 0.5),  # 4min57 0.21297718584537506
        # (1024, 1024, 30, 0.12, None, False, True, 0.5),  # 2min34 0.25963714718818665
        # (1024, 1024, 30, 0.2, None, False, True, 0.5),  # 1min51 0.31409069895744324
        # (1024, 1024, 20, 0, None, False, False, 0.5),  # 5min25 2min29 0.18987375497817993
        # (1024, 1024, 20, 0.05, None, False, True, 0.5),  # 3min3 0.17194810509681702
        # (1024, 1024, 20, 0.12, None, False, True, 0.5),  # 2min15 0.19407868385314941
        # (1024, 1024, 20, 0.2, None, False, True, 0.5),  # 1min48 0.2832985818386078
        (1024, 1024, 30, 0.12, None, False, False, 0.26),
        (512, 2048, 30, 0.12, "anime", True, False, 0.4),
    ],
)
def test_flux_dev_base(
    height: int,
    width: int,
    num_inference_steps: int,
    cache_threshold: float,
    lora_name: str | None,
    use_qencoder: bool,
    cpu_offload: bool,
    expected_lpips: float,
):
    run_test_flux_dev(
        precision="int4",
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        use_qencoder=use_qencoder,
        cpu_offload=cpu_offload,
        lora_name=lora_name,
        lora_scale=1,
        cache_threshold=cache_threshold,
        max_dataset_size=16,
        expected_lpips=expected_lpips,
    )
