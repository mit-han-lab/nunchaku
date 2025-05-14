import pytest

from nunchaku.utils import is_turing

from .utils import run_test


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
@pytest.mark.parametrize(
    "height,width,use_qencoder,expected_lpips",
    [(1024, 1024, True, 0.1356), (1920, 1024, True, 0.1445)],
)
def test_int4_schnell_qencoder(height: int, width: int, use_qencoder: bool, expected_lpips: float):
    run_test(
        precision="int4",
        height=height,
        width=width,
        use_qencoder=use_qencoder,
        expected_lpips=expected_lpips,
    )
