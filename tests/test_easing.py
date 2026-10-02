import numpy as np
import pytest

from skitter.core.easing import (
    ease_in_cubic,
    ease_in_out_cubic,
    ease_out_back,
    ease_out_cubic,
    lerp,
    linear,
    progress,
)


@pytest.mark.parametrize(
    "curve", [linear, ease_in_cubic, ease_out_cubic, ease_in_out_cubic, ease_out_back]
)
def test_curves_hit_endpoints(curve):
    np.testing.assert_allclose(curve(np.array([0.0, 1.0])), [0.0, 1.0], atol=1e-12)


def test_progress_clamps_and_vectorizes():
    np.testing.assert_allclose(progress(1.0, np.array([0.0, 0.5, 2.0]), 1.0), [1.0, 0.5, 0.0])


def test_lerp():
    result = lerp(np.array([0.0, 10.0]), np.array([10.0, 20.0]), 0.25)
    np.testing.assert_allclose(result, [2.5, 12.5])
