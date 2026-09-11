import pytest
import warnings
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_compile():
    from ctrex.projectors.static_projectors import SIRTProjector
    with warnings.catch_warnings(record=True) as w:
        a = SIRTProjector()  # try to compile CUDA code

    assert len(w) == 0, f"Warnings during CUDA compile: {w}"
