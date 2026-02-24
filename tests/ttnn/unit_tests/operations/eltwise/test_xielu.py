# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp


@pytest.mark.parametrize("alpha_p, alpha_n", [(0.3, 0.1), (0.5, 1.0), (1.0, 0.5)])
def test_xielu(alpha_p, alpha_n, device):
    torch_input = torch.rand([4, 4], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn.xielu)
    torch_output = golden_fn(torch_input, alpha_p=alpha_p, alpha_n=alpha_n)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.xielu(ttnn_input, alpha_p=alpha_p, alpha_n=alpha_n)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_ulp(torch_output, ttnn_output, 1)


def test_xielu_golden(device):
    torch_input_tensor = torch.rand([4, 4], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn.xielu)
    torch_output_tensor = golden_fn(torch_input_tensor, alpha_p=0.8, alpha_n=0.8)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.xielu(input_tensor, alpha_p=0.8, alpha_n=0.8)
    output = ttnn.to_torch(output)
    assert_with_ulp(torch_output_tensor, output, 1)
