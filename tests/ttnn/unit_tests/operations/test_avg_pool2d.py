import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc


def run_once(device, input_shape, kernel_size, stride, padding, dilation):
    batch_size, in_c, in_h, in_w = input_shape
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, in_c))
    input_tensor = ttnn.from_torch(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.avg_pool2d(input_tensor, batch_size, in_h, in_w, in_c, kernel_size, stride, padding, dilation)

    expected_output = torch.nn.functional.avg_pool2d(
        torch_input, kernel_size, stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None
    )
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = torch.reshape(output_tensor, expected_output.shape)
    assert_with_pcc(expected_output, output_tensor, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4096}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, dilation",
    [
        ((1, 128, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1)),
        ((1, 256, 28, 28), (2, 2), (2, 2), (0, 0), (1, 1)),
        ((1, 192, 56, 56), (2, 2), (2, 2), (0, 0), (1, 1)),
        ((1, 160, 7, 7), (2, 2), (2, 2), (0, 0), (1, 1)),
        ((1, 256, 56, 56), (13, 13), (1, 1), (0, 0), (1, 1)),
        pytest.param((1, 512, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="low pcc")),
        pytest.param(
            (1, 384, 28, 28),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0"),
        ),
        pytest.param(
            (1, 1056, 14, 14),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0"),
        ),
        pytest.param(
            (1, 640, 14, 14),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0"),
        ),
        pytest.param(
            (1, 896, 14, 14),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0"),
        ),
        pytest.param(
            (1, 24, 56, 56),
            (2, 2),
            (2, 2),
            (0, 0),
            (1, 1),
            marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0"),
        ),
        pytest.param((1, 40, 28, 28), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="low pcc")),
        pytest.param((1, 80, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="low pcc")),
        pytest.param((1, 112, 14, 14), (2, 2), (2, 2), (0, 0), (1, 1), marks=pytest.mark.xfail(reason="low pcc")),
        pytest.param(
            (1, 384, 35, 35),
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            marks=pytest.mark.xfail(reason="in_ntiles_c % MAX_TILES_PER_REDUCTION != 0"),
        ),
        pytest.param((1, 1024, 17, 17), (3, 3), (1, 1), (1, 1), (1, 1), marks=pytest.mark.xfail(reason="low pcc")),
        pytest.param((1, 1536, 8, 8), (3, 3), (1, 1), (1, 1), (1, 1), marks=pytest.mark.xfail(reason="low pcc")),
    ],
)
def test_avg_pool2d(device, input_shape, kernel_size, stride, padding, dilation):
    run_once(device, input_shape, kernel_size, stride, padding, dilation)
