from importlib import import_module

import pytest
import torch

try:
    CompressedTensorsMxInt4MoE = import_module(
        "sglang.srt.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a4_mxint4_moe"
    ).CompressedTensorsMxInt4MoE
except ImportError as exc:
    pytest.skip(
        f"compressed-tensors MxInt4 MoE import unavailable: {exc}",
        allow_module_level=True,
    )


def _pack_compressed_tensors_int32(values: torch.Tensor) -> torch.Tensor:
    assert values.dtype == torch.int8
    assert values.dim() == 2
    assert values.shape[1] % 8 == 0
    shifted = (values.to(torch.int32) + 8) & 0x0F
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    return (
        (shifted.reshape(values.shape[0], -1, 8) << shifts)
        .sum(dim=-1)
        .to(torch.int32)
    )


def _unpack_signed_uint8(packed: torch.Tensor) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    out = torch.empty(packed.shape[0], packed.shape[1] * 2, dtype=torch.int8)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return torch.where(out >= 8, out - 16, out)


def test_prepare_static_weights_for_warp_decode_swaps_flashinfer_w13_order():
    scheme = object.__new__(CompressedTensorsMxInt4MoE)

    up_rows = torch.tensor(
        [[-8, -7, -6, -5, -4, -3, -2, -1], [0, 1, 2, 3, 4, 5, 6, 7]],
        dtype=torch.int8,
    )
    gate_rows = torch.tensor(
        [[7, 6, 5, 4, 3, 2, 1, 0], [-1, -2, -3, -4, -5, -6, -7, -8]],
        dtype=torch.int8,
    )
    # FusedMoE's FlashInfer TRTLLM load path stores w13 as [up, gate].
    w13_loaded = torch.cat([up_rows, gate_rows], dim=0).unsqueeze(0)
    w2_loaded = torch.tensor(
        [[1, 2, 3, 4, -1, -2, -3, -4]], dtype=torch.int8
    ).unsqueeze(0)
    w13_int32 = _pack_compressed_tensors_int32(w13_loaded[0]).unsqueeze(0)
    w2_int32 = _pack_compressed_tensors_int32(w2_loaded[0]).unsqueeze(0)
    w13_scales = torch.arange(4, dtype=torch.float32).view(1, 4, 1).to(torch.bfloat16)
    w2_scales = torch.ones(1, 1, 1, dtype=torch.bfloat16)

    w13_warp, w13_scales_warp, w2_warp, _ = (
        scheme.prepare_static_weights_for_warp_decode(
            w13_int32,
            w2_int32,
            w13_scales,
            w2_scales,
            num_experts=1,
        )
    )

    assert torch.equal(
        _unpack_signed_uint8(w13_warp[0]), torch.cat([gate_rows, up_rows], dim=0)
    )
    assert torch.equal(
        w13_scales_warp[0].float(), torch.tensor([[2.0], [3.0], [0.0], [1.0]])
    )
    assert torch.equal(_unpack_signed_uint8(w2_warp[0]), w2_loaded[0])


def test_prepare_tiled_down_weights_keeps_each_expert_tile_contiguous():
    scheme = object.__new__(CompressedTensorsMxInt4MoE)
    scheme.group_size = 32

    w2 = torch.arange(2 * 2 * 128, dtype=torch.uint8).view(2, 2, 128)
    w2_scales = torch.arange(2 * 2 * 8, dtype=torch.float32).view(2, 2, 8).to(
        torch.bfloat16
    )

    w2_tiled, w2_scales_tiled = scheme.prepare_tiled_down_weights_for_warp_decode(
        w2, w2_scales
    )

    assert w2_tiled.shape == (2, 1, 2, 128)
    assert w2_scales_tiled.shape == (2, 1, 2, 8)
    assert w2_tiled.stride() == (256, 256, 128, 1)
    assert w2_scales_tiled.stride() == (16, 16, 8, 1)
    assert torch.equal(w2_tiled[0, 0, 0], w2[0, 0])
    assert torch.equal(w2_tiled[0, 0, 1], w2[1, 0])
    assert torch.equal(w2_tiled[1, 0, 0], w2[0, 1])
    assert torch.equal(w2_scales_tiled[0, 0, 0], w2_scales[0, 0])
    assert torch.equal(w2_scales_tiled[0, 0, 1], w2_scales[1, 0])


def test_prepare_tiled_down_weights_rejects_unverified_group_size():
    scheme = object.__new__(CompressedTensorsMxInt4MoE)
    scheme.group_size = 64

    w2 = torch.arange(2 * 2 * 128, dtype=torch.uint8).view(2, 2, 128)
    w2_scales = torch.arange(2 * 2 * 4, dtype=torch.float32).view(2, 2, 4).to(
        torch.bfloat16
    )

    with pytest.raises(AssertionError, match="group size 32"):
        scheme.prepare_tiled_down_weights_for_warp_decode(w2, w2_scales)
