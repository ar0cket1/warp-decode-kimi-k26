import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("FLASHINFER_RUN_KIMI_K26_WARP_DECODE_TESTS") != "1",
    reason="Kimi K2.6 warp-decode tests allocate full 384-expert tensors; set "
    "FLASHINFER_RUN_KIMI_K26_WARP_DECODE_TESTS=1 on a Blackwell GPU to run.",
)


HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 256
NUM_EXPERTS = 384
TOP_K = 8
GROUP_SIZE = 32
DOWN_INTERMEDIATE_TILE = 256


def _require_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, _ = torch.cuda.get_device_capability()
    if major not in (10, 12) and os.environ.get("SGLANG_KIMI_WARP_DECODE_ALLOW_HOPPER") != "1":
        pytest.skip("Blackwell-class CUDA device is required")


def _signed_int4_packed(shape, device):
    return torch.randint(
        0, 256, (*shape[:-1], shape[-1] // 2), dtype=torch.uint8, device=device
    )


def _dequant_row(packed_row, scale_row, logical_size):
    nibbles = torch.empty(logical_size, dtype=torch.int8, device=packed_row.device)
    low = packed_row & 0x0F
    high = (packed_row >> 4) & 0x0F
    nibbles[0::2] = low.to(torch.int8)
    nibbles[1::2] = high.to(torch.int8)
    nibbles = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    scales = scale_row.repeat_interleave(GROUP_SIZE)
    return nibbles.to(torch.float32) * scales.to(torch.float32)


def _make_inputs(batch_size):
    device = torch.device("cuda")
    torch.manual_seed(1234 + batch_size)
    hidden_states = torch.randn(
        batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    )
    routing_logits = torch.full(
        (batch_size, NUM_EXPERTS), -20.0, dtype=torch.float32, device=device
    )
    selected = torch.arange(TOP_K, device=device)
    routing_logits[:, selected] = torch.arange(TOP_K, 0, -1, device=device).float()
    routing_bias = torch.zeros(NUM_EXPERTS, dtype=torch.bfloat16, device=device)

    w13 = _signed_int4_packed((NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE), device)
    w13_scale = torch.rand(
        NUM_EXPERTS,
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE // GROUP_SIZE,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.001
    w2 = _signed_int4_packed((NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), device)
    w2_scale = torch.rand(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE // GROUP_SIZE,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.001
    return hidden_states, routing_logits, routing_bias, w13, w13_scale, w2, w2_scale


def _reference_selected_dims(
    hidden_states, routing_logits, routing_bias, w13, w13_scale, w2, w2_scale, dims
):
    scores = routing_logits.sigmoid()
    _, topk_ids = torch.topk(scores + routing_bias.float(), k=TOP_K, dim=-1)
    topk_weights = scores.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    ref = torch.empty(
        hidden_states.shape[0], len(dims), dtype=torch.float32, device=hidden_states.device
    )
    x = hidden_states.float()
    for token in range(hidden_states.shape[0]):
        interm = torch.empty(
            TOP_K, INTERMEDIATE_SIZE, dtype=torch.float32, device=hidden_states.device
        )
        for route in range(TOP_K):
            expert = int(topk_ids[token, route])
            for neuron in range(INTERMEDIATE_SIZE):
                gate = _dequant_row(
                    w13[expert, neuron], w13_scale[expert, neuron], HIDDEN_SIZE
                )
                up = _dequant_row(
                    w13[expert, INTERMEDIATE_SIZE + neuron],
                    w13_scale[expert, INTERMEDIATE_SIZE + neuron],
                    HIDDEN_SIZE,
                )
                gate_dot = torch.dot(x[token], gate)
                up_dot = torch.dot(x[token], up)
                interm[route, neuron] = torch.nn.functional.silu(gate_dot) * up_dot

        for out_i, dim in enumerate(dims):
            acc = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
            for route in range(TOP_K):
                expert = int(topk_ids[token, route])
                down = _dequant_row(
                    w2[expert, dim], w2_scale[expert, dim], INTERMEDIATE_SIZE
                )
                acc += topk_weights[token, route] * torch.dot(interm[route], down)
            ref[token, out_i] = acc
    return ref


def _topk_from_logits(routing_logits, routing_bias):
    scores = routing_logits.sigmoid()
    _, topk_ids = torch.topk(scores + routing_bias.float(), k=TOP_K, dim=-1)
    topk_weights = scores.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids.to(torch.int32), topk_weights.to(torch.float32)


def _prepare_tiled_down_weights(w2, w2_scale):
    num_experts, hidden_size, packed_intermediate = w2.shape
    intermediate = packed_intermediate * 2
    num_intermediate_tiles = intermediate // DOWN_INTERMEDIATE_TILE
    tile_payload_bytes = DOWN_INTERMEDIATE_TILE // 2
    scale_groups_per_tile = DOWN_INTERMEDIATE_TILE // GROUP_SIZE
    w2_tiled = (
        w2.view(num_experts, hidden_size, num_intermediate_tiles, tile_payload_bytes)
        .permute(1, 2, 0, 3)
        .contiguous()
    )
    w2_scale_tiled = (
        w2_scale.view(
            num_experts,
            hidden_size,
            num_intermediate_tiles,
            scale_groups_per_tile,
        )
        .permute(1, 2, 0, 3)
        .contiguous()
    )
    return w2_tiled, w2_scale_tiled


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64])
def test_kimi_k26_mxint4_warp_decode_selected_dims(batch_size):
    _require_blackwell()
    from flashinfer.fused_moe import kimi_k26_mxint4_warp_decode_moe_with_topk

    inputs = _make_inputs(batch_size)
    hidden_states, routing_logits, routing_bias, w13, w13_scale, w2, w2_scale = inputs
    topk_ids, topk_weights = _topk_from_logits(routing_logits, routing_bias)
    output = kimi_k26_mxint4_warp_decode_moe_with_topk(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states=hidden_states,
        gemm1_weights=w13,
        gemm1_weights_scale=w13_scale,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_scale,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
    )[0]
    torch.cuda.synchronize()

    dims = [0, 1, 17, 255, 1024, HIDDEN_SIZE - 1]
    ref = _reference_selected_dims(
        hidden_states, routing_logits, routing_bias, w13, w13_scale, w2, w2_scale, dims
    )
    actual = output[:, dims].float()
    assert (
        torch.nn.functional.cosine_similarity(actual.flatten(), ref.flatten(), dim=0)
        >= 0.999996
    )
    assert torch.max(torch.abs(actual - ref)) <= 0.001953


@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
def test_kimi_k26_tiled_down_matches_row_major_down_full_output(batch_size):
    _require_blackwell()
    from flashinfer.fused_moe import (
        kimi_k26_mxint4_warp_decode_moe_with_topk,
        kimi_k26_mxint4_warp_decode_tiled_down_moe_with_topk,
    )

    inputs = _make_inputs(batch_size)
    hidden_states, routing_logits, routing_bias, w13, w13_scale, w2, w2_scale = inputs
    topk_ids, topk_weights = _topk_from_logits(routing_logits, routing_bias)
    w2_tiled, w2_scale_tiled = _prepare_tiled_down_weights(w2, w2_scale)

    row_major_output = kimi_k26_mxint4_warp_decode_moe_with_topk(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states=hidden_states,
        gemm1_weights=w13,
        gemm1_weights_scale=w13_scale,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_scale,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
    )[0]
    tiled_output = kimi_k26_mxint4_warp_decode_tiled_down_moe_with_topk(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states=hidden_states,
        gemm1_weights=w13,
        gemm1_weights_scale=w13_scale,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w2_tiled,
        gemm2_weights_scale=w2_scale_tiled,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
    )[0]
    torch.cuda.synchronize()

    actual = tiled_output.float()
    expected = row_major_output.float()
    assert (
        torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        >= 0.999996
    )
    assert torch.max(torch.abs(actual - expected)) <= 0.001953


@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
def test_kimi_k26_internal_topk_matches_precomputed_topk_full_output(batch_size):
    _require_blackwell()
    from flashinfer.fused_moe import (
        kimi_k26_mxint4_warp_decode_moe,
        kimi_k26_mxint4_warp_decode_moe_with_topk,
    )

    inputs = _make_inputs(batch_size)
    hidden_states, routing_logits, routing_bias, w13, w13_scale, w2, w2_scale = inputs
    topk_ids, topk_weights = _topk_from_logits(routing_logits, routing_bias)

    precomputed_output = kimi_k26_mxint4_warp_decode_moe_with_topk(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states=hidden_states,
        gemm1_weights=w13,
        gemm1_weights_scale=w13_scale,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_scale,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
    )[0]
    internal_output = kimi_k26_mxint4_warp_decode_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        gemm1_weights=w13,
        gemm1_weights_scale=w13_scale,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_scale,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=2,
    )[0]
    torch.cuda.synchronize()

    actual = internal_output.float()
    expected = precomputed_output.float()
    assert (
        torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        >= 0.999996
    )
    assert torch.max(torch.abs(actual - expected)) <= 0.001953


@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
def test_kimi_k26_tiled_down_internal_topk_matches_row_major_internal_topk(batch_size):
    _require_blackwell()
    from flashinfer.fused_moe import (
        kimi_k26_mxint4_warp_decode_moe,
        kimi_k26_mxint4_warp_decode_tiled_down_moe,
    )

    inputs = _make_inputs(batch_size)
    hidden_states, routing_logits, routing_bias, w13, w13_scale, w2, w2_scale = inputs
    w2_tiled, w2_scale_tiled = _prepare_tiled_down_weights(w2, w2_scale)

    row_major_output = kimi_k26_mxint4_warp_decode_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        gemm1_weights=w13,
        gemm1_weights_scale=w13_scale,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_scale,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=2,
    )[0]
    tiled_output = kimi_k26_mxint4_warp_decode_tiled_down_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        gemm1_weights=w13,
        gemm1_weights_scale=w13_scale,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w2_tiled,
        gemm2_weights_scale=w2_scale_tiled,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=2,
    )[0]
    torch.cuda.synchronize()

    actual = tiled_output.float()
    expected = row_major_output.float()
    assert (
        torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        >= 0.999996
    )
    assert torch.max(torch.abs(actual - expected)) <= 0.001953
