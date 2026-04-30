#!/usr/bin/env python3
"""Microbenchmark for the Kimi K2.6 MxInt4 warp-decode MoE API.

Run on a Blackwell GPU with a FlashInfer build that includes
``kimi_k26_mxint4_warp_decode_moe``.

``warp-decode`` includes routing, gate/up, and down. ``warp-decode-compute``
is the two-kernel compute-only path with precomputed top-k and should not be
compared directly against TRTLLM end-to-end MoE latency. The
``warp-decode-tiled-down-compute`` backend keeps gate/up row-major and uses the
decode-only tiled W_down layout.
"""

import argparse
import os
import time

import torch

from flashinfer.autotuner import autotune
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe import (
    convert_to_block_layout,
    kimi_k26_mxint4_warp_decode_moe,
    kimi_k26_mxint4_warp_decode_moe_with_topk,
    kimi_k26_mxint4_warp_decode_tiled_down_moe_with_topk,
    trtllm_mxint4_block_scale_moe,
)
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)


HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 256
NUM_EXPERTS = 384
TOP_K = 8
GROUP_SIZE = 32
DOWN_INTERMEDIATE_TILE = 256
DOWN_EXPERT_TILE_STRIDE_BYTES = 128
TUNE_MAX_NUM_TOKENS = 4096
ROUTED_SCALING_FACTOR = 2.827


def packed_int4(shape, device):
    return torch.randint(
        0, 256, (*shape[:-1], shape[-1] // 2), dtype=torch.uint8, device=device
    )


def make_case(batch_size, device):
    torch.manual_seed(2026 + batch_size)
    hidden_states = torch.randn(
        batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    )
    routing_logits = torch.randn(
        batch_size, NUM_EXPERTS, dtype=torch.float32, device=device
    )
    routing_bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device=device) * 0.01
    w13_warp = packed_int4((NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE), device)
    w13_scale = torch.rand(
        NUM_EXPERTS,
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE // GROUP_SIZE,
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = packed_int4((NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), device)
    w2_scale = torch.rand(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE // GROUP_SIZE,
        dtype=torch.bfloat16,
        device=device,
    )
    output = torch.empty(
        batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    )
    # TRTLLM-Gen's fused-gate helper expects the SGLang-loaded [up, gate] order,
    # while the warp-decode API uses logical [gate, up] rows for direct streaming.
    w13_trtllm = torch.cat(
        [w13_warp[:, INTERMEDIATE_SIZE:], w13_warp[:, :INTERMEDIATE_SIZE]], dim=1
    ).contiguous()
    w13_scale_trtllm = torch.cat(
        [
            w13_scale[:, INTERMEDIATE_SIZE:],
            w13_scale[:, :INTERMEDIATE_SIZE],
        ],
        dim=1,
    ).contiguous()
    trtllm_static = prepare_trtllm_weights(w13_trtllm, w13_scale_trtllm, w2, w2_scale)
    w2_tiled, w2_scale_tiled = prepare_tiled_down_weights(w2, w2_scale)
    topk_ids, topk_weights = topk_from_logits(routing_logits, routing_bias)
    return {
        "hidden_states": hidden_states,
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "w13_warp": w13_warp,
        "w13_scale_warp": w13_scale,
        "w2_warp": w2,
        "w2_scale_warp": w2_scale,
        "w2_tiled_down": w2_tiled,
        "w2_scale_tiled_down": w2_scale_tiled,
        "trtllm_static": trtllm_static,
        "output": output,
    }


def topk_from_logits(routing_logits, routing_bias):
    scores = routing_logits.sigmoid()
    _, topk_ids = torch.topk(scores + routing_bias.float(), k=TOP_K, dim=-1)
    topk_weights = scores.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * ROUTED_SCALING_FACTOR
    return topk_ids.to(torch.int32), topk_weights.to(torch.float32)


def prepare_trtllm_weights(w13, w13_scale, w2, w2_scale):
    epilogue_tile_m = 128
    block_k = 128
    cache = {}
    gemm1_weights = []
    gemm1_scales = []
    gemm2_weights = []
    gemm2_scales = []
    for expert in range(NUM_EXPERTS):
        w13_expert = w13[expert].view(torch.uint8)
        w13_perm = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13_expert, epilogue_tile_m
        )
        gemm1_weights.append(
            convert_to_block_layout(
                w13_expert[w13_perm.to(w13.device)].contiguous(), block_k
            )
        )

        w13_scale_expert = w13_scale[expert].view(torch.bfloat16)
        w13_scale_perm = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13_scale_expert, epilogue_tile_m, num_elts_per_sf=32
        )
        gemm1_scales.append(
            block_scale_interleave(
                w13_scale_expert[w13_scale_perm.to(w13_scale.device)].contiguous()
            )
        )

        w2_expert = w2[expert].view(torch.uint8)
        w2_perm = get_w2_permute_indices_with_cache(cache, w2_expert, epilogue_tile_m)
        gemm2_weights.append(
            convert_to_block_layout(
                w2_expert[w2_perm.to(w2.device)].contiguous(), block_k
            )
        )

        w2_scale_expert = w2_scale[expert].view(torch.bfloat16)
        w2_scale_perm = get_w2_permute_indices_with_cache(
            cache, w2_scale_expert, epilogue_tile_m, num_elts_per_sf=16
        )
        gemm2_scales.append(
            block_scale_interleave(
                w2_scale_expert[w2_scale_perm.to(w2_scale.device)].contiguous()
            )
        )

    return {
        "gemm1_weights": torch.stack(gemm1_weights),
        "gemm1_scales": torch.stack(gemm1_scales).view(torch.bfloat16),
        "gemm2_weights": torch.stack(gemm2_weights),
        "gemm2_scales": torch.stack(gemm2_scales).view(torch.bfloat16),
    }


def prepare_tiled_down_weights(w2, w2_scale):
    num_experts, hidden_size, packed_intermediate = w2.shape
    intermediate = packed_intermediate * 2
    num_intermediate_tiles = intermediate // DOWN_INTERMEDIATE_TILE
    tile_payload_bytes = DOWN_INTERMEDIATE_TILE // 2
    scale_groups_per_tile = DOWN_INTERMEDIATE_TILE // GROUP_SIZE
    assert tile_payload_bytes == DOWN_EXPERT_TILE_STRIDE_BYTES
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


def run_warp_decode_compute(case):
    return kimi_k26_mxint4_warp_decode_moe_with_topk(
        topk_ids=case["topk_ids"],
        topk_weights=case["topk_weights"],
        hidden_states=case["hidden_states"],
        gemm1_weights=case["w13_warp"],
        gemm1_weights_scale=case["w13_scale_warp"],
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=case["w2_warp"],
        gemm2_weights_scale=case["w2_scale_warp"],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        output=case["output"],
    )[0]


def run_warp_decode_tiled_down_compute(case):
    return kimi_k26_mxint4_warp_decode_tiled_down_moe_with_topk(
        topk_ids=case["topk_ids"],
        topk_weights=case["topk_weights"],
        hidden_states=case["hidden_states"],
        gemm1_weights=case["w13_warp"],
        gemm1_weights_scale=case["w13_scale_warp"],
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=case["w2_tiled_down"],
        gemm2_weights_scale=case["w2_scale_tiled_down"],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        output=case["output"],
    )[0]


def run_warp_decode(case):
    return kimi_k26_mxint4_warp_decode_moe(
        routing_logits=case["routing_logits"],
        routing_bias=case["routing_bias"],
        hidden_states=case["hidden_states"],
        gemm1_weights=case["w13_warp"],
        gemm1_weights_scale=case["w13_scale_warp"],
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=case["w2_warp"],
        gemm2_weights_scale=case["w2_scale_warp"],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        routing_method_type=2,
        output=case["output"],
    )[0]


def run_trtllm(case):
    static = case["trtllm_static"]
    with autotune(True):
        return trtllm_mxint4_block_scale_moe(
            routing_logits=case["routing_logits"],
            routing_bias=case["routing_bias"],
            hidden_states=case["hidden_states"],
            gemm1_weights=static["gemm1_weights"],
            gemm1_weights_scale=static["gemm1_scales"],
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=static["gemm2_weights"],
            gemm2_weights_scale=static["gemm2_scales"],
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            n_group=1,
            topk_group=1,
            intermediate_size=INTERMEDIATE_SIZE,
            local_expert_offset=0,
            local_num_experts=NUM_EXPERTS,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            routing_method_type=2,
            output=case["output"],
            tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
        )[0]


def estimate_weight_bytes(batch_size):
    w13_bytes_per_route = 2 * INTERMEDIATE_SIZE * (HIDDEN_SIZE // 2)
    w13_scale_bytes_per_route = (
        2 * INTERMEDIATE_SIZE * (HIDDEN_SIZE // GROUP_SIZE) * 2
    )
    w2_bytes_per_route = HIDDEN_SIZE * (INTERMEDIATE_SIZE // 2)
    w2_scale_bytes_per_route = HIDDEN_SIZE * (INTERMEDIATE_SIZE // GROUP_SIZE) * 2
    return (
        batch_size
        * TOP_K
        * (
            w13_bytes_per_route
            + w13_scale_bytes_per_route
            + w2_bytes_per_route
            + w2_scale_bytes_per_route
        )
    )


def estimate_warp_intermediate_bytes(batch_size):
    intermediate_values = batch_size * TOP_K * INTERMEDIATE_SIZE
    return intermediate_values * 2 * 2  # bf16 write in gate/up, bf16 read in down


def measure(name, batch_size, case, fn, warmup, iters, kernel_count, notes):
    for _ in range(warmup):
        fn(case)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(case)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0 / iters
    tokens_s = batch_size * 1000.0 / latency_ms
    weight_bandwidth_gbs = estimate_weight_bytes(batch_size) / (latency_ms / 1000.0) / 1e9
    intermediate_mb = estimate_warp_intermediate_bytes(batch_size) / 1e6
    print(
        f"{name},{batch_size},{latency_ms:.4f},{tokens_s:.2f},"
        f"{weight_bandwidth_gbs:.2f},{kernel_count},{intermediate_mb:.2f},{notes}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=[
            "trtllm",
            "warp-decode",
            "warp-decode-compute",
            "warp-decode-tiled-down-compute",
        ],
        default=[
            "trtllm",
            "warp-decode",
            "warp-decode-compute",
            "warp-decode-tiled-down-compute",
        ],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, _ = torch.cuda.get_device_capability()
    if major not in (10, 12) and os.environ.get("SGLANG_KIMI_WARP_DECODE_ALLOW_HOPPER") != "1":
        raise RuntimeError("Blackwell-class CUDA device is required")

    device = torch.device("cuda")
    print(
        "backend,batch_size,latency_ms,tokens_per_s,"
        "effective_weight_bandwidth_gbs,kernel_count,intermediate_traffic_mb,notes"
    )
    for batch_size in args.batch_sizes:
        case = make_case(batch_size, device)
        if "trtllm" in args.backends:
            measure(
                "trtllm",
                batch_size,
                case,
                run_trtllm,
                args.warmup,
                args.iters,
                "baseline",
                "routing+packed_gemm+finalize",
            )
        if "warp-decode" in args.backends:
            measure(
                "warp-decode",
                batch_size,
                case,
                run_warp_decode,
                args.warmup,
                args.iters,
                3,
                "routing+gate_up+down",
            )
        if "warp-decode-compute" in args.backends:
            measure(
                "warp-decode-compute",
                batch_size,
                case,
                run_warp_decode_compute,
                args.warmup,
                args.iters,
                2,
                "precomputed_topk+gate_up+down",
            )
        if "warp-decode-tiled-down-compute" in args.backends:
            measure(
                "warp-decode-tiled-down-compute",
                batch_size,
                case,
                run_warp_decode_tiled_down_compute,
                args.warmup,
                args.iters,
                2,
                "precomputed_topk+gate_up+tiled_down",
            )
        del case
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
