# Warp Decode for Kimi K2.6

This repository packages the Kimi K2.6 warp-decode implementation work, the SGLang/FlashInfer patch sets, and the latest verified 8x B200 benchmark artifacts.

This work is based on the Warp Decode idea described in Cursor's technical blog post: [Warp Decode: Faster MoE inference at batch size one](https://cursor.com/blog/warp-decode). Credit for the core row-major warp-decode concept goes to the Cursor team; this repository is an independent Kimi K2.6/SGLang/FlashInfer implementation and benchmark package.

The implementation has three benchmark variants:

- `sota`: regular prefill kernel and regular FlashInfer/TRTLLM decode MoE kernel.
- `warp-decode`: regular prefill kernel and Cursor-style row-major warp decode MoE kernel.
- `warp-decode-tiled-down`: regular prefill kernel and custom tiled `W_down` warp decode MoE kernel.

Important status: the latest verified B200 run showed that the row-major and tiled warp decode kernels were active during decode, but they were slower than the FlashInfer/TRTLLM baseline for this TP=8 Kimi K2.6 setup. The current patches include a later internal-topk integration follow-up that still needs fresh GPU validation.

## Latest Verified Results

Setup: 8x B200, `moonshotai/Kimi-K2.6`, INT4, TP=8, EP=1, random dataset, ISL=1024, OSL=1024, concurrency 1 and 4.

| Variant | Prefill MoE backend | Decode MoE backend | Concurrency | Output tok/s total | Output tok/s/GPU | Runtime verified | CUDA active markers | Fallback markers |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `sota` | `flashinfer_trtllm` | `flashinfer_trtllm` | 1 | 146.82 | 18.35 | false | 0 | 0 |
| `sota` | `flashinfer_trtllm` | `flashinfer_trtllm` | 4 | 500.61 | 62.58 | false | 0 | 0 |
| `warp-decode` | `flashinfer_trtllm` | `flashinfer_kimi_warp_decode` | 1 | 80.66 | 10.08 | true | 8 | 16 |
| `warp-decode` | `flashinfer_trtllm` | `flashinfer_kimi_warp_decode` | 4 | 218.99 | 27.37 | true | 8 | 16 |
| `warp-decode-tiled-down` | `flashinfer_trtllm` | `flashinfer_kimi_warp_decode_tiled_down` | 1 | 79.73 | 9.97 | true | 8 | 16 |
| `warp-decode-tiled-down` | `flashinfer_trtllm` | `flashinfer_kimi_warp_decode_tiled_down` | 4 | 216.81 | 27.10 | true | 8 | 16 |

Full result artifacts are in `results/b200_context_fixed/`.

## Custom Tiled `W_down` Variant

The `warp-decode-tiled-down` path is a custom extension on top of the Cursor-style row-major warp decode implementation. The intent is to address one of the main remaining decode bottlenecks for routed MoE: after the router selects `TOP_K=8` experts, the down projection has to read weight rows for several different experts. In a normal expert-major layout, those reads can jump across large expert matrices, which is bad for cache locality and memory transaction efficiency during batch-size-one or low-concurrency decode.

The custom path keeps `W_gate` and `W_up` in their existing expert-major format, but gives only the MoE down-projection weights a decode-specific layout:

```text
W_down_decode[output_hidden_tile][intermediate_tile][expert][packed_tile_payload]
```

The intuition is that a warp computing a fixed output-hidden tile walks through the intermediate dimension and needs the same routed experts for every intermediate tile. By placing the selected experts' down-projection tiles close together for each `(output_hidden_tile, intermediate_tile)` pair, the kernel can read the 8 routed expert tiles from a compact neighborhood instead of bouncing across full expert-major matrices.

Implementation details in this repository:

- Only `W_down` gets the tiled decode layout; `W_gate` and `W_up` stay in the normal format.
- Each expert tile's packed INT4 payload remains internally contiguous; the layout is expert-interleaved at tile granularity, not scalar-interleaved across experts.
- Tile addressing is designed around a fixed base for each `(output_hidden_tile, intermediate_tile)`, so expert addresses are computed as `tile_group_base + expert_id * expert_tile_stride`.
- The decode kernel is specialized for Kimi K2.6's fixed routed `TOP_K=8` and unrolls the routed expert accumulation.
- The implementation uses the same MXINT4 group-size assumption as the Kimi K2.6 serving path used during this work, with `group_size=32`.
- The path avoids runtime token grouping, padding-based dispatch, scatter/combine buffers, and hot-path repacking during decode; the extra layout work is intended to happen before serving/decode execution.
- The B200 ptxas validation included in `results/b200_context_fixed/ptxas_checks/` showed no local-memory spills for the tiled down kernel.

This was expected to improve on plain row-major warp decode by reducing the memory-latency penalty from non-contiguous routed expert accesses in `W_down`. In the latest verified B200 end-to-end result, however, the tiled path did not improve throughput over the row-major warp-decode path, so the current implementation should be treated as an experimental layout idea rather than a validated speedup.

## Contents

- `patches/sglang-kimi-k26-warp-decode.patch`: SGLang integration patch.
- `patches/flashinfer-trtllmgen-moe-kimi-k26-warp-decode.patch`: FlashInfer/TRTLLM-gen MoE kernel patch.
- `overlays/`: untracked/new files from the working trees that are not included in normal `git diff` output.
- `results/b200_context_fixed/`: latest verified 8x B200 result set with runtime marker validation.
- `docs/results.md`: concise result table and interpretation.

## Base Revisions

The patches were produced against these local base revisions:

- SGLang: `714173555c1e9bbc82b2daeeb404d3acfdc83083`
- FlashInfer fork checkout: `64ed071e23bf8d5d2d5af5c91577e5b8e036a1cf`

## Benchmark Command

The intended InferenceX-style benchmark command was:

```bash
python3 benchmark/kimi_k26_warp_decode/bench_kimi_k26_e2e.py \
  --variant both \
  --inferencex-dir /Users/aumdesai/4kernels/inferencex_ref \
  --concurrency 1,4 \
  --input-len 1024 \
  --output-len 1024 \
  --output-dir kimi_k26_inferencex_budget
```

On the B200 node this was run against `moonshotai/Kimi-K2.6` with 8 GPUs, INT4, TP=8, EP=1.

## Applying Patches

From clean checkouts matching the base revisions:

```bash
cd /path/to/sglang
git apply /path/to/warp-decode-kimi-k26/patches/sglang-kimi-k26-warp-decode.patch
cp -R /path/to/warp-decode-kimi-k26/overlays/sglang/. .

cd /path/to/flashinfer-trtllmgen-moe
git apply /path/to/warp-decode-kimi-k26/patches/flashinfer-trtllmgen-moe-kimi-k26-warp-decode.patch
cp -R /path/to/warp-decode-kimi-k26/overlays/flashinfer-trtllmgen-moe/. .
```

## Current Technical Read

The implementation is not currently a validated reproduction of Cursor's reported speedups. In the latest verified 8x B200 run, the active row-major and tiled warp decode implementations underperformed the FlashInfer/TRTLLM baseline.

It is currently unclear why this implementation is so much slower than the behavior reported in Cursor's Warp Decode blog. Plausible explanations include:

- The Kimi K2.6 INT4/TP=8 shape may interact very differently with warp-level decomposition than the model, quantization format, and sharding setup Cursor benchmarked.
- The FlashInfer/TRTLLM baseline used here may already avoid enough practical routing overhead, or may use stronger fused/persistent MoE scheduling, making it a harder baseline than the one assumed by this implementation.
- The current row-major warp kernels may be functionally close to the blog design but not performance-equivalent at the CUDA instruction/memory-transaction level.
- The implementation still materializes intermediate gate/up outputs and top-k state, so any residual memory traffic or launch/capture overhead may erase the intended decode benefit.
- The local TP shard size, shared-expert behavior, MXINT4 scale layout, and W2 access pattern may be leaving the active warp path memory-inefficient despite having no ptxas spills.

The next debugging target should be an isolated MoE-layer benchmark comparing FlashInfer/TRTLLM MxINT4 MoE versus the row-major warp decode path, with kernel timing split across top-k, gate/up, and down projection.
