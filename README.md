# Warp Decode for Kimi K2.6

This repository packages the Kimi K2.6 warp-decode implementation work, the SGLang/FlashInfer patch sets, and the latest verified 8x B200 benchmark artifacts.

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

The next debugging target should be an isolated MoE-layer benchmark comparing FlashInfer/TRTLLM MxINT4 MoE versus the row-major warp decode path, with kernel timing split across top-k, gate/up, and down projection.
