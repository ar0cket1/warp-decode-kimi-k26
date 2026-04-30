# Warp Decode for Kimi K2.6

This repository packages the Kimi K2.6 warp-decode implementation work, the SGLang/FlashInfer patch sets, and the reported 8x B200 benchmark artifacts.

The implementation has three benchmark variants:

- `sota`: regular prefill kernel and regular FlashInfer/TRTLLM decode MoE kernel.
- `warp-decode`: regular prefill kernel and Cursor-style row-major warp decode MoE kernel.
- `warp-decode-tiled-down`: regular prefill kernel and custom tiled `W_down` warp decode MoE kernel.

Important status: the B200 run with active CUDA/Python markers showed that the row-major and tiled warp decode kernels were actually active during decode, but they were slower than the FlashInfer/TRTLLM baseline for this TP=8 Kimi K2.6 setup. The current patches include a later internal-topk integration follow-up that still needs fresh GPU validation.

## Contents

- `patches/sglang-kimi-k26-warp-decode.patch`: SGLang integration patch.
- `patches/flashinfer-trtllmgen-moe-kimi-k26-warp-decode.patch`: FlashInfer/TRTLLM-gen MoE kernel patch.
- `overlays/`: untracked/new files from the working trees that are not included in normal `git diff` output.
- `results/b200_initial/`: initial 8x B200 result set where warp variants appeared nearly identical to SOTA.
- `results/b200_context_fixed/`: context-fixed 8x B200 result set with runtime marker validation.
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

The implementation is not currently a validated reproduction of Cursor's reported speedups. The active-kernel B200 run is useful because it rules out the earlier "same kernel accidentally used for every variant" explanation for the context-fixed result.

The next debugging target should be an isolated MoE-layer benchmark comparing FlashInfer/TRTLLM MxINT4 MoE versus the row-major warp decode path, with kernel timing split across top-k, gate/up, and down projection.
