# Reported Results

## Latest Verified B200 Run

This run verified that the warp-decode paths were active through Python and CUDA markers.

Setup: 8x B200, `moonshotai/Kimi-K2.6`, INT4, TP=8, EP=1, random dataset, ISL=1024, OSL=1024.

| Variant | Prefill MoE backend | Decode MoE backend | Concurrency | Output tok/s total | Output tok/s/GPU | Runtime verified | CUDA active markers | Fallback markers |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `sota` | `flashinfer_trtllm` | `flashinfer_trtllm` | 1 | 146.82 | 18.35 | false | 0 | 0 |
| `sota` | `flashinfer_trtllm` | `flashinfer_trtllm` | 4 | 500.61 | 62.58 | false | 0 | 0 |
| `warp-decode` | `flashinfer_trtllm` | `flashinfer_kimi_warp_decode` | 1 | 80.66 | 10.08 | true | 8 | 16 |
| `warp-decode` | `flashinfer_trtllm` | `flashinfer_kimi_warp_decode` | 4 | 218.99 | 27.37 | true | 8 | 16 |
| `warp-decode-tiled-down` | `flashinfer_trtllm` | `flashinfer_kimi_warp_decode_tiled_down` | 1 | 79.73 | 9.97 | true | 8 | 16 |
| `warp-decode-tiled-down` | `flashinfer_trtllm` | `flashinfer_kimi_warp_decode_tiled_down` | 4 | 216.81 | 27.10 | true | 8 | 16 |

The full CSV and plot are in `results/b200_context_fixed/`.

## ptxas Check

The B200 `sm_100a` ptxas check reported no spills for the warp decode kernels:

- Tiled down kernel: 40 registers, 0 spill stores, 0 spill loads.
- Row-major down kernel: 32 registers, 0 spill stores, 0 spill loads.
- Gate/up kernel: 32 registers, 0 spill stores, 0 spill loads.
- Top-k kernel: 32 registers, 0 spill stores, 0 spill loads.

The raw ptxas log is in `results/b200_context_fixed/ptxas_checks/kimi_k26_warp_decode_sm100a.ptxas.log`.

## Interpretation

The current evidence says:

- The warp kernels were active, but the active implementation was much slower than the FlashInfer/TRTLLM baseline.
- The active result does not yet reproduce Cursor Warp Decode speedups.
- The current local patches include a later internal-topk follow-up that was not benchmarked in the reported B200 result set.
