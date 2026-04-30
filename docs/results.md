# Reported Results

## B200 Context-Fixed Run

This run verified that the warp-decode paths were active through Python and CUDA markers.

| Variant | Concurrency | Output tok/s total | Output tok/s/GPU | Runtime verified | CUDA active markers | Fallback markers |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `sota` | 1 | 146.82 | 18.35 | false | 0 | 0 |
| `sota` | 4 | 500.61 | 62.58 | false | 0 | 0 |
| `warp-decode` | 1 | 80.66 | 10.08 | true | 8 | 16 |
| `warp-decode` | 4 | 218.99 | 27.37 | true | 8 | 16 |
| `warp-decode-tiled-down` | 1 | 79.73 | 9.97 | true | 8 | 16 |
| `warp-decode-tiled-down` | 4 | 216.81 | 27.10 | true | 8 | 16 |

The full CSV and plot are in `results/b200_context_fixed/`.

## Initial B200 Run

This run showed all variants within noise of each other, which was later treated as suspicious because CUDA graph capture could route the warp variants through the fallback TRTLLM MoE path.

| Variant | Concurrency | Output tok/s total | Output tok/s/GPU |
| --- | ---: | ---: | ---: |
| `sota` | 1 | 139.29 | 17.41 |
| `sota` | 4 | 480.31 | 60.04 |
| `warp-decode` | 1 | 139.53 | 17.44 |
| `warp-decode` | 4 | 481.21 | 60.15 |
| `warp-decode-tiled-down` | 1 | 139.56 | 17.45 |
| `warp-decode-tiled-down` | 4 | 481.33 | 60.17 |

The full CSV and plot are in `results/b200_initial/`.

## ptxas Check

The B200 `sm_100a` ptxas check reported no spills for the warp decode kernels:

- Tiled down kernel: 40 registers, 0 spill stores, 0 spill loads.
- Row-major down kernel: 32 registers, 0 spill stores, 0 spill loads.
- Gate/up kernel: 32 registers, 0 spill stores, 0 spill loads.
- Top-k kernel: 32 registers, 0 spill stores, 0 spill loads.

The raw ptxas log is in `results/b200_initial/ptxas_checks/kimi_k26_warp_decode_sm100a.ptxas.log`.

## Interpretation

The current evidence says:

- The original near-identical result was probably not a meaningful speedup; it likely measured the same/fallback decode behavior.
- After fixing CUDA graph forward-batch context, the warp kernels were active, but the active implementation was much slower than the FlashInfer/TRTLLM baseline.
- The active result does not yet reproduce Cursor Warp Decode speedups.
- The current local patches include a later internal-topk follow-up that was not benchmarked in the reported B200 result set.
