#!/usr/bin/env python3
"""InferenceX Kimi K2.6 serving benchmark launcher.

The benchmark keeps the full SGLang inference stack fixed and only changes
``--moe-runner-backend`` across the current FlashInfer TRTLLM path, the
Cursor-copy Kimi warp-decode MoE path, and the custom tiled-W_down decode path.

The client benchmark is the SemiAnalysis InferenceX benchmark_serving.py
harness. Defaults are a budget version of the Kimi K2.5 B200 INT4 InferenceX
conditions: synthetic random-token requests, random_range_ratio=0.8,
request_rate=inf, ignore_eos, num_prompts=concurrency*10,
warmups=2*concurrency, TP=8, EP=1, DP attention off, and 1K input / 1K output
tokens at concurrency 1 and 4.

Both warp-decode variants still use the baseline FlashInfer TRTLLM MoE path for
prefill. They only switch decode MoE when SGLang is in decode mode and the
supported shape guards match.

* x-axis: output tokens/s/user = total output tokens/s / max concurrency
* y-axis: total tokens/s/gpu = total token throughput / tensor parallel size
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_concurrency_list(value: str) -> list[int]:
    values: list[int] = []
    for item in value.replace(",", " ").split():
        concurrency = int(item)
        if concurrency <= 0:
            raise argparse.ArgumentTypeError("concurrency values must be positive")
        values.append(concurrency)
    if not values:
        raise argparse.ArgumentTypeError("at least one concurrency value is required")
    return values


def parse_request_rate(value: str) -> float:
    if value.lower() in {"inf", "infinity"}:
        return float("inf")
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("request rate must be positive or inf")
    return parsed


def request_rate_to_cli(value: float) -> str:
    return "inf" if value == float("inf") else str(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a budget InferenceX-style Kimi K2.6 serving curve."
    )
    parser.add_argument(
        "--variant",
        choices=["sota", "warp-decode", "warp-decode-tiled-down", "both", "all"],
        required=True,
        help=(
            "Backend variant to benchmark. 'both' and 'all' run sota, "
            "warp-decode, and warp-decode-tiled-down."
        ),
    )
    parser.add_argument("--model-path", default="moonshotai/Kimi-K2.6")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path/name for the synthetic workload. Defaults to model path.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("kimi_k26_bench_out"))
    parser.add_argument(
        "--inferencex-dir",
        type=Path,
        default=None,
        help=(
            "Path to a clone of https://github.com/SemiAnalysisAI/InferenceX. "
            "Defaults to $INFERENCEX_DIR, ../inferencex_ref, or ../InferenceX."
        ),
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help=(
            "SGLang server --max-running-requests. Defaults to the current "
            "benchmark point's concurrency to match vLLM --max-num-seqs=$CONC."
        ),
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.95,
        help=(
            "SGLang --mem-fraction-static. Default mirrors InferenceX's "
            "vLLM --gpu-memory-utilization 0.95 setting."
        ),
    )
    parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        help=(
            "Do not pass --disable-radix-cache to SGLang. InferenceX K2.5 "
            "vLLM uses --no-enable-prefix-caching, so this is off by default."
        ),
    )
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument("--startup-timeout-s", type=int, default=900)
    parser.add_argument(
        "--require-cuda-warp-decode-marker",
        action="store_true",
        help=(
            "For warp-decode variants, also require the FlashInfer C++ wrapper "
            "to emit its CUDA-active marker. This proves the local extension was "
            "rebuilt with the marker instrumentation."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print server and InferenceX commands without launching SGLang.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Do not pass --trust-remote-code to the SGLang server.",
    )

    workload = parser.add_argument_group("InferenceX K2.5 B200 INT4 workload")
    workload.add_argument(
        "--isl",
        "--input-len",
        dest="isl",
        type=int,
        default=1024,
        help="Input tokens.",
    )
    workload.add_argument(
        "--osl",
        "--output-len",
        dest="osl",
        type=int,
        default=1024,
        help="Output tokens.",
    )
    workload.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="SGLang --context-length. Defaults to isl + osl + 256, like InferenceX.",
    )
    workload.add_argument(
        "--concurrency",
        type=parse_concurrency_list,
        default=parse_concurrency_list("1,4"),
        help=(
            "Comma or space separated max concurrency points. Defaults to the "
            "budget InferenceX-style pair: 1,4."
        ),
    )
    workload.add_argument(
        "--prompts-per-concurrency",
        type=int,
        default=10,
        help="num_prompts = max(min_num_prompts, concurrency * this value).",
    )
    workload.add_argument(
        "--min-num-prompts",
        type=int,
        default=10,
        help="Minimum prompts per point. Useful for the concurrency=1 point.",
    )
    workload.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.8,
        help="InferenceX random sequence length lower bound ratio.",
    )
    workload.add_argument(
        "--request-rate",
        type=parse_request_rate,
        default=float("inf"),
        help="Request arrival rate. InferenceX K2.5 default is inf.",
    )
    workload.add_argument(
        "--warmup-multiplier",
        type=int,
        default=2,
        help="num_warmups = warmup_multiplier * concurrency. InferenceX uses 2.",
    )
    workload.add_argument(
        "--extra-bench-args",
        default="",
        help="Additional arguments passed to InferenceX benchmark_serving.py.",
    )

    plot = parser.add_argument_group("plotting")
    plot.add_argument(
        "--skip-plot",
        action="store_true",
        help="Only write JSON/CSV results, without generating a PNG curve.",
    )
    plot.add_argument(
        "--reference-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with interactivity_tok_s_per_user and "
            "token_throughput_per_gpu_tok_s columns to overlay."
        ),
    )
    args = parser.parse_args()
    fixed_server_flags = {
        "--tensor-parallel-size",
        "--tp-size",
        "--data-parallel-size",
        "--dp-size",
        "--expert-parallel-size",
        "--ep-size",
        "--ep",
        "--moe-a2a-backend",
        "--enable-dp-attention",
        "--quantization",
        "--moe-runner-backend",
        "--context-length",
        "--max-running-requests",
        "--mem-fraction-static",
    }
    extra_server_args = set(shlex.split(args.extra_server_args))
    overridden = sorted(fixed_server_flags & extra_server_args)
    if overridden:
        parser.error(
            "These server flags are fixed by the benchmark and cannot appear in "
            f"--extra-server-args: {', '.join(overridden)}"
        )
    if len(args.concurrency) > 2:
        parser.error("budget mode accepts at most two concurrency points; use 1,4")
    return args


def backend_for_variant(variant: str) -> str:
    if variant == "sota":
        return "flashinfer_trtllm"
    if variant == "warp-decode":
        return "flashinfer_kimi_warp_decode"
    if variant == "warp-decode-tiled-down":
        return "flashinfer_kimi_warp_decode_tiled_down"
    raise ValueError(variant)


def prefill_backend_for_variant(_: str) -> str:
    return "flashinfer_trtllm"


def decode_backend_for_variant(variant: str) -> str:
    return backend_for_variant(variant)


def variants_to_run(variant: str) -> list[str]:
    if variant in {"both", "all"}:
        return ["sota", "warp-decode", "warp-decode-tiled-down"]
    return [variant]


def warp_decode_marker_variant(variant: str) -> str | None:
    if variant == "warp-decode":
        return "row-major"
    if variant == "warp-decode-tiled-down":
        return "tiled-W_down"
    return None


def resolve_inferencex_dir(args: argparse.Namespace, repo_root: Path) -> Path:
    candidates = []
    if args.inferencex_dir is not None:
        candidates.append(args.inferencex_dir)
    if os.environ.get("INFERENCEX_DIR"):
        candidates.append(Path(os.environ["INFERENCEX_DIR"]))
    candidates.extend(
        [
            repo_root.parent / "inferencex_ref",
            repo_root.parent / "InferenceX",
        ]
    )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        benchmark = candidate / "utils" / "bench_serving" / "benchmark_serving.py"
        config = candidate / ".github" / "configs" / "nvidia-master.yaml"
        if benchmark.exists() and config.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find an InferenceX checkout. Clone it next to this repo with:\n"
        "  git clone https://github.com/SemiAnalysisAI/InferenceX.git ../inferencex_ref\n"
        "or pass --inferencex-dir /path/to/InferenceX."
    )


def wait_for_server(host: str, port: int, timeout_s: int):
    import urllib.request

    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"SGLang server did not become healthy at {url}")


def terminate_server(proc: subprocess.Popen[Any]):
    if proc.poll() is not None:
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=60)


def write_command_header(log_file: Path, cmd: list[str]):
    with log_file.open("w", encoding="utf-8") as f:
        f.write("$ " + shlex.join(cmd) + "\n\n")


def validate_warp_decode_activation(
    *,
    args: argparse.Namespace,
    variant: str,
    server_log_path: Path,
) -> dict[str, Any]:
    expected_variant = warp_decode_marker_variant(variant)
    if expected_variant is None:
        return {
            "warp_decode_runtime_verified": False,
            "warp_decode_python_active_markers": 0,
            "warp_decode_cuda_active_markers": 0,
            "warp_decode_fallback_markers": 0,
        }

    # Child worker processes write directly to this file; give the filesystem a
    # short beat so a flushed marker is visible before we inspect it.
    time.sleep(1)
    log_text = server_log_path.read_text(encoding="utf-8", errors="replace")
    python_marker = f"SGLANG_KIMI_WARP_DECODE_ACTIVE variant={expected_variant}"
    cuda_marker = f"SGLANG_KIMI_WARP_DECODE_CUDA_ACTIVE variant={expected_variant}"
    python_active_count = log_text.count(python_marker)
    cuda_active_count = log_text.count(cuda_marker)
    fallback_count = log_text.count("SGLANG_KIMI_WARP_DECODE_FALLBACK")

    if python_active_count == 0:
        raise RuntimeError(
            f"{variant} completed but no runtime warp-decode active marker was "
            f"found in {server_log_path}. This run cannot be trusted as a "
            "warp-decode benchmark."
        )
    if args.require_cuda_warp_decode_marker and cuda_active_count == 0:
        raise RuntimeError(
            f"{variant} emitted the Python warp-decode marker, but no CUDA "
            f"wrapper marker was found in {server_log_path}. Rebuild/reinstall "
            "the FlashInfer extension before using this run for kernel claims."
        )

    return {
        "warp_decode_runtime_verified": True,
        "warp_decode_python_active_markers": python_active_count,
        "warp_decode_cuda_active_markers": cuda_active_count,
        "warp_decode_fallback_markers": fallback_count,
    }


def benchmark_point_paths(
    args: argparse.Namespace,
    variant: str,
    concurrency: int,
    *,
    num_prompts: int | None = None,
    suffix: str = "",
) -> tuple[int, Path, Path]:
    if num_prompts is None:
        num_prompts = max(
            args.min_num_prompts, concurrency * args.prompts_per_concurrency
        )
    stem = (
        f"{variant}_isl{args.isl}_osl{args.osl}_conc{concurrency}"
        f"_prompts{num_prompts}"
    )
    if suffix:
        stem = f"{stem}_{suffix}"
    bench_log_path = args.output_dir / f"{stem}.log"
    result_json_path = args.output_dir / f"{stem}.json"
    return num_prompts, bench_log_path, result_json_path


def build_benchmark_command(
    *,
    args: argparse.Namespace,
    inferencex_dir: Path,
    variant: str,
    concurrency: int,
    num_prompts_override: int | None = None,
    warmups_override: int | None = None,
    suffix: str = "",
) -> tuple[list[str], int, Path, Path]:
    num_prompts, bench_log_path, result_json_path = benchmark_point_paths(
        args,
        variant,
        concurrency,
        num_prompts=num_prompts_override,
        suffix=suffix,
    )
    benchmark_cmd = [
        sys.executable,
        str(inferencex_dir / "utils" / "bench_serving" / "benchmark_serving.py"),
        "--backend",
        "sglang",
        "--base-url",
        f"http://{args.host}:{args.port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        args.model_path,
        "--tokenizer",
        args.tokenizer or args.model_path,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(num_prompts),
        "--random-input-len",
        str(args.isl),
        "--random-output-len",
        str(args.osl),
        "--random-range-ratio",
        str(args.random_range_ratio),
        "--request-rate",
        request_rate_to_cli(args.request_rate),
        "--max-concurrency",
        str(concurrency),
        "--ignore-eos",
        "--num-warmups",
        str(
            warmups_override
            if warmups_override is not None
            else args.warmup_multiplier * concurrency
        ),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--result-dir",
        str(args.output_dir),
        "--result-filename",
        result_json_path.name,
        "--disable-tqdm",
        "--save-result",
        "--trust-remote-code",
        *shlex.split(args.extra_bench_args),
        "--metadata",
        "reference_config=kimik2.5-int4-b200-vllm",
        "reference_hardware=B200",
        "reference_precision=INT4",
        "reference_framework=vLLM",
        f"total_gpus={args.tp_size}",
        "precision=INT4",
        f"tensor_parallelism={args.tp_size}",
        "expert_parallelism=1",
        "dp_attention=false",
        "moe_a2a_backend=none",
        f"prefill_moe_backend={prefill_backend_for_variant(variant)}",
        f"decode_moe_backend={decode_backend_for_variant(variant)}",
        f"sglang_moe_backend={backend_for_variant(variant)}",
    ]
    return benchmark_cmd, num_prompts, bench_log_path, result_json_path


def run_benchmark_point(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    inferencex_dir: Path,
    env: dict[str, str],
    variant: str,
    concurrency: int,
) -> dict[str, Any]:
    benchmark_cmd, num_prompts, bench_log_path, result_json_path = (
        build_benchmark_command(
            args=args,
            inferencex_dir=inferencex_dir,
            variant=variant,
            concurrency=concurrency,
        )
    )
    if result_json_path.exists():
        result_json_path.unlink()

    write_command_header(bench_log_path, benchmark_cmd)
    with bench_log_path.open("a", encoding="utf-8") as bench_log:
        subprocess.run(
            benchmark_cmd,
            stdout=bench_log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
            env=env,
            cwd=repo_root,
        )

    result = read_result_json(result_json_path)
    return normalize_result(
        result=result,
        args=args,
        variant=variant,
        concurrency=concurrency,
        num_prompts=num_prompts,
        result_json_path=result_json_path,
        bench_log_path=bench_log_path,
    )


def read_result_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_result(
    *,
    result: dict[str, Any],
    args: argparse.Namespace,
    variant: str,
    concurrency: int,
    num_prompts: int,
    result_json_path: Path,
    bench_log_path: Path,
) -> dict[str, Any]:
    output_tps = as_float(result.get("output_throughput"))
    total_tps = as_float(
        result.get("total_throughput", result.get("total_token_throughput"))
    )
    duration_s = as_float(result.get("duration"))
    input_tps = as_float(result.get("input_throughput"))
    if input_tps == 0.0 and duration_s > 0.0:
        input_tps = as_float(result.get("total_input_tokens")) / duration_s
    actual_concurrency = as_float(result.get("concurrency"), default=float(concurrency))
    target_interactivity = output_tps / max(float(concurrency), 1.0)
    actual_interactivity = output_tps / max(actual_concurrency, 1.0)

    return {
        "variant": variant,
        "moe_backend": backend_for_variant(variant),
        "prefill_moe_backend": prefill_backend_for_variant(variant),
        "decode_moe_backend": decode_backend_for_variant(variant),
        "benchmark_harness": "InferenceX",
        "reference_config": "kimik2.5-int4-b200-vllm",
        "model_path": args.model_path,
        "total_gpus": args.tp_size,
        "precision": "INT4",
        "tensor_parallelism": args.tp_size,
        "expert_parallelism": 1,
        "dp_attention": False,
        "moe_a2a_backend": "none",
        "dataset_name": "random",
        "isl": args.isl,
        "osl": args.osl,
        "random_range_ratio": args.random_range_ratio,
        "request_rate": request_rate_to_cli(args.request_rate),
        "tp_size": args.tp_size,
        "max_concurrency": concurrency,
        "actual_concurrency": actual_concurrency,
        "num_prompts": num_prompts,
        "completed": result.get("completed"),
        "duration_s": duration_s,
        "request_throughput_req_s": result.get("request_throughput"),
        "input_throughput_tok_s": input_tps,
        "output_throughput_tok_s": output_tps,
        "total_throughput_tok_s": total_tps,
        "token_throughput_per_gpu_tok_s": total_tps / max(float(args.tp_size), 1.0),
        "input_token_throughput_per_gpu_tok_s": input_tps
        / max(float(args.tp_size), 1.0),
        "output_token_throughput_per_gpu_tok_s": output_tps
        / max(float(args.tp_size), 1.0),
        "interactivity_tok_s_per_user": target_interactivity,
        "target_interactivity_tok_s_per_user": target_interactivity,
        "actual_interactivity_tok_s_per_user": actual_interactivity,
        "mean_e2e_latency_ms": result.get("mean_e2el_ms"),
        "p90_e2e_latency_ms": result.get("p90_e2el_ms"),
        "p99_e2e_latency_ms": result.get("p99_e2el_ms"),
        "mean_ttft_ms": result.get("mean_ttft_ms"),
        "median_ttft_ms": result.get("median_ttft_ms"),
        "p99_ttft_ms": result.get("p99_ttft_ms"),
        "mean_tpot_ms": result.get("mean_tpot_ms"),
        "median_tpot_ms": result.get("median_tpot_ms"),
        "p99_tpot_ms": result.get("p99_tpot_ms"),
        "mean_itl_ms": result.get("mean_itl_ms"),
        "median_itl_ms": result.get("median_itl_ms"),
        "p90_itl_ms": result.get("p90_itl_ms"),
        "p99_itl_ms": result.get("p99_itl_ms"),
        "result_json": str(result_json_path),
        "benchmark_log": str(bench_log_path),
    }


def write_summary_csv(rows: list[dict[str, Any]], path: Path):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_reference_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_curve(rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plot_skipped=matplotlib_unavailable error={exc}")
        return

    metric_specs = [
        ("token_throughput_per_gpu_tok_s", "Token Throughput per GPU (tok/s/gpu)"),
        (
            "input_token_throughput_per_gpu_tok_s",
            "Input Token Throughput per GPU (tok/s/gpu)",
        ),
        (
            "output_token_throughput_per_gpu_tok_s",
            "Output Token Throughput per GPU (tok/s/gpu)",
        ),
    ]
    fig_height = 3.3 * len(metric_specs) + 0.5
    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(9, fig_height), sharex=True)
    variants = []
    for row in rows:
        if row["variant"] not in variants:
            variants.append(row["variant"])

    for ax, (metric_key, ylabel) in zip(axes, metric_specs):
        for variant in variants:
            variant_rows = [row for row in rows if row["variant"] == variant]
            variant_rows.sort(
                key=lambda row: as_float(row["interactivity_tok_s_per_user"])
            )
            xs = [
                as_float(row["interactivity_tok_s_per_user"])
                for row in variant_rows
            ]
            ys = [as_float(row[metric_key]) for row in variant_rows]
            ax.plot(xs, ys, marker="o", label=variant)
            for row, x, y in zip(variant_rows, xs, ys):
                ax.annotate(
                    f"c{row['max_concurrency']}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(4, 5),
                    fontsize=8,
                )

        if reference_rows:
            reference_rows.sort(
                key=lambda row: as_float(row.get("interactivity_tok_s_per_user"))
            )
            xs = [
                as_float(row.get("interactivity_tok_s_per_user"))
                for row in reference_rows
            ]
            ys = [as_float(row.get(metric_key)) for row in reference_rows]
            label = reference_rows[0].get("variant") or "reference"
            ax.plot(xs, ys, marker="s", linestyle="--", label=label)

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_title("Kimi K2.6 InferenceX-Style Throughput Curve")
    axes[-1].set_xlabel("Interactivity (output tok/s/user)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def launch_server(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    env: dict[str, str],
    variant: str,
    concurrency: int,
    server_cmd: list[str] | None = None,
    log_suffix: str = "",
) -> tuple[subprocess.Popen[Any], Any, Path]:
    suffix = f"_{log_suffix}" if log_suffix else ""
    server_log_path = args.output_dir / f"server_{variant}_conc{concurrency}{suffix}.log"
    if server_cmd is None:
        server_cmd = build_server_command(args=args, variant=variant, concurrency=concurrency)
    server_env = server_env_for_variant(env, variant)

    server_log = server_log_path.open("w", encoding="utf-8")
    server_log.write("$ " + shlex.join(server_cmd) + "\n\n")
    server_log.flush()
    proc = subprocess.Popen(
        server_cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
        env=server_env,
        cwd=repo_root,
        preexec_fn=os.setsid,
    )
    return proc, server_log, server_log_path


def build_server_command(
    *, args: argparse.Namespace, variant: str, concurrency: int
) -> list[str]:
    backend = backend_for_variant(variant)
    max_running_requests = args.max_running_requests or concurrency
    server_cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tp_size),
        "--data-parallel-size",
        "1",
        "--expert-parallel-size",
        "1",
        "--moe-a2a-backend",
        "none",
        "--context-length",
        str(args.context_length or (args.isl + args.osl + 256)),
        "--max-running-requests",
        str(max_running_requests),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--quantization",
        "compressed-tensors",
        "--moe-runner-backend",
        backend,
        *shlex.split(args.extra_server_args),
    ]
    if not args.no_trust_remote_code:
        server_cmd.append("--trust-remote-code")
    if not args.enable_prefix_cache:
        server_cmd.append("--disable-radix-cache")

    return server_cmd


def server_env_for_variant(env: dict[str, str], variant: str) -> dict[str, str]:
    server_env = env.copy()
    if variant in {"warp-decode", "warp-decode-tiled-down"}:
        server_env["SGLANG_REQUIRE_KIMI_WARP_DECODE"] = "1"
        server_env["SGLANG_KIMI_WARP_DECODE_MARK_ACTIVE"] = "1"
    return server_env


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    inferencex_dir = resolve_inferencex_dir(args, repo_root)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/inferencex-pycache")
    pythonpath = os.pathsep.join(
        [
            str(repo_root / "python"),
            str(inferencex_dir / "utils" / "bench_serving"),
        ]
    )
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    if args.dry_run:
        for variant in variants_to_run(args.variant):
            for concurrency in args.concurrency:
                server_cmd = build_server_command(
                    args=args, variant=variant, concurrency=concurrency
                )
                benchmark_cmd, _, _, _ = build_benchmark_command(
                    args=args,
                    inferencex_dir=inferencex_dir,
                    variant=variant,
                    concurrency=concurrency,
                )
                server_env = server_env_for_variant(env, variant)
                print(f"variant={variant} concurrency={concurrency}")
                if "SGLANG_REQUIRE_KIMI_WARP_DECODE" in server_env:
                    print("server_env SGLANG_REQUIRE_KIMI_WARP_DECODE=1")
                if "SGLANG_KIMI_WARP_DECODE_MARK_ACTIVE" in server_env:
                    print("server_env SGLANG_KIMI_WARP_DECODE_MARK_ACTIVE=1")
                print("server_cmd " + shlex.join(server_cmd))
                print("benchmark_cmd " + shlex.join(benchmark_cmd))
        return

    rows: list[dict[str, Any]] = []
    for variant in variants_to_run(args.variant):
        for concurrency in args.concurrency:
            proc, server_log, server_log_path = launch_server(
                args=args,
                repo_root=repo_root,
                env=env,
                variant=variant,
                concurrency=concurrency,
            )
            try:
                wait_for_server(args.host, args.port, args.startup_timeout_s)
                print(f"running variant={variant} concurrency={concurrency}")
                row = run_benchmark_point(
                    args=args,
                    repo_root=repo_root,
                    inferencex_dir=inferencex_dir,
                    env=env,
                    variant=variant,
                    concurrency=concurrency,
                )
                row.update(
                    validate_warp_decode_activation(
                        args=args,
                        variant=variant,
                        server_log_path=server_log_path,
                    )
                )
                rows.append(row)
            finally:
                terminate_server(proc)
                server_log.close()
                print(f"server_log_{variant}_conc{concurrency}={server_log_path}")

    summary_csv = args.output_dir / (
        f"inferencex_style_kimi_k26_{args.variant}_isl{args.isl}_osl{args.osl}.csv"
    )
    write_summary_csv(rows, summary_csv)

    plot_path = args.output_dir / (
        f"inferencex_style_kimi_k26_{args.variant}_isl{args.isl}_osl{args.osl}.png"
    )
    if not args.skip_plot:
        plot_curve(rows, load_reference_rows(args.reference_csv), plot_path)
        print(f"plot={plot_path}")

    print(f"summary_csv={summary_csv}")
    for row in rows:
        print(
            "result "
            f"variant={row['variant']} "
            f"concurrency={row['max_concurrency']} "
            f"interactivity={as_float(row['interactivity_tok_s_per_user']):.2f} "
            f"total_tok_s_gpu={as_float(row['token_throughput_per_gpu_tok_s']):.2f} "
            f"input_tok_s_gpu={as_float(row['input_token_throughput_per_gpu_tok_s']):.2f} "
            f"output_tok_s_gpu={as_float(row['output_token_throughput_per_gpu_tok_s']):.2f}"
        )


if __name__ == "__main__":
    main()
