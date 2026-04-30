#!/usr/bin/env python3
"""Compile Kimi K2.6 warp-decode kernels with ptxas -v and reject spills."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


KERNEL_LAUNCHER = "csrc/trtllm_fused_moe_kernel_launcher.cu"
KIMI_KERNEL_MARKER = "kimi_k26_warp_decode"
TILED_DOWN_MARKER = "kimi_k26_moe_down_warp_decode_tiled_kernel"


def import_module_path(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    return Path(spec.origin).parent


def find_flashinfer_cubin_include() -> tuple[Path | None, Path | None]:
    try:
        import flashinfer_cubin
    except ImportError:
        return None, None

    cubin_dir = Path(flashinfer_cubin.get_cubin_dir())
    include_dirs = sorted(cubin_dir.glob("*/batched_gemm-*/include"))
    return cubin_dir, include_dirs[0] if include_dirs else None


def build_command(args: argparse.Namespace, repo_root: Path, output_obj: Path) -> list[str]:
    cuda_home = Path(args.cuda_home or os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    nvcc = Path(args.nvcc) if args.nvcc else cuda_home / "bin" / "nvcc"
    if not nvcc.exists():
        found = shutil.which("nvcc")
        if found is None:
            raise RuntimeError("nvcc not found; set --nvcc or CUDA_HOME")
        nvcc = Path(found)
        cuda_home = nvcc.parent.parent

    cubin_dir, bmm_include = find_flashinfer_cubin_include()
    tvm_ffi_dir = import_module_path("tvm_ffi")
    deep_gemm_dir = import_module_path("deep_gemm")

    include_flags = []
    include_paths = [
        cubin_dir,
        bmm_include,
        repo_root / "csrc" / "nv_internal",
        repo_root / "csrc" / "nv_internal" / "include",
    ]
    system_include_paths = [
        Path(sysconfig_path("include")),
        cuda_home / "include",
        cuda_home / "include" / "cccl",
        tvm_ffi_dir / "include" if tvm_ffi_dir else None,
        repo_root / "include",
        repo_root / "csrc",
        repo_root / "3rdparty" / "cutlass" / "include",
        repo_root / "3rdparty" / "cutlass" / "tools" / "util" / "include",
        deep_gemm_dir / "include" if deep_gemm_dir else None,
        repo_root / "3rdparty" / "spdlog" / "include",
    ]
    for path in include_paths:
        if path is not None:
            include_flags += ["-I", str(path)]
    for path in system_include_paths:
        if path is not None:
            include_flags += ["-isystem", str(path)]

    arch = args.arch
    return [
        str(nvcc),
        "-DPy_LIMITED_API=0x03090000",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
        *include_flags,
        "--compiler-options=-fPIC",
        "--expt-relaxed-constexpr",
        "-static-global-template-stub=false",
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_FP4_E2M1",
        "-std=c++17",
        "--threads=1",
        "-use_fast_math",
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
        "-DNDEBUG",
        "-O3",
        "-DTLLM_GEN_EXPORT_INTERFACE",
        "-DTLLM_GEN_EXPORT_FLASHINFER",
        "-DTLLM_ENABLE_CUDA",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP4",
        "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
        '-DTLLM_GEN_GEMM_CUBIN_PATH="batched_gemm/"',
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "-Xptxas",
        "-v",
        "-c",
        str(repo_root / KERNEL_LAUNCHER),
        "-o",
        str(output_obj),
    ]


def sysconfig_path(name: str) -> str:
    import sysconfig

    return sysconfig.get_paths()[name]


def parse_ptxas_log(text: str) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    current: dict[str, int | str] | None = None
    for line in text.splitlines():
        if "Compiling entry function" in line:
            current = None
            if KIMI_KERNEL_MARKER in line:
                match = re.search(r"'([^']+)'", line)
                current = {"kernel": match.group(1) if match else line}
                rows.append(current)
            continue

        if current is None:
            continue

        spill = re.search(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
            line,
        )
        if spill:
            current["stack_frame_bytes"] = int(spill.group(1))
            current["spill_store_bytes"] = int(spill.group(2))
            current["spill_load_bytes"] = int(spill.group(3))
            continue

        regs = re.search(r"Used (\d+) registers", line)
        if regs:
            current["registers"] = int(regs.group(1))

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="100a", help="CUDA arch suffix, e.g. 90a or 100a")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cuda-home")
    parser.add_argument("--nvcc")
    parser.add_argument("--output-dir", type=Path, default=Path("ptxas_checks"))
    parser.add_argument("--max-tiled-down-registers", type=int, default=64)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_obj = args.output_dir / f"kimi_k26_warp_decode_sm{args.arch}.o"
    output_log = args.output_dir / f"kimi_k26_warp_decode_sm{args.arch}.ptxas.log"

    cmd = build_command(args, repo_root, output_obj)
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_log.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        print(result.stdout)
        print(f"nvcc failed; log={output_log}", file=sys.stderr)
        return result.returncode

    rows = parse_ptxas_log(result.stdout)
    if not rows:
        print(result.stdout)
        print("No Kimi warp-decode ptxas entries found", file=sys.stderr)
        return 1

    failed = False
    for row in rows:
        kernel = str(row["kernel"])
        regs = int(row.get("registers", -1))
        spill_stores = int(row.get("spill_store_bytes", -1))
        spill_loads = int(row.get("spill_load_bytes", -1))
        print(
            f"{kernel}: registers={regs} "
            f"spill_stores={spill_stores} spill_loads={spill_loads}"
        )
        if spill_stores != 0 or spill_loads != 0:
            failed = True
        if TILED_DOWN_MARKER in kernel and regs > args.max_tiled_down_registers:
            failed = True

    print(f"log={output_log}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
