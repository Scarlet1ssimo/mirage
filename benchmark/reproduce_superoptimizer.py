#!/usr/bin/env python3
"""Reproduce the academic Mirage compiler using existing Forest/XLA fixtures."""

import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("repro_superoptimizer_worker.py")
FOREST_CASES = [
    "attention",
    "gqa",
    "layernorm_matmul",
    "rms_ffn_swiglu",
    "efficient_vit",
]
AE_CASES = ["attention", "gqa", "rmsnorm", "gated_mlp"]


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def timed(result):
    latency = result.get("latency_ms", {}).get("median")
    throughput = result.get("effective_tflops")
    return (
        result.get("status") in ("ok", "measured_invalid")
        and isinstance(latency, (int, float))
        and math.isfinite(latency)
        and latency > 0
        and isinstance(throughput, (int, float))
        and math.isfinite(throughput)
        and throughput > 0
    )


def select(results):
    measured = [r for r in results if timed(r)]
    valid = [
        r
        for r in measured
        if r["status"] == "ok"
        and r.get("correctness", {}).get("passed") is True
        and r.get("post_validation", {}).get("passed") is True
    ]
    fastest = lambda rr: (
        min(rr, key=lambda r: r["latency_ms"]["median"]) if rr else None
    )
    return fastest(valid), fastest(measured)


def resolve_gpus(identifiers):
    devices = []
    fields = ["index", "uuid", "name", "memory.total", "power.limit", "driver_version"]
    for identifier in identifiers:
        lines = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    identifier,
                    "--query-gpu=" + ",".join(fields),
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            .strip()
            .splitlines()
        )
        if len(lines) != 1:
            raise ValueError(
                f"GPU identifier must select one physical GPU: {identifier}"
            )
        values = [s.strip() for s in lines[0].split(",")]
        devices.append(dict(zip(fields, values)))
    if len({d["uuid"] for d in devices}) != len(devices):
        raise ValueError("Duplicate physical GPU identifiers")
    return devices


def cache_for(kind, case, batch, cache_dir=None):
    directory = cache_dir or REPO / "benchmark" / (
        "saved_mugraphs_hopper" if kind == "hopper" else "saved_mugraphs"
    )
    name = {
        "attention": "mha",
        "gqa": "gqa",
        "rms_ffn_swiglu": "rmsnorm",
        "gated_mlp": "gated_mlp",
    }[case]
    return directory / f"{name}_bs{batch}.json"


def command_output(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=15
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)


def export_candidate(result, destination):
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(result["directory"])
    for name in [
        "result.json",
        "command.json",
        "nvcc-command.json",
        "generated.py",
        "generated-original.cu",
    ]:
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    if (source / "generated_codes/0/test0.cu").exists():
        shutil.copy2(
            source / "generated_codes/0/test0.cu", destination / "generated.cu"
        )
    candidate = Path(result["candidate_graph"])
    shutil.copy2(candidate, destination / "candidate_graph.json")
    original = source.parent / "search/input_graph.json"
    if not original.exists():
        original = source.parent / "export/input_graph.json"
    if original.exists():
        shutil.copy2(original, destination / "InputIR.json")
    graph = json.loads(candidate.read_text())
    tuning = dict(result.get("tuning", {}))
    tuning["threadblock_graphs"] = [
        {
            k: op["bgraph"][k]
            for k in ("grid_dim", "block_dim", "forloop_range", "reduction_dimx")
            if k in op["bgraph"]
        }
        for op in graph
        if op["op_type"] == "kn_customized_op"
    ]
    write_json(destination / "tuning.json", tuning)
    write_json(
        destination / "source-hashes.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.iterdir()
            if p.is_file() and p.name != "source-hashes.json"
        },
    )


def summarize(output):
    rows, attempts = [], []
    for case_dir in sorted((output / "cases").glob("*/B*")):
        case = json.loads((case_dir / "case.json").read_text())
        results = []
        for p in sorted(case_dir.glob("*/result.json")):
            if p.parent.name in ("winner", "fastest_timed"):
                continue
            result = json.loads(p.read_text())
            result["directory"] = str(p.parent)
            attempts.append(result)
            results.append(result)
        winner, fastest = select(results)
        if winner:
            export_candidate(winner, case_dir / "winner")
        if fastest:
            export_candidate(fastest, case_dir / "fastest_timed")
        metrics = fastest.get("correctness", {}) if fastest else {}
        row = {
            "suite": case["suite"],
            "workload": "RMSNorm+Matmul (AE)"
            if case["suite"] == "ae" and case["case"] == "rms_ffn_swiglu"
            else case["case"],
            "B": case["B"],
            "source": case["source"],
            "cache_kind": case.get("cache_kind"),
            "validated_tflops": winner["effective_tflops"] if winner else None,
            "validated_latency_ms": winner["latency_ms"]["median"] if winner else None,
            "validated_backend": winner.get("backend") if winner else None,
            "validated_candidate_index": winner.get("candidate_index")
            if winner
            else None,
            "fastest_timed_tflops": fastest["effective_tflops"] if fastest else None,
            "fastest_timed_latency_ms": fastest["latency_ms"]["median"]
            if fastest
            else None,
            "fastest_timed_valid": bool(fastest and select([fastest])[0]),
            "backend": fastest.get("backend") if fastest else None,
            "candidate_index": fastest.get("candidate_index") if fastest else None,
            "output_nan": metrics.get("output_nan"),
            "output_inf": metrics.get("output_inf"),
            "output_elements": metrics.get("output_elements"),
            "relative_l2": metrics.get("relative_l2"),
            "normalized_max_abs": metrics.get("normalized_max_abs"),
            "status": "ok"
            if winner
            else "measured_invalid"
            if fastest
            else case["status"],
            "candidate_count": case.get("candidate_count", 0),
            "attempted_candidates": case.get("attempted_candidates", 0),
            "candidate_prefix_truncated": case.get("candidate_prefix_truncated", False),
            "fp32_adapter": fastest.get("fp32_accumulation_adapter", False)
            if fastest
            else case["fp32_accum"],
            "fold_zero_epsilon": case["fold_zero_epsilon"],
            "physical_gpu_uuid": case["gpu_uuid"],
            "semantics": fastest.get("semantics") if fastest else None,
            "case_directory": str(case_dir),
        }
        rows.append(row)
    if rows:
        with (output / "performance.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / "attempts.json", attempts)
    write_json(
        output / "failure-distribution.json",
        {
            "attempt_statuses": dict(
                collections.Counter(r.get("status") for r in attempts)
            ),
            "failed_stages": dict(
                collections.Counter(
                    r.get("stage", "bootstrap")
                    for r in attempts
                    if r.get("status") not in ("ok", "measured_invalid")
                )
            ),
            "case_statuses": dict(collections.Counter(r["status"] for r in rows)),
        },
    )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["forest", "ae"], default="forest")
    p.add_argument(
        "--source", choices=["auto", "fresh", "cache", "direct"], default="auto"
    )
    p.add_argument(
        "--gpus",
        nargs="+",
        required=True,
        help="Explicit physical GPU IDs or UUIDs; one sequential worker per GPU",
    )
    p.add_argument(
        "--output", type=Path, required=True, help="New, empty output directory"
    )
    p.add_argument("--forest-root", type=Path, default=REPO.parent / "forest")
    p.add_argument(
        "--fixture-dir",
        type=Path,
        help="XLA fixture root containing <case>/B*/inputs.npz, reference.npy, settings.json",
    )
    p.add_argument("--cases", nargs="+")
    p.add_argument("--batch-sizes", nargs="+", type=int)
    p.add_argument(
        "--backends", nargs="+", choices=["cuda", "triton"], default=["cuda"]
    )
    p.add_argument("--cache-kind", choices=["ordinary", "hopper"], default="ordinary")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--candidate-indices", nargs="+", type=int)
    p.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="0: all, subject to the case budget",
    )
    p.add_argument(
        "--allow-invalid-output",
        action="store_true",
        help="Time NaN/precision-failing output and label measured_invalid",
    )
    p.add_argument(
        "--fp32-accum",
        action="store_true",
        help="Optional generated-code SM80 MMA/cuBLAS adapter; excludes Hopper WGMMA",
    )
    p.add_argument("--fold-zero-epsilon", action="store_true")
    p.add_argument("--decompose-norm", action="store_true")
    p.add_argument("--norm-grid", action="store_true")
    p.add_argument("--max-tb-ops", type=int, default=-1)
    p.add_argument("--search-config", choices=["mlp", "attention"])
    p.add_argument(
        "--target-cc",
        type=int,
        choices=[80, 90],
        default=0,
        help="Default: actual GPU compute capability",
    )
    p.add_argument("--pipeline-stages", type=int, default=2)
    p.add_argument("--num-warp-groups", type=int, default=2)
    p.add_argument(
        "--triton-path",
        type=Path,
        help="Optional isolated Triton 2.3.1 --target directory",
    )
    p.add_argument("--search-budget-s", type=float, default=180)
    p.add_argument("--candidate-budget-s", type=float, default=180)
    p.add_argument("--case-budget-s", type=float, default=600)
    a = p.parse_args()
    allowed = FOREST_CASES if a.suite == "forest" else AE_CASES
    cases = a.cases or allowed
    if any(c not in allowed for c in cases):
        p.error(f"{a.suite} cases must be selected from {allowed}")
    if (
        len(set(cases)) != len(cases)
        or a.batch_sizes
        and len(set(a.batch_sizes)) != len(a.batch_sizes)
    ):
        p.error("Duplicate cases or batch sizes")
    if any(
        not math.isfinite(v) or v <= 0
        for v in (a.search_budget_s, a.candidate_budget_s, a.case_budget_s)
    ):
        p.error("Time budgets must be finite and positive")
    if (
        a.max_candidates < 0
        or a.batch_sizes
        and any(b < 1 for b in a.batch_sizes)
        or a.candidate_indices
        and any(i < 0 for i in a.candidate_indices)
    ):
        p.error("Batch sizes must be positive; candidate indices/counts nonnegative")
    if (
        len(set(a.backends)) != len(a.backends)
        or a.candidate_indices
        and len(set(a.candidate_indices)) != len(a.candidate_indices)
    ):
        p.error("Duplicate backends or candidate indices")
    if a.pipeline_stages < 1 or a.num_warp_groups < 1:
        p.error("Pipeline stages and warp groups must be positive")
    if a.suite == "forest" and (a.fixture_dir is None or not a.fixture_dir.is_dir()):
        p.error(
            "Forest suite requires an existing --fixture-dir; see REPRODUCE_SUPEROPTIMIZER.md"
        )
    if not (a.forest_root / "benchmarks/src/forest_benchmarks/timing.py").exists():
        p.error(
            "--forest-root must contain the existing Forest benchmark timing implementation"
        )
    if a.source == "cache" and a.suite != "ae":
        p.error("Provided caches have AE shapes; Forest suite must use fresh or direct")
    a.source = (
        ("fresh" if a.suite == "forest" else "cache")
        if a.source == "auto"
        else a.source
    )
    a.output = a.output.resolve()
    if a.output.exists() and any(a.output.iterdir()):
        p.error(
            "Output directory must be empty; existing measurements are never overwritten"
        )
    a.output.mkdir(parents=True, exist_ok=True)
    for name in ("forest_root", "fixture_dir", "triton_path", "cache_dir"):
        if getattr(a, name) is not None:
            setattr(a, name, getattr(a, name).resolve())
    devices = resolve_gpus(a.gpus)
    (a.output / "cases").mkdir()
    environment = {
        "arguments": vars(a),
        "gpus": devices,
        "python": sys.version,
        "mirage_commit": command_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "forest_commit": command_output(
            ["git", "-C", str(a.forest_root), "rev-parse", "HEAD"]
        ),
        "nvcc": command_output(["nvcc", "--version"]),
        "compiler": command_output(["g++", "--version"]),
        "pip": command_output([sys.executable, "-m", "pip", "list", "--format=json"]),
        "protocol": {
            "seed": 1729,
            "warmup": 10,
            "measurement_ms": 200,
            "flush_l2": True,
            "relative_l2_limit": 0.005,
            "normalized_max_abs_limit": 0.02,
            "search_threads": 8,
        },
    }
    write_json(a.output / "environment.json", environment)
    (a.output / "source.patch").write_text(
        command_output(["git", "-C", str(REPO), "diff", "HEAD"])
    )
    for script in (Path(__file__), WORKER):
        shutil.copy2(script, a.output / script.name)
    stopped = threading.Event()
    process_lock = threading.Lock()
    processes = set()
    interrupted = [None]

    def stop(signum, _frame):
        interrupted[0] = signum
        stopped.set()
        with process_lock:
            for proc in processes:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)

    def launch(extra, directory, device, budget, case):
        directory.mkdir(parents=True)
        args = [
            sys.executable,
            "-u",
            str(WORKER),
            "--case",
            case["case"],
            "--B",
            str(case["B"]),
            "--shape",
            a.suite,
            "--forest-root",
            str(a.forest_root),
            "--directory",
            str(directory),
            "--target-cc",
            str(a.target_cc),
            "--pipeline-stages",
            str(a.pipeline_stages),
            "--num-warp-groups",
            str(a.num_warp_groups),
            "--max-tb-ops",
            str(a.max_tb_ops),
            *extra,
        ]
        for flag, value in [
            ("fixture-dir", a.fixture_dir),
            ("triton-path", a.triton_path),
            ("search-config", a.search_config),
        ]:
            if value is not None:
                args += ["--" + flag, str(value)]
        for flag in (
            "allow-invalid-output",
            "fp32-accum",
            "fold-zero-epsilon",
            "decompose-norm",
            "norm-grid",
        ):
            if getattr(a, flag.replace("-", "_")):
                args.append("--" + flag)
        write_json(directory / "command.json", args)
        env = os.environ.copy()
        env.update(
            MIRAGE_ROOT=str(REPO),
            CUDA_VISIBLE_DEVICES=device["uuid"],
            OMP_NUM_THREADS="8",
            OPENBLAS_NUM_THREADS="1",
        )
        env["PYTHONPATH"] = (
            str(REPO / "python") + os.pathsep + env.get("PYTHONPATH", "")
        )
        started = time.monotonic()
        with (directory / "run.log").open("w") as log:
            with process_lock:
                if stopped.is_set():
                    return {"status": "cancelled"}
                proc = subprocess.Popen(
                    args,
                    cwd=directory,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.add(proc)
            try:
                code = proc.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                code = 124
            finally:
                with process_lock:
                    processes.discard(proc)
        path = directory / "result.json"
        result = (
            json.loads(path.read_text()) if path.exists() else {"stage": "bootstrap"}
        )
        result.update(
            case=case["case"],
            B=case["B"],
            shape_set=a.suite,
            exit_code=code,
            gpu_uuid=device["uuid"],
            physical_gpu_index=device["index"],
            wall_s=time.monotonic() - started,
            budget_s=budget,
            directory=str(directory),
        )
        if "--candidate" in extra:
            result["candidate_graph"] = extra[extra.index("--candidate") + 1]
        if code != 0:
            result["status"] = (
                "cancelled"
                if stopped.is_set()
                else "timeout"
                if code == 124
                else "process_failure"
            )
        write_json(path, result)
        print(
            case["case"],
            case["B"],
            directory.name,
            result.get("status"),
            result.get("effective_tflops"),
            flush=True,
        )
        return result

    def run_case(case, device):
        directory = a.output / "cases" / case["case"] / f"B{case['B']}"
        directory.mkdir(parents=True)
        case.update(
            suite=a.suite,
            source=a.source,
            cache_kind=a.cache_kind if a.source == "cache" else None,
            fp32_accum=a.fp32_accum,
            fold_zero_epsilon=a.fold_zero_epsilon,
            gpu_uuid=device["uuid"],
            status="running",
        )
        write_json(directory / "case.json", case)
        begin = time.monotonic()
        if a.source in ("fresh", "direct"):
            mode = "search" if a.source == "fresh" else "export"
            result = launch(
                ["--mode", mode],
                directory / mode,
                device,
                min(a.search_budget_s, a.case_budget_s),
                case,
            )
            case["status"] = result.get("status", "failed")
            cache = directory / "search/fresh_candidates.json"
            if a.source == "direct":
                original = directory / "export/input_graph.json"
                cache = directory / "original-as-candidate.json"
                if original.exists():
                    write_json(cache, [json.loads(original.read_text())])
        else:
            cache = cache_for(a.cache_kind, case["case"], case["B"], a.cache_dir)
            case["status"] = "cache_missing" if not cache.exists() else "not_measured"
        if cache.exists() and not stopped.is_set():
            try:
                graphs = json.loads(cache.read_text())
                if not isinstance(graphs, list):
                    raise ValueError("Expected an array of candidate graphs")
            except (ValueError, OSError) as error:
                case.update(status="checkpoint_parse_failure", error=str(error))
                graphs = []
            case["candidate_count"] = len(graphs)
            write_json(
                directory / "provenance.json",
                {
                    "source": str(cache),
                    "kind": a.source,
                    "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                    "count": len(graphs),
                },
            )
            indices = (
                a.candidate_indices
                if a.candidate_indices is not None
                else range(len(graphs))
            )
            indices = [i for i in indices if i < len(graphs)]
            if a.max_candidates:
                indices = indices[: a.max_candidates]
            attempted = set()
            for index in indices:
                if stopped.is_set():
                    break
                candidate = directory / f"candidate-{index}.json"
                write_json(candidate, graphs[index])
                for backend in a.backends:
                    left = a.case_budget_s - (time.monotonic() - begin)
                    if left < 1 or stopped.is_set():
                        break
                    result = launch(
                        [
                            "--mode",
                            "evaluate",
                            "--backend",
                            backend,
                            "--candidate",
                            str(candidate),
                        ],
                        directory / f"{backend}-{index}",
                        device,
                        min(a.candidate_budget_s, left),
                        case,
                    )
                    result["candidate_index"] = index
                    write_json(directory / f"{backend}-{index}" / "result.json", result)
                    attempted.add(index)
                    case["status"] = result.get("status", "failed")
                if time.monotonic() - begin >= a.case_budget_s:
                    break
            case["attempted_candidates"] = len(attempted)
            case["candidate_prefix_truncated"] = len(attempted) < len(graphs)
        if case["status"] == "ok" and case.get("attempted_candidates", 0) == 0:
            case["status"] = "no_timed_candidates"
        case["wall_s"] = time.monotonic() - begin
        write_json(directory / "case.json", case)

    jobs = []
    for case in cases:
        batches = a.batch_sizes or (
            [1, 8]
            if a.suite == "ae" and case == "attention"
            else [1]
            if case == "gated_mlp"
            else [1, 16]
        )
        jobs += [
            {"case": "rms_ffn_swiglu" if case == "rmsnorm" else case, "B": b}
            for b in batches
        ]

    def run_device(device, subset):
        for case in subset:
            if stopped.is_set():
                break
            run_case(case, device)

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(devices)
        ) as executor:
            futures = [
                executor.submit(run_device, device, jobs[i :: len(devices)])
                for i, device in enumerate(devices)
            ]
            for future in futures:
                future.result()
    finally:
        rows = summarize(a.output)
        write_json(
            a.output / "run.json",
            {
                "interrupted_by_signal": interrupted[0],
                "case_count": len(rows),
                "validated_cases": sum(r["status"] == "ok" for r in rows),
                "timed_cases": sum(r["fastest_timed_tflops"] is not None for r in rows),
            },
        )
    return 128 + interrupted[0] if interrupted[0] else 0


if __name__ == "__main__":
    raise SystemExit(main())
