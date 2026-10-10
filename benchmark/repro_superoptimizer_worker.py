"""Worker for reproduce_superoptimizer.py; uses the existing Forest/XLA protocol."""

import argparse, importlib.util, json, math, os, sys, time, traceback, types
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--forest-root", type=Path, required=True)
p.add_argument("--fixture-dir", type=Path)
p.add_argument("--triton-path", type=Path)
p.add_argument("--target-cc", type=int, default=0)
p.add_argument("--pipeline-stages", type=int, default=2)
p.add_argument("--num-warp-groups", type=int, default=2)
p.add_argument("--allow-invalid-output", action="store_true")
p.add_argument("--case", required=True)
p.add_argument("--B", type=int, required=True)
p.add_argument("--shape", choices=["forest", "ae"], default="forest")
p.add_argument("--mode", choices=["search", "evaluate", "export"], required=True)
p.add_argument("--norm-grid", action="store_true")
p.add_argument("--max-tb-ops", type=int, default=-1)
p.add_argument("--search-config", choices=["mlp", "attention"])
p.add_argument("--precise-math", action="store_true")
p.add_argument("--fp32-accum", action="store_true")
p.add_argument("--decompose-norm", action="store_true")
p.add_argument("--fold-zero-epsilon", action="store_true")
p.add_argument("--backend", choices=["cuda", "triton"], default="cuda")
p.add_argument("--candidate", type=Path)
p.add_argument("--directory", type=Path, required=True)
a = p.parse_args()
a.directory.mkdir(parents=True, exist_ok=True)
os.chdir(a.directory)

if a.backend == "triton" and a.triton_path:
    sys.path.insert(0, str(a.triton_path))
import fcntl

gpu_lock = open(
    "/tmp/mirage-benchmark-"
    + os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    + ".lock",
    "w",
)
fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX)
import numpy as np
import torch
import triton
import mirage as mi
import mirage.core as mc
import hashlib
from mirage.kernel import gen_empty_tensor
from reproduce_superoptimizer import write_json

OUT = Path(__file__).resolve().parent
FIX = a.fixture_dir
ROOT = a.forest_root.resolve()
for name, path in [("forest_benchmarks", ROOT / "benchmarks/src/forest_benchmarks")]:
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    sys.modules[name] = m
from forest_benchmarks.timing import measure

spec = importlib.util.spec_from_file_location(
    "numerics", ROOT / "codegen/src/forest/codegen/numerics.py"
)
num = importlib.util.module_from_spec(spec)
spec.loader.exec_module(num)
torch.set_grad_enabled(False)
torch.manual_seed(1729)
np.random.seed(1729)
physical_cc = (
    torch.cuda.get_device_properties(0).major * 10
    + torch.cuda.get_device_properties(0).minor
)
target_cc = a.target_cc or physical_cc
if target_cc not in (80, 90) or target_cc > physical_cc:
    raise ValueError(
        "This AE compiler supports target CC 80/90; requested target cannot exceed the physical GPU"
    )
start = time.monotonic()
r = {
    "status": "running",
    "allow_invalid_output": a.allow_invalid_output,
    "core_sha256": hashlib.sha256(Path(mc.__file__).read_bytes()).hexdigest(),
    "stage": "graph",
    "case": a.case,
    "B": a.B,
    "shape_set": a.shape,
    "backend": a.backend,
    "seed": 1729,
    "software": {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": sys.version,
    },
    "numerical_limits": {"relative_l2": 0.005, "normalized_max_abs": 0.02},
    "timing": {"do_bench": True, "flush_l2": True, "warmup": 10, "rep_ms": 200},
    "norm_grid": a.norm_grid,
    "max_tb_ops": a.max_tb_ops,
    "search_config": a.search_config,
    "precise_math": a.precise_math,
    "fp32_accumulation_adapter": a.fp32_accum and a.backend == "cuda",
    "decompose_norm": a.decompose_norm,
    "fold_zero_epsilon": a.fold_zero_epsilon,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "target_cc": target_cc,
    "physical_cc": physical_cc,
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
}
meta = []


def save():
    r["elapsed_s"] = time.monotonic() - start
    write_json(a.directory / "result.json", r)


def stage(s):
    r["stage"] = s
    save()
    print("STAGE", s, flush=True)


def graph_input(g, dims, strides=None, constant=None):
    if strides is None:
        st = []
        n = 1
        for d in reversed(dims):
            st.append(n)
            n *= d
        strides = tuple(reversed(st))
    meta.append({"shape": list(dims), "strides": list(strides), "constant": constant})
    # Cython accepts dense views with transposed matrix dimensions; Python wrapper
    # only accepts a whole-tensor row/column ordering. No input data transformation.
    return g.cygraph.new_input(tuple(dims), tuple(strides), mi.float16)


def rms(g, x, d):
    if not a.decompose_norm:
        return g.rms_norm(x, normalized_shape=(d,))
    count = graph_input(g, (1,) * x.num_dims, constant=d)
    return g.div(x, g.sqrt(g.div(g.reduction(g.square(x), x.num_dims - 1), count)))


def graph():
    g = mi.new_kernel_graph()
    b = a.B
    if a.case in ("attention", "gqa"):
        if a.shape == "forest":
            outer = b * (32 if a.case == "attention" else 8)
            m = 4096 * (1 if a.case == "attention" else 4)
            d = 128
            n = 4096
        else:
            outer = b * (32 if a.case == "attention" else 2)
            m = 16 if a.case == "attention" else 256
            d = 64
            n = 4096
        q = graph_input(g, (outer, m, d))
        k = graph_input(
            g, (outer, d, n), strides=(d * n, 1, d) if a.shape == "forest" else None
        )
        v = graph_input(g, (outer, n, d))
        z = g.matmul(q, k)
        if a.shape == "forest":
            z = g.div(z, graph_input(g, (1, 1, 1), constant=math.sqrt(d)))
        e = g.exp(z)
        s = g.reduction(e, 2)
        o = g.matmul(g.div(e, s), v)
    elif a.case == "layernorm_matmul":
        x = graph_input(g, (b, 8192))
        y = graph_input(g, (8192, 28672))
        negative_mean = g.mul(
            g.reduction(x, 1), graph_input(g, (1, 1), constant=-1 / 8192)
        )
        centered = g.add(x, negative_mean)
        o = g.matmul(rms(g, centered, 8192), y)
    elif a.case == "rms_ffn_swiglu":
        if a.shape == "ae":
            x = graph_input(g, (2 * b, 4096))
            w = graph_input(g, (4096, 6144))
            o = g.matmul(g.rms_norm(x, normalized_shape=(4096,)), w)
        else:
            x = graph_input(g, (b * 4096, 96))
            w = graph_input(g, (96, 192))
            v = graph_input(g, (96, 192))
            u = graph_input(g, (192, 96))
            z = rms(g, x, 96)
            o = g.matmul(g.mul(g.silu(g.matmul(z, w)), g.matmul(z, v)), u)
    elif a.case == "efficient_vit":
        q = graph_input(g, (b, 128, 128))
        k = graph_input(g, (b, 128, 128))
        v = graph_input(g, (b, 128, 128), strides=(128 * 128, 1, 128))
        kv = g.matmul(k, v)
        den = g.matmul(q, g.reduction(k, 2))
        o = (
            g.div(g.matmul(q, kv), den)
            if a.fold_zero_epsilon
            else g.div(
                g.matmul(q, kv), g.add(den, graph_input(g, (1, 1, 1), constant=1e-15))
            )
        )
    elif a.case == "gated_mlp":
        x = graph_input(g, (8, 4096))
        w = graph_input(g, (4096, 4096))
        v = graph_input(g, (4096, 4096))
        o = g.mul(g.silu(g.matmul(x, w)), g.matmul(x, v))
    else:
        raise ValueError(a.case)
    g.mark_output(o)
    g.to_json(str(a.directory / "input_graph.json"))
    r["inputs"] = meta
    save()
    return g


def tensors_and_reference():
    if a.shape == "forest":
        if FIX is None:
            raise ValueError(
                "Forest suite requires --fixture-dir containing inputs.npz/reference.npy/settings.json"
            )
        folder = (
            FIX
            / {
                "gqa": "gqa-attention",
                "layernorm_matmul": "layernorm-matmul",
                "rms_ffn_swiglu": "rms-ffn-swiglu",
                "efficient_vit": "efficient-vit",
            }.get(a.case, a.case)
            / f"B{a.B}"
        )
        settings = json.loads((folder / "settings.json").read_text())
        r["forest_settings"] = settings
        r["effective_flops"] = settings["effective_flops"]
        with np.load(folder / "inputs.npz") as f:
            xs = [
                torch.from_numpy(f[f"input{i}"].copy()).cuda()
                for i in range(len(f.files))
            ]
        ref = torch.from_numpy(np.load(folder / "reference.npy")).cuda()
        if a.case in ("attention", "gqa"):
            outer = a.B * (32 if a.case == "attention" else 8)
            m = 4096 * (1 if a.case == "attention" else 4)
            xs = [
                xs[0].view(outer, m, 128),
                xs[1].view(outer, 4096, 128).transpose(-1, -2),
                xs[2].view(outer, 4096, 128),
                torch.full(
                    (1, 1, 1), math.sqrt(128), dtype=torch.float16, device="cuda"
                ),
            ]
            ref = ref.view(outer, m, 128)
        elif a.case == "layernorm_matmul":
            xs.append(torch.full((1, 1), -1 / 8192, dtype=torch.float16, device="cuda"))
            if a.decompose_norm:
                xs.append(torch.full((1, 1), 8192, dtype=torch.float16, device="cuda"))
        elif a.case == "rms_ffn_swiglu" and a.decompose_norm:
            xs.append(torch.full((1, 1), 96, dtype=torch.float16, device="cuda"))
        elif a.case == "efficient_vit":
            xs[2] = xs[2].transpose(-1, -2)
            if not a.fold_zero_epsilon:
                xs.append(
                    torch.full((1, 1, 1), 1e-15, dtype=torch.float16, device="cuda")
                )
            r["epsilon_fp16_underflow"] = True
    else:
        xs = [torch.randn(m["shape"], device="cuda", dtype=torch.float16) for m in meta]
        if a.case in ("attention", "gqa"):
            q, k, v = [x.double() for x in xs]
            ref = torch.softmax(q @ k, dim=-1) @ v
            r["effective_flops"] = (
                4 * q.shape[0] * q.shape[1] * k.shape[-1] * q.shape[-1]
            )
            r["semantics"] = "AE unscaled, noncausal attention"
        elif a.case == "rms_ffn_swiglu":
            x, w = [x.double() for x in xs]
            ref = (x / x.square().mean(-1, keepdim=True).sqrt()) @ w
            r["effective_flops"] = 2 * x.shape[0] * x.shape[1] * w.shape[1]
            r["semantics"] = "AE RMSNorm + one Matmul; not Forest full FFN"
        elif a.case == "gated_mlp":
            x, w, v = [x.double() for x in xs]
            gate = x @ w
            ref = torch.nn.functional.silu(gate) * (x @ v)
            r["effective_flops"] = 4 * x.shape[0] * x.shape[1] * w.shape[1]
            r["semantics"] = "AE gated MLP; no RMSNorm or final Matmul"
    return xs, ref


def check(x, ref):
    metrics = num.numerical_metrics(x, ref)
    metrics["output_nan"] = torch.isnan(x).sum().item()
    metrics["output_inf"] = torch.isinf(x).sum().item()
    metrics["output_elements"] = x.numel()
    metrics["passed"] = (
        metrics["output_nonfinite"] == 0
        and metrics["reference_nonfinite"] == 0
        and metrics["relative_l2"] < 0.005
        and metrics["normalized_max_abs"] < 0.02
    )
    return metrics


save()
try:
    if a.mode == "export":
        g = graph()
        r.update(status="ok", lowering_only=True)
    elif a.mode == "search":
        g = graph()
        stage("search")
        graphs = mi.search(
            g.cygraph,
            griddims=[(a.B, 1, 1)] if a.norm_grid else None,
            franges=[1] if a.norm_grid else None,
            max_num_threadblock_graph_op=a.max_tb_ops,
            max_num_kernel_graph_op=10 if a.shape == "forest" else -1,
            num_search_thread=8,
            previous_checkpoint=str(a.directory / "fresh_candidates.json"),
            verbose=True,
            default_config=a.search_config
            or ("attention" if a.case in ("attention", "gqa") else "mlp"),
        )
        r.update(
            status="ok" if graphs else "no_candidates", candidate_count=len(graphs)
        )
    else:
        stage("load_candidate")
        g = mi.new_kernel_graph()
        g.from_json(str(a.candidate))
        meta = [
            {
                "shape": list(g.cygraph.get_input_dtensor_shape_and_stride(t)[0]),
                "strides": list(g.cygraph.get_input_dtensor_shape_and_stride(t)[1]),
                "constant": None,
            }
            for t in g.cygraph.get_input_dtensors()
        ]
        r["inputs"] = meta
        stage("load_inputs")
        xs, ref = tensors_and_reference()
        r["output_shape"] = list(ref.shape)
        save()
        stage("compile")
        if a.backend == "cuda":
            import mirage.kernel as kernel_module

            original_command = kernel_module.get_cc_cmd

            def compiler_command(*args, **kwargs):
                cmd = original_command(*args, **kwargs)
                if a.precise_math:
                    cmd = [x for x in cmd if x != "-use_fast_math"]
                (a.directory / "nvcc-command.json").write_text(
                    json.dumps(cmd, indent=2)
                )
                return cmd

            kernel_module.get_cc_cmd = compiler_command
            if a.fp32_accum:
                import mirage.kernel as kernel_module

                original_generator = kernel_module.generate_cuda_program

                def adapted_generator(*args, **kwargs):
                    result = original_generator(*args, **kwargs)
                    (a.directory / "generated-original.cu").write_text(
                        result["code"] + kernel_module.HARD_CODE
                    )
                    if "Hopper_Matmul<" in result["code"]:
                        raise ValueError(
                            "FP32 adapter supports SM80 MMA/cuBLAS only, not Hopper WGMMA; use ordinary cache or disable --fp32-accum"
                        )
                    replacements = sum(
                        result["code"].count(token)
                        for token in (
                            "SM80_16x8x16_F16F16F16F16_TN",
                            "SM80_16x8x8_F16F16F16F16_TN",
                            "CUBLAS_COMPUTE_16F",
                        )
                    )
                    if not replacements:
                        raise ValueError(
                            "No supported FP16 accumulation sites found; refusing to label this kernel FP32-adapted"
                        )
                    r["fp32_replacements"] = replacements
                    save()
                    result["code"] = (
                        result["code"]
                        .replace(
                            "SM80_16x8x16_F16F16F16F16_TN",
                            "SM80_16x8x16_F32F16F16F32_TN",
                        )
                        .replace(
                            "SM80_16x8x8_F16F16F16F16_TN", "SM80_16x8x8_F32F16F16F32_TN"
                        )
                        .replace("CUBLAS_COMPUTE_16F", "CUBLAS_COMPUTE_32F")
                    )
                    return result

                kernel_module.generate_cuda_program = adapted_generator
            res = g.compile(
                inputs=xs,
                target_cc=target_cc,
                file_id=0,
                pipeline_stages=a.pipeline_stages,
                num_warp_groups=a.num_warp_groups,
            )
            if not g.valid_kernels():
                raise RuntimeError(g.get_error_message())
            g.backend = "cuda"
            # Preallocate exactly as Mirage.cuda_call does, so benchmark includes kernel
            # execution and launch, without repeatedly allocating large scratch/output.
            scratch = torch.empty(res["buf_size"], dtype=torch.uint8, device="cuda")
            out = [
                gen_empty_tensor(
                    m["alloc_size"], m["shape"], m["strides"], device="cuda"
                )
                for m in res["output_directives"]
            ]
            prof = torch.empty(
                res["profiler_buf_size"], dtype=torch.uint64, device="cuda"
            )
            ip = [x.data_ptr() for x in xs]
            op = [x.data_ptr() for x in out]
            stream = torch.cuda.current_stream().cuda_stream

            def run():
                g.run(ip, op, scratch.data_ptr(), stream, prof.data_ptr())

            r["tuning"] = {
                k: res[k]
                for k in (
                    "buf_size",
                    "max_smem_size",
                    "output_directives",
                    "profiler_buf_size",
                )
            }
            r["tuning"].update(
                target_cc=target_cc,
                pipeline_stages=a.pipeline_stages,
                num_warp_groups=a.num_warp_groups,
                buffer_policy="preallocated",
            )
        else:
            os.environ["KERNELS_PATH"] = str(
                Path(__file__).resolve().parents[1]
                / "include/mirage/triton_transpiler/runtime"
            )
            res = mi.generate_triton_program(g.cygraph, target_cc=target_cc)
            path = a.directory / "generated.py"
            path.write_text(res["code"])
            sp = importlib.util.spec_from_file_location("generated", path)
            mod = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(mod)
            out = [
                torch.empty(shape, device="cuda", dtype=torch.float16)
                for shape in res["output_shapes"]
            ]

            def run():
                mod.execute_mugraph(*xs, *out)

            r["tuning"] = {
                "target_cc": target_cc,
                "output_shapes": res["output_shapes"],
                "buffer_policy": "outputs preallocated, intermediates as emitted by Mirage",
            }
        stage("validation")
        run()
        torch.cuda.synchronize()
        r["correctness"] = check(out[0], ref)
        save()
        if not r["correctness"]["passed"] and not a.allow_invalid_output:
            raise ValueError("numerical validation failed")
        stage("measure")
        r["cooldown"] = {}
        lat = measure(
            run,
            warmup=10,
            benchmark_ms=200,
            cooldown_s=5,
            device=torch.device("cuda:0"),
            cooldown_report=r["cooldown"],
            phase_directory=a.directory,
        )
        r["post_validation"] = check(out[0], ref)
        if not r["post_validation"]["passed"] and not a.allow_invalid_output:
            raise ValueError("numerical validation failed after timing")
        if not math.isfinite(lat["median"]) or lat["median"] <= 0:
            raise ValueError("invalid timing")
        r.update(
            status="ok"
            if r["correctness"]["passed"] and r["post_validation"]["passed"]
            else "measured_invalid",
            latency_ms=lat,
            effective_tflops=r["effective_flops"] / (lat["median"] * 1e9),
        )
except Exception as e:
    r.update(
        status="incorrect" if "numerical validation failed" in str(e) else "failed",
        error=traceback.format_exc(),
    )
    (a.directory / "error.log").write_text(r["error"])
    print(r["error"], flush=True)
save()
print("RESULT", r["status"], r.get("effective_tflops"), flush=True)
