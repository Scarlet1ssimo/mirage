import mirage as mi
import torch
import torch.nn.functional as F
import triton

D = 96
K = 192
N = 96
EPS = 1e-5
WARMUP_ITERS = 25
BENCH_ITERS = 100
QUANTILES = (0.5, 0.1, 0.9)
M_VALUES = [2 ** i for i in range(11, 12)]


def rmsnorm_swiglu_pytorch(
    X: torch.Tensor,
    WV: torch.Tensor,
    U: torch.Tensor,
    eps: float = EPS,
):
    normalized_X = F.rms_norm(X, normalized_shape=X.shape[-1:], eps=eps)

    X_WV = torch.matmul(normalized_X, WV.T)
    gate, value = X_WV.chunk(2, dim=-1)
    swiglu = F.silu(gate) * value

    output = torch.matmul(swiglu, U.T)
    return output


def build_opt_graph(graph):
    configs = [
        {"backend": "triton", "warmup_iters": 16, "profile_iters": 80},
    ]
    last_error = None
    for cfg in configs:
        try:
            print(f"Trying superoptimize config: {cfg}")
            return graph.superoptimize(**cfg), cfg
        except Exception as err:
            last_error = err
            print(f"Superoptimize failed with {cfg}: {err}")
    raise RuntimeError(f"All superoptimize attempts failed: {last_error}")


def benchmark_ms(fn, warmup_iters: int, bench_iters: int) -> float:
    # Match triton-eval-bench timing path (median from triton.testing.do_bench).
    ms, _, _ = triton.testing.do_bench(
        fn,
        warmup=warmup_iters,
        rep=bench_iters,
        quantiles=list(QUANTILES),
    )
    return float(ms)


def get_compile_fn(fn):
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    return torch.compile(fn, mode="max-autotune")


if __name__ == "__main__":
    mi.set_gpu_device_id(0)
    mirage_dtype = mi.float16
    torch_dtype = torch.float16
    summary_rows = []

    for M in M_VALUES:
        print(f"\n================ Sweep M={M} ================")
        print("Searching for Graph 1 (RMSNorm + MatMul for Gate+Value)...")
        graph1 = mi.new_kernel_graph()
        x_node = graph1.new_input(dims=(M, D), dtype=mirage_dtype)
        WV_T_node = graph1.new_input(dims=(D, 2 * K), dtype=mirage_dtype)

        normalized_x_node = graph1.rms_norm(x_node, normalized_shape=(D,))
        UV = graph1.matmul(normalized_x_node, WV_T_node)
        graph1.mark_output(UV)
        opt_graph1, cfg1 = build_opt_graph(graph1)
        print(f"Chosen config for graph1: {cfg1}")

        print("Searching for Graph 2 (Down Projection)...")
        graph2 = mi.new_kernel_graph()
        swiglu_node = graph2.new_input(dims=(M, K), dtype=mirage_dtype)
        U_T_node = graph2.new_input(dims=(K, N), dtype=mirage_dtype)

        O_node = graph2.matmul(swiglu_node, U_T_node)
        graph2.mark_output(O_node)
        opt_graph2, cfg2 = build_opt_graph(graph2)
        print(f"Chosen config for graph2: {cfg2}")

        @torch.inference_mode()
        def rmsnorm_swiglu_mirage(X: torch.Tensor, WV: torch.Tensor, U: torch.Tensor):
            WV_T = WV.T.contiguous()
            U_T = U.T.contiguous()

            UV_out = opt_graph1(inputs=[X, WV_T])[0]

            gate, value = UV_out.chunk(2, dim=-1)
            swiglu_out = F.silu(gate) * value

            final_out = opt_graph2(inputs=[swiglu_out, U_T])[0]
            return final_out

        print("\n--- Testing Correctness ---")
        # Match triton-eval-bench input generation.
        X_t = torch.randn(M, D, dtype=torch_dtype, device="cuda:0")
        WV_t = torch.randn(2 * K, D, dtype=torch_dtype, device="cuda:0")
        U_t = torch.randn(N, K, dtype=torch_dtype, device="cuda:0")

        out_pt = rmsnorm_swiglu_pytorch(X_t, WV_t, U_t, eps=0.0)
        out_mi = rmsnorm_swiglu_mirage(X_t, WV_t, U_t)

        compiled_rmsnorm_swiglu = get_compile_fn(rmsnorm_swiglu_pytorch)
        # Trigger compile ahead of timing so latency excludes one-time compilation.
        _ = compiled_rmsnorm_swiglu(X_t, WV_t, U_t, eps=0.0)
        torch.cuda.synchronize()
        out_pt_compile = compiled_rmsnorm_swiglu(X_t, WV_t, U_t, eps=0.0)

        abs_diff = torch.abs(out_pt - out_mi)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        has_nan = torch.isnan(out_pt).any().item(
        ) or torch.isnan(out_mi).any().item()
        allclose = (not has_nan) and torch.allclose(
            out_pt, out_mi, rtol=3e-2, atol=5e-2)
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Has NaN: {has_nan}")
        print(f"Allclose (rtol=3e-2, atol=5e-2): {allclose}")

        compile_abs_diff = torch.abs(out_pt - out_pt_compile)
        compile_max_diff = compile_abs_diff.max().item()
        compile_mean_diff = compile_abs_diff.mean().item()
        compile_has_nan = torch.isnan(out_pt_compile).any().item()
        compile_allclose = (not compile_has_nan) and torch.allclose(
            out_pt, out_pt_compile, rtol=3e-2, atol=5e-2
        )
        print(f"Compile max absolute difference vs eager: {compile_max_diff:.6f}")
        print(f"Compile mean absolute difference vs eager: {compile_mean_diff:.6f}")
        print(f"Compile has NaN: {compile_has_nan}")
        print(f"Compile allclose vs eager (rtol=3e-2, atol=5e-2): {compile_allclose}")

        print("\n--- Running Performance Benchmark ---")
        # Use a fresh compiled callable for timing to match benchmark runner behavior.
        compiled_rmsnorm_swiglu_bench = get_compile_fn(rmsnorm_swiglu_pytorch)

        def run_torch_compile_bench():
            with torch.inference_mode():
                return compiled_rmsnorm_swiglu_bench(X_t, WV_t, U_t, eps=0.0)

        latency_ms = benchmark_ms(
            lambda: rmsnorm_swiglu_mirage(X_t, WV_t, U_t),
            WARMUP_ITERS,
            BENCH_ITERS,
        )
        compile_latency_ms = benchmark_ms(
            run_torch_compile_bench,
            WARMUP_ITERS,
            BENCH_ITERS,
        )

        matmul1_flops = 2.0 * M * D * (2 * K)
        matmul2_flops = 2.0 * M * K * N
        triton_eval_flops = matmul1_flops + matmul2_flops
        matmul_tflops = triton_eval_flops / (latency_ms * 1e-3) / 1e12

        print(f"Mirage Mean Latency (End-to-End): {latency_ms:.4f} ms")
        print(f"Effective TFLOPS (End-to-End): {matmul_tflops:.3f}")

        compile_tflops_est = triton_eval_flops / (compile_latency_ms * 1e-3) / 1e12
        print(f"torch.compile Mean Latency (End-to-End): {compile_latency_ms:.4f} ms")
        print(
            f"torch.compile Effective TFLOPS (matmul+elementwise-est): {compile_tflops_est:.3f}"
        )

        speedup = latency_ms / compile_latency_ms
        summary_rows.append(
            {
                "M": M,
                "mirage_ms": latency_ms,
                "compile_ms": compile_latency_ms,
                "mirage_tflops": matmul_tflops,
                "compile_tflops": compile_tflops_est,
                "speedup": speedup,
            }
        )

    print("\n================ Final Summary (one row per M) ================")
    print(
        "M\tMirage_ms\tCompile_ms\tMirage_TFLOPS\tCompile_TFLOPS\tCompile_speedup"
    )
    for row in summary_rows:
        print(
            f"{row['M']}\t{row['mirage_ms']:.4f}\t{row['compile_ms']:.4f}\t"
            f"{row['mirage_tflops']:.3f}\t{row['compile_tflops']:.3f}\t{row['speedup']:.2f}x"
        )
