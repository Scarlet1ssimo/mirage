import mirage as mi
import torch

@torch.inference_mode()
def layernorm_matmul_pytorch_plain(
    x: torch.Tensor,
    y_t: torch.Tensor,
    eps: float,
):
    # Match mugraph math order as much as possible for fair correctness checks.
    k = x.shape[-1]
    inv_k = 1.0 / float(k)
    mean = x.sum(dim=-1, keepdim=True) * inv_k
    centered = x - mean
    var = centered.square().sum(dim=-1, keepdim=True) * inv_k
    normalized = centered / torch.sqrt(var + eps)
    return torch.matmul(normalized, y_t)


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
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(bench_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / bench_iters

if __name__ == "__main__":
    mi.set_gpu_device_id(0)
    mirage_dtype = mi.float16
    torch_dtype = torch.float16

    M, K, N = 1024, 1024, 1024
    EPS = 1e-5
    WARMUP_ITERS = 20
    BENCH_ITERS = 200

    print("Building Mirage Graph for LayerNorm + MatMul (LayerNorm in mugraph)...")
    graph = mi.new_kernel_graph()
    x_node = graph.new_input(dims=(M, K), dtype=mirage_dtype)
    y_t_node = graph.new_input(dims=(K, N), dtype=mirage_dtype)

    inv_k_node = graph.new_input(dims=(M, 1), dtype=mirage_dtype)
    neg_one_node = graph.new_input(dims=(M, 1), dtype=mirage_dtype)
    eps_node = graph.new_input(dims=(M, 1), dtype=mirage_dtype)
    one_node = graph.new_input(dims=(M, 1), dtype=mirage_dtype)

    mean_sum = graph.reduction(x_node, 1)
    mean = graph.mul(mean_sum, inv_k_node)
    neg_mean = graph.mul(mean, neg_one_node)
    centered = graph.add(x_node, neg_mean)
    var_sum = graph.reduction(graph.square(centered), 1)
    var = graph.mul(var_sum, inv_k_node)
    std = graph.sqrt(graph.add(var, eps_node))
    inv_std = graph.div(one_node, std)
    x_norm = graph.mul(centered, inv_std)
    out_node = graph.matmul(x_norm, y_t_node)
    graph.mark_output(out_node)
    opt_graph, chosen_cfg = build_opt_graph(graph)
    print(f"Chosen config: {chosen_cfg}")

    x = torch.randn(M, K, dtype=torch_dtype, device="cuda:0")
    y = torch.randn(N, K, dtype=torch_dtype, device="cuda:0")
    y_t = y.T.contiguous()
    inv_k_t = torch.full((M, 1), 1.0 / float(K), dtype=torch_dtype, device="cuda:0")
    neg_one_t = torch.full((M, 1), -1.0, dtype=torch_dtype, device="cuda:0")
    eps_t = torch.full((M, 1), EPS, dtype=torch_dtype, device="cuda:0")
    one_t = torch.ones((M, 1), dtype=torch_dtype, device="cuda:0")

    @torch.inference_mode()
    def run_reference():
        return layernorm_matmul_pytorch_plain(x, y_t, eps=EPS)

    @torch.inference_mode()
    def run_mirage():
        return opt_graph(inputs=[x, y_t, inv_k_t, neg_one_t, eps_t, one_t])[0]

    print("\n--- Testing Correctness ---")
    out_pt = run_reference()
    out_mi = run_mirage()
    abs_diff = torch.abs(out_pt - out_mi)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    allclose = torch.allclose(out_pt, out_mi, rtol=3e-2, atol=5e-2)
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Allclose (rtol=3e-2, atol=5e-2): {allclose}")

    print("\n--- Running Performance Benchmark ---")
    latency_ms = benchmark_ms(run_mirage, WARMUP_ITERS, BENCH_ITERS)
    matmul_flops = 2.0 * M * N * K
    layernorm_flops_est = 7.0 * M * K
    matmul_tflops = matmul_flops / (latency_ms * 1e-3) / 1e12
    end2end_tflops_est = (matmul_flops + layernorm_flops_est) / (latency_ms * 1e-3) / 1e12
    print(f"Mirage Mean Latency (End-to-End): {latency_ms:.4f} ms")
    print(f"Effective TFLOPS (matmul-only): {matmul_tflops:.3f}")
    print(f"Effective TFLOPS (matmul+layernorm-est): {end2end_tflops_est:.3f}")