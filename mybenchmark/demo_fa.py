import math
import os

import mirage as mi
import torch

H = int(os.getenv("FA_H", "2"))
M = int(os.getenv("FA_M", "128"))
N = int(os.getenv("FA_N", "1024"))
D = int(os.getenv("FA_D", "64"))
L = int(os.getenv("FA_L", "64"))
WARMUP_ITERS = int(os.getenv("FA_WARMUP", "10"))
BENCH_ITERS = int(os.getenv("FA_BENCH", "80"))


@torch.inference_mode()
def flashattn_pytorch_plain(
    q_scaled: torch.Tensor, k_t: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    attn_scores = torch.matmul(q_scaled, k_t)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_weights, v)


def build_opt_graph(graph):
    configs = [
        {
            "backend": "triton",
            "config": "attention",
            "warmup_iters": 6,
            "profile_iters": 24,
        },
        {
            "backend": "cuda",
            "config": "attention",
            "warmup_iters": 6,
            "profile_iters": 24,
        },
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

    print(f"FA shape: H={H}, M={M}, N={N}, D={D}, L={L}")

    print("Building Mirage Graph for Attention...")
    graph = mi.new_kernel_graph()
    q_node = graph.new_input(dims=(H, M, D), dtype=mirage_dtype)
    k_t_node = graph.new_input(dims=(H, D, N), dtype=mirage_dtype)
    v_node = graph.new_input(dims=(H, N, L), dtype=mirage_dtype)

    a_node = graph.matmul(q_node, k_t_node)
    e_node = graph.exp(a_node)
    s_node = graph.reduction(e_node, 2)
    norm_a_node = graph.div(e_node, s_node)
    o_node = graph.matmul(norm_a_node, v_node)
    graph.mark_output(o_node)

    print("Superoptimizing...")
    opt_graph, chosen_cfg = build_opt_graph(graph)
    print(f"Chosen config: {chosen_cfg}")

    q = torch.randn(H, M, D, dtype=torch_dtype, device="cuda:0")
    k_t = torch.randn(H, D, N, dtype=torch_dtype, device="cuda:0")
    v = torch.randn(H, N, L, dtype=torch_dtype, device="cuda:0")
    q_scaled = (q / math.sqrt(D)).contiguous()

    def run_mirage():
        return opt_graph(inputs=[q_scaled, k_t, v])[0]

    print("\n--- Testing Correctness ---")
    out_pt = flashattn_pytorch_plain(q_scaled, k_t, v)
    out_mi = run_mirage()
    abs_diff = torch.abs(out_pt - out_mi)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    has_nan = torch.isnan(out_pt).any().item() or torch.isnan(out_mi).any().item()
    allclose = (not has_nan) and torch.allclose(out_pt, out_mi, rtol=5e-2, atol=6e-2)
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Has NaN: {has_nan}")
    print(f"Allclose (rtol=5e-2, atol=6e-2): {allclose}")

    print("\n--- Running Performance Benchmark ---")
    latency_ms = benchmark_ms(run_mirage, WARMUP_ITERS, BENCH_ITERS)
    flops = 2.0 * H * M * N * D + 2.0 * H * M * N * L
    tflops = flops / (latency_ms * 1e-3) / 1e12
    print(f"Mirage Mean Latency (End-to-End): {latency_ms:.4f} ms")
    print(f"Effective TFLOPS (matmul-equivalent): {tflops:.3f}")