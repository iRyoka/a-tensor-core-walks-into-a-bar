import torch
import triton.testing as tt

device = 'cuda'

batch_size = 1024
# batch_size = 512
vec_sizes = [32_768 * i for i in range(1, 5)]  # example varying problem size
n_batches = 10

providers = ["Single-buffered", "Double-buffered"]

configs = [tt.Benchmark(
    x_names=["vec_size"],
    x_vals=vec_sizes,
    line_arg="pipeline",
    line_vals=providers,
    line_names=providers,
    styles=[("blue", "-"), ("green", "-")],
    ylabel="Time (ms)",
    plot_name="Memory-bound pipeline",
    args={"batch_size": batch_size, "n_batches": n_batches},
)]

@tt.perf_report(configs)
def benchmark(vec_size, pipeline, batch_size, n_batches):
    # Prepare CPU data for this vec_size
    cpu_data = [torch.randn(batch_size, vec_size, device='cpu') for _ in range(n_batches)]
    cpu_results = [torch.empty(batch_size, vec_size, device='cpu') for _ in range(n_batches)]

    def single_buffered():
        torch.cuda.synchronize()
        for i in range(n_batches):
            inp_gpu = cpu_data[i].to(device)
            out_gpu = torch.empty_like(inp_gpu, device=device)
            out_gpu.copy_(inp_gpu)
            cpu_results[i].copy_(out_gpu, non_blocking=False)
        torch.cuda.synchronize()

    def double_buffered():
        cpu_buffers = [torch.empty_like(cpu_data[0], pin_memory=True) for _ in range(2)]
        gpu_buffers = [torch.empty_like(cpu_data[0], device=device) for _ in range(2)]
        result_buffers = [torch.empty_like(cpu_data[0], device=device) for _ in range(2)]

        stream0 = torch.cuda.Stream()
        stream1 = torch.cuda.Stream()

        torch.cuda.synchronize()
        for i in range(n_batches + 1):
            buf_idx = i % 2
            prev_idx = (i - 1) % 2

            if i < n_batches:
                cpu_buffers[buf_idx].copy_(cpu_data[i], non_blocking=True)

            if i > 0:
                with torch.cuda.stream(stream0):
                    gpu_buffers[prev_idx].copy_(cpu_buffers[prev_idx], non_blocking=True)
                    result_buffers[prev_idx].copy_(gpu_buffers[prev_idx])
                with torch.cuda.stream(stream1):
                    cpu_results[prev_idx].copy_(result_buffers[prev_idx], non_blocking=True)

        torch.cuda.synchronize()

    fn_map = {"Single-buffered": single_buffered, "Double-buffered": double_buffered}

    # Run benchmark (no extra args, just default iterations)
    ms = tt.do_bench(fn_map[pipeline])
    return ms

