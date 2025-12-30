# Cuda for curious kids: intro

# 1. What is CUDA (Compute Unified Device Architecture)

> [!NOTE] Main references: 
> https://docs.nvidia.com/cuda/cuda-programming-guide/index.html (very good!)
> https://www.cse.iitd.ac.in/~rijurekha/col730_2022/cudabook.pdf (also very good, but somewhat obsolete in fine details)
> 

1. hardware architecture specification. Allows producing CUDA-compatible GPU devices. Note that TPUs and other inference devices do not use or implement CUDA. Defines `compute capability` -- a set of qualitative (e.g. Tensor Cores, Ray Tracing Cores, TF32 support) and quantitative (SM memory sizes, max number of threads per SM) hardware features. Essentially CC is a version of SM (see later for the definition of SM)
2. Programming model (threads, blocks, grids, kernels, warps (doubles as SIMD lanes in hardware))
3. several layers of API:
	1. CUDA Driver API -- low level C API, includes JIT for translating `PTX` (Parallel Thread Execution -- *virtual* instruction set; partly device-agnostic) into `SASS` (Streaming ASSembly -- physical device assembly instructions set = (almost) machine code). In particular, Driver API defines software features e.g. Unified Memory (backed by Page Migration Engine in hardware).
	2. CUDA Runtime API -- C++ - ish level abstraction. Does mostly(!) whatever Driver API does but trades some control for friendliness. C code written with CUDA Driver or Runtime API is translated to PTX by `nvcc` (compiler).
	3. CUDA Tile API (CUDA 13.1+) removes some thread-level management (SIMT) burden. Can be thought of a C version of Triton.
```c
// Excerpt from
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/vectorAdd/vectorAdd.cu
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/vectorAddDrv/vectorAddDrv.cpp

// You are allowed to do this as a first cuda call
#include <cuda_runtime.h>
float *d_A = NULL;
err = cudaMalloc((void **)&d_A, size);

// --------------------------------------------------
// But Driver API would require smth like
#include <cuda.h>
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>
#include <builtin_types.h>

// Init CUDA and a device driver
checkCudaErrors(cuInit(0));
CUdevice cuDevice = findCudaDeviceDRV(argc, (const char **)argv);;
// Create context (~workspace for a program)
CUcontext cuContext;
CUctxCreateParams ctxCreateParams = {};
checkCudaErrors(cuCtxCreate(&cuContext, &ctxCreateParams, 0, cuDevice));

CUdeviceptr d_A;
checkCudaErrors(cuMemAlloc(&d_A, size));
```

4. The Toolkit: 
	1. dev tools like nvcc, cuda-gdb, profilers
	2. core libraries like cuBLAS (linear algebra), cuDNN (deep learning), cuFFT (Fourier transform), cuSPARSE, cuRAND et al.**

# 2. Hardware Model
CUDA is designed to work in a heterogeneous system, i.e. a host (CPU el al) and device (GPU) tandem.
Device code is comprised of kernels (functions) that are launched (executed) asynchronously relative to host.

![GPU Model](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-cpu-system-diagram.png)

A GPU's compute hardware is partitioned into Graphics Processing Clusters (GPC), then Streaming Multiprocessors (SM), then cores (Tensor, RT and CUDA/SP) and hardware threads called **lanes**. Each SM physically owns two types of memory: a. L1 cache + shared memory is a single physical back called Unified Data Cache (UDC) and can be partitioned based on runtime config of a kernel b. register file. 
Apart from that a GPU also possesses a VRAM also called Device Global Memory or High Band Memory (HBM), an L2-cache (usually not managed by a programmer) and interfaces: 
 - PCIe/SXM (Server eXpress Module) (device -- host);
 - NVLINK (device - device);
 - InfiniBand (network level device -- device).
# 3. Basics of the CUDA Programming Model
![Program Model](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-thread-blocks.png)

A **kernel** (device function) is launched on a device as a collection of massive amount of **threads**. Those threads are grouped in **thread blocks**, which are in turn grouped in a **Grid**. All blocks have the same size (number of threads) and dimensions (threads may be indexed by 1, 2 or 3 indices). Partitioning of threads in blocks is defined by user via an **execution configuration**. N.B.: the necessity to run different machine code based on execution configuration is is one of the reasons that CUDA has jit and PTX.
```c
__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host code
int main()
{
    int n = 1 << 20;              // number of elements
    int threadsPerBlock = 256;    // block size (threads)
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    // allocate vectors a, b, c 

    // Kernel launch: <<< grid, block >>> is the execution configuration
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);

    cudaDeviceSynchronize();
    return 0;
}

// General syntax is kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args);
```
Each thread block is executed on a single SM but not necessarily in order. Blocks cannot migrate to different SMs and usually are executed until completion (the exception being the Dynamic Parallelism feature that enables kernels to launch other kernels and may result in the calling kernel to be suspended to VRAM). 

Within an SM threads are organized in groups of 32 called Warps (weaving not Star Trek!) which are scheduled and executed in a SIMT manner (same instruction multiple threads) to simplify the scheduling and exploit effective memory access patterns such as global memory coalescing and shared memory bank access patterns. 

Conceptually, a warp behaves like a 32-lane SIMD vector executing in lockstep. However, unlike CPU SIMD units, NVIDIA GPUs do not implement warps using a single wide vector ALU. Instead, warp instructions are executed using many scalar execution units (CUDA cores), with the hardware issuing and completing operations over one or more cycles as needed.

Most CUDA instructions execute on scalar FP32 or INT32 ALUs, while Tensor Core operations are warp-level matrix instructions that are internally partitioned and scheduled across dedicated matrix execution units depending on the instruction shape and available hardware resources.
![block scheduling](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/thread-block-scheduling.png)

|               | Threads within the same warp                                                                                           | Same block, different warps                     | Different blocks                                              |
| ------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------- |
| **Run order** | SIMT                                                                                                                   | may synchronize, but don't have to              | no guarantees on scheduling: any order, any timing            |
| Common memory | limited access (warp shuffle instructions) to the register of threads in the same warp (e.g. `__shfl_up(val, offset)`) | can use shared memory of an SM to exchange data | Generally should not have any cross-dependencies in execution |
![block scheduling](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/thread-block-scheduling.png)

> [!NOTE] For an extra credit
 given cc > 9.0 thread blocks may be grouped in Thread Block Clusters. TBC's are executed in a single GPC and allow TBC-wide synchronization and a number of reduction-like operations. This allows cooperative reductions and broadcasts across blocks in the cluster. See [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html) for details.

The instructions (memory copying, kernels) are scheduled sequentially within instruction queues called **CUDA streams**. All blocks of the same kernel/grid are scheduled within the same stream and the next instruction on the stream is scheduled only once all blocks complete (i.e. kernel-level scheduling is still sequential). The instructions placed on different streams are scheduled asynchronously unless synchronized explicitly.
### Black sheeps
1. Since a block must be executed in a single SM, the SM's **occupancy** directly depends on the scarce resource of available registers. E.g. if a block utilizes more than half of the SM's registers, no other block from the same grid may be executed concurrently which wastes compute power. This is called **register pressure**. Smaller blocks of other grids may however still be executed if resources allow that.
2. The same applies to other SM resources: threads per SM, size of the shared memory
3. A warp executes the same instruction for all threads. A conditionals/branching occurring in the same warp may cause the threads to follow different execution paths. This situation is called **warp divergence**, reduces performance and is handled with **masking** (lanes that should not follow an instruction are temporarily disabled). Note that WD is a *performance* issue and not a one of *correctness*.
![warp divergence](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/active-warp-lanes.png)
# Specs of A4000

| Resource / Limit                  | Value (CC 8.6)             | Notes                        |
| --------------------------------- | -------------------------- | ---------------------------- |
| **Warp size**                     | 32 threads                 | Fixed across NVIDIA GPUs     |
| **Register file per SM**          | 64K 32-bit registers       | Shared by all resident warps |
| **Max registers per thread**      | 255                        | Compiler-enforced limit      |
| **Max threads per SM**            | 1,536                      | 48 warps per SM              |
| **Max warps per SM**              | 48                         | Occupancy upper bound        |
| **Max thread blocks per SM**      | 16                         | Architectural cap            |
| **Shared memory per SM**          | 100 KB                     | Unified L1 + shared pool     |
| **Max shared memory per block**   | 99 KB                      | Configurable                 |
| **L1 cache per SM**               | Up to 128 KB (shared + L1) | Ampere unified design        |
| **L2 cache (chip-wide)**          | ~4–6 MB                    | GA104 ≈ 4 MB                 |
| **Special function units (SFUs)** | 16 per SM                  | sin/cos/rsqrt                |
| **Tensor Cores per SM**           | 4 (3rd-gen)                | FP16 / BF16 / TF32           |
| **FP64 units per SM**             | 2                          | 1/32 rate vs FP32            |
| **FP32 CUDA cores per SM**        | 128                        | Dual-issue capable           |
| **INT32 units per SM**            | 64                         | Parallel with FP32           |
| **SM issue width**                | 4 warp schedulers / SM     | 1 warp/cycle each            |
Note that with 4 warp schedulers and 128 cuda cores we technically can run instructions for only 4 warps per cycle. However up to 48 warps may be resident (code loaded, memory allocated). This is fine since memory ops have huge latencies (e.g. accessing HBM may take 400-800 cycles), thus *most* for warps stall almost all the time, so having few computing ones is fine.

More specs (including SM features can be found at)
https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications
and https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy

Some data may also be obtained directly from the device via CUDA Driver API or python wrappers
```python
import torch

if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)

    print(f"Device index: {idx}")
    print(f"Name: {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Total memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"SM count: {props.multi_processor_count}")
    print(f"Max threads / SM: {props.max_threads_per_multi_processor}")
    print(f"Shared memory / block: {props.shared_memory_per_block / 1024:.1f} KB")
    print(f"Shared memory / SM: {props.shared_memory_per_multiprocessor / 1024:.1f} KB")
    print(f"Registers per SM: {props.regs_per_multiprocessor / 1024:.1f}K 32bit registers")
    print(f"Warp size: {props.warp_size}")
# Device index: 0
# Name: NVIDIA RTX A4000
# Compute capability: 8.6
# Total memory: 15.73 GB
# SM count: 48
# Max threads / SM: 1536
# Shared memory / block: 48.0 KB # this is the default limit can can be reconfigured up to 99KB
# Shared memory / SM: 100.0 KB
# Registers per SM: 64.0K 32bit registers
# Warp size: 32

```

### A note on reported performance
Note that Tensor Cores perf (e.g. [A4000 Data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs21/rtx-a4000/nvidia-rtx-a4000-datasheet.pdf))  is usually reported with *sparsity* at BF16 data type. For Ampere arch Tensor cores use the same hardware paths for BF16, FP16 and TF32, so `effective_flops = reported_flops / 2 ~= 76.7FLOPS`. Note that FP32 is supported by CUDA (not Tensor!) cores only. However more recent architectures usually either report FLOPS/dtype explicitly (https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306) or  solely for BFLOAT16 using sparsity.
Further, CUDA core FLOPS are computed based on the available FMA's each of which is comprised of 2 float ops,. e.g.
``` 
48 SMs * 128 FP32 CUDA cores * 1.56 Ghz boost clock * 2 ops = 19.2 TFLOPS
``` 
This is peak theoretical FLOPS. The real value may be about 50-90% of that even for compute bound scenarios.
### Appendix A. Floating point data types

`Value = (-1)^sign × 1.mantissa × 2^(exponent - bias)`

| Format                  | Total bits | Mantissa bits | Exponent bits | Approx. representable range | Approx. precision     | Accumulation / Notes                                                              |     |
| ----------------------- | ---------- | ------------- | ------------- | --------------------------- | --------------------- | --------------------------------------------------------------------------------- | --- |
| **FP64 (double)**       | 64         | 52            | 11            | ~±1.8 × 10^308              | ~15–17 decimal digits | Accumulates in FP64; HPC, scientific computing                                    |     |
| **FP32 (single)**       | 32         | 23            | 8             | ~±3.4 × 10^38               | ~6–9 decimal digits   | Accumulates in FP32; standard DL ops                                              |     |
| **TF32 (tensor float)** | 19         | 10            | 8             | ~±1.7 × 10^38               | ~3–4 decimal digits   | Multiply in TF32, accumulation in FP32; used in matmuls, softmax, convolutions    |     |
| **FP16 (half)**         | 16         | 10            | 5             | ~±6.5 × 10^4                | ~3 decimal digits     | Accumulation in FP16 or FP32 (Tensor Cores default to FP32 for matmuls)           |     |
| **BF16 (bfloat16)**     | 16         | 7             | 8             | ~±3.4 × 10^38               | ~2–3 decimal digits   | Accumulation in FP32 on Tensor Cores; used in training and normalization          |     |
| **FP8 (E4M3)**          | 8          | 3             | 4             | ~±240                       | ~1 decimal digit      | Multiply in FP8, accumulation in FP16/FP32 (Hopper / H100); LLM inference         |     |
| **FP8 (E5M2)**          | 8          | 2             | 5             | ~±57,344                    | ~1 decimal digit      | Multiply in FP8, accumulation in FP16/FP32; large range variant for LLM inference |     |

# What's next?
1. torch profiling and *simple* tricks to speed things up a bit (or slow them down dramatically)
2. understanding and mitigating memory bound vs compute bound
3. triton (CUDA kernels for the weak)
4. nccl/torch.distributed beyond DDP