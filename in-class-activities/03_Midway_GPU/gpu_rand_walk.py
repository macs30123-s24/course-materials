import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as clrand
import pyopencl.tools as cltools
from pyopencl.scan import GenericScanKernel
import matplotlib.pyplot as plt
import time

def sim_rand_walks(n_runs):
    # Set up context and command queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    t0 = time.time()

    # Generate an array of Normal Random Numbers on GPU of length n_sims*n_steps
    n_steps = 100
    rand_gen = clrand.PhiloxGenerator(ctx)
    ran = rand_gen.normal(queue, (n_runs * n_steps), np.float32, mu=0, sigma=1)

    # Establish boundaries for each simulated walk (i.e. start and end)
    # Necessary so that we perform scan only within rand walks and not between
    seg_boundaries = [1] + [0] * (n_steps - 1)
    seg_boundaries = np.array(seg_boundaries, dtype=np.uint8)
    seg_boundary_flags = np.tile(seg_boundaries, int(n_runs))
    seg_boundary_flags = cl_array.to_device(queue, seg_boundary_flags)

    # GPU: Define Segmented Scan Kernel, scanning simulations: f(n-1) + f(n)
    seg_scan = GenericScanKernel(ctx, np.float32,
                arguments="float *ary, char *segflags, float *out",
                input_expr="ary[i]",
                neutral="0",
                scan_expr="across_seg_boundary ? b : (a+b)",
                is_segment_start_expr="segflags[i]",
                output_statement="out[i] = item + 100",
                options=[])

    dev_result = cl_array.empty_like(ran)

    # Enqueue and Run Scan Kernel
    seg_scan(ran, seg_boundary_flags, dev_result)

    # Get results back on CPU to plot and do final calcs, just as last week
    r_walks_all = dev_result.get() \
                            .reshape(n_runs, n_steps) \
                            .transpose()

    average_finish = np.mean(r_walks_all[-1])
    std_finish = np.std(r_walks_all[-1])
    final_time = time.time()
    time_elapsed = final_time - t0

    print("Simulated %d Random Walks in: %f seconds"
                % (n_runs, time_elapsed))
    print("Average final position: %f, Standard Deviation: %f"
                % (average_finish, std_finish))

if __name__ == "__main__":
    sim_rand_walks(n_runs=10000)