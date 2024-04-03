import pyopencl as cl
import pyopencl.clrandom as clrand
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel
import time
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n_runs = 10**8

############################################################
# NumPy Solution
############################################################

t_np0 = time.time()

# Simulate Random Coordinates in Unit Square:
ran = np.random.uniform(low=-1, high=1, size=(2, n_runs))

# Identify Random Coordinates that fall within Unit Circle and count them
result = ran[0]**2 + ran[1]**2 <= 1
n_in_circle = np.sum(result)
pi_np = 4 * n_in_circle / n_runs
t_np1 = time.time()
print("Time Elapsed (CPU): ", t_np1 - t_np0)

############################################################
# Data, Memory Operations
############################################################

t0 = time.time()
x_dev = clrand.rand(queue, (n_runs), np.float32, a=-1, b=1)
y_dev = clrand.rand(queue, (n_runs), np.float32, a=-1, b=1)

z = np.empty(len(x_dev), dtype=np.int32)
z_dev = cl_array.to_device(queue, z)
t_mem = time.time() - t0
print("Time Elapsed (Memory Ops): ", t_mem)

############################################################
# Pythonic Map Operation
############################################################

t1 = time.time()
map_result = (y_dev**2 + x_dev**2) <= 1
n_total = map_result.get()
n_in = np.sum(n_total)
pi_map = 4 * n_in / n_runs
t_map = time.time()
print("Time Elapsed (Pythonic Map): ", t_map - t1)

############################################################
# Pythonic Map + Reduce Operation
############################################################

t1 = time.time()
map_result = (y_dev**2 + x_dev**2) <= 1
n_in_dev = cl_array.sum(map_result)
n_in_host = n_in_dev.get()
pi_map = 4 * n_in_host / n_runs
t_map = time.time()
print("Time Elapsed (Pythonic Map + Reduce): ", t_map - t1)

############################################################
# Elementwise Map
############################################################

t2 = time.time()
mknl = ElementwiseKernel(ctx,
        "float *x, float *y, int *z",
        "z[i] = (x[i]*x[i] + y[i]*y[i]) <= 1 ? 1 : 0"
        )
rknl = ReductionKernel(ctx, np.float32,
            neutral="0",
            reduce_expr="a+b",
            map_expr="z[i]",
            arguments="float *z"
        )

mknl(x_dev, y_dev, z_dev)
n_in = rknl(z_dev).get()
pi_emap = 4 * n_in / n_runs
t_emap = time.time()
print("Time Elapsed (Elementwise Map + Reduction Kernel): ", t_emap - t2)

############################################################
# Combined Map/Reduce Kernel
############################################################

t3 = time.time()
rknl = ReductionKernel(ctx, np.float32,
        neutral="0",
        reduce_expr="a+b",
        map_expr="(x[i]*x[i] + y[i]*y[i]) <= 1 ? 1 : 0",
        arguments="float *x, float *y"
        )
n_in = rknl(x_dev, y_dev).get()
pi_map_reduce = 4 * n_in / n_runs
t_map_reduce = time.time()
print("Time Elapsed (Combined Map + Reduce): ", t_map_reduce - t3)
