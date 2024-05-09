import cupy as cp
from cupyx.profiler import benchmark

def rand_walk_gpu(N):
    # generate on GPU + use single precision floats
    steps_dev = cp.random.normal(loc=0, scale=1, size=(N, 100), dtype='float')

    # perform cumulative sum, and reduction operations on GPU
    steps_dev[:, 0] = 0
    r_walks_dev = 100 + cp.cumsum(steps_dev, axis=1)

    average_finish_dev = cp.mean(r_walks_dev[:, -1])
    std_finish_dev = cp.std(r_walks_dev[:, -1])

    # Bring average and standard deviation to CPU host
    average_finish = average_finish_dev.get()
    std_finish = std_finish_dev.get()
    return average_finish, std_finish

if __name__ == '__main__':
    N = 10 ** 7

    # Benchmark GPU/CPU time (repeat 10 times)
    print(f'Benchmarking Results for {N} runs:\n', benchmark(rand_walk_gpu, (N,), n_repeat=10))

    # Compute average and standard deviation
    avg, std = rand_walk_gpu(N)
    print(f'Simulation Average: {avg.round(3)}\nStandard Deviation: {std.round(3)}')
