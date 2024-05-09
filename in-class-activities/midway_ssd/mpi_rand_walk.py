from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time

def sim_rand_walks_parallel(n_runs, plot=False):
    # Get rank of process and overall size of communicator:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Start time:
    t0 = time.time()

    # Evenly distribute number of simulation runs across processes
    N = int(n_runs / size)

    # Simulate N random walks (of 100 steps) on each MPI Process
    steps = np.random.normal(loc=0, scale=1, size=(N, 100))
    steps[:, 0] = 0
    r_walks_array = 100 + np.cumsum(steps, axis=1)

    # Gather all simulation arrays to buffer of expected size/dtype on rank 0
    r_walks_all = None
    if rank == 0:
        r_walks_all = np.empty([N * size, 100], dtype='float')
    comm.Gather(sendbuf=r_walks_array, recvbuf=r_walks_all, root=0)

    # Print/plot simulation results on rank 0
    if rank == 0:
        # Calculate time elapsed after computing mean and std
        average_finish = np.mean(r_walks_all[:, -1])
        std_finish = np.std(r_walks_all[:, -1])
        time_elapsed = time.time() - t0

        # Print time elapsed + simulation results
        print(f'Simulated {n_runs} Random Walks in: {round(time_elapsed, 3)} seconds on {size} MPI processes')
        print(f'Average final position: {round(average_finish, 3)}, Standard Deviation: {round(std_finish, 3)}')

        if plot:
            # Plot simulations and save to file
            plt.plot(r_walks_all.transpose())
            plt.savefig(f'r_walk_nprocs{size}_nruns{n_runs}.png')

    return

if __name__ == '__main__':
    N = 10 ** 7
    sim_rand_walks_parallel(N)
