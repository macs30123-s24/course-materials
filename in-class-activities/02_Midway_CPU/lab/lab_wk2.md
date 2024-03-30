# Lab - Week 2 - `sbatch` and MPI

## Ex1. Sbatch Configurations

```bash
#!/bin/bash

#SBATCH --job-name=mpi
#SBATCH --output=mpi.out
#SBATCH --ntasks=4
#SBATCH --partition=amd
#SBATCH --account=macs30123

module load python mpich

mpirun python3 ./mpi_rand_walk.py
```

1. Consider the sbatch script above. Describe in words what each line of the sbatch script does.
2. Create a file called `lab_wk2.sbatch` on Midway 3 and copy the script above into it. Try to modify the script to accomplish the following (see [the `sbatch` documentation in the RCC user guide](https://rcc-uchicago.github.io/user-guide/slurm/sbatch/)):
    * Request only 3 CPU cores from the `caslake` partition
    * Add a file location where your error logs can be logged (what happens if you don't specify an error or console output log location?)
    * Ensure that all of the CPU cores will be on the same node (why might we want this to be the case?)
    * Request Midway 3 to send you emails with updates about your job.
3. Submit this job on Midway 3. 
    * After you submit it, practice checking on the status of your running job.
    * Take a look at your output. Does it look similar to what we saw in class?
    * Practice canceling the job (as well as all running jobs for your user account).
4. A common practice when debugging MPI jobs is to test your code on 1 process (mapped to 1 core), 2 processes (mapped to 2 cores), and 3 processes (mapped to 3 cores) before scaling up to higher numbers of cores (why might these be good test cases?).
    * Write a Bash `for` loop in your script that increments the number of MPI processes from 1 to 3 (check the `mpi_loop_job.sbatch` code for an example of writing a bash loop).
    * Practice submitting your job again and checking its status.

## Ex2. Code Compilation

1. Describe in words the differences between the following `numba`-compiled versions of the same code (consider both the different function definitions, as well as the different function signatures in the `@cc.export` lines).
2. Add code to the function that is compiled ahead of time to enable it to work with 32-bit floats as well.

```python
from numba.pycc import CC
from numba import vectorize, jit, prange
import numpy as np

@jit(nopython=True)
def distance_jit(lon1, lat1, lon2, lat2):
    '''
    Calculate the circle distance between two points
    on the earth (specified in decimal degrees)
    '''
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # 6367 km is the radius of the Earth
    km = 6367 * c
    m = km * 1000
    return m


cc = CC('aot')
@cc.export('distance', 'f8(f8,f8,f8,f8)')
@cc.export('distance_v', 'f8[:](f8[:],f8[:],f8,f8)')
def distance_numba(lon1, lat1, lon2, lat2):
    '''                                                                         
    Calculate the circle distance between two points                            
    on the earth (specified in decimal degrees)
    
    (distance: Numba-accelerated; distance_v: Numba-accelerated + vectorized)
    '''
    # convert decimal degrees to radians                        
    lon1, lat1 = map(np.radians, [lon1, lat1])
    lon2, lat2 = map(np.radians, [lon2, lat2])

    # haversine formula                                                         
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # 6367 km is the radius of the Earth                                        
    km = 6367 * c
    m = km * 1000
    return m
cc.compile()
```

## Ex3. MPI Communications

1. Consider the `mpi4py` code below. Provide comments to describe what is happening.
2. The "YOUR CODE HERE" comments below are locations where MPI communications should go in order for this code to run. Which MPI communications are best for this application? Why?
3. Implement the MPI communications in the code and run the code on the Midway Cluster using a `sbatch` script (using 1, 2, and 3 processes again to test your solution). Does it produce the expected result?

```python
from mpi4py import MPI
import numpy as np
import time

def simple_square_parallel():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    numbers = None
    if rank == 0:        
        t0 = time.time()
        np.random.seed(rank)
        numbers = np.linspace(1, 100, 100)
    
    # Split up numbers and send equal work to each process
    # via a MPI communication (stored in an array `sub_numbers`)
    # YOUR CODE HERE

    squared_sub_numbers = np.square(sub_numbers)

    # Collect all of the values in `squared_sub_numbers` and
    # send back to rank 0 via a MPI communication
    # YOUR CODE HERE

    if rank == 0:
        max_index = np.argmax(all_squared_numbers)
        max_square = all_squared_numbers[max_index]
        original_number = numbers[max_index]
        elapsed_time = time.time() - t0

        print("Original number with max square:", original_number)
        print("Max square:", max_square)
        print("Computation time:", elapsed_time)

if __name__ == '__main__':
    simple_square_parallel()
```

## Assignment Walk-through