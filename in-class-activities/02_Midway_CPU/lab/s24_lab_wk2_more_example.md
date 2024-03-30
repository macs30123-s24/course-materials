# Week 2 Lab Session Code Example

## 0. Base code for calculating $\pi$ with monte carlo simulation

```python
import numpy as np
import random
import time

def monte_carlo_pi(nsamples):
    acc = 0
    for _ in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

start = time.time()
print(monte_carlo_pi(1000000))
print("Time taken: ", time.time() - start)
```

## 1. Example with Numba

### (1) Example with JIT (Just-in-time Compiling)
Use the decorator `@jit` to set the function to Just-in-time compile. using parameter nopython=True or using decorator `@njit`, would make the function to run without using python compiler at all (makes it faster than just calling jit).

Lazy compilation lets numba decide the optimal way to compile the code, so no need to tell the function signature.
```python
import numpy as np
import random
import time
from numba import jit

@jit(nopython=True)
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

start = time.time()
print(monte_carlo_pi(1000000))
print("Time taken: ", time.time() - start)
```

### (2) Example with AOT (Ahead-of-time Compiling)
This pre-compiles the code and saves it as a shared library. This is useful when you want to use the function in another script or language. Since it is pre-compiling your code, you need the function signature. You can make function be exported in different function signature by using the decorator @cc.export('function_name', 'function_signature').
```python
import numpy as np
import random
import time
from numba.pycc import CC

cc = CC('example_1')
@cc.export('monte_carlo_pi_double', 'f8(i8)')
@cc.export('monte_carlo_pi_float', 'f4(i4)')
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

cc.compile()
```
```python
import example_1

start = time.time()
print(example_1.monte_carlo_pi_double(1000000))
print("Time taken: ", time.time() - start)
```

## 2. Example with Cython (Using jupyter notebook)
Running Cython code in jupyter notebook is easier than running it in a script. You can load cython with `%load_ext Cython` and use the magic command `%%cython` to compile the code.

```python
%load_ext Cython
```
```python
%%cython
from libc.stdlib cimport rand, RAND_MAX

cpdef double monte_carlo_pi(int nsamples):
    cdef int i, acc = 0
    cdef double x
    cdef double y

    for i in range(nsamples):
        x = rand()/RAND_MAX
        y = rand()/RAND_MAX
        if (x*x + y*y) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples
```
```python
start = time.time()
print(example_1.monte_carlo_pi(1000000))
print("Time taken: ", time.time() - start)
```

## 3. Example with MPI

Save the following code as `monte_carlo_pi_mpi.py`.
```python
from mpi4py import MPI
import numpy as np
import time
import random

def monte_carlo_pi(nsamples):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    random.seed(1234)

    start_time = time.time()
    N = int(nsamples / size)
    acc = np.zeros(N)

    for i in range(N):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc[i] = 1

    acc_total = None

    if rank == 0:
        acc_total = np.empty(size * N)

    comm.Gather(sendbuf=acc, recvbuf=acc_total, root=0)

    if rank == 0:
        sum_acc = np.sum(acc_total)
        print('the estimated pi value is: ', 4.0 * sum_acc / nsamples)
        finish_time = time.time() - start_time
        print(f'the process took {finish_time} seconds')

    return 0

def main():
  monte_carlo_pi(nsamples)

nsamples = 10000

if __name__ == "__main__":
  main()
```

Save the following code as `monte_carlo_pi_mpi.sbatch`, and submit the job using `sbatch ./monte_carlo_pi_mpi.sbatch`.
```
#!/bin/bash

#SBATCH --job-name=motne_carlo_pi
#SBATCH --output=monte_carlo_pi.out
#SBATCH --error=monte_carlo_pi.err
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --partition=caslake
#SBATCH --account=macs30123

module load python mpich

mpirun python3 -n 4 ./monte_carlo_pi_mpi.py
```
