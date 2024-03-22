# Install packages we'll be using on Midway 3
module load python mpich cuda
pip install --user numba==0.57.1 numpy==1.22.4 mpi4py-mpich==3.1.5 \
                   pyopencl==2024.1 rasterio==1.3.9
