from time import time
import matplotlib
#this will prevent the figure from popping up
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel

# CUDA-C, particular C datatypes will correspond directly to particular Python NumPy datatypes
# ~> means corresponds to
# CUDA C int ~> NumPy int32
# CUDA C float ~> NumPy float32
# PyCUDA ::complex<float> ~> Numpy complex64
# """ xxx """: larger inline CUDA kernels in Python
# Mandelbrot set is 2D arrays in Python, real and complex, but ElementwiseKernel will automatically translate everything into a 1D set

# pycuda::complex<float> c = lattice[i]: lattice point
# pycuda::complex<float> z(0,0): 1st zero ~> real part, 2nd zero ~> imaginary part
# for loop: Run serially over j, entire piece of code will be parallelized across i
mandel_ker = ElementwiseKernel(
"pycuda::complex<float> *lattice, float *mandelbrot_graph, int max_iters, float upper_bound",
"""
mandelbrot_graph[i] = 1;

pycuda::complex<float> c = lattice[i]; 
pycuda::complex<float> z(0,0);

for (int j = 0; j < max_iters; j++)
    {
    
     z = z*z + c;
     
     if(abs(z) > upper_bound)
         {
          mandelbrot_graph[i] = 0;
          break;
         }

    }
         
""",
"mandel_ker")
# mandel_ker: kernel internal CUDA C name

def gpu_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):

    # we set up our complex lattice as such
    real_vals = np.matrix(np.linspace(real_low, real_high, width), dtype=np.complex64)
    imag_vals = np.matrix(np.linspace( imag_high, imag_low, height), dtype=np.complex64) * 1j
    # typecast the result from a NumPy matrix type to a two-dimensional NumPy array (since PyCUDA can only handle NumPy array types, not matrix types)
    mandelbrot_lattice = np.array(real_vals + imag_vals.transpose(), dtype=np.complex64)    
    
    # gpuarray.to_array function only can operate on NumPy array types
    # copy complex lattice to the GPU
    mandelbrot_lattice_gpu = gpuarray.to_gpu(mandelbrot_lattice)

    # allocate an empty array on the GPU, similar to malloc on C but don't need to deallocate or free this memeory later since gpuarray object destructor taking care of memory clean-up automatically
    mandelbrot_graph_gpu = gpuarray.empty(shape=mandelbrot_lattice.shape, dtype=np.float32)

    # 1. mandelbrot_lattice_gpu: complex lattice of points (NumPy complex64 type)
    # 2. mandelbrot_graph_gpu: pointer to a two-dimensional floating point array (NumPy float32 type) that will indicate which elements are members of the Mandelbrot set
    # 3. np.int32(max_iters): integer indicating the maximum number of iterations for each point
    # 4. np.float32(upper_bound): upper bound for each point used for determining membership in the Mandelbrot class
    # Be  careful in typecasting everything that goes into the GPU
    mandel_ker( mandelbrot_lattice_gpu, mandelbrot_graph_gpu, np.int32(max_iters), np.float32(upper_bound))

    # Retrieve the Mandelbrot set generated from the GPU back into CPU space     
    mandelbrot_graph = mandelbrot_graph_gpu.get()
    
    return mandelbrot_graph


if __name__ == '__main__':

    t1 = time()
    mandel = gpu_mandelbrot(512,512,-2,2,-2,2,256, 2)
    t2 = time()

    mandel_time = t2 - t1

    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.savefig('mandelbrot.png', dpi=fig.dpi)
    t2 = time()

    dump_time = t2 - t1

    print('It took {} seconds to calculate the Mandelbrot graph.'.format(mandel_time))
    print('It took {} seconds to dump the image.'.format(dump_time))

# PyCUDA also has to compile and link our CUDA C code at runtime, and the time it takes to make the memory transfers to and from the GPU (will ahve extra overhead)