import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.scan import InclusiveScanKernel

seq = np.array([1,2,3,4],dtype=np.int32)
seq_gpu = gpuarray.to_gpu(seq)
# Construct kernel 
# 1. input/output type: NumPy int32
# 2. string: "a+b"
# "a+b" in GPU ~> lambda a,b: a + b in Python
sum_gpu = InclusiveScanKernel(np.int32, "a+b")
print(sum_gpu(seq_gpu).get())
print(np.cumsum(seq))
