### Questions
1. In simple_element_kernel_example0.py, we don't consider the memory transfers to and from the GPU in measuring the time for the GPU computation. Try measuring the time that the gpuarray functions, to_gpu and get, take with the Python time command. Would you say it's worth offloading this particular function onto the GPU, with the memory transfer times in consideration?

2. In Chapter 1, Why GPU Programming?, we had a discussion of Amdahl's Law, which gives us some idea of the gains we can potentially get by offloading portions of a program onto a GPU. Name two issues that we have seen in this chapter that Amdahl's law does not take into consideration.

3. Modify gpu_mandel0.py to use smaller and smaller lattices of complex numbers, and compare this to the same lattices CPU version of the program. Can we choose a small enough lattice such that the CPU version is actually faster than the GPU version?

4. Create a kernel with ReductionKernel that takes two complex64 arrays on the GPU of the same length and returns the absolute largest element among both arrays.

5. What happens if a gpuarray object reaches end-of-scope in Python?

6. Why do you think we need to define neutral when we use ReductionKernel?

7. If in ReductionKernel we set reduce_expr ="a > b ? a : b", and we are operating on int32 types, then what should we set "neutral" to?