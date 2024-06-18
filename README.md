# Hands-On GPU Programming with Python and CUDA

<a href="https://www.packtpub.com/application-development/hands-gpu-programming-python-and-cuda?utm_source=github&utm_medium=repository&utm_campaign=9781788993913 "><img src="https://d255esdrn735hr.cloudfront.net/sites/default/files/imagecache/ppv4_main_book_cover/B10306.png" alt="Hands-On GPU Programming with Python and CUDA" height="256px" align="right"></a>

This is the code repository for [Hands-On GPU Programming with Python and CUDA](https://www.packtpub.com/application-development/hands-gpu-programming-python-and-cuda?utm_source=github&utm_medium=repository&utm_campaign=9781788993913 ), published by Packt.

**Explore high-performance parallel computing with CUDA**

## What is this book about?
Hands-On GPU Programming with Python and CUDA hits the ground running: you’ll start by learning how to apply Amdahl’s Law, use a code profiler to identify bottlenecks in your Python code, and set up an appropriate GPU programming environment. You’ll then see how to “query” the GPU’s features and copy arrays of data to and from the GPU’s own memory.

This book covers the following exciting features:
* Launch GPU code directly from Python 
* Write effective and efficient GPU kernels and device functions 
* Use libraries such as cuFFT, cuBLAS, and cuSolver 
* Debug and profile your code with Nsight and Visual Profiler 
* Apply GPU programming to datascience problems 
* Build a GPU-based deep neuralnetwork from scratch 
* Explore advanced GPU hardware features, such as warp shuffling 

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1788993918) today!

## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
cublas.cublasDestroy(handle)
print 'cuBLAS returned the correct value: %s' % np.allclose(np.dot(A,x), y_gpu.get())
```

**Following is what you need for this book:**
Hands-On GPU Programming with Python and CUDA is for developers and data scientists who want to learn the basics of effective GPU programming to improve performance using Python code. You should have an understanding of first-year college or university-level engineering mathematics and physics, and have some experience with Python as well as in any C-based programming language such as C, C++, Go, or Java.

With the following software and hardware list you can run all code files present in the book (Chapter 1-12).
### Software and Hardware List
| Chapter  | Software required                    | OS required                         |
| -------- | ------------------------------------ | ----------------------------------- |
| 1-11     | Anaconda 5 (Python 2.7 version)      | Windows, Linux                      |
| 2-11     | CUDA 9.2, CUDA 10.x                  | Windows, Linux                      |
| 2-11     | PyCUDA (latest)                      | Windows, Linux                      |
| 7        | Scikit-CUDA (latest)                 | Windows, Linux                      |
| 2-11     | Visual Studio Community 2015         | Windows                             |
| 2-11     | GCC, GDB, Eclipse                    | Linux                               |


| Chapter  | Hardware required                    | OS required                         |
| -------- | ------------------------------------ | ----------------------------------- |
| 1-11     | 64-bit Intel/AMD PC                  | Windows, Linux                      |
| 1-11     | 4 Gigabytes RAM                      | Windows, Linux                      |
| 2-11     | NVIDIA GPU (GTX 1050 or better)      | Windows, Linux                      |


We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://www.packtpub.com/sites/default/files/downloads/9781788993913_ColorImages.pdf).

## Get to Know the Author
**Dr Brian Tuomanen**
has been working with CUDA and general-purpose GPU programming since 2014. He received his bachelor of science in electrical engineering from the University of Washington in Seattle, and briefly worked as a software engineer before switching to mathematics for graduate school. He completed his PhD in mathematics at the University of Missouri in Columbia, where he first encountered GPU programming as a means for studying scientific problems. Dr. Tuomanen has spoken at the US Army Research Lab about general-purpose GPU programming and has recently led GPU integration and development at a Maryland-based start-up company. He currently works as a machine learning specialist (Azure CSI) for Microsoft in the Seattle area.

## Notes
### Chapter 01: Why GPU Programming? Technical requirements, Parallelization and Amdahl's Law, Code Profiling
-  CUDA (Pronunciation: coo-duh), a framework for general-purpose GPU (GPGPU) programming from NVIDIA
- Amdahl's Law
    - A method to estimate potential speedup we can get by offloading a program or algorithm onto a GPU/ for a program that can be at least partially parallelized
    - Speedup = 1/((1-p)+p/N)
    - p = parallelizable proportion of execution time for code in original serial program
    - N = Number of processor cores
- Latency: Duration of performing a single computation.
- Power of GPU - TONS of cores MORE than in a CPU, which provides throughput
- Throughput: Number of computations that can be performed simultaneously
- "A GPU is like a very wide city road that is designed to handle many slower-moving cars at once (high throughput, high latency), whereas a CPU is like a narrow highway that can only admit a few cars at once, but can get each individual car to its destination much quicker (low throughput, low latency)."
- ⬆ number of cores of GPU, ⬆ in throughput
- Example: 
    - CPU - 11th Gen Intel(R) Core (TM) i7-11850H @2.50GHz: 8 cores
    - GPU - [NVIDIA RTX A2000](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/rtx-a2000/nvidia-rtx-a2000-datasheet.pdf): 3328 CUDA Cores
    - GPU - [NVIDIA RTX A4000](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs21/rtx-a4000/nvidia-rtx-a4000-datasheet.pdf): 6144 CUDA Cores
- Exploit throughput, know how to properly parallelize to split workload to run in parallel on multiple processors simultaneously
- Building a house (N number of laborers, N times as fast, N =  speedup of parallelizing a task over the serial version of a task) = parallelizable task
- Profiling code with cPython module, check out [Chapter01/mandelbrot0.ipynb](Chapter01/mandelbrot0.ipynb)
- Advantage of GPU over CPU = ⬆ throughput, execute more parallel code simultaneously on GPU than on a CPU
- "GPU cannot make recursive algorithms or nonparallelizable algorithms somewhat faster"
- Serial vs Parallelizable