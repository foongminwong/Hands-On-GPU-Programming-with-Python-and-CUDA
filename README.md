# Hands-On GPU Programming with Python and CUDA - Notes
<a href="https://www.packtpub.com/application-development/hands-gpu-programming-python-and-cuda?utm_source=github&utm_medium=repository&utm_campaign=9781788993913 "></a>

This is the forked code repository for [Hands-On GPU Programming with Python and CUDA](https://www.packtpub.com/application-development/hands-gpu-programming-python-and-cuda?utm_source=github&utm_medium=repository&utm_campaign=9781788993913 ), published by Packt.

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

### Chapter 02: Setting Up GPU Programming Environment, Technical & Hardware Requirements, GPU Drivers Installation, Setup C++ & Python development environment on Windows & Linux
- Setup Instructions setup available for Ubuntu LTS (long-term support)-based Linux OS (e.g.,  Xubuntu, Kubuntu, or Linux Mint) or Windows 10
- Fast pre-optimized NumPy/SciPy packages <= Intel's Math Kernel Library (MKL)
- Requiremenet: an NVIDIA brand GPU. The CUDA Toolkit is proprietary for NVIDIA cards, so it won't work for programming Intel HD or Radeon GPUs.
- Linux:
    - Open Terminal: `Ctrl + Alt + T`
    - Check processor: `lscpu`
    - Check memory capacity: `free -g` - Display total number of proper memory & memory in swap space
    - Check for Nvidia GPU: `lspci | grep -e "NVIDIA"`
    - Nouveau - an open-source default driver for NVIDIA GPUs in Ubuntu, buts does not work with CUDA
    - Install proprietary driver by adding `graphics-drivers` repository to package manager
        - `sudo add-apt-repository ppa:graphics-drivers/ppa`
        - Synchronize system with the new repo: `sudo apt-get update`
        - Press `Windows + R` and search `software and drivers`
        - Software & Drivers -> Additional Drivers -> Choose the lastest stable proprietary drivers for the GPU (e.g., Using NVIDIA driver metapackage from nvidia-driver-XXX (open source)) -> Apply Changes -> Wait... -> Press `Windows + A` -> Enter `nvidia-settings` -> NVIDIA X Server Settings manager
    - Setup GCC, Eclipse IDE, and graphical dependencies: 
        - `sudo apt-get update`
        - `sudo apt-get install build-essential binutils gdb eclipse-cdt`
            - `build-essential` - gcc, g++ compilers & other utilities such as make
            - `binutils` - useful utilities, such as the LD linker
            - `gdb` - debugger
            - `Eclipse` - IDE
        - `sudo apt-get install freeglut3 freeglut3-dev libxi-dev libxmu-dev` - allow our system to run graphical (OpenGL) stuff included with the CUDA Toolkit
    - Download & Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
        - `.deb` package (recommended) & a `.run` file (This installs outside of the system's Advanced Package Tool - APT system, which copies the appropriate files to the system's /usr binary and library directories - do this if you know Linux well and know how to work with the Ubuntu's system APT system or repos)
        - Edit `.bashrc` file to configure PATH and LD_SYSTEM_CONFIG environment variables so that the system can find the appropriate binary executable and library files needed for CUDA.
        ```
        export PATH="/usr/local/cuda/bin:${PATH}
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
        ```
    - Install PyCUDA
        - Check where the python binary `which python` else edit `~/.bashrc` file by adding `export PATH="/home/${USER}/anaconda2/bin:${PATH}"`
        - `pip install pycuda`



- Windows
    - Control Panel -> Device Manager
    - ![alt text](image-1.png)
    - CUDA is very particular about which driver has been installed by the user.
    - "Windows has better CUDA driver compatibility and a more user-friendly installation than Linux."
    - Download & Install GPU drivers for Windows on [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx?lang=en-us)
        - ![alt text](image-2.png)
        - ![alt text](image-3.png)
        - ![alt text](image.png)
    - [Install C/C++ on Windows - Guide](https://medium.com/@farid.rahimoff/installing-c-compiler-for-windows-10-and-11-step-by-st-51e760443b02)
    - Download & Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), nvcc compiler
        - ![alt text](image-4.png)
        - ![alt text](image-5.png)
        - ![alt text](image-6.png)
        - Check the NVIDIA compiler version information: `nvcc --version`
        - ![alt text](image-9.png)
    - Install PyCUDA: `pip install pycuda`
        - ![alt text](image-8.png)
        - If you received error `ImportError: DLL load failed while importing _driver: The specified module could not be found.` when running [Chapter03/deviceQuery.ipynb](Chapter03/deviceQuery.ipynb), add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin` to PATH