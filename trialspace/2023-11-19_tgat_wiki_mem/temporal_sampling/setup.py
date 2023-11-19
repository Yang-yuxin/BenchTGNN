from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME
from torch.utils.cpp_extension import CppExtension, CUDAExtension

# In any case, include the CPU version
modules = [
    CppExtension('temporal_sampling.cpu',
                 ['src/cpu/searchsorted_cpu_wrapper.cpp']),
]

# If nvcc is available, add the CUDA extension
if CUDA_HOME:
    modules.append(
        CUDAExtension('temporal_sampling.cuda',
                      ['src/cuda/searchsorted_cuda_wrapper.cpp',
                       'src/cuda/searchsorted_cuda_kernel.cu'])
    )

# Now proceed to setup
setup(
    name='temporal_sampling',
    version='1.1',
    description='A searchsorted implementation for pytorch',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    ext_modules=modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)
