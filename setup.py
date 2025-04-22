import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = glob.glob('cuda/kaconvolution.cpp')+glob.glob('cuda/kaconvolution_kernel.cu')


setup(
    name="kaconvolution_cuda",
    ext_modules=[
        CUDAExtension(name="kaconvolution_cuda",
                    sources=sources, extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2", "-lineinfo", "-use_fast_math"]})
    ],
    cmdclass={'build_ext': BuildExtension}
)