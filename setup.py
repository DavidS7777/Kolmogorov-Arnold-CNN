import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

sources_g = glob.glob('*group.cpp')+glob.glob('*group.cu')
sources_t = glob.glob('*test.cpp')+glob.glob('*test.cu')
# ext_modules = [
#     CUDAExtension(
#         'kaconvolution',
#         ['kaconvolution.cpp', 'rational_kernel_group.cu'],
#     ),
# ]

setup(
    name="kaconvolution",
    ext_modules=[
        CUDAExtension(name="kaconvolution_cu",
                      sources=sources_g,
                      
                      ),
        CUDAExtension(name="kaconvolutiontest",
                    sources=sources_t, extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2", "-lineinfo", "-use_fast_math"]})
    ],
    cmdclass={'build_ext': BuildExtension}
)