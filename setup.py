import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

sources = glob.glob('*.cpp')+glob.glob('*.cu')

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
                      sources=sources,
                      
                      )
    ],
    cmdclass={'build_ext': BuildExtension}
)