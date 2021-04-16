import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

include_dirs = os.path.dirname(os.path.abspath(__file__))

setup(
    name='SpaVar',
    ext_modules=[
        CUDAExtension('SpaVar', [
            'src/SV_cuda.cpp',
            'src/SV_kernel.cu',
            ],
            include_dirs=[include_dirs]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
