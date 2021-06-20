import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

include_dirs = os.path.dirname(os.path.abspath(__file__))

setup(
    name='SpaMat',
    ext_modules=[
        CUDAExtension('SpaMat', [
            'src/SM_cuda.cpp',
            'src/SM_kernel.cu',
            ],
            include_dirs=[include_dirs]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
