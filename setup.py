# Standard Library
import glob
import os
import subprocess
import io
import re

# Import from third library
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension

with io.open("up/__init__.py", "rt", encoding="utf8") as f:
    __version__ = re.search(r'__version__ = "(\D*)(.*?)"', f.read(), re.M).group(2)


def _find_cuda_home():
    # guess rule 3 of torch.utils.cpp_extension
    nvcc = subprocess.check_output(['which', 'nvcc']).decode().rstrip('\r\n')
    cuda_home = os.path.dirname(os.path.dirname(nvcc))
    print(f'find cuda home:{cuda_home}')
    return cuda_home


# remember to overwrite PyTorch auto-detected cuda home which
# may not be our expected
torch.utils.cpp_extension.CUDA_HOME = _find_cuda_home()
CUDA_HOME = torch.utils.cpp_extension.CUDA_HOME

CORE_DIR = 'up'
EXT_DIR = 'extensions'


def read_requirements():
    reqs = []
    with open('requirements.txt', 'r') as fin:
        for line in fin.readlines():
            reqs.append(line.strip())
    return reqs


def recursive_glob(base_dir, pattern):
    files = []
    for root, subdirs, subfiles in os.walk(base_dir):
        files += glob.glob(os.path.join(root, pattern))
    return files


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sources_dir = os.path.join(this_dir, CORE_DIR, EXT_DIR, "csrc")

    source_cpu = recursive_glob(sources_dir, "*.cpp")
    source_cuda = recursive_glob(sources_dir, "*.cu")

    sources = source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []
    assert torch.cuda.is_available() and CUDA_HOME is not None

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    # sources = [os.path.join(extensions_dir, s) for s in sources]
    print(f'sources:{sources}')

    include_dirs = [sources_dir]

    ext_modules = [
        extension(
            f"{CORE_DIR}.{EXT_DIR}._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="UP",
    version=__version__,
    author="Model ToolChain",
    description="United Perception",
    author_email="",
    python_requires='>=3.6',
    packages=find_packages(),
    classifiers=(
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux"),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
