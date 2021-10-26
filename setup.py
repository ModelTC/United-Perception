from setuptools import setup, find_packages
from eod import __version__


def read_requirements():
    reqs = []
    with open('requirements.txt', 'r') as fin:
        for line in fin.readlines():
            reqs.append(line.strip())
    return reqs


setup(
    name="EOD",
    version=__version__,
    author="The Great Cold",
    description="Easy and Efficient Object Detector",
    author_email="",
    python_requires='>=3.6',
    packages=find_packages(),
    classifiers=(
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux"),
    install_requires=read_requirements()
)
