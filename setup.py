from setuptools import setup, find_packages
from glob import glob
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "cpp_backend",
        sorted(glob("cpp_backend/*.cpp")),
    ),
]

setup(name='minitorch', version='1.0', packages=find_packages(), ext_modules=ext_modules)