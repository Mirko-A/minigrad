from setuptools import setup, find_packages
from glob import glob
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "Backend",
        sorted(glob("minitorch/backend/*.cpp")),
        sorted(glob("minitorch/backend/*.h")),
        extra_compile_args=["/DEBUG"]
    )
]

setup(name='minitorch', version='1.0', packages=find_packages(), ext_modules=ext_modules)