from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os, shutil

env = "config_processing.env"
env_default = "scripts/env.default.sh"

if not os.path.exists(env):
    shutil.copyfile(env_default, env)


extensions = [
    Extension(
        "utils.processing_methods",
        ["utils/processing_methods.pyx"],
        include_dirs=[numpy.get_include()],
    )
]
setup(
    name="fishcluster_wip",
    version="0.1",
    author="MRB",
    # author_email="luka.staerk@mailbox.org",
    # description="A package to analyze trajectories",
    ext_modules=cythonize(extensions),
    # install_requires=['requirement'],
    packages=find_packages(),
    include_package_data=True,
)
