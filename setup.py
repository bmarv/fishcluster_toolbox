from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os
import shutil

env = "config.env"
env_default = "config.env.default"

if not os.path.exists(env):
    shutil.copyfile(env_default, env)


extensions = [
    Extension(
        "processing.processing_methods",
        ["processing/processing_methods.pyx"],
        include_dirs=[numpy.get_include()],
    )
]
setup(
    name="fishcluster_toolbox",
    version="0.1",
    author="MRB",
    author_email="beesemarvin@gmail.com",
    description="Preprocessing and Unsupervised Learning\
         of behavioural trajectories",
    ext_modules=cythonize(extensions),
    # install_requires=['requirement'],
    packages=find_packages(),
    include_package_data=True,
)
