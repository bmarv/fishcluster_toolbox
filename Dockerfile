FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

COPY requirements.txt /tmp/requirements.txt
COPY --from=continuumio/miniconda3 /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

RUN conda create -n rapids-24.08 -c rapidsai -c conda-forge -c nvidia  \
    rapids=24.08 python=3.9 'cuda-version>=11.4,<=11.8' -y

RUN conda init; . ~/.bashrc; conda activate rapids-24.08; pip install -r /tmp/requirements.txt
RUN conda init; . ~/.bashrc; conda activate rapids-24.08; \
    pip install torch==1.11.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115
RUN . ~/.bashrc; conda activate rapids-24.08; pip install wandb

# fixing dask & pandas installations
RUN . ~/.bashrc; conda activate rapids-24.08; \
    pip install "dask[dataframe]" --upgrade; \
    pip install cupy-cuda11x>=12.0.0;

RUN apt update
RUN apt install nano less build-essential -y

# resetting the Threading layer to tbb to prevent unsafe fork() calls in OpenMP using umap
# -> https://stackoverflow.com/questions/68131348/training-a-python-umap-model-hangs-in-a-multiprocessing-process
# -> https://github.com/numba/numba/issues/7148 
# -> https://github.com/numba/numba/issues/6108#issuecomment-675365997 
RUN export LD_LIBRARY_PATH=/opt/conda/envs/rapids-24.08/lib/libtbb.so 

COPY . /workspace
WORKDIR /workspace
RUN . ~/.bashrc; conda activate rapids-24.08; python /workspace/setup.py build_ext --inplace

RUN chmod +x /workspace/misc/docker_entrypoint.sh

ENTRYPOINT ["/workspace/misc/docker_entrypoint.sh"]