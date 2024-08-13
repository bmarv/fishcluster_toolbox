FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

COPY requirements.txt /tmp/requirements.txt
COPY --from=continuumio/miniconda3 /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

RUN conda create -n rapids-22.04 -c rapidsai -c nvidia -c conda-forge \
    rapids=22.04 python=3.9 cudatoolkit=11.5 -y

RUN conda init; . ~/.bashrc; conda activate rapids-22.04; pip install -r /tmp/requirements.txt
RUN conda init; . ~/.bashrc; conda activate rapids-22.04; \
    pip install torch==1.11.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115
RUN . ~/.bashrc; conda activate rapids-22.04; pip install wandb

RUN apt update
RUN apt install nano less build-essential -y

# resetting the Threading layer to tbb to prevent unsafe fork() calls in OpenMP using umap
# -> https://stackoverflow.com/questions/68131348/training-a-python-umap-model-hangs-in-a-multiprocessing-process
# -> https://github.com/numba/numba/issues/7148 
# -> https://github.com/numba/numba/issues/6108#issuecomment-675365997 
RUN export LD_LIBRARY_PATH=/opt/conda/envs/rapids-22.04/lib/libtbb.so 

COPY . /workspace
WORKDIR /workspace
RUN . /root/.bashrc; python /workspace/setup.py build_ext --inplace

RUN chmod +x /workspace/misc/docker_entrypoint.sh

ENTRYPOINT ["/workspace/misc/docker_entrypoint.sh"]