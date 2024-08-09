FROM ubuntu:22.04

COPY environment.yml /tmp/environment.yml
COPY --from=continuumio/miniconda3 /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

RUN conda env create --name fishcluster_toolbox --file /tmp/environment.yml -y

RUN conda init; . /root/.bashrc; conda config --set auto_activate_base false; echo "conda activate fishcluster_toolbox" >> /root/.bashrc
RUN . /root/.bashrc; pip install --upgrade tbb

RUN apt update
RUN apt install nano less build-essential -y

# resetting the Threading layer to tbb to prevent unsafe fork() calls in OpenMP using umap
# -> https://stackoverflow.com/questions/68131348/training-a-python-umap-model-hangs-in-a-multiprocessing-process
RUN export LD_LIBRARY_PATH=/opt/conda/envs/fishcluster_toolbox/lib/libtbb.so 

COPY . /workspace
WORKDIR /workspace
RUN . /root/.bashrc; python /workspace/setup.py build_ext --inplace

RUN chmod +x /workspace/misc/docker_entrypoint.sh

ENTRYPOINT ["/workspace/misc/docker_entrypoint.sh"]