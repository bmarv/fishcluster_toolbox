FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

COPY requirements.txt /tmp/requirements.txt
COPY --from=continuumio/miniconda3 /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

RUN conda create -n training_cpu python=3.9 -y

RUN conda init; . ~/.bashrc; conda activate training_cpu; pip install -r /tmp/requirements.txt; pip install torch
RUN . ~/.bashrc; conda activate training_cpu; pip install wandb; pip install --upgrade tbb

RUN apt update
RUN apt install nano less -y

RUN . ~/.bashrc; conda config --set auto_activate_base false; echo "conda activate training_cpu" >> ~/.bashrc
# resetting the Threading layer to tbb to prevent unsafe fork() calls in OpenMP using umap
# -> https://stackoverflow.com/questions/68131348/training-a-python-umap-model-hangs-in-a-multiprocessing-process
RUN export LD_LIBRARY_PATH=/opt/conda/envs/training_cpu/lib/libtbb.so 

# MANUAL
RUN echo 'echo "\n\nWelcome to Container ${HOSTNAME} of the fishcluster_toolbox image. \
    \nPlease make sure to specify the API-Key of Weights and Biases using \
    \n\t>>wandb login \
    \nA new training run can be started with \
    \n\t>>python train.py --n_neighbors xx --min_dist x.x --device x \
    \nData should be stored in /mnt/ \
    \nFor gpu-acceleration using cuda, please specify --device 0"' >> ~/.bashrc


COPY . /workspace

WORKDIR /workspace


CMD ["bash"]

# sudo docker run -it --gpus all -v /media/marv/Extreme_SSD2/PE_tracks_final:/mnt/ --env-file=private.env fishcluster_toolbox:v1 /bin/bash
# command to run the dockerfile interactively with gpu-acceleration, a mounted volume and a specified env-file for wandb-login credentials