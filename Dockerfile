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
RUN apt install nano less -y

RUN . ~/.bashrc; conda config --set auto_activate_base false; echo "conda activate rapids-22.04" >> ~/.bashrc
RUN echo 'echo "\n\nWelcome to Container ${HOSTNAME} of the fishcluster_toolbox image. \
    \nPlease make sure to specify the API-Key of Weights and Biases using \
    \n\t>>wandb login \
    \nA new training run can be started with \
    \n\t>>python train.py --n_neighbors xx --min_dist x.x --device x --data "/path"\
    \nData should be stored in /mnt/ \
    \nFor gpu-acceleration using cuda, please specify --device 0"' >> ~/.bashrc


COPY . /workspace

WORKDIR /workspace


CMD ["bash"]

# sudo docker run -it --gpus all -v /media/marv/Extreme_SSD2/PE_tracks_final:/mnt/ --env-file=private.env fishcluster_toolbox:v1 /bin/bash
# command to run the dockerfile interactively with gpu-acceleration, a mounted volume and a specified env-file for wandb-login credentials