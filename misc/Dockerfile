FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

COPY requirements.txt /tmp/requirements.txt
COPY --from=continuumio/miniconda3 /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

RUN conda create -n rapids-24.06 -c rapidsai-nightly -c conda-forge -c nvidia  \
    rapids=24.06 python=3.9 cuda-version=12.2 -y

RUN conda init; . ~/.bashrc; conda activate rapids-24.06; pip install -r /tmp/requirements.txt; pip install torch
RUN . ~/.bashrc; conda activate rapids-24.06; pip install wandb
# apt installations
RUN apt update
RUN apt install nano less -y

COPY . /workspace

WORKDIR /workspace

CMD ["bash"]