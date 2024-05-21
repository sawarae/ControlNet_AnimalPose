FROM sawarae/miniconda:py39cuda124
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH
RUN apt install -y git libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
COPY environment.yaml .
RUN conda env create -f environment.yaml && \
    conda init && \
    echo "conda activate ControlPose" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV ControlPose && \
    PATH /opt/conda/envs/ControlPose/bin:$PATH

SHELL ["conda", "run", "-n", "ControlPose", "/bin/bash", "-c"]
RUN pip install -U openmim && mim install mmengine && mim install "mmcv==2.1.0" "mmdet>=3.0.0"
RUN git clone https://github.com/open-mmlab/mmpose.git && cd mmpose && pip install -v -e .