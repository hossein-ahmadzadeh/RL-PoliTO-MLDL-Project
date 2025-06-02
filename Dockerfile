FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-distutils python3.9-dev \
    python3-pip git wget unzip curl libgl1-mesa-glx libosmesa6-dev \
    libglfw3 patchelf libglew-dev libgl1-mesa-dev libxrandr-dev libxinerama-dev \
    libxcursor-dev libxi-dev && \
    rm -rf /var/lib/apt/lists/*

# Link python3.9 as default and upgrade pip safely
RUN ln -s /usr/bin/python3.9 /usr/bin/python && \
    python -m pip install --upgrade pip==23.3.1

# Install PyTorch with CUDA 11.8 first, with hash checking disabled
RUN pip install --no-cache-dir --disable-pip-version-check \
    torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy and install remaining Python requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Download and install MuJoCo
RUN mkdir -p /root/.mujoco && \
    cd /root/.mujoco && \
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar -xvzf mujoco210-linux-x86_64.tar.gz && \
    rm mujoco210-linux-x86_64.tar.gz

COPY mjkey.txt /root/.mujoco/mjkey.txt

ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

# Set working directory and copy project files
WORKDIR /app
COPY . /app

CMD ["bash"]
