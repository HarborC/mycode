FROM docker.1ms.run/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装基础工具
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# 接受 conda TOS 并初始化 shell
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda init bash

# 安装 Node.js 和 npm
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# 安装 claude-code 和 happy-coder
RUN npm install -g @anthropic-ai/claude-code happy-coder && \
    echo 'export HAPPY_SERVER_URL=https://harborchen.top' >> ~/.bashrc

# 配置 claude-code
RUN mkdir -p ~/.claude && \
    printf '{\n  "env": {\n    "ANTHROPIC_BASE_URL": "https://ai.vdian.net/api",\n    "ANTHROPIC_AUTH_TOKEN": "cr_2cb9004a7287b6c8c54e455acdc02145c6302318b42993a3ce9a1ad35043ed97"\n  }\n}\n' > ~/.claude/settings.json

# 创建 conda 虚拟环境并安装 PyTorch
RUN conda create -n test python=3.12 -y && \
    conda run -n test pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu130

# 验证安装
RUN conda --version && node --version && npm --version

CMD ["/bin/bash", "--login"]
