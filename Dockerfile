FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt update && \
    apt install --no-install-recommends -y python3.10 python3-pip && \
    apt clean && \
    rm -rf /var/lib/apt/lists/ \
    pip3 install --no-cache-dir torch torchvision torchaudio torchmetrics 
