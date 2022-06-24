FROM python:3.8

RUN apt-get update -y && apt-get upgrade -y

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu113