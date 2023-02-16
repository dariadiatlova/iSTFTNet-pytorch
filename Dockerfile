FROM python:3.8

WORKDIR /app
COPY . /app

RUN apt-get update -y
RUN apt-get install -y --no-install-recommends build-essential gcc libsndfile1
RUN apt-get install -y vim

RUN pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install -r requirements.txt

ENTRYPOINT [ "bash" ]