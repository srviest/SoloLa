FROM mtgupf/essentia:ubuntu18.04-v2.1_beta5

RUN apt-get update -y && apt-get -y install python3.6-dev python3-pip libsndfile1-dev ffmpeg libfftw3-dev libfftw3-doc

ENV PYTHONUNBUFFERED 1

WORKDIR /solola

RUN pip3 install --upgrade pip

COPY requirements-base.txt requirements-base.txt
RUN pip3 install -r requirements-base.txt

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /solola 