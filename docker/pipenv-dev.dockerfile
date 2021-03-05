FROM mtgupf/essentia:ubuntu18.04-v2.1_beta5

RUN apt-get update -y && apt-get -y install python3.6-dev python3-pip libsndfile1-dev ffmpeg libfftw3-dev libfftw3-doc

WORKDIR /solola
COPY . /solola 

RUN pip3 install pipenv 
RUN python3 -V
RUN pipenv sync

# # RUN chmod +x docker/entry.sh
# EXPOSE 5000