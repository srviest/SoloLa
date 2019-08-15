FROM mtgupf/essentia

# update system package & installation
RUN apt-get update -y && apt-get install -y python3-pip libfftw3-dev libfftw3-doc ffmpeg 
WORKDIR /
RUN git clone https://github.com/SoloLa-Platform/solola
WORKDIR /solola
RUN sh prerequisites.sh