FROM mtgupf/essentia

# update system package
RUN apt-get update -y && apt-get -y install python3-pip ffmpeg
COPY . /solola
WORKDIR /solola
COPY prerequisites.sh /solola/prerequisites.sh
RUN sh prerequisites.sh
