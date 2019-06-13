FROM mtgupf/essentia

# update system package
RUN apt-get update -y
RUN apt-get -y install python3-pip -y
RUN apt-get install libav-tools -y
COPY . /soloLa
WORKDIR /soloLa
COPY prerequisites.sh /soloLa/prerequisites.sh
RUN sh prerequisites.sh
