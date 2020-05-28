FROM mtgupf/essentia

# update system package
RUN apt-get update -y && apt-get -y install python3-pip

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY . /usr/src/services
WORKDIR /usr/src/services



# Install solola transcription algorithm

COPY ./prerequisites.sh /usr/src/services/prerequisites.sh
RUN sh prerequisites.sh

# Install http server and server framework

COPY ./install_service_dependency.sh /usr/src/services/install_service_dependency.sh
RUN sh install_service_dependency.sh

