
FROM mtgupf/essentia:ubuntu18.04-v2.1_beta5
# production or development
ARG FLASK_ENV
ENV FLASK_ENV=${FLASK_ENV}
# update system package
RUN apt-get update -y && apt-get -y install python3-pip libsndfile1-dev

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR false

WORKDIR /solola
COPY . /solola 

RUN pip3 install pipenv 
RUN pipenv install --deploy --ignore-pipfile

RUN chmod +x docker/entry.sh
EXPOSE 5000
ENTRYPOINT ["./docker/entry.sh"]