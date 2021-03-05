# SoloLa with docker

We highly recommend to solola with docker
> https://hub.docker.com/repository/docker/ykhorizon/solola_api

## Dependencies

We use essentia docker image as base images:
> https://hub.docker.com/r/mtgupf/essentia
Other dependency check out [SoloLa Requirements](https://github.com/SoloLa-Platform/SoloLa/tree/dev_version)

### Usage 
<pre>
sudo docker run -ti --rm -v /home/ykhorizon/workspace/solola/solola:/solola ykhorizon/solola_api:stable pipenv run cli
</pre>
