# Usage

Requirements
- [docker engine](https://docs.docker.com/engine/install/)


## Images

If you want to develop on solola, use the following commands
```bash
# build dev image
docker build -f docker/cli.dockerfile -t solola_cli:dev .
# enter container with bash
docker run -it -v ${PWD}:/solola solola_cli:dev bash
```

## CLI

Run CLI to get transcription results
```bash
docker pull ykhorizon/solola_cli:latest
cd solola
# ${PWD} means current command line location
docker run -v ${PWD}/inputs:/solola/inputs -v ${PWD}/outputs:/solola/outputs solola_cli:latest -o /solola/outputs/old_licks "inputs/old_licks/*.wav
``




