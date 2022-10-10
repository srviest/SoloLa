# Usage

Requirements
- [docker engine](https://docs.docker.com/engine/install/)


Run CLI to get transcription results
```bash
docker pull ykhorizon/solola_cli:latest
cd solola
# ${PWD} means current command line location
docker run -v ${PWD}/inputs:/solola/inputs -v ${PWD}/outputs:/solola/outputs solola_cli:latest -o /solola/outputs/old_licks "inputs/old_licks/*.wav
``




