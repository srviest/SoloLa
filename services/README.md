# Solola http API Service (WIP)

## Introduction
If you want to use integrated service(Flask), you use follow usage instructions

## Development

```bash
# enter docker env
docker run -it --rm -p 5000:5000 -v $(pwd):/solola solola_api:0.0.1 bash

# gunicorn
pipenv run gunicorn -b 0.0.0.0:5000 -w 4 --chdir=/solola services/API/api:app

# flask dev server
pipenv run python3 services/API/api.py
```

testing entry.sh
```bash
docker run -it --rm -p 5000:5000 -v [wsl_project_path]:/solola solola_api:0.0.1
sh entry.sh
```

## production
```bash
sudo docker run -it --rm -p 5000:5000 ykhorizon/solola_api:0.0.0
```
# Build Image
<pre>
cd [project_root]
docker build -t solola_api:[tag] -f docker/pipenv-dev.dockerfile .
</pre>


# Development without docker

```bash
python services/manage.py run -h localhost
```

# Development in docker pipenv 

```bash
# in pipenv venv
python3 services/API/api.py
```