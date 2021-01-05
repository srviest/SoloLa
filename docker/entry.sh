#!/bin/sh
# version 1
cd services/
path=$(pipenv --venv)
echo "$path"
eval "$path/bin/gunicorn -b :5000 -w 4 wsgi:app"
# version 2

pipenv run gunicorn -b 0.0.0.0:5000 -w 4 --chdir=/home/solola/services wsgi:app
