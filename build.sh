#!/usr/bin/env bash
pip cache purge
pip install -r requirements.txt
python manage.py collectstatic --noinput
python manage.py migrate