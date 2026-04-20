@echo off
title Django Server
call venv\Scripts\activate
python manage.py runserver
pause