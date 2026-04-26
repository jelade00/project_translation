web: gunicorn django_project.wsgi --workers 1 --threads 4 --timeout 3000
worker: celery -A django_project worker --loglevel=info --concurrency=1