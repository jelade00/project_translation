import os
import uuid
import json
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.http import FileResponse, Http404, JsonResponse
from django.views.decorators.http import require_http_methods

import static_ffmpeg
static_ffmpeg.add_paths()

from .tasks import process_video_task  # импортируем задачу Celery

# ---------- Файловое хранилище статусов ----------
TASK_STATUS_DIR = os.path.join(settings.MEDIA_ROOT, 'task_statuses')
os.makedirs(TASK_STATUS_DIR, exist_ok=True)

def get_status_file_path(task_id):
    return os.path.join(TASK_STATUS_DIR, f"{task_id}.json")

def save_task_status(task_id, data):
    with open(get_status_file_path(task_id), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def get_task_status(task_id):
    path = get_status_file_path(task_id)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# ---------- Django views ----------
@require_http_methods(["GET", "POST"])
def index_page(request):
    if request.method == "POST" and request.FILES.get('video_file'):
        video_file = request.FILES['video_file']
        ext = os.path.splitext(video_file.name)[1].lower()
        if ext not in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            return JsonResponse({'error': 'Неподдерживаемый формат'}, status=400)

        tmp_dir = os.path.join(settings.MEDIA_ROOT, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, video_file.name)
        with default_storage.open(tmp_path, 'wb+') as dest:
            for chunk in video_file.chunks():
                dest.write(chunk)

        task_id = str(uuid.uuid4())
        save_task_status(task_id, {'status': 'processing'})
        # Асинхронный запуск задачи Celery
        process_video_task.delay(task_id, tmp_path, video_file.name)

        return JsonResponse({'task_id': task_id, 'status': 'processing'})

    return render(request, 'index-page.html')

def task_status(request, task_id):
    status = get_task_status(task_id)
    if status is None:
        return JsonResponse({'status': 'not_found'})
    return JsonResponse(status)

def download_result(request, filename):
    file_path = os.path.join(settings.MEDIA_ROOT, 'results', filename)
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=filename)
    raise Http404("Файл не найден")

def about_page(request):
    return render(request, 'about.html')