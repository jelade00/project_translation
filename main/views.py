import os
import uuid
import threading
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.http import FileResponse, Http404, JsonResponse
from django.views.decorators.http import require_http_methods
import static_ffmpeg
static_ffmpeg.add_paths()

from .utils import save_task_status, get_task_status, process_video_task  # process_video_task теперь в utils

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
        # Запускаем фоновый поток (без Celery)
        thread = threading.Thread(target=process_video_task, args=(task_id, tmp_path, video_file.name))
        thread.daemon = True
        thread.start()

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

def serve_video(request, filename):
    file_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', filename)
    if not os.path.exists(file_path):
        raise Http404("Видео не найдено")
    if os.path.getsize(file_path) == 0:
        raise Http404("Видеофайл повреждён (нулевой размер)")
    import mimetypes
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = 'video/mp4'
    response = FileResponse(open(file_path, 'rb'), content_type=mime_type)
    response['Accept-Ranges'] = 'bytes'
    response['Content-Disposition'] = f'inline; filename="{filename}"'
    return response