import sys
import os
import time
import gc
import threading
import uuid
import shutil
import asyncio
import aiohttp
import json
import subprocess
import mimetypes
from asyncio import Semaphore
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.http import FileResponse, Http404, JsonResponse
from django.views.decorators.http import require_http_methods
from django.urls import reverse

import static_ffmpeg
static_ffmpeg.add_paths()

from faster_whisper import WhisperModel

# ---------- Глобальная инициализация (ускоренная) ----------
WHISPER_MODEL_SIZE = "base"
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=2,
    num_workers=1
)

translation_cache = {}

# ---------- Файловое хранилище статусов ----------
TASK_STATUS_DIR = os.path.join(settings.MEDIA_ROOT, 'task_statuses')
os.makedirs(TASK_STATUS_DIR, exist_ok=True)

def get_status_file_path(task_id):
    return os.path.join(TASK_STATUS_DIR, f"{task_id}.json")

def save_task_status(task_id, data):
    file_path = get_status_file_path(task_id)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def get_task_status(task_id):
    file_path = get_status_file_path(task_id)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def delete_task_status(task_id):
    file_path = get_status_file_path(task_id)
    if os.path.exists(file_path):
        os.remove(file_path)

# ---------- Вспомогательные функции ----------
def serve_video(request, filename):
    file_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', filename)
    if not os.path.exists(file_path):
        raise Http404("Видео не найдено")
    if os.path.getsize(file_path) == 0:
        raise Http404("Видеофайл повреждён (нулевой размер)")
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = 'video/mp4'
    response = FileResponse(open(file_path, 'rb'), content_type=mime_type)
    response['Content-Disposition'] = f'inline; filename="{filename}"'
    return response

def safe_remove(file_path, max_attempts=5, delay=0.2):
    for attempt in range(max_attempts):
        try:
            os.remove(file_path)
            return True
        except PermissionError:
            if attempt < max_attempts - 1:
                time.sleep(delay)
                gc.collect()
            else:
                raise

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ---------- Асинхронный перевод ----------
async def translate_text_async(session, text, semaphore):
    if text in translation_cache:
        return translation_cache[text]
    async with semaphore:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "en",
            "tl": "ru",
            "dt": "t",
            "q": text
        }
        try:
            async with session.get(url, params=params, timeout=10) as resp:
                data = await resp.json()
                translated = data[0][0][0] if data and data[0] else text
                translation_cache[text] = translated
                return translated
        except Exception as e:
            sys.stderr.write(f"Ошибка перевода: {e}\n")
            sys.stderr.flush()
            return text

async def translate_batch_async(texts, max_concurrent=5):
    semaphore = Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = [translate_text_async(session, t, semaphore) for t in texts]
        return await asyncio.gather(*tasks)

def run_async_translation(texts):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(translate_batch_async(texts))
    finally:
        loop.close()

# ---------- Основная задача обработки видео (без объединения фрагментов) ----------
def process_video_task(task_id, tmp_path, original_filename):
    sys.stderr.write(f"[DEBUG] process_video_task вызвана для {task_id}\n")
    sys.stderr.flush()
    save_task_status(task_id, {'status': 'processing'})
    audio_path = None
    working_video = None
    try:
        # 1. Проверка свободного места
        total, used, free = shutil.disk_usage(settings.MEDIA_ROOT)
        sys.stderr.write(f"[DEBUG] Свободно на диске: {free // (2**20)} МБ\n")
        sys.stderr.flush()
        if free < 500 * 1024 * 1024:
            raise Exception(f"Недостаточно места на диске: {free // (2**20)} МБ")

        # 2. Конвертация в MP4 – только если исходный файл не .mp4
        ext = os.path.splitext(original_filename)[1].lower()
        if ext != '.mp4':
            base, _ = os.path.splitext(tmp_path)
            converted_path = base + '_converted.mp4'
            sys.stderr.write(f"[DEBUG] Конвертация {tmp_path} -> {converted_path}\n")
            sys.stderr.flush()
            cmd_convert = [
                'ffmpeg', '-i', tmp_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-c:a', 'aac', '-b:a', '128k',
                '-movflags', '+faststart',
                '-y', converted_path
            ]
            result = subprocess.run(cmd_convert, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise Exception(f"Не удалось сконвертировать видео: {result.stderr}")
            safe_remove(tmp_path)
            working_video = converted_path
        else:
            working_video = tmp_path
            sys.stderr.write(f"[DEBUG] Пропускаем конвертацию, файл уже MP4\n")
            sys.stderr.flush()

        # 3. Извлечение аудио через ffmpeg
        audio_path = working_video.replace('.mp4', '_temp.wav')
        sys.stderr.write(f"[DEBUG] Извлечение аудио {working_video} -> {audio_path}\n")
        sys.stderr.flush()
        cmd_audio = [
            'ffmpeg', '-i', working_video,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', audio_path
        ]
        result = subprocess.run(cmd_audio, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise Exception(f"Не удалось извлечь аудио: {result.stderr}")

        # 4. Распознавание речи (быстрые настройки)
        sys.stderr.write(f"[DEBUG] Распознавание речи через Whisper (модель {WHISPER_MODEL_SIZE})\n")
        sys.stderr.flush()
        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=1,
            language="en",
            vad_filter=False,
        )
        segments_list = list(segments)
        sys.stderr.write(f"[DEBUG] Распознано {len(segments_list)} фрагментов, язык: {info.language}\n")
        sys.stderr.flush()

        # 5. Переводим каждый фрагмент отдельно (без объединения в предложения)
        fragment_texts = [seg.text for seg in segments_list]
        batch_size = 20
        translated_texts = []
        for i in range(0, len(fragment_texts), batch_size):
            batch = fragment_texts[i:i+batch_size]
            sys.stderr.write(f"[DEBUG] Перевод фрагментов {i+1}-{min(i+batch_size, len(fragment_texts))}\n")
            sys.stderr.flush()
            batch_translated = run_async_translation(batch)
            translated_texts.extend(batch_translated)

        # 6. Формирование вывода (HTML, plain, subtitles)
        output_lines_html = []
        output_lines_plain = []
        subtitles = []
        for seg, trans in zip(segments_list, translated_texts):
            start_str = format_time(seg.start)
            end_str = format_time(seg.end)
            html_line = f'<div class="fragment"><span class="timestamp" data-time="{seg.start}" data-end="{seg.end}">[{start_str} -> {end_str}]</span> {seg.text}<br>- {trans}<br><br></div>'
            output_lines_html.append(html_line)
            plain_line = f"[{start_str} -> {end_str}] {seg.text}\n- {trans}\n\n"
            output_lines_plain.append(plain_line)
            subtitles.append({
                'start': seg.start,
                'end': seg.end,
                'en': seg.text,
                'ru': trans
            })

        result_text_html = "".join(output_lines_html)
        result_text_plain = "".join(output_lines_plain)

        # 7. Сохранение текстового результата
        result_filename = f"{os.path.splitext(original_filename)[0]}_translated.txt"
        result_dir = os.path.join(settings.MEDIA_ROOT, 'results')
        os.makedirs(result_dir, exist_ok=True)
        result_file_path = os.path.join(result_dir, result_filename)
        with open(result_file_path, 'w', encoding='utf-8') as f:
            f.write(result_text_plain)

        # 8. Сохранение видео для плеера
        video_dir = os.path.join(settings.MEDIA_ROOT, 'processed_videos')
        os.makedirs(video_dir, exist_ok=True)
        video_name, _ = os.path.splitext(original_filename)
        saved_video_name = f"{video_name}_{task_id}.mp4"
        saved_video_path = os.path.join(video_dir, saved_video_name)
        shutil.copy2(working_video, saved_video_path)
        video_url = reverse('serve_video', args=[saved_video_name])

        save_task_status(task_id, {
            'status': 'completed',
            'result_text_html': result_text_html,
            'result_text_plain': result_text_plain,
            'file_url': settings.MEDIA_URL + 'results/' + result_filename,
            'video_url': video_url,
            'subtitles': subtitles
        })

        sys.stderr.write(f"[DEBUG] Задача {task_id} успешно завершена\n")
        sys.stderr.flush()

        # 9. Очистка временных файлов
        if working_video and os.path.exists(working_video):
            safe_remove(working_video)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

    except Exception as e:
        import traceback
        sys.stderr.write(f"[ERROR] Task {task_id} failed: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        save_task_status(task_id, {'status': 'failed', 'error': str(e)})
        safe_remove(tmp_path)
        if working_video and os.path.exists(working_video):
            safe_remove(working_video)
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

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