import os
import time
import gc
import threading
import uuid
import shutil
import asyncio
import aiohttp
import json
from asyncio import Semaphore
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.http import FileResponse, Http404, JsonResponse
from django.views.decorators.http import require_http_methods
from django.urls import reverse
import mimetypes

import static_ffmpeg
static_ffmpeg.add_paths()

from moviepy import VideoFileClip
from faster_whisper import WhisperModel

# ---------- Глобальная инициализация ----------
WHISPER_MODEL_SIZE = "small"
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=4,
    num_workers=2
)

translation_cache = {}

# ---------- Файловое хранилище статусов ----------
def get_status_file_path(task_id):
    return os.path.join(settings.TASK_STATUS_DIR, f"{task_id}.json")

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

def serve_video(request, filename):
    file_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', filename)
    if not os.path.exists(file_path):
        raise Http404("Видео не найдено")
    if os.path.getsize(file_path) == 0:
        raise Http404("Видеофайл повреждён (нулевой размер)")
    # Определяем MIME-тип
    import mimetypes
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

def convert_to_mp4(input_path, output_path):
    try:
        with VideoFileClip(input_path) as clip:
            clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)
        return True
    except Exception as e:
        print(f"Ошибка конвертации в MP4: {e}")
        return False

def merge_segments_into_sentences(segments):
    sentences = []
    current_text = ""
    current_start = None
    current_end = None

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if current_start is None:
            current_start = seg.start
        current_text += " " + text
        current_end = seg.end
        if any(text.rstrip().endswith(p) for p in ('.', '!', '?')):
            sentences.append({
                'start': current_start,
                'end': current_end,
                'text': current_text.strip()
            })
            current_text = ""
            current_start = None
            current_end = None

    if current_text:
        sentences.append({
            'start': current_start,
            'end': current_end,
            'text': current_text.strip()
        })
    return sentences

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
            print(f"Ошибка перевода: {e}")
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

# ---------- Основная задача обработки видео ----------
def process_video_task(task_id, tmp_path, original_filename):
    save_task_status(task_id, {'status': 'processing'})
    audio_path = None
    working_video = None
    try:
        # 1. Принудительная конвертация в MP4 (даже если файл уже .mp4)
        import subprocess
        ext = os.path.splitext(original_filename)[1].lower()
        converted_path = tmp_path.replace(ext, '.mp4')
        print(f"Конвертация {tmp_path} -> {converted_path} через ffmpeg")
        cmd_convert = [
            'ffmpeg', '-i', tmp_path,
            '-c:v', 'libx264', '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart',
            '-y', converted_path
        ]
        result = subprocess.run(cmd_convert, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Не удалось сконвертировать видео: {result.stderr}")
        working_video = converted_path
        safe_remove(tmp_path)  # оригинал больше не нужен

        # 2. Извлечение аудио через ffmpeg
        audio_path = working_video.replace('.mp4', '_temp.wav')
        cmd_audio = [
            'ffmpeg', '-i', working_video,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', audio_path
        ]
        result = subprocess.run(cmd_audio, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Не удалось извлечь аудио: {result.stderr}")

        # 3. Распознавание речи (faster-whisper)
        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=1,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        segments_list = list(segments)
        print(f"[INFO] Распознано {len(segments_list)} фрагментов, язык: {info.language}")

        # 4. Объединение в предложения и перевод (без изменений)
        sentences = merge_segments_into_sentences(segments_list)
        sentence_texts = [s['text'] for s in sentences]
        batch_size = 20
        translated_texts = []
        for i in range(0, len(sentence_texts), batch_size):
            batch = sentence_texts[i:i+batch_size]
            print(f"[INFO] Перевод предложений {i+1}-{min(i+batch_size, len(sentence_texts))}")
            batch_translated = run_async_translation(batch)
            translated_texts.extend(batch_translated)

        output_lines_html = []
        output_lines_plain = []
        subtitles = []
        for sent, trans in zip(sentences, translated_texts):
            start_str = format_time(sent['start'])
            end_str = format_time(sent['end'])
            html_line = f'<div class="fragment"><span class="timestamp" data-time="{sent["start"]}" data-end="{sent["end"]}">[{start_str} -> {end_str}]</span> {sent["text"]}<br>- {trans}<br><br></div>'
            output_lines_html.append(html_line)
            plain_line = f"[{start_str} -> {end_str}] {sent['text']}\n- {trans}\n\n"
            output_lines_plain.append(plain_line)
            subtitles.append({
                'start': sent['start'],
                'end': sent['end'],
                'en': sent['text'],
                'ru': trans
            })

        result_text_html = "".join(output_lines_html)
        result_text_plain = "".join(output_lines_plain)

        # 5. Сохранение результатов
        result_filename = f"{os.path.splitext(original_filename)[0]}_translated.txt"
        result_dir = os.path.join(settings.MEDIA_ROOT, 'results')
        os.makedirs(result_dir, exist_ok=True)
        result_file_path = os.path.join(result_dir, result_filename)
        with open(result_file_path, 'w', encoding='utf-8') as f:
            f.write(result_text_plain)

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

        # 6. Очистка
        safe_remove(working_video)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}")
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