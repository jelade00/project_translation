# #############################################################################
# Автор: LinguaScan
# Описание: Основной файл с логикой обработки видео, распознаванием речи,
#          переводом и статусами задач.
# #############################################################################

# ------------------------------------------------------------
# Стандартные библиотеки Python
# ------------------------------------------------------------
import sys          # вывод сообщений в stderr (логи Railway)
import os           # работа с файловой системой
import time         # замеры времени выполнения
import gc           # сборщик мусора (принудительная очистка)
import threading    # потоки – обработка видео в фоне
import uuid         # уникальные ID для задач
import shutil       # копирование файлов
import asyncio      # асинхронный код (для переводов)
import aiohttp      # асинхронные HTTP-запросы (к Google Translate)
import json         # сохранение статусов задач в JSON-файлы
import subprocess   # запуск внешних программ (ffmpeg)
import mimetypes    # определение MIME-типов (для видео)

# ------------------------------------------------------------
# Django-специфичные импорты
# ------------------------------------------------------------
from asyncio import Semaphore
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.http import FileResponse, Http404, JsonResponse
from django.views.decorators.http import require_http_methods
from django.urls import reverse

# ------------------------------------------------------------
# Сторонние библиотеки
# ------------------------------------------------------------
import static_ffmpeg                 # автоматическая установка FFmpeg
import nltk                          # токенизация текста на предложения
from nltk.tokenize import sent_tokenize
from faster_whisper import WhisperModel

# ------------------------------------------------------------
# Загрузка данных для NLTK (один раз при старте приложения)
# ------------------------------------------------------------
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# ------------------------------------------------------------
# Глобальная настройка модели распознавания речи faster-whisper
# ------------------------------------------------------------
# Модель "base" – хороший баланс скорости и качества на моём сервере.
# Параметры:
#   device="cpu"                    – нет видеокарты, используем CPU
#   compute_type="int8"             – 8-битное квантование (экономия памяти)
#   cpu_threads=2                   – задействуем 2 ядра процессора
#   num_workers=4                   – внутренняя параллельность (экспериментально)
WHISPER_MODEL_SIZE = "base"
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=2,
    num_workers=4
)
# Запись в лог, что модель загружена (для отладки на Railway)
sys.stderr.write("[DEBUG] Модель Whisper загружена\n")

# ------------------------------------------------------------
# Настройка пути к FFmpeg (static_ffmpeg загружает бинарник)
# ------------------------------------------------------------
static_ffmpeg.add_paths()
FFMPEG_PATH = static_ffmpeg.ffmpeg_path
sys.stderr.write(f"[DEBUG] FFmpeg path: {FFMPEG_PATH}\n")

# ------------------------------------------------------------
# Кэш для переводов (чтобы не переводить одинаковые фразы повторно)
# ------------------------------------------------------------
translation_cache = {}

# ------------------------------------------------------------
# Файловое хранилище статусов задач
# ------------------------------------------------------------
# Все статусы хранятся в папке media/task_statuses/ в виде JSON-файлов.
# Это позволяет избежать проблем с несколькими воркерами Gunicorn.
TASK_STATUS_DIR = os.path.join(settings.MEDIA_ROOT, 'task_statuses')
os.makedirs(TASK_STATUS_DIR, exist_ok=True)

def get_status_file_path(task_id):
    """Возвращает путь к JSON-файлу статуса для задачи task_id."""
    return os.path.join(TASK_STATUS_DIR, f"{task_id}.json")

def save_task_status(task_id, data):
    """Сохраняет статус задачи в JSON-файл."""
    file_path = get_status_file_path(task_id)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def get_task_status(task_id):
    """Читает статус задачи из JSON-файла или возвращает None, если файла нет."""
    file_path = get_status_file_path(task_id)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def delete_task_status(task_id):
    """Удаляет файл статуса (не используется в основном коде, но оставлено на всякий случай)."""
    file_path = get_status_file_path(task_id)
    if os.path.exists(file_path):
        os.remove(file_path)

# ------------------------------------------------------------
# Функция отдачи видео для плеера
# ------------------------------------------------------------
def serve_video(request, filename):
    """
    Отдаёт видеофайл из папки processed_videos.
    Установка правильного MIME-типа и заголовков для inline-воспроизведения.
    """
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

# ------------------------------------------------------------
# Вспомогательная функция безопасного удаления файлов (с повторными попытками)
# ------------------------------------------------------------
def safe_remove(file_path, max_attempts=5, delay=0.2):
    """
    Иногда на Windows файл может быть временно заблокирован.
    Делаем несколько попыток удаления с паузами.
    """
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

# ------------------------------------------------------------
# Форматирование секунд в ЧЧ:ММ:СС
# ------------------------------------------------------------
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ------------------------------------------------------------
# Ключевая функция: объединение фрагментов Whisper в предложения
# ------------------------------------------------------------
def merge_segments_into_sentences(segments, max_words=70, max_duration=15.0):
    """
    Объединяет фрагменты Faster-Whisper в осмысленные предложения.
    Алгоритм:
      1. Накопливание фрагментов в буфере вместе с их временем окончания.
      2. Использование NLTK для разбиения накопленного текста на предложения.
      3. Для каждого полного предложения (кроме последнего незаконченного)
         определение времени окончания по последнему фрагменту, который покрывает конец предложения.
      4. Если знаков препинания нет и текст становится слишком длинным (по словам/времени),
         принудительный разрыв.
    Возвращает список словарей: { start, end, text }.
    """
    sentences = []
    # Список кортежей: (текст_фрагмента, время_окончания). Нужно для точных таймкодов.
    buffer_items = []
    buffer_start = None
    buffer_text = ""

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        # Инициализация начала буфера при первом фрагменте
        if not buffer_items:
            buffer_start = seg.start

        buffer_items.append((text, seg.end))
        buffer_text = " ".join(t[0] for t in buffer_items)

        # Разбиваем накопленный текст на предложения (NLTK учитывает точки, !, ?)
        tokenized = sent_tokenize(buffer_text)

        # Если есть хотя бы одно полное предложение (т.е. в списке >1 элемента)
        if len(tokenized) > 1:
            pos = 0
            # Обрабатываем все предложения, кроме последнего (оно может быть незаконченным)
            for i in range(len(tokenized) - 1):
                sent = tokenized[i]
                sent_len = len(sent)
                end_pos = pos + sent_len
                # Поиск фрагмента, который покрывает символ end_pos
                cum_len = 0
                sent_end_time = buffer_start
                for t, end_t in buffer_items:
                    cum_len += len(t) + 1   # +1 за пробел между фрагментами
                    if cum_len >= end_pos:
                        sent_end_time = end_t
                        break
                # Добавление готового предложения с вычисленным временем окончания
                sentences.append({
                    'start': buffer_start,
                    'end': sent_end_time,
                    'text': sent
                })
                # Сдвиг начала следующего предложения на время окончания предыдущего
                buffer_start = sent_end_time
                pos = end_pos
            # Последнее (незаконченное) предложение остаётся в буфере
            last_sent = tokenized[-1]
            # Сброс буфера, оставляя только этот остаток
            buffer_items = [(last_sent, buffer_items[-1][1])]
            buffer_text = last_sent
        else:
            # Нет ни одного законченного предложения – проверяем лимиты для принудительного разрыва
            duration = buffer_items[-1][1] - buffer_start
            word_count = len(buffer_text.split())
            if word_count > max_words or duration > max_duration:
                sentences.append({
                    'start': buffer_start,
                    'end': buffer_items[-1][1],
                    'text': buffer_text
                })
                buffer_items = []
                buffer_start = None
                buffer_text = ""

    # Если в конце видео остался текст (без точки), добавляем его как есть
    if buffer_items:
        sentences.append({
            'start': buffer_start,
            'end': buffer_items[-1][1],
            'text': buffer_text
        })

    return sentences

# ------------------------------------------------------------
# Асинхронный перевод текста через Google Translate (прямые HTTP-запросы)
# ------------------------------------------------------------
async def translate_text_async(session, text, semaphore):
    """
    Перевод одного текста с английского на русский.
    Используется бесплатный API translate.googleapis.com.
    Кэширование результата, чтобы не переводить повторно одну и ту же фразу.
    """
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
            # Пишем ошибку в лог, но не прерываем обработку
            sys.stderr.write(f"Ошибка перевода: {e}\n")
            sys.stderr.flush()
            return text

async def translate_batch_async(texts, max_concurrent=5):
    """
    Перевод списка текстов параллельно (одновременно не более max_concurrent запросов).
    """
    semaphore = Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = [translate_text_async(session, t, semaphore) for t in texts]
        return await asyncio.gather(*tasks)

def run_async_translation(texts):
    """
    Запуск асинхронного выполнения translate_batch_async из синхронного контекста.
    Создание нового event loop и его закрытие после завершения.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(translate_batch_async(texts))
    finally:
        loop.close()

# ------------------------------------------------------------
# Основная задача обработки видео (выполняется в фоновом потоке)
# ------------------------------------------------------------
def process_video_task(task_id, tmp_path, original_filename):
    """
    Выполняется в отдельном потоке после того, как файл уже загружен на сервер.
    Этапы:
      1. Проверка свободного места.
      2. Конвертация в MP4 (если файл не .mp4). Использование ffmpeg.
      3. Извлечение аудио в WAV (16 кГц, моно) через ffmpeg.
      4. Распознавание речи с помощью faster-whisper.
      5. Объединение фрагментов в предложения (merge_segments_into_sentences).
      6. Пакетный асинхронный перевод предложений.
      7. Формирование HTML, plain-текста и субтитров.
      8. Сохранение результата и статуса.
      9. Очистка временных файлов.
    """
    import time
    start_total = time.time()

    sys.stderr.write(f"\n[DEBUG] === НАЧАЛО ОБРАБОТКИ ЗАДАЧИ {task_id} ===\n")
    sys.stderr.flush()
    save_task_status(task_id, {'status': 'processing'})

    audio_path = None
    working_video = None

    try:
        # ---- 1. Проверка свободного места ----
        t = time.time()
        total, used, free = shutil.disk_usage(settings.MEDIA_ROOT)
        sys.stderr.write(f"[TIME] Проверка диска: {time.time()-t:.2f} сек\n")
        sys.stderr.write(f"[DEBUG] Свободно на диске: {free // (2**20)} МБ\n")
        sys.stderr.flush()
        if free < 500 * 1024 * 1024:
            raise Exception(f"Недостаточно места на диске: {free // (2**20)} МБ")

        # ---- 2. Конвертация в MP4 (только если исходный файл не .mp4) ----
        ext = os.path.splitext(original_filename)[1].lower()
        if ext != '.mp4':
            base, _ = os.path.splitext(tmp_path)
            converted_path = base + '_converted.mp4'
            sys.stderr.write(f"[DEBUG] Конвертация {tmp_path} -> {converted_path}\n")
            sys.stderr.flush()
            t = time.time()
            cmd_convert = [
                FFMPEG_PATH, '-i', tmp_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-c:a', 'aac', '-b:a', '128k',
                '-movflags', '+faststart',
                '-y', converted_path
            ]
            result = subprocess.run(cmd_convert, capture_output=True, text=True, timeout=600)
            sys.stderr.write(f"[TIME] Конвертация ffmpeg: {time.time()-t:.2f} сек\n")
            if result.returncode != 0:
                raise Exception(f"Не удалось сконвертировать видео: {result.stderr}")
            safe_remove(tmp_path)
            working_video = converted_path
        else:
            working_video = tmp_path
            sys.stderr.write(f"[DEBUG] Пропускаем конвертацию, файл уже MP4\n")
            sys.stderr.flush()

        # ---- 3. Извлечение аудио через ffmpeg ----
        base_audio = os.path.splitext(working_video)[0]
        audio_path = base_audio + '_temp.wav'
        sys.stderr.write(f"[DEBUG] Извлечение аудио {working_video} -> {audio_path}\n")
        sys.stderr.flush()
        cmd_audio = [
            FFMPEG_PATH, '-i', working_video,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', audio_path
        ]
        result = subprocess.run(cmd_audio, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise Exception(f"Не удалось извлечь аудио: {result.stderr}")

        # ---- 4. Распознавание речи через faster-whisper ----
        sys.stderr.write(f"[DEBUG] Распознавание речи через Whisper (модель {WHISPER_MODEL_SIZE})\n")
        sys.stderr.flush()
        t = time.time()
        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=1,        # жадный поиск – быстрее
            language="en",
            vad_filter=False,   # отключение VAD для ускорения
        )
        segments_list = list(segments)
        sys.stderr.write(f"[TIME] Распознавание речи: {time.time()-t:.2f} сек\n")
        sys.stderr.write(f"[DEBUG] Распознано {len(segments_list)} фрагментов\n")
        sys.stderr.flush()

        # ---- 5. Объединение фрагментов в предложения ----
        sentences = merge_segments_into_sentences(segments_list)
        sys.stderr.write(f"[DEBUG] Объединено в {len(sentences)} предложений\n")
        sys.stderr.flush()

        # ---- 6. Перевод предложений (пакетами по 20) ----
        sentence_texts = [s['text'] for s in sentences]
        batch_size = 20
        translated_texts = []
        t_translate_start = time.time()
        for i in range(0, len(sentence_texts), batch_size):
            batch = sentence_texts[i:i+batch_size]
            sys.stderr.write(f"[DEBUG] Перевод предложений {i+1}-{min(i+batch_size, len(sentence_texts))}\n")
            sys.stderr.flush()
            t_batch = time.time()
            batch_translated = run_async_translation(batch)
            sys.stderr.write(f"[TIME] Пакет {i//batch_size+1} переведён за {time.time()-t_batch:.2f} сек\n")
            translated_texts.extend(batch_translated)
        sys.stderr.write(f"[TIME] Общее время перевода всех предложений: {time.time()-t_translate_start:.2f} сек\n")

        # ---- 7. Формирование HTML, plain-текста и списка субтитров ----
        t = time.time()
        output_lines_html = []
        output_lines_plain = []
        subtitles = []
        for sent, trans in zip(sentences, translated_texts):
            start_str = format_time(sent['start'])
            end_str = format_time(sent['end'])
            # HTML-строка с обёрткой .fragment (нужна для парсинга субтитров на клиенте)
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
        sys.stderr.write(f"[TIME] Формирование HTML и субтитров: {time.time()-t:.2f} сек\n")

        # ---- 8. Сохранение текстового результата ----
        t = time.time()
        result_filename = f"{os.path.splitext(original_filename)[0]}_translated.txt"
        result_dir = os.path.join(settings.MEDIA_ROOT, 'results')
        os.makedirs(result_dir, exist_ok=True)
        result_file_path = os.path.join(result_dir, result_filename)
        with open(result_file_path, 'w', encoding='utf-8') as f:
            f.write(result_text_plain)
        sys.stderr.write(f"[TIME] Сохранение текстового файла: {time.time()-t:.2f} сек\n")

        # ---- 9. Сохранение видео для плеера (копирование в processed_videos) ----
        t = time.time()
        video_dir = os.path.join(settings.MEDIA_ROOT, 'processed_videos')
        os.makedirs(video_dir, exist_ok=True)
        video_name, _ = os.path.splitext(original_filename)
        saved_video_name = f"{video_name}_{task_id}.mp4"
        saved_video_path = os.path.join(video_dir, saved_video_name)
        shutil.copy2(working_video, saved_video_path)
        video_url = reverse('serve_video', args=[saved_video_name])
        sys.stderr.write(f"[TIME] Копирование видео: {time.time()-t:.2f} сек\n")

        # ---- 10. Финальное сохранение статуса задачи ----
        t = time.time()
        save_task_status(task_id, {
            'status': 'completed',
            'result_text_html': result_text_html,
            'result_text_plain': result_text_plain,
            'file_url': settings.MEDIA_URL + 'results/' + result_filename,
            'video_url': video_url,
            'subtitles': subtitles
        })
        sys.stderr.write(f"[TIME] Сохранение статуса: {time.time()-t:.2f} сек\n")

        sys.stderr.write(f"[DEBUG] Задача {task_id} успешно завершена\n")
        sys.stderr.write(f"[TIME] ОБЩЕЕ ВРЕМЯ ОБРАБОТКИ: {time.time()-start_total:.2f} сек\n")
        sys.stderr.flush()

        # ---- 11. Очистка временных файлов ----
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

# ------------------------------------------------------------
# Django views (обработчики HTTP-запросов)
# ------------------------------------------------------------
@require_http_methods(["GET", "POST"])
def index_page(request):
    """Главная страница. Обработка загрузки видео и запуск фоновой задачи."""
    if request.method == "POST" and request.FILES.get('video_file'):
        video_file = request.FILES['video_file']
        ext = os.path.splitext(video_file.name)[1].lower()
        if ext not in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            return JsonResponse({'error': 'Неподдерживаемый формат'}, status=400)

        # Создание временной папки и сохранение загруженного файла
        tmp_dir = os.path.join(settings.MEDIA_ROOT, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, video_file.name)
        with default_storage.open(tmp_path, 'wb+') as dest:
            for chunk in video_file.chunks():
                dest.write(chunk)

        task_id = str(uuid.uuid4())
        save_task_status(task_id, {'status': 'processing'})
        # Запуск обработки в отдельном потоке, чтобы не блокировать ответ клиенту
        thread = threading.Thread(target=process_video_task, args=(task_id, tmp_path, video_file.name))
        thread.daemon = True
        thread.start()

        return JsonResponse({'task_id': task_id, 'status': 'processing'})

    return render(request, 'index-page.html')

def task_status(request, task_id):
    """Возвращает статус задачи (processing, completed, failed) в формате JSON."""
    status = get_task_status(task_id)
    if status is None:
        return JsonResponse({'status': 'not_found'})
    return JsonResponse(status)

def download_result(request, filename):
    """Скачивание готового текстового файла с переводом."""
    file_path = os.path.join(settings.MEDIA_ROOT, 'results', filename)
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=filename)
    raise Http404("Файл не найден")

def about_page(request):
    """Страница «О проекте»."""
    return render(request, 'about.html')