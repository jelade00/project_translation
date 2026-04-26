import os
import json
import time
import gc
import asyncio
import aiohttp
from asyncio import Semaphore
from django.conf import settings
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

# ---------- Whisper и перевод ----------
WHISPER_MODEL_SIZE = "base"  # или "small", но для Railway лучше "base"
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=2,
    num_workers=1
)

translation_cache = {}

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

# ---------- Вспомогательные функции ----------
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