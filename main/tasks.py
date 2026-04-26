import os
import subprocess
import shutil
from django.conf import settings
from celery import shared_task
from django.urls import reverse

from .views import save_task_status, safe_remove, format_time
from .views import merge_segments_into_sentences, run_async_translation, whisper_model

@shared_task
def process_video_task(task_id, tmp_path, original_filename):
    save_task_status(task_id, {'status': 'processing'})
    audio_path = None
    working_video = None
    try:
        # 1. Конвертация в MP4 (если нужно)
        ext = os.path.splitext(original_filename)[1].lower()
        if ext != '.mp4':
            base, _ = os.path.splitext(tmp_path)
            converted_path = base + '_converted.mp4'
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

        # 2. Извлечение аудио
        audio_path = working_video.replace('.mp4', '_temp.wav')
        cmd_audio = [
            'ffmpeg', '-i', working_video,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audio_path
        ]
        result = subprocess.run(cmd_audio, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise Exception(f"Не удалось извлечь аудио: {result.stderr}")

        # 3. Распознавание речи
        segments, info = whisper_model.transcribe(
            audio_path, beam_size=1, language="en",
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500)
        )
        segments_list = list(segments)

        # 4. Объединение в предложения и перевод
        sentences = merge_segments_into_sentences(segments_list)
        sentence_texts = [s['text'] for s in sentences]

        batch_size = 20
        translated_texts = []
        for i in range(0, len(sentence_texts), batch_size):
            batch = sentence_texts[i:i+batch_size]
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
            subtitles.append({'start': sent['start'], 'end': sent['end'], 'en': sent['text'], 'ru': trans})

        result_text_html = "".join(output_lines_html)
        result_text_plain = "".join(output_lines_plain)

        # 5. Сохранение файлов
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

        # Очистка временных файлов
        if working_video and os.path.exists(working_video):
            safe_remove(working_video)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

    except Exception as e:
        save_task_status(task_id, {'status': 'failed', 'error': str(e)})
        safe_remove(tmp_path)
        if working_video and os.path.exists(working_video):
            safe_remove(working_video)
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass