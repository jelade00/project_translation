import static_ffmpeg

def ensure_ffmpeg():
    print("Проверка FFmpeg...")
    static_ffmpeg.add_paths()
    print("FFmpeg готов к использованию.")

if __name__ == "__main__":
    ensure_ffmpeg()
    print("Установка FFmpeg завершена. Теперь можно запускать основной скрипт.")