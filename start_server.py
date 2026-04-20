import os
import subprocess
import sys

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    manage_path = os.path.join(base_dir, 'manage.py')

    if not os.path.exists(manage_path):
        print("Ошибка: manage.py не найден. Запустите скрипт из папки проекта.")
        sys.exit(1)

    # Поиск python внутри виртуального окружения
    venv_names = ['venv', '.venv', 'env', '.env']
    python_exe = None
    for venv_name in venv_names:
        venv_path = os.path.join(base_dir, venv_name)
        if os.path.exists(venv_path):
            # Windows
            candidate = os.path.join(venv_path, 'Scripts', 'python.exe')
            if os.path.exists(candidate):
                python_exe = candidate
                break
            # Linux / macOS
            candidate = os.path.join(venv_path, 'bin', 'python')
            if os.path.exists(candidate):
                python_exe = candidate
                break

    if python_exe is None:
        python_exe = sys.executable
        print("Виртуальное окружение не найдено. Использую системный Python.")

    # Запуск сервера
    cmd = [python_exe, manage_path, 'runserver']
    print("Запуск сервера:", ' '.join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()