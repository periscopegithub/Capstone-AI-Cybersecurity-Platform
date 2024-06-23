import subprocess
import time
import os


def start_redis():
    redis_path = os.path.join("Redis", "start.bat")
    subprocess.Popen([redis_path])
    print("Redis started")


def start_ngrok():
    ngrok_path = os.path.join("ngrok", "ngrok.exe")
    subprocess.Popen(
        [ngrok_path, "http", "--domain=lynx-adapting-stork.ngrok-free.app", "5000"]
    )
    print("Ngrok started")


def start_celery():
    celery_command = "celery -A celery_app.celery worker --pool=solo --loglevel=info"
    subprocess.Popen(celery_command, shell=True)
    print("Celery started")


def start_flask():
    flask_process = subprocess.Popen(["python", "survey.py"])
    return flask_process


if __name__ == "__main__":
    # Start Redis
    print("Starting Redis...")
    redis_process = start_redis()
    time.sleep(5)  # Give some time for Redis to start

    # Start Ngrok
    print("Starting Ngrok...")
    ngrok_process = start_ngrok()
    time.sleep(5)  # Give some time for Ngrok to start

    # Start Celery
    print("Starting Celery...")
    celery_process = start_celery()
    time.sleep(5)  # Give some time for Celery to start

    # Start Flask
    print("Starting Flask app...")
    flask_process = start_flask()
