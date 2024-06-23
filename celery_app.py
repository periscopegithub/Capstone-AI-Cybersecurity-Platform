from celery import Celery


def make_celery(app_name, broker, backend):
    celery = Celery(app_name, backend=backend, broker=broker)
    return celery


celery = make_celery(
    "survey_app", "redis://localhost:6379/0", "redis://localhost:6379/0"
)

# Update the Celery configuration
celery.conf.update(
    worker_concurrency=4,
)

# Import the tasks module to ensure tasks are registered
import tasks
