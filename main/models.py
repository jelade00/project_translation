from django.db import models

class TaskStatus(models.Model):
    task_id = models.CharField(max_length=36, unique=True)
    status = models.CharField(max_length=20)  # processing, completed, failed
    result = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.task_id} - {self.status}"