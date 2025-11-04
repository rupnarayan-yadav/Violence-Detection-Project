# detection/models.py
from django.db import models

class DetectionResult(models.Model):
    video_name = models.CharField(max_length=255)
    video_file = models.FileField(upload_to='violence_snippets/', null=True, blank=True)
    is_violent = models.BooleanField(default=False)
    confidence_score = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.video_name} - Violent: {self.is_violent}"