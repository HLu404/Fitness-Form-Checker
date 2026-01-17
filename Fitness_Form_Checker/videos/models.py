from django.db import models

class Video(models.Model):
    original = models.FileField(upload_to='videos/original/')
    uploaded_at = models.DateTimeField(auto_now_add=True)