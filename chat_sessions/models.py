from django.db import models
from user_management.models import CustomUser  # Import your custom user model
import uuid

class ChatSession(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    title = models.CharField(max_length=255)
    description = models.TextField()
    pdf_file = models.FileField(upload_to='chat_sessions/pdfs/')
    creator = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='created_chat_sessions')

    def __str__(self):
        return self.title
