from rest_framework import serializers
from .models import ChatSession

class ChatSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatSession
        fields = ('id', 'title', 'description', 'pdf_file', 'creator')

class ConversationSerializer(serializers.Serializer):
    session_id = serializers.UUIDField(required=False)
    userMessage = serializers.CharField()
    AIMessage = serializers.CharField(required=False)