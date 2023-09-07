from django.urls import path
from .views import ChatSessionListView, ChatSessionDetailView, ConverseView

urlpatterns = [
    path('chat-sessions/', ChatSessionListView.as_view(), name='chat-session-list'),
    path('chat-sessions/<int:pk>/', ChatSessionDetailView.as_view(), name='chat-session-detail'),
    path('converse/<str:session_id>/', ConverseView.as_view(), name="converse")
]
