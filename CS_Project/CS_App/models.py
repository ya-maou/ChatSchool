# CS_Project/CS_App/models.py
from django.db import models
from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone

class ChatHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_histories', null=True)
    title = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    summary = models.TextField(blank=True, default="")

    def __str__(self):
        return self.title


class Message(models.Model):
    chat_history = models.ForeignKey(ChatHistory, on_delete=models.CASCADE, related_name='messages')
    username = models.CharField(max_length=50)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    model_name = models.CharField(max_length=100, blank=True, null=True)
    related_links = models.JSONField(default=list, blank=True)
    image_links = models.JSONField(default=list, blank=True)
    duration = models.FloatField(null=True, blank=True)  

    def __str__(self):
        return f'{self.username}: {self.content[:30]}...'

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    nickname = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return self.user.username

class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user} - {self.created_at:%Y-%m-%d %H:%M}"

class Event(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='events', null=True, blank=True)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    date = models.DateField()

    def __str__(self):
        return f"{self.user.username} - {self.title} ({self.date})"

class PinnedSimpleArticle(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    article_id = models.CharField(max_length=255, default='0')
    title = models.CharField(max_length=255)
    date = models.CharField(max_length=255, null=True, blank=True)
    url = models.URLField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return f'{self.user.username} - {self.title}'

class EmailVerificationCode(models.Model):
    email = models.EmailField(unique=True)
    code = models.CharField(max_length=4)
    created_at = models.DateTimeField(auto_now_add=True)

    def is_valid(self):
        return (timezone.now() - self.created_at).total_seconds() < 300
    
from pgvector.django import VectorField 

class ChatLog(models.Model):
    user_id = models.CharField(max_length=255)
    question = models.TextField()
    answer = models.TextField(blank=True, null=True)
    embedding = VectorField(dimensions=768, blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.user_id}] {self.question[:30]}..."

class LineUser(models.Model):
    line_user_id = models.CharField(max_length=64, unique=True)
    preferred_model = models.CharField(max_length=50, default='rag_main')  # 預設模型
    language = models.CharField(max_length=10, default='zh-TW')
    notifications_enabled = models.BooleanField(default=True)
    in_feedback_mode = models.BooleanField(default=False)


class Line_Feedback(models.Model):
    line_user = models.ForeignKey(LineUser, on_delete=models.CASCADE, null=True, blank=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return f"{self.line_user_id} - {self.created_at}"

