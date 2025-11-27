from django.contrib import admin
from .models import (
    Message, ChatHistory, Feedback, Event, UserProfile
)

# ========== ChatHistory ==========
@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'created_at')  
    list_filter = ('user',)
    search_fields = ('title',)

# ========== Message ==========
@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('chat_history', 'username', 'timestamp', 'model_name', 'related_links')
    search_fields = ('chat_history__title', 'username', 'content')
    list_filter = ('timestamp',)

# ========== Feedback ==========
@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('user', 'created_at', 'content')
    search_fields = ('content', 'user__username')
    list_filter = ('created_at',)

# ========== Event ==========
@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ('title', 'date')
    search_fields = ('title',)
    list_filter = ('date',)

# ========== UserProfile ==========
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'nickname')
    search_fields = ('user__username', 'nickname')
