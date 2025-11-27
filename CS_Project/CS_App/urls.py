# C:\Users\User\Studio\Django\ChatSchool\CS_Project\CS_App\urls.py

from django.urls import path
from django.contrib.auth import views as auth_views

# 從 views 套件中匯入各個模組
from .views import authentication
from .views import chat
from .views import calendar
from .views import lineBot
from .views import ortherViews

urlpatterns = [
    # === Chat 頁面與互動 ===
    path('', chat.chat_view, name='chat'),
    path('chat/', chat.chat_view, name='chat'),
    path('send_message/', chat.send_message, name='send_message'),
    path('chat_api/', chat.chat_api, name='chat_api'),
    path('chat_history_api/', chat.chat_history_api, name='chat_history_api'),
    path('chat_history/<int:chat_id>/', chat.chat_view, name='chat_history_detail'),
    path('chat_history_api/<int:chat_id>/', chat.delete_chat_history, name='delete_chat_history'),

    # === LINE Bot Webhook ===
    path('line/webhook/', lineBot.line_webhook, name='line_webhook'),

    # === 使用者驗證與設定 ===
    path('login_api/', authentication.login_or_register, name='login_or_register'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    path('settings/', authentication.settings_view, name='settings'),
    path('settings/update/', authentication.update_settings, name='update_settings'),
    path('settings/delete/', authentication.delete_account, name='delete_account'),

    # === 日曆與事件管理 ===
    path('calendar/', calendar.Calendar_view, name='calendar'),
    path('events/', calendar.event_list, name='event-list'),
    path('pin_event/', calendar.pin_event, name='pin_event'),
    path('get_pinned_events/', calendar.get_pinned_events, name='get_pinned_events'),
    
    # === 其他功能 ===
    path('reward/', ortherViews.Reward_view, name='reward'),
    path('guide/', ortherViews.Guide_view, name='guide'),
    path('submit-feedback/', ortherViews.submit_feedback, name='submit_feedback'),

    path('create_event/', calendar.create_event, name='create_event'),
    path('my_events/', calendar.my_event_list, name='my_event_list'),
    path('get_user_events/', calendar.get_user_events, name='get_user_events'),

    path('update_event/<int:event_id>/', calendar.update_event, name='update_event'),
    path('forgot-password/', authentication.forgot_password, name='forgot_password'),
    path('delete_event/<int:event_id>/', calendar.delete_event, name='delete_event'),

    path('get_links', chat.get_links, name='get_links'),
    path('set_reminder/', calendar.set_reminder, name='set_reminder'),
]