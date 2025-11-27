# CS_App/views/authentication.py
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User
from django.contrib.auth import login, logout, authenticate, update_session_auth_hash
from django.contrib import messages
from django.conf import settings
from django.contrib.auth import get_backends

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.mail import send_mail
from django.utils import timezone

from ..models import UserProfile, EmailVerificationCode
from ..models import EmailVerificationCode

import json
import random
def login_or_register(request):
    """
    處理使用者登入、註冊及發送驗證碼的請求。
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'message': '無效的JSON格式', 'success': False}, status=400)

        email = data.get('email')
        username = data.get('username')  # 修正: 前端傳的是 username
        password = data.get('password')
        verification_code = data.get('verification_code')

        # ======================================================
        # 1. 發送驗證碼（只有 email）
        # ======================================================
        if email and not password and not verification_code:
            try:
                ver_code = EmailVerificationCode.objects.get(email=email)
                if (timezone.now() - ver_code.created_at).total_seconds() < 30:
                    return JsonResponse({'message': '請稍後再試。', 'success': False})
            except EmailVerificationCode.DoesNotExist:
                pass

            # 生成驗證碼
            code = str(random.randint(1000, 9999))
            print(f"Generated verification code for {email}: {code}")

            EmailVerificationCode.objects.update_or_create(
                email=email,
                defaults={'code': code, 'created_at': timezone.now()}
            )

            try:
                send_mail(
                    '您的驗證碼',
                    f'您的驗證碼是: {code}',
                    settings.EMAIL_HOST_USER,
                    [email],
                    fail_silently=False,
                )
                return JsonResponse({'message': '驗證碼已寄送至您的電子郵件信箱。', 'success': True})
            except Exception as e:
                return JsonResponse({'message': f'寄送驗證碼失敗：{str(e)}', 'success': False})

        # ======================================================
        # 2. 登入 / 註冊（email + password + verification_code）
        # ======================================================
        if not (email and password and verification_code):
            return JsonResponse({'message': '資料不足，請確認欄位內容。', 'success': False})

        if not username:
            return JsonResponse({'message': '請提供使用者名稱。', 'success': False})

        # 驗證驗證碼
        try:
            ver_code = EmailVerificationCode.objects.get(email=email)
            if not ver_code.is_valid() or ver_code.code != verification_code:
                return JsonResponse({'message': '驗證碼無效或已過期', 'success': False})
        except EmailVerificationCode.DoesNotExist:
            return JsonResponse({'message': '請先取得有效的驗證碼', 'success': False})

        # ======================================================
        # 2-1. 登入
        # ======================================================
        user = authenticate(request, username=username, password=password)

        if user is not None:
            ver_code.delete()
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')
            return JsonResponse({'message': '登入成功', 'success': True})

        # ======================================================
        # 2-2. 註冊
        # ======================================================
        if User.objects.filter(email=email).exists():
            return JsonResponse({'message': '電子郵件或密碼錯誤', 'success': False})

        try:
            # 正確: Django 必須有 username
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password
            )

            profile, created = UserProfile.objects.get_or_create(user=user)
            profile.nickname = username
            profile.save()

            ver_code.delete()
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')

            return JsonResponse({'message': '註冊成功', 'success': True})

        except Exception as e:
            return JsonResponse({'message': f'註冊失敗: {str(e)}', 'success': False})

    return JsonResponse({'message': '無效的請求', 'success': False}, status=405)


@login_required
def settings_view(request):
    user = request.user
    profile, _ = UserProfile.objects.get_or_create(user=user)
    context = {
        'user_email': user.username,
        'user': user,
        'nickname': profile.nickname,
    }
    return render(request, 'settings.html', context)



@login_required
def update_settings(request):
    if request.method == 'POST':
        user = request.user
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            data = request.POST

        scope = data.get('scope')
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        username = data.get('username')

        if not current_password or not user.check_password(current_password):
            return JsonResponse({'success': False, 'message': '目前密碼不正確。'})

        if scope == 'username' and username:
            if User.objects.filter(username=username).exclude(pk=user.pk).exists():
                return JsonResponse({'success': False, 'message': '此使用者名稱已被使用。'})
            user.username = username

        if scope == 'password' and new_password:
            user.set_password(new_password)
            update_session_auth_hash(request, user)

        user.save()
        return JsonResponse({'success': True, 'message': '設定已更新。'})

    return JsonResponse({'success': False, 'message': '無效請求'}, status=405)

def forgot_password(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'message': '無效的JSON格式', 'success': False}, status=400)
        
        email = data.get('email')
        verification_code = data.get('verification_code')
        new_password = data.get('new_password')

        # 處理「發送驗證碼」請求
        if not verification_code and not new_password:
            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                return JsonResponse({'message': '查無此電子郵件帳號。', 'success': False}, status=404)

            # 生成並發送驗證碼
            code = str(random.randint(1000, 9999))
            EmailVerificationCode.objects.update_or_create(
                email=email,
                defaults={'code': code, 'created_at': timezone.now()}
            )
            
            try:
                send_mail(
                    '您的密碼重設驗證碼',
                    f'您的密碼重設驗證碼是: {code}',
                    settings.EMAIL_HOST_USER,
                    [email],
                    fail_silently=False,
                )
                return JsonResponse({'message': '驗證碼已寄送至您的電子郵件信箱。', 'success': True})
            except Exception as e:
                return JsonResponse({'message': f'寄送驗證碼失敗：{str(e)}', 'success': False})

        # 處理「重設密碼」請求
        else:
            # 1. 驗證驗證碼
            try:
                ver_code = EmailVerificationCode.objects.get(email=email)
                if not ver_code.is_valid() or ver_code.code != verification_code:
                    return JsonResponse({'message': '驗證碼無效或已過期', 'success': False})
            except EmailVerificationCode.DoesNotExist:
                return JsonResponse({'message': '請先取得有效的驗證碼。', 'success': False})

            # 2. 驗證碼正確，重設密碼
            try:
                user = User.objects.get(email=email)
                user.set_password(new_password)
                user.save()
                return JsonResponse({'message': '密碼已成功重設。', 'success': True})
            except User.DoesNotExist:
                return JsonResponse({'message': '帳號不存在。', 'success': False})

    return JsonResponse({'message': '無效的請求', 'success': False}, status=405)

@login_required
def delete_account(request):
    if request.method == 'POST':
        user = request.user
        logout(request)
        user.delete()
        return redirect('chat')
    return redirect('settings')