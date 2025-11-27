# calendar.py
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_http_methods
from django.contrib.auth.decorators import login_required
from django.db import connections
import json
import re
from datetime import datetime, date
from django.shortcuts import render, get_object_or_404
from django.utils import timezone
from django.core.mail import send_mail
from django.conf import settings
from background_task import background

from ..models import PinnedSimpleArticle, Event  # 確保導入正確

@login_required
def Calendar_view(request):
    return render(request, 'calendar.html')


def normalize_date(raw_date):
    if not raw_date:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw_date, fmt).date().isoformat()
        except ValueError:
            continue
    match_roc = re.match(r'^(\d{2,3})年(\d{1,2})月(\d{1,2})日$', raw_date)
    if match_roc:
        try:
            year = int(match_roc.group(1)) + 1911
            month = int(match_roc.group(2))
            day = int(match_roc.group(3))
            return date(year, month, day).isoformat()
        except ValueError:
            pass
    match_west = re.match(r'^(\d{4})年(\d{1,2})月(\d{1,2})日$', raw_date)
    if match_west:
        try:
            year = int(match_west.group(1))
            month = int(match_west.group(2))
            day = int(match_west.group(3))
            return date(year, month, day).isoformat()
        except ValueError:
            pass
    return None


def event_list(request):
    year = request.GET.get("year")
    month = request.GET.get("month")
    if not (year and month):
        return JsonResponse({"status": "error", "message": "Missing year or month"}, status=400)
    try:
        year = int(year)
        month = int(month)
        start_date = date(year, month, 1)
        end_date = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    except ValueError:
        return JsonResponse({"status": "error", "message": "Invalid year or month"}, status=400)

    results = []
    pinned_article_ids = set()
    if request.user.is_authenticated:
        pinned_article_ids = set(
            PinnedSimpleArticle.objects.filter(user=request.user).values_list('article_id', flat=True)
        )

    try:
        with connections['postgreSQL'].cursor() as cursor:
            cursor.execute("SELECT id, url, title, date, summary FROM ARTICLES")
            rows = cursor.fetchall()
        for article_id, url, title, raw_date, summary in rows:
            if not raw_date:
                continue
            date_list = [d.strip() for d in raw_date.split(",") if d.strip()]
            for d in date_list:
                norm_date = normalize_date(d)
                if not norm_date:
                    continue
                try:
                    dt = datetime.strptime(norm_date, "%Y-%m-%d").date()
                    if not (start_date <= dt < end_date):
                        continue
                except ValueError:
                    continue
                is_pinned = str(article_id) in pinned_article_ids
                results.append({
                    "id": article_id,
                    "url": url,
                    "title": title or "無標題",
                    "date": norm_date,
                    "description": summary or "（無內容）",
                    "is_pinned": is_pinned,
                })
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse(results, safe=False)


@login_required
def my_event_list(request):
    """
    獲取使用者自訂事件列表
    """
    year = request.GET.get("year")
    month = request.GET.get("month")
    if not (year and month):
        return JsonResponse({"status": "error", "message": "Missing year or month"}, status=400)
    try:
        start_date = date(int(year), int(month), 1)
        end_date = date(int(year) + 1, 1, 1) if int(month) == 12 else date(int(year), int(month) + 1, 1)
    except ValueError:
        return JsonResponse({"status": "error", "message": "Invalid year or month"}, status=400)

    events = Event.objects.filter(
        user=request.user,
        date__gte=start_date,
        date__lt=end_date
    ).values('id', 'title', 'description', 'date')

    data = [{"id": e['id'], "title": e['title'], "description": e['description'],
             "date": e['date'].isoformat(), "is_user_event": True} for e in events]
    return JsonResponse(data, safe=False)


@login_required
@require_POST
def create_event(request):
    title = request.POST.get('title')
    description = request.POST.get('description', '')
    date_str = request.POST.get('date')

    if not all([title, date_str]):
        return JsonResponse({"success": False, "message": "標題和日期為必填項"})

    try:
        event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        Event.objects.create(
            user=request.user,
            title=title,
            description=description,
            date=event_date
        )
        return JsonResponse({"success": True, "message": "事件新增成功"})
    except ValueError:
        return JsonResponse({"success": False, "message": "日期格式錯誤"})
    except Exception as e:
        return JsonResponse({"success": False, "message": str(e)})


@login_required
@require_POST
def pin_event(request):
    user = request.user
    article_id = request.POST.get('id')
    title = request.POST.get('title')
    date_val = request.POST.get('date')
    url = request.POST.get('url', '')
    description = request.POST.get('description', '')
    is_user_event = request.POST.get('is_user_event') == 'true'

    if not all([article_id, title]):
        return JsonResponse({"success": False, "message": "缺少事件ID或標題"}, status=400)

    pin_id = f"user_event_{article_id}" if is_user_event else article_id

    if is_user_event:
        if not Event.objects.filter(id=article_id, user=user).exists():
            return JsonResponse({"success": False, "message": "找不到自訂事件或您沒有權限"}, status=403)

    try:
        pinned = PinnedSimpleArticle.objects.get(user=user, article_id=pin_id)
        pinned.delete()
        return JsonResponse({"success": True, "message": "已取消釘選", "action": "unpinned"})
    except PinnedSimpleArticle.DoesNotExist:
        PinnedSimpleArticle.objects.create(
            user=user,
            article_id=pin_id,
            title=title,
            date=date_val,
            url=url,
            description=description
        )
        return JsonResponse({"success": True, "message": "已釘選", "action": "pinned"})


@login_required
def get_pinned_events(request):
    pinned = PinnedSimpleArticle.objects.filter(user=request.user).order_by('-id')
    data = []
    for p in pinned:
        is_user_event = str(p.article_id).startswith('user_event_')
        data.append({
            "id": p.article_id,
            "title": p.title,
            "date": p.date,
            "url": p.url,
            "description": p.description,
            "is_user_event": is_user_event,
        })
    return JsonResponse(data, safe=False)


@login_required
def get_user_events(request):
    """
    獲取使用者自訂事件列表（側邊欄用）
    """
    events = Event.objects.filter(user=request.user).order_by('-date')
    data = [{
        "id": e.id,
        "title": e.title,
        "date": e.date.isoformat(),
        "description": e.description,
    } for e in events]
    return JsonResponse(data, safe=False)


@login_required
@require_http_methods(["POST", "PUT"])
def update_event(request, event_id):
    event = get_object_or_404(Event, pk=event_id, user=request.user)
    data = json.loads(request.body.decode('utf-8')) if request.method == 'PUT' else request.POST

    title = data.get('title')
    description = data.get('description', '')
    date_str = data.get('date')

    if not all([title, date_str]):
        return JsonResponse({"success": False, "message": "標題和日期為必填項"}, status=400)

    try:
        event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return JsonResponse({"success": False, "message": "日期格式錯誤"}, status=400)

    event.title = title
    event.description = description
    event.date = event_date
    event.save()

    pinned_id = f"user_event_{event_id}"
    try:
        pinned_article = PinnedSimpleArticle.objects.get(user=request.user, article_id=pinned_id)
        pinned_article.title = title
        pinned_article.date = event_date.isoformat()
        pinned_article.description = description
        pinned_article.save()
    except PinnedSimpleArticle.DoesNotExist:
        pass

    return JsonResponse({"success": True, "message": "事件修改成功"})


@login_required
@require_POST
def delete_event(request, event_id):
    try:
        event = Event.objects.get(pk=event_id, user=request.user)
        event.delete()
        return JsonResponse({"success": True, "message": "事件刪除成功"})
    except Event.DoesNotExist:
        return JsonResponse({"success": False, "message": "找不到事件或您沒有權限刪除此事件"}, status=404)
    except Exception as e:
        return JsonResponse({"success": False, "message": str(e)}, status=500)


@background()
def send_event_reminder_email(user_email, event_title, event_content, event_date, event_url=""):
    subject = f"【ChatSchool-提醒】: {event_title}"
    message = (
        f"「{event_title}」\n\n {event_content} \n\n 提醒時間：{event_date}。\n\n"
        f"{'詳細資訊：' + event_url if event_url else ''}"
    )
    from_email = settings.DEFAULT_FROM_EMAIL or settings.EMAIL_HOST_USER
    recipient_list = [user_email]
    send_mail(subject, message, from_email, recipient_list, fail_silently=False)


@login_required
@require_POST
def set_reminder(request):
    try:
        data = json.loads(request.body or "{}")
        event_id = str(data.get('event_id') or "").strip()
        is_user_event = str(data.get('is_user_event')).lower() == 'true'
        reminder_time_str = data.get('reminder_time')  # "YYYY-MM-DDTHH:MM"

        if not event_id or not reminder_time_str:
            return JsonResponse({"success": False, "message": "遺失必要參數"}, status=400)

        naive_dt = datetime.strptime(reminder_time_str, "%Y-%m-%dT%H:%M")
        local_tz = timezone.get_current_timezone()
        reminder_dt = timezone.make_aware(naive_dt, local_tz)

        if reminder_dt <= timezone.now():
            return JsonResponse({"success": False, "message": "無法設定過去時間的提醒"}, status=400)

        user_email = (request.user.email or "").strip()
        if not user_email:
            return JsonResponse({"success": False, "message": "您的帳號未設定 Email，無法寄送提醒"}, status=400)

        event_title = ""
        event_content = ""
        event_url = ""
        if is_user_event:
            original_event_id_str = event_id.split('_')[-1]
            original_event_id = int(original_event_id_str)
            event = get_object_or_404(Event, id=original_event_id, user=request.user)
            event_title = event.title or "(未命名事件)"
            event_content = event.description or "(無內容)"
        else:
            pinned = get_object_or_404(PinnedSimpleArticle, user=request.user, article_id=event_id)
            event_title = pinned.title or "(未命名事件)"
            event_content = pinned.description or "(無內容)"
            event_url = getattr(pinned, "url", "") or ""

        event_time_str = reminder_dt.astimezone(local_tz).strftime("%Y-%m-%d %H:%M")
        send_event_reminder_email(
            user_email=user_email,
            event_title=event_title,
            event_content=event_content,
            event_date=event_time_str,
            event_url=event_url,
            schedule=reminder_dt
        )

        return JsonResponse({"success": True, "message": "提醒設定成功"})
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "message": "無效的 JSON 格式"}, status=400)
    except Exception as e:
        print(f"[set_reminder] {type(e).__name__}: {e}")
        return JsonResponse({"success": False, "message": "發生未知錯誤"}, status=500)
