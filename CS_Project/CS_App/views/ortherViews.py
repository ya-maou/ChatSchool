# ortherViews.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
from ..models import Feedback, Event

@csrf_exempt
@require_http_methods(["POST"])
def submit_feedback(request):
    try:
        data = json.loads(request.body)
        content = data.get('content', '').strip()
        if not content:
            return JsonResponse({'success': False, 'message': '內容不得為空'}, status=400)
        Feedback.objects.create(
            user=request.user if request.user.is_authenticated else None,
            content=content
        )
        return JsonResponse({'success': True, 'message': '已收到您的意見，感謝回饋！'})
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)}, status=500)

def Reward_view(request):
    return render(request, 'reward.html')

def Guide_view(request):
    return render(request, 'guide.html')