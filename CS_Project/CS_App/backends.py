from django.contrib.auth.backends import ModelBackend
from django.contrib.auth.models import User

class EmailAuthBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        """
        擴展 Django 的 ModelBackend 以允許使用電子郵件登入。
        """
        try:
            # 嘗試使用 username 參數作為 email 來尋找使用者
            user = User.objects.get(email=username)
        except User.DoesNotExist:
            # 如果找不到，則回傳 None
            return None
        
        # 使用者的 check_password 函式會自動比對密碼
        if user.check_password(password):
            return user
        return None

    def get_user(self, user_id):
        """
        Django session 用來取回使用者物件。
        """
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None