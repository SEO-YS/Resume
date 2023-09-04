from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # 추가 필드를 정의할 수 있습니다 (예: 전화번호, 별명 등)

    def __str__(self):
        return self.user.username