import os

from django.db import models
from django.contrib.auth.models import AbstractBaseUser,BaseUserManager
# Create your models here.
from rest_framework.authtoken.models import Token
import base64
from django.conf import settings

GENDER_CHOICES = (
    (0, 'Female'),
    (1, 'Male'),
    (2, 'Not to disclose')
)
# User Manager -> 유저 DB 에 등록 해줌 (장고 제공)
# 필요한 부분만 재정의 중

class MyUserManager(BaseUserManager):
    def create_user(self, email, nickname, gender,password):
        if not email:
            raise ValueError('The given email mist be set')
        user = self.model(
            email=self.normalize_email(email),
            gender=gender,
            nickname=nickname,
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, nickname, gender=1,password=None):
        # Super User 만들기
        user = self.create_user(email, password, gender, nickname)
        user.is_admin = True
        user.save(using=self._db)
        return user

class User(AbstractBaseUser):# Abstract User 상속받음
    email = models.EmailField(verbose_name="email", max_length=255, unique=True)
    gender = models.SmallIntegerField(choices=GENDER_CHOICES, default=2)
    nickname = models.CharField(max_length=20)
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)
    objects = MyUserManager()
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['nickname']
    # Primary key 필요하다면 사용 -> 바꿔도됨
    """ def __str__(self):
        return "<%d %s>" % (self.pk, self.email)"""
    #기본
    def __str__(self):
        return self.email

    # 신경 ㄴㄴ
    def get_full_name(self):
        # The user is identified by their email address
        return self.email

    def get_short_name(self):
        # The user is identified by their email address
        return self.email

    def has_perm(self, perm, obj=None):
        "Does the user have a specific permission?"
        # Simplest possible answer: Yes, always
        return True

    def has_module_perms(self, app_label):
        "Does the user have permissions to view the app `app_label`?"
        # Simplest possible answer: Yes, always
        return True

    @property
    def is_staff(self):
        "Is the user a member of staff?"
        # Simplest possible answer: All admins are staff
        return self.is_admin


def upload_to_local(instance, filename):
    extension = os.path.splitext(filename)[-1].lower() # 확장자 추출
    name = instance.owner.split('.')[0]
    return ("/".join(
        ["video",name]
    ))+extension


class File(models.Model):
    owner = models.EmailField(verbose_name="email", max_length=255,null=True)
    file = models.FileField(upload_to=upload_to_local,null=True)


def upload_to_originImage(instance,filename):
    extension = os.path.splitext(filename)[-1].lower()  # 확장자 추출
    return ("/".join(
        ["userimage",'origin'+ instance.owner.email]
    )) + extension


def upload_to_changedImage(instance,filename):
    extension = os.path.splitext(filename)[-1].lower()  # 확장자 추출
    return ("/".join(
        ["userimage", 'changed'+instance.owner.email]
    )) + extension


class UserImage(models.Model):
    owner = models.ForeignKey(User,related_name='images', on_delete=models.CASCADE, null=True)
    title = models.CharField(max_length=250)
    score = models.CharField(max_length=20)
    originImage = models.TextField(null=True)
    changedImage = models.TextField(null=True)