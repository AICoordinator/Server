import base64
import time
from http.client import HTTPResponse
#from msilib.schema import File
from math import floor
from urllib import response
from django.contrib.auth import authenticate, logout
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from tensorflow.python.client import device_lib
from . import models
from .forms import FileForm
from .models import User, File,UserImage
from .serializers import UserSerializer,UserImageSerializer,ImageSerializer
from .run import run_test
import os
import torch.nn.functional as F
from PIL import Image
# Create your views here.
from django.conf import settings

class SignupAPI(APIView):
    def post(self, request):
        user = User.objects.create_user(email=request.data['email'],gender=request.data['gender'],nickname=request.data['gender'],password=request.data['password'])
        user.save()
        serializer = UserSerializer(user)
        return Response(serializer.data)

    # 추가 구현 예정
    def get(self, request):
        queryset = User.objects.all()
        serializer = UserSerializer(queryset, many=True)
        return Response(serializer.data)


class LoginAPI(APIView):
    def post(self,request):
        user = authenticate(email=request.data['email'], password=request.data['password'])
        if user is not None:
            serializer = UserSerializer(user)
            print(serializer.data)
            return Response(serializer.data)
    def get(self, request):
            print(1)
            return Response(status=200)


class LogoutAPI(APIView):
    def get(self,request):
        # 토큰 삭제
        request.user.auth_token.delete()
        logout(request)
        return Response('User Logged out successfully')


class result(APIView):
    def post(self,request):
        userVideo = request.FILES.get('video')
        newV = File(file=userVideo, owner='ata97@naver.com')
        newV.save()

        start = time.time()
        unique_key = str(floor(time.time() * 100))
        run_test('ata97@naver.com', unique_key, 5, 30, 30)
        end = time.time()
        print(f"total time : {end - start: .5f} sec")
        data = UserImage.objects.filter(owner__email='ata97@naver.com', title__icontains=unique_key).order_by('-score')
        serializer_data = ImageSerializer(data, many=True)
        return Response(serializer_data.data)

class AICommunication(APIView):
    def get(self, request):
        # 사진 - 값 순으로 정렬 됨.
        """print(time.time())
        a = str(floor(time.time() * 100))
        user_email = 'ata97@naver.com'+'/'+a
        print(user_email)"""
        unique_key = str(floor(time.time() * 100))
        run_test('ata97@naver.com', unique_key, 5, 30, 30)
        data = UserImage.objects.filter(owner__email='ata97@naver.com',title__icontains=unique_key)
        serializer_data = ImageSerializer(data,many=True)
        return Response(serializer_data.data)

