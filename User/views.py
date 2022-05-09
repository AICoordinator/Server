from http.client import HTTPResponse
from telnetlib import STATUS
#from msilib.schema import File
from urllib import response
from django.contrib.auth import authenticate, logout
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from django.conf import settings

from . import models
from .forms import FileForm
from .models import User, File, Result
from .serializers import Base64StringField, ImageSerializer, UserSerializer, ImageSerializer
from pathlib import Path
#from .test import test,get_opt 
import json
import base64
from pprint import pprint
from django.http import JsonResponse

from pathlib import Path
import os

from User import serializers

class SignupAPI(APIView):
    def post(self, request):
        user = User.objects.create_user(email=request.data['email'],nickname=request.data['nickname'],gender=request.data['gender'],password=request.data['password'])
        user.save()
        #print(request.data['email'] + request.data['nickname'] + request.data['gender'])
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

#  동영상 받아와서 AI로 넘기는 view
class ResultAPI(APIView):
    def post(self, request):
        if request.method == 'POST':
            userVideo = request.FILES.get('video')
            print(userVideo)
            newV = File()
            newV.file = userVideo
            newV.save()  # 저장 됨
            # for i in range(1, 11):
            #     with open(settings.MEDIA_ROOT + "/images/" + str(i) + ".jpeg", "rb") as image_file:
            #         images.append(base64.b64encode(image_file.read()).decode('utf-8'))
            # for i in range(1, 11):
            #     images.append(str(i))
            # jsondata = json.dumps({'list' : images})
            images = []
            for i in range(1, 11):
                with open(settings.MEDIA_ROOT + "/images/" + str(i) +  ".jpeg", "rb") as image_file:
                        images.append(base64.b64encode(image_file.read()).decode('utf-8'))
            result = Result.objects.all()
            result.image1 = images[0]
            result.image2 = images[1]
            result.image3 = images[2]
            result.image4 = images[3]
            result.image5 = images[4]
            result.image6 = images[5]
            result.image7 = images[6]
            result.image8 = images[7]
            result.image9 = images[8]
            result.image10 = images[9]
            serializer = Base64StringField(result)

            print("hello")
            print(serializer.data)
            print("hello")
            return Response(serializer.data)
        else:
            form = FileForm()
            return Response(status = 500)
    


