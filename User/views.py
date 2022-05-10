from http.client import HTTPResponse
#from msilib.schema import File
from urllib import response
from django.contrib.auth import authenticate, logout
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
MEDIA_URL = '/media/'
from . import models
from .forms import FileForm
from .models import User, File
from .serializers import UserSerializer
from .run import run_test
# Create your views here.

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


#  동영상 받아와서 AI로 넘기는 view
def result(request):
    if request.method == 'POST':
        userVideo = request.FILES.get('video')
        print(userVideo)
        newV = File()
        newV.file = userVideo
        newV.save()  # 저장 됨
        result = run_test('ata97@naver.com','User/samples/test.mp4', 'User/test', 5, 30,
                          'User/checkpoints/best_model.pth', 'User/output', 30, '0')
        # run_test('User/samples/test.mp4', 'User/test', 5, 30, 'User/checkpoints', '/User/output', 30, 4)
        # newV.delete()
        print("POST START")
    else:
        form = FileForm()
    
    return HttpResponseRedirect('/user/success')


class AICommunication(APIView):
    def get(self, request):
        # 사진 - 값 순으로 정렬 됨.
        result = run_test('ata97@naver.com','User/samples/test.mp4','User/test',5,30,'User/checkpoints/best_model.pth','User/output',30,'0')

        return Response(result)
