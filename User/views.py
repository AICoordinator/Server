from django.contrib.auth import authenticate, logout
from django.shortcuts import render, get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authtoken.models import Token

from . import models
from .models import User
from .serializers import UserSerializer


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
            return Response({"sex":"sex"})


class LogoutAPI(APIView):
    def get(self,request):
        # 토큰 삭제
        request.user.auth_token.delete()
        logout(request)
        return Response('User Logged out successfully')