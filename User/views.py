from django.contrib.auth import authenticate, logout
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from .models import User
from .serializers import UserSerializer
# Create your views here.


class SignupAPI(APIView):
    def post(self, request):
        user = User.objects.create_user(username=request.data['email'], password=request.data['password'])
        user.save()
        token = Token.objects.create(user=user)
        return Response({"Token": token.key})
    # 추가 구현 예정
    def get(self, request):
        queryset = User.objects.all()
        serializer = UserSerializer(queryset, many=True)
        return Response(serializer.data)


class LoginAPI(APIView):
    def post(self,request):
        user = authenticate(username=request.data['email'], password=request.data['password'])
        if user is not None:
            token = Token.objects.get(user=user)
            # 로그인 하면 토큰 부여 예정 -> Android 에서가지고 있다가 사용 가능
            return Response({"Token": token.key})
        else:
            return Response(status=401)


class LogoutAPI(APIView):
    def get(self,request):
        # 토큰 삭제
        request.user.auth_token.delete()
        logout(request)
        return Response('User Logged out successfully')
