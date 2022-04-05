from django.contrib.auth import authenticate, logout
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from .models import User
from .serializers import UserSerializer, loginSerializer


# Create your views here.


class SignupAPI(APIView):
    def post(self, request):
        user = User.objects.create_user(email=request.data['email'],gender=request.data['gender'],nickname=request.data['gender'],password=request.data['password'])
        user.save()
        token = Token.objects.create(user=user)
        serializer = UserSerializer(user)
        return Response({"User": serializer.data, "Token": token.key})
    # 추가 구현 예정
    def get(self, request):
        queryset = User.objects.all()
        serializer = UserSerializer(queryset, many=True)
        return Response(serializer.data)


class LoginAPI(APIView):
    def post(self,request):
        print(request.data)
        user = authenticate(email=request.data['email'], password=request.data['password'])
        if user is not None:
            token = Token.objects.get(user=user)
            serializer = loginSerializer(user)
            serializer.getToken(token)
            # 로그인 하면 토큰 부여 예정 -> Android 에서가지고 있다가 사용 가능
            return Response(serializer.data)
        else:
            return Response(status=401)


class LogoutAPI(APIView):
    def get(self,request):
        # 토큰 삭제
        request.user.auth_token.delete()
        logout(request)
        return Response('User Logged out successfully')