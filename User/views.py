import time
from math import floor
from django.contrib.auth import authenticate, logout
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import User, File,UserImage
from .serializers import UserSerializer,UserImageSerializer,ImageSerializer
from .run import run_test
import os
import torch.nn.functional as F
from PIL import Image
import time
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
        print(request.auth)
        user = request.user

        userVideo = request.FILES.get('video')
        newV = File(file=userVideo, owner= user.email)
        newV.save()

        start = time.time()
        unique_key = str(floor(time.time() * 100))
        run_test(user.email, unique_key, 5, 30, 30)
        end = time.time()

        print(f"total time : {end - start: .5f} sec")
        data = UserImage.objects.filter(owner__email=user.email, title__icontains=unique_key).order_by('-score')
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

#  동영상 받아와서 AI로 넘기는 view Test Code
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
            time.sleep(20)
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

class ProfileAPI(APIView):
    def post(self, request):