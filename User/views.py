from http.client import HTTPResponse
#from msilib.schema import File
from urllib import response
from django.contrib.auth import authenticate, logout
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authtoken.models import Token

from . import models
from .forms import FileForm
from .models import User, File
from .serializers import UserSerializer
<<<<<<< HEAD
from .run import run_test
=======
from pathlib import Path
from .test import test,get_opt
import json

from pathlib import Path
import os

import torch
from .dataset import ImageDatasetTest
from .networks import RegressionNetwork
import argparse
import os
import torch.nn.functional as F
from PIL import Image
>>>>>>> parent of 22f656c (complete receiving result images from server)
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

<<<<<<< HEAD

def result(request):
    if request.method == 'POST':
        userVideo = request.FILES.get('video')
        print(userVideo)
        newV = File()
        newV.file = userVideo
        newV.save()  # 저장 됨
        result = run_test('ata97@naver.com', 'User/samples/test.mp4', 'User/test', 5, 30,
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


#  동영상 받아와서 AI로 넘기는 view Test Code
"""
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
    """
=======

#  동영상 받아와서 AI로 넘기는 view
def result(request):
    if request.method == 'POST':
        userVideo = request.FILES.get('video')
        print(userVideo)
        newV = File()
        newV.file = userVideo
        newV.save()  # 저장 됨
        print("POST START")
    else:
        form = FileForm()
    
    return HttpResponseRedirect('/user/success')

>>>>>>> parent of 22f656c (complete receiving result images from server)

class AICommunication(APIView):
    def get(self,request):
        import sys
        a = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))))+'\\AI_Model\\attractiveness'
        sys.path.append(a)
        import linkingTest
        linkingTest.linkingTest()
        return Response(a)

    # def get(self, request):
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #     # Define model
    #     model = RegressionNetwork()
    #     # Load checkpoint
    #     model.load_state_dict(torch.load("User/checkpoints/8000.pth"))
    #     # Define dataloader
    #     test_dataset = ImageDatasetTest(data_dir="User/samples")
    #     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)
    #     # Train the model
    #     data = test(model, test_loader, "User/output")
    #     print(data)
    #     return Response(data)
