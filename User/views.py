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
        print("POST START")
    else:
        form = FileForm()
    
    return HttpResponseRedirect('/user/success')


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
