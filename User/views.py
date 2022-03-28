from django.shortcuts import render
from rest_framework import viewsets
from User.models import User
from User.serializers import UserSerializer
# Create your views here.

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer