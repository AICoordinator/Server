
from django.contrib import admin
from django.urls import path, include
from User import views
from rest_framework import routers

app_name = 'User'

urlpatterns = [
    path('', include('rest_framework.urls', namespace='rest_framework_category')),
]