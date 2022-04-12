
from django.contrib import admin
from django.urls import path, include
from User import views
from rest_framework import routers

app_name = 'User'


urlpatterns = [
    path("signup", views.SignupAPI.as_view()),
    path("logout", views.LogoutAPI.as_view()),
    path("login", views.LoginAPI.as_view()),
    path("result", views.result)
]