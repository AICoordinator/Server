
from django.contrib import admin
from django.urls import path, include
from User import views
from rest_framework import routers
from django.conf import settings
from django.conf.urls.static import static

app_name = 'User'


urlpatterns = [
    path("signup", views.SignupAPI.as_view()),
    path("logout", views.LogoutAPI.as_view()),
    path("login", views.LoginAPI.as_view()),
    path("result", views.ResultAPI.as_view()),
    #path("ai", views.AICommunication.as_view())
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)