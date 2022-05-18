
from .models import User,UserImage
from rest_framework import serializers
from rest_framework.authtoken.models import Token
from django.conf import settings

class UserSerializer(serializers.ModelSerializer):
    token = serializers.SerializerMethodField()
    class Meta:
        model = User
        fields = ('email','gender', 'nickname','token')

    def get_token(self, obj):
        token, created = Token.objects.get_or_create(user=obj)
        return token.key


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserImage
        fields = ('title', 'score', 'originImage', 'changedImage')


class UserImageSerializer(serializers.ModelSerializer):
    images = ImageSerializer(many=True,read_only=True)

    class Meta:
        model = User
        fields = ('email', 'images')

    def to_representation(self, instance):
        response = super().to_representation(instance)
        response["images"] = sorted(response["images"], key=lambda x: x["score"],reverse=True)
        return response
