
from User.models import User
from rest_framework import serializers
from rest_framework.authtoken.models import Token

class UserSerializer(serializers.ModelSerializer):
    token = serializers.SerializerMethodField()
    class Meta:
        model = User
        fields = ('email','gender', 'nickname','token')

    def get_token(self, obj):
        token, created = Token.objects.get_or_create(user=obj)
        return token.key






