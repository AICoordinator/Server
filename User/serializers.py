
from User.models import User
from rest_framework import serializers


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('email', 'gender', 'nickname')


class loginSerializer(serializers.ModelSerializer):
    token = serializers.SerializerMethodField(method_name='getToken')

    class Meta:
        model = User
        fields = ('email', 'gender', 'nickname')

    def getToken(self,obj):
        return obj