
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
<<<<<<< HEAD
"""
class Base64StringField(serializers.ModelSerializer):
    
    class Meta:
        model = Result
        fields = ('image1','image2','image3','image4','image5',
        'image6','image7','image8','image9','image10')

class ImageSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Result
        fields = '__all__''"""
=======


>>>>>>> parent of 22f656c (complete receiving result images from server)
