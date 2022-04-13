from django import forms

class FileForm(forms.Form):
    videofile = forms.FileField()