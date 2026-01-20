from django import forms
from .models import Video

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ['original']
        widgets = {
            'original': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'video/*',
            })
        }
        labels = {
            'original': 'Upload Video'
        }