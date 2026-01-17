from django.urls import path
from . import views

app_name = "videos"

urlpatterns = [
    path("", views.homepage, name="home"),
    path("upload/", views.upload, name="upload"),
    path("stream/<int:video_id>/", views.stream, name="stream"),
    path("video-feed/<int:video_id>/", views.video_feed, name="video_feed"),
]
