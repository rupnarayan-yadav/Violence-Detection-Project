# detection/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # This is the URL for your main page
    path('results/', views.show_results, name='show_results'),
    
    # This is the URL for the video stream
    path('video_stream/', views.video_stream, name='video_stream'),
]