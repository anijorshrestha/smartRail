from django.urls import path
from smartWindow import views

urlpatterns = [
    path('', views.smartwindow, name='main'),
    path('video_feed_1/', views.video_feed_1, name="video-feed-1"),
    path('geolocation/', views.geolocation, name="geolocation"),

]
