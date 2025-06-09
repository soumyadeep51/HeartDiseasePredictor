from django.urls import path
from .views import predict_heart_disease,home

urlpatterns = [
    path('predict/', predict_heart_disease, name='predict_heart_disease'),
    path('home/',home,name="home")
]
