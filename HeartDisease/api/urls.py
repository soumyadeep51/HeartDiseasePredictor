from django.urls import path
from .views import predict,home,result

urlpatterns = [
    path('predict/', predict, name='predict'),
    path('',home,name="home"),
    path('result/', result, name='result'),
]
