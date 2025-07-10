from django.urls import path
from .views import predict,home,result,register,login_view

urlpatterns = [
    path('predict/', predict, name='predict'),
    path('',home,name="home"),
    path('result/', result, name='result'),
    path('register/',register,name='register'),
    path('login/',login_view,name='login')
]
