from django.urls import path
from .views import predict,home,result,register,login_view,submit_feedback,parameters,prediction_form

urlpatterns = [
    path('predict/', predict, name='predict'),
    path('',home,name="home"),
    path('result/', result, name='result'),
    path('register/',register,name='register'),
    path('login/',login_view,name='login'),
    path('feedback/',submit_feedback,name='feedback'),
     path('parameters/',parameters,name='parameters'),
    path('prediction_form/',prediction_form,name='prediction_form')
]
