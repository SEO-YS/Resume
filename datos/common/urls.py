from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from datos.views import base_views as dv

app_name = 'common'

urlpatterns = [
    path('login/', auth_views.LoginView.as_view(template_name='common/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('signup/', views.signup, name='signup'),
    path('cctv/', views.cctv, name='cctv'),
    path('conclusion/', views.conclusion, name='conclusion'),
    path('population/', views.population, name='population'),
    path('seoul_pop/', dv.index, name='seoul_pop'),
    # path('seoul_pop/', views.seoul_pop, name='seoul_pop'),
]