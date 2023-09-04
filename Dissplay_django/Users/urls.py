from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'Users'

urlpatterns=[
    # path('login/', auth_views.LoginView.as_view(template_name='Users/login.html'), name='login'),
    # path('signup/', views.signup, name='signup'),
    path('signup/', views.signup2, name='signup'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('login/', auth_views.LoginView.as_view(template_name='Users/login2.html'), name='login'),
    path('find_id/', views.find_id, name='find_id'),

]