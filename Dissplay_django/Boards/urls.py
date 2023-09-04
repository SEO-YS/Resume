from django.urls import path
from .views import base_views, question_views, answer_views
from . import views

app_name = 'Boards'

urlpatterns = [
    # base_views.py
    path('board/', base_views.board, name='board'),
    path('board2/', base_views.board2, name='board2'),
    path('board3/', base_views.board3, name='board3'),
    path('board4/', base_views.board4, name='board4'),
    path('board5/', base_views.board5, name='board5'),
    path('question/<int:question_id>/', base_views.detail, name='detail'),
    path('question2/<int:question_id>/', base_views.detail2, name='detail2'),
    path('question3/<int:question_id>/', base_views.detail3, name='detail3'),
    path('question4/<int:question_id>/', base_views.detail4, name='detail4'),
    path('question5/<int:question_id>/', base_views.detail5, name='detail5'),

    # question_views.py
    path('question/create/',
         question_views.question_create, name='question_create'),
    path('question/create/<int:tabnum>/',
         question_views.question_create, name='question_create'),

    # question_modify
    path('question/modify/<int:question_id>/',
         question_views.question_modify, name='question_modify'),
    path('question2/modify/<int:question_id>/',
         question_views.question_modify2, name='question_modify2'),
    path('question3/modify/<int:question_id>/',
         question_views.question_modify3, name='question_modify3'),
    path('question4/modify/<int:question_id>/',
         question_views.question_modify4, name='question_modify4'),
    path('question5/modify/<int:question_id>/',
         question_views.question_modify5, name='question_modify5'),

    # question_delete
    path('question/delete/<int:question_id>/',
         question_views.question_delete, name='question_delete'),
    path('question2/delete/<int:question_id>/',
         question_views.question_delete2, name='question_delete2'),
    path('question3/delete/<int:question_id>/',
         question_views.question_delete3, name='question_delete3'),
    path('question4/delete/<int:question_id>/',
         question_views.question_delete4, name='question_delete4'),
    path('question5/delete/<int:question_id>/',
         question_views.question_delete5, name='question_delete5'),


    # answer_views.py
    path('answer/create/<int:question_id>/',
         answer_views.answer_create, name='answer_create'),
    path('answer/modify/<int:answer_id>/',
         answer_views.answer_modify, name='answer_modify'),
    path('answer/delete/<int:answer_id>/',
         answer_views.answer_delete, name='answer_delete'),

    path('question/vote/<int:question_id>/', question_views.question_vote, name='question_vote'),
    path('answer/vote/<int:answer_id>/', answer_views.answer_vote, name='answer_vote'),



    # path('board/', views.board, name='board'),
    # # path('board2/', views.board2, name='board2'),
    # path('board2/', views.board2, name='board2'),
    # path('board3/', views.board3, name='board3'),
    #
    # # path('', views.index, name='index'), # 2-5번
    # path('<int:question_id>/', views.detail, name='detail'),
    # path('answer/create/<int:question_id>/', views.answer_create, name='answer_create'),
    # path('question/create/', views.question_create, name='question_create'),

]