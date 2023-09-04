from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class Question(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='author_question')
    subject = models.CharField(max_length=200)
    content = models.TextField()
    create_date = models.DateTimeField()
    modify_date = models.DateTimeField(null=True, blank=True)
    thumbnail = models.ImageField(upload_to='media/',blank=True, null=True)
    voter = models.ManyToManyField(User,  related_name='voter_question')
    file = models.FileField(upload_to='questions/', blank=True)


    # 추천인 추가

    def __str__(self):
        return self.subject
class Question2(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='author_question2')
    subject = models.CharField(max_length=200)
    content = models.TextField()
    create_date = models.DateTimeField()
    modify_date = models.DateTimeField(null=True, blank=True)
    voter = models.ManyToManyField(User,  related_name='voter_question2')  # 추천인 추가
    file = models.FileField(upload_to='questions2/', blank=True)

    def __str__(self):
        return self.subject

class Question3(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='author_question3')
    subject = models.CharField(max_length=200)
    content = models.TextField()
    create_date = models.DateTimeField()
    modify_date = models.DateTimeField(null=True, blank=True)
    voter = models.ManyToManyField(User,  related_name='voter_question3')  # 추천인 추가
    file = models.FileField(upload_to='questions3/', blank=True)

    def __str__(self):
        return self.subject

class Question4(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='author_question4')
    subject = models.CharField(max_length=200)
    content = models.TextField()
    content2 = models.TextField(default='')
    create_date = models.DateTimeField()
    modify_date = models.DateTimeField(null=True, blank=True)   # 어떤조건이든 값을 비워둘 수 있음
    voter = models.ManyToManyField(User, related_name='voter_question4')  # 추천인 추가
    file = models.FileField(upload_to='questions4/', blank=True)

    def __str__(self):
        return self.subject
class Question5(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='author_question5')
    subject = models.CharField(max_length=200)
    content = models.TextField()
    content2 = models.TextField(default='')
    create_date = models.DateTimeField()
    modify_date = models.DateTimeField(null=True, blank=True)   # 어떤조건이든 값을 비워둘 수 있음
    voter = models.ManyToManyField(User, related_name='voter_question5')  # 추천인 추가
    file = models.FileField(upload_to='questions5/', blank=True)

    def __str__(self):
        return self.subject
class Answer(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='author_answer')
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    content = models.TextField()
    create_date = models.DateTimeField()
    modify_date = models.DateTimeField(null=True, blank=True)
    voter = models.ManyToManyField(User, related_name='voter_answer')


