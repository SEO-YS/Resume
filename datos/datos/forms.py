from django import forms
from datos.models import Question, Answer, Question2, Question3, Question4

class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question  # 사용할 모델
        fields = ['subject', 'content', 'content2']  # QuestionForm에서 사용할 Question 모델의 속성
        labels = {
            'subject': '제목',
            'content': '내용',
            'content2': '내용2',
        }  

class AnswerForm(forms.ModelForm):
    class Meta:
        model = Answer
        fields = ['content']
        labels = {
            'content': '답변내용',
        }


class QuestionForm2(forms.ModelForm):
    class Meta:
        model = Question2  # 사용할 모델
        fields = ['subject', 'content', 'content2']  # QuestionForm에서 사용할 Question 모델의 속성

        labels = {
            'subject': '제목',
            'content': '내용',
            'content2': '내용2',
        }

class QuestionForm3(forms.ModelForm):
    class Meta:
        model = Question3  # 사용할 모델
        fields = ['subject', 'content', 'content2']  # QuestionForm에서 사용할 Question 모델의 속성
        labels = {
            'subject': '제목',
            'content': '내용',
            'content2': '내용2',
        }
class QuestionForm4(forms.ModelForm):
    class Meta:
        model = Question4  # 사용할 모델
        fields = ['subject', 'content', 'content2']  # QuestionForm에서 사용할 Question 모델의 속성
        labels = {
            'subject': '제목',
            'content': '내용',
            'content2': '내용2',
        }

