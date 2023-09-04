from django import forms
from Boards.models import Question, Answer, Question2, Question3, Question4, Question5


class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question  # 사용할 모델
        fields = ['subject', 'content', 'file']  # QuestionForm에서 사용할 Question 모델의 속성
        # widgets = {
        #     'subject': forms.TextInput(attrs={'class': 'form-control'}),
        #     'content': forms.Textarea(attrs={'class': 'form-control', 'rows': 10}),
        # }
        labels = {
            'subject': '제목',
            'content': '내용',
            'file':'file'
        }
class QuestionForm2(forms.ModelForm):
    class Meta:
        model = Question2  # 사용할 모델
        fields = ['subject', 'content', 'file']  # QuestionForm에서 사용할 Question 모델의 속성
        labels = {
            'subject': '제목',
            'content': '내용',
            'file': 'file'
        }

class QuestionForm3(forms.ModelForm):
    class Meta:
        model = Question3  # 사용할 모델
        fields = ['subject', 'content', 'file']  # QuestionForm에서 사용할 Question 모델의 속성
        labels = {
            'subject': '제목',
            'content': '내용',
            'file': 'file'
        }
class QuestionForm4(forms.ModelForm):
    class Meta:
        model = Question4  # 사용할 모델
        fields = ['subject', 'content', 'file']  # QuestionForm에서 사용할 Question 모델의 속성
        labels = {
            'subject': '제목',
            'content': '내용',
            'file': 'file'
        }

class QuestionForm5(forms.ModelForm):
    class Meta:
        model = Question5  # 사용할 모델
        fields = ['subject', 'content', 'file']  # QuestionForm에서 사용할 Question 모델의 속성
        labels = {
            'subject': '제목',
            'content': '내용',
            'file': 'file'
        }


class AnswerForm(forms.ModelForm):
    class Meta:
        model = Answer
        fields = ['content']
        labels = {
            'content': '답변내용',
        }

