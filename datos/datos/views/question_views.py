from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone

from ..forms import QuestionForm, QuestionForm2, QuestionForm3, QuestionForm4

from ..models import Question, Question2, Question3

@login_required(login_url='common:login')   # 어노테이션 함수, 자동으로 로그인 화면으로 이동
def question_create(request, tabnum):
    if request.method == 'POST':
        temp_post = request.POST
        str_direct = ''
        if tabnum == 1 :
            form = QuestionForm(temp_post)
            str_direct = 'datos:index'
        elif tabnum == 2 :
            form = QuestionForm2(temp_post)
            str_direct = 'common:cctv'
        elif tabnum == 3 :
            form = QuestionForm3(temp_post)
            str_direct = 'common:population'
        elif tabnum == 4 :
            form = QuestionForm4(temp_post)
            str_direct = 'common:conclusion'

        if form.is_valid():
            question = form.save(commit=False) # 임시 저장하여 question 객체를 리턴받는다.
            question.author = request.user  # author 속성에 로그인 계정 저장
            question.create_date = timezone.now() # 실제 저장을 위해 작성일시를 설정한다.
            question.save()  # 데이터를 실제로 저장한다.
            return redirect(str_direct)
    else:
        # 202300509_kkh 시작
        # form = QuestionForm() if tabnum == 1 else (QuestionForm2() if tabnum == 2  else (QuestionForm3() if tabnum == 3  else QuestionForm4()))
        # 202300509_kkh 끝
        if tabnum == 1 :
            form = QuestionForm()
        elif tabnum == 2 :
            form = QuestionForm2()
        elif tabnum == 3 :
            form = QuestionForm3()
        elif tabnum == 4 :
            form = QuestionForm4()
    context = {'form': form, 'tabnum' : tabnum}
    return render(request, 'datos/question_form.html', context)

@login_required(login_url='common:login')
def question_modify(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user != question.author:
        messages.error(request, '수정권한이 없습니다')
        return redirect('datos:detail', question_id=question.id)
    if request.method == "POST":
        form = QuestionForm(request.POST, instance=question)
        if form.is_valid():
            question = form.save(commit=False)
            question.modify_date = timezone.now()  # 수정일시 저장
            question.save()
            return redirect('datos:detail', question_id=question.id)
    else:
        form = QuestionForm(instance=question)
    context = {'form': form}
    return render(request, 'datos/question_form.html', context)


@login_required(login_url='common:login')
def question_delete(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user != question.author:
        messages.error(request, '삭제권한이 없습니다')
        return redirect('datos:detail', question_id=question.id)
    question.delete()
    return redirect('datos:index')


@login_required(login_url='common:login')
def question_vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user == question.author:
        messages.error(request, '본인이 작성한 글은 추천할수 없습니다')
    else:
        question.voter.add(request.user)
    return redirect('datos:detail', question_id=question.id)
