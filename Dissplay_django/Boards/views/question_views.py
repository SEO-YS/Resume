from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone

from ..forms import QuestionForm, QuestionForm2, QuestionForm3, QuestionForm4, QuestionForm5
from ..models import Question, Question2, Question3, Question4, Question5

@login_required(login_url='Users:login')
def question_create(request, tabnum):
    print('1')
    if request.method == 'POST':
        str_direct = ''
        if tabnum == 1:
            print('hi')
            form = QuestionForm(request.POST, request.FILES)
            str_direct = 'Boards:board'
        elif tabnum == 2:
            form = QuestionForm2(request.POST, request.FILES)
            str_direct = 'Boards:board2'
        elif tabnum == 3:
            form = QuestionForm3(request.POST, request.FILES)
            str_direct = 'Boards:board3'
        elif tabnum == 4:
            form = QuestionForm4(request.POST, request.FILES)
            str_direct = 'Boards:board4'
        elif tabnum == 5:
            form = QuestionForm5(request.POST, request.FILES)
            str_direct = 'Boards:board5'
        if form.is_valid():
            print(form)
            question = form.save(commit=False)
            print('end')
            question.author = request.user
            question.create_date = timezone.now()
            print(request.FILES)
            file = request.FILES['file']
            question.save()
            return redirect(str_direct)
    else:
        if tabnum == 1:
            form = QuestionForm()
        elif tabnum == 2:
            form = QuestionForm2()
        elif tabnum == 3:
            form = QuestionForm3()
        elif tabnum == 4:
            form = QuestionForm4()
        elif tabnum == 5:
            form = QuestionForm5()

    context = {'form': form, 'tabnum': tabnum}
    return render(request, 'Boards/question_form.html', context)

@login_required(login_url='Users:login')
def question_modify(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user != question.author:
        messages.error(request, '수정권한이 없습니다')
        return redirect('Boards:detail', question_id=question.id)
    if request.method == "POST":
        form = QuestionForm(request.POST, instance=question)
        if form.is_valid():
            question = form.save(commit=False)
            question.modify_date = timezone.now()  # 수정일시 저장
            question.save()
            return redirect('Boards:detail', question_id=question.id)
    else:
        form = QuestionForm(instance=question)
    context = {'form': form}
    return render(request, 'Boards/question_form.html', context)

@login_required(login_url='Users:login')
def question_modify2(request, question_id):
    question = get_object_or_404(Question2, pk=question_id)
    if request.user != question.author:
        messages.error(request, '수정권한이 없습니다')
        return redirect('Boards:detail2', question_id=question.id)
    if request.method == "POST":
        form = QuestionForm2(request.POST, instance=question)
        if form.is_valid():
            question = form.save(commit=False)
            question.modify_date = timezone.now()  # 수정일시 저장
            question.save()
            return redirect('Boards:detail2', question_id=question.id)
    else:
        form = QuestionForm2(instance=question)
    context = {'form': form}
    return render(request, 'Boards/question_form.html', context)


@login_required(login_url='Users:login')
def question_modify3(request, question_id):
    question = get_object_or_404(Question3, pk=question_id)
    if request.user != question.author:
        messages.error(request, '수정권한이 없습니다')
        return redirect('Boards:detail3', question_id=question.id)
    if request.method == "POST":
        form = QuestionForm3(request.POST, instance=question)
        if form.is_valid():
            question = form.save(commit=False)
            question.modify_date = timezone.now()  # 수정일시 저장
            question.save()
            return redirect('Boards:detail3', question_id=question.id)
    else:
        form = QuestionForm3(instance=question)
    context = {'form': form}
    return render(request, 'Boards/question_form.html', context)


@login_required(login_url='Users:login')
def question_modify4(request, question_id):
    question = get_object_or_404(Question4, pk=question_id)
    if request.user != question.author:
        messages.error(request, '수정권한이 없습니다')
        return redirect('Boards:detail4', question_id=question.id)
    if request.method == "POST":
        form = QuestionForm4(request.POST, instance=question)
        if form.is_valid():
            question = form.save(commit=False)
            question.modify_date = timezone.now()  # 수정일시 저장
            question.save()
            return redirect('Boards:detail4', question_id=question.id)
    else:
        form = QuestionForm4(instance=question)
    context = {'form': form}
    return render(request, 'Boards/question_form.html', context)


@login_required(login_url='Users:login')
def question_modify5(request, question_id):
    question = get_object_or_404(Question5, pk=question_id)
    if request.user != question.author:
        messages.error(request, '수정권한이 없습니다')
        return redirect('Boards:detail5', question_id=question.id)
    if request.method == "POST":
        form = QuestionForm5(request.POST, instance=question)
        if form.is_valid():
            question = form.save(commit=False)
            question.modify_date = timezone.now()  # 수정일시 저장
            question.save()
            return redirect('Boards:detail5', question_id=question.id)
    else:
        form = QuestionForm5(instance=question)
    context = {'form': form}
    return render(request, 'Boards/question_form.html', context)


## 삭제 ###
@login_required(login_url='common:login')
def question_delete(request, question_id):
    question = get_object_or_404(Question, pk=question_id) # Question 숫자 바꾸면 각게시판 삭제 컨트롤 가능
    if request.user != question.author:
        messages.error(request, '삭제권한이 없습니다')
        return redirect('Boards:detail', question_id=question.id)
    question.delete()
    return redirect('mainpage')

@login_required(login_url='common:login')
def question_delete2(request, question_id):
    question = get_object_or_404(Question2, pk=question_id) # Question 숫자 바꾸면 각게시판 삭제 컨트롤 가능
    if request.user != question.author:
        messages.error(request, '삭제권한이 없습니다')
        return redirect('datos:detail2', question_id=question.id)
    question.delete()
    return redirect('mainpage')

@login_required(login_url='common:login')
def question_delete3(request, question_id):
    question = get_object_or_404(Question3, pk=question_id) # Question 숫자 바꾸면 각게시판 삭제 컨트롤 가능
    if request.user != question.author:
        messages.error(request, '삭제권한이 없습니다')
        return redirect('Boards:detail3', question_id=question.id)
    question.delete()
    return redirect('Boards:board3')


@login_required(login_url='common:login')
def question_delete4(request, question_id):
    question = get_object_or_404(Question4, pk=question_id) # Question 숫자 바꾸면 각게시판 삭제 컨트롤 가능
    if request.user != question.author:
        messages.error(request, '삭제권한이 없습니다')
        return redirect('Boards:detail5', question_id=question.id)
    question.delete()
    return redirect('Boards:board4')


@login_required(login_url='common:login')
def question_delete5(request, question_id):
    question = get_object_or_404(Question5, pk=question_id) # Question 숫자 바꾸면 각게시판 삭제 컨트롤 가능
    if request.user != question.author:
        messages.error(request, '삭제권한이 없습니다')
        return redirect('Boards:detail5', question_id=question.id)
    question.delete()
    return redirect('Boards:board5')




### 작동안됨 ##
# @login_required(login_url='Users:login')
# def question_delete(request, question_id):
#     question = get_object_or_404(Question, pk=question_id)
#     tabnum = question.tabnum  # 게시글이 속한 게시판의 tabnum 값을 가져옵니다
#
#     # tabnum에 따라 리디렉션할 URL을 결정합니다
#     if tabnum == 1:
#         redirect_url = 'Boards:board'
#     elif tabnum == 2:
#         redirect_url = 'Boards:board2'
#     elif tabnum == 3:
#         redirect_url = 'Boards:board3'
#     elif tabnum == 4:
#         redirect_url = 'Boards:board4'
#     else:
#         # 유효하지 않은 tabnum일 경우 메인 페이지로 리디렉션합니다
#         redirect_url = 'mainpage'
#
#     if request.user != question.author:
#         messages.error(request, '삭제권한이 없습니다')
#         return redirect('Boards:detail', question_id=question.id)
#
#     question.delete()
#     return redirect(redirect_url)
@login_required(login_url='Users:login')
def question_vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user == question.author:
        messages.error(request, '본인이 작성한 글은 추천할수 없습니다')
    else:
        question.voter.add(request.user)
    return redirect('Boards:detail', question_id=question.id)