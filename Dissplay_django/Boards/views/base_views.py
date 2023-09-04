from django.core.paginator import Paginator
from django.shortcuts import render, get_object_or_404
from ..models import Question, Question2, Question3, Question4, Question5
from django.db.models import Q

def board(request):
    page = request.GET.get('page', '1')  # 페이지
    kw = request.GET.get('kw', '')  # 검색어
    question_list = Question.objects.order_by('-create_date')
    print(question_list)
    if kw:
        question_list = question_list.filter(
            Q(subject__icontains=kw) |  # 제목 검색
            Q(content__icontains=kw) |  # 내용 검색
            Q(answer__content__icontains=kw) |  # 답변 내용 검색
            Q(author__username__icontains=kw) |  # 질문 글쓴이 검색
            Q(answer__author__username__icontains=kw)  # 답변 글쓴이 검색
        ).distinct()
    paginator = Paginator(question_list, 10)  # 페이지당 10개씩 보여주기
    page_obj = paginator.get_page(page)
    max_index = len(paginator.page_range)
    print(page_obj)
    context = {'question_list': page_obj, 'page': page, 'kw': kw, 'max_index': max_index}

    return render(request, "Boards/databoards1.html", context)

def board2(request):
        page = request.GET.get('page', '1')  # 페이지
        kw = request.GET.get('kw', '')  # 검색어
        question_list = Question2.objects.order_by('-create_date')
        if kw:
            question_list = question_list.filter(
                Q(subject__icontains=kw) |  # 제목 검색
                Q(content__icontains=kw) |  # 내용 검색
                Q(answer__content__icontains=kw) |  # 답변 내용 검색
                Q(author__username__icontains=kw) |  # 질문 글쓴이 검색
                Q(answer__author__username__icontains=kw)  # 답변 글쓴이 검색
            ).distinct()
        paginator = Paginator(question_list, 10)  # 페이지당 10개씩 보여주기
        page_obj = paginator.get_page(page)
        max_index = len(paginator.page_range)
        context = {'question_list': page_obj, 'page': page, 'kw': kw, 'max_index': max_index}

        return render(request, "Boards/databoards2.html", context)
def board3(request):
        page = request.GET.get('page', '1')  # 페이지
        kw = request.GET.get('kw', '')  # 검색어
        question_list = Question3.objects.order_by('-create_date')
        if kw:
            question_list = question_list.filter(
                Q(subject__icontains=kw) |  # 제목 검색
                Q(content__icontains=kw) |  # 내용 검색
                Q(answer__content__icontains=kw) |  # 답변 내용 검색
                Q(author__username__icontains=kw) |  # 질문 글쓴이 검색
                Q(answer__author__username__icontains=kw)  # 답변 글쓴이 검색
            ).distinct()
        paginator = Paginator(question_list, 10)  # 페이지당 10개씩 보여주기
        page_obj = paginator.get_page(page)
        max_index = len(paginator.page_range)
        context = {'question_list': page_obj, 'page': page, 'kw': kw, 'max_index': max_index}

        return render(request, "Boards/databoards3.html", context)
def board4(request):
    page = request.GET.get('page', '1')  # 페이지
    print('q')
    kw = request.GET.get('kw', '')  # 검색어

    question_list = Question4.objects.order_by('-create_date')
    print(question_list)
    if kw:
        question_list = question_list.filter(
            Q(subject__icontains=kw) |  # 제목 검색
            Q(content__icontains=kw) |  # 내용 검색
            Q(answer__content__icontains=kw) |  # 답변 내용 검색
            Q(author__username__icontains=kw) |  # 질문 글쓴이 검색
            Q(answer__author__username__icontains=kw)  # 답변 글쓴이 검색
        ).distinct()
    else:
        print('h1')

    paginator = Paginator(question_list, 10)  # 페이지당 10개씩 보여주기
    page_obj = paginator.get_page(page)
    max_index = len(paginator.page_range)
    context = {'question_list': page_obj, 'page': page, 'kw': kw, 'max_index': max_index}

    return render(request, "Boards/databoards4.html", context)

def board5(request):
    page = request.GET.get('page', '1')  # 페이지
    print('q')
    kw = request.GET.get('kw', '')  # 검색어

    question_list = Question5.objects.order_by('-create_date')
    print(question_list)
    if kw:
        question_list = question_list.filter(
            Q(subject__icontains=kw) |  # 제목 검색
            Q(content__icontains=kw) |  # 내용 검색
            Q(answer__content__icontains=kw) |  # 답변 내용 검색
            Q(author__username__icontains=kw) |  # 질문 글쓴이 검색
            Q(answer__author__username__icontains=kw)  # 답변 글쓴이 검색
        ).distinct()
    else:
        print('h1')

    paginator = Paginator(question_list, 10)  # 페이지당 10개씩 보여주기
    page_obj = paginator.get_page(page)
    max_index = len(paginator.page_range)
    context = {'question_list': page_obj, 'page': page, 'kw': kw, 'max_index': max_index}

    return render(request, "Boards/databoards5.html", context)

### detail ###
def detail(request, question_id):
    question = get_object_or_404(Question, id=question_id)
    context = {'question': question}
    return render(request, 'Boards/question_detail.html', context)

def detail2(request, question_id):
    question = get_object_or_404(Question2, id=question_id)
    context = {'question': question}
    return render(request, 'Boards/question_detail2.html', context)

def detail3(request, question_id):
    question = get_object_or_404(Question3, id=question_id)
    context = {'question': question}
    return render(request, 'Boards/question_detail.html', context)


def detail4(request, question_id):
    question = get_object_or_404(Question4, id=question_id)
    context = {'question': question}
    return render(request, 'Boards/question_detail4.html', context)

def detail5(request, question_id):
    question = get_object_or_404(Question5, id=question_id)
    context = {'question': question}
    return render(request, 'Boards/question_detail5.html', context)

############# 게시판 ############
# 여기서 긁어왔슴
# def index(request):
#     page = request.GET.get('page', '1')  # 페이지
#     question_list = Question.objects.order_by('-create_date')
#     paginator = Paginator(question_list, 10)  # 페이지당 10개씩 보여주기
#     page_obj = paginator.get_page(page)
#     context = {'question_list': page_obj}
#     return render(request, 'Boards/question_list_였던것.html', context)
    # return render(request, 'Boards/databoards4.html', context)