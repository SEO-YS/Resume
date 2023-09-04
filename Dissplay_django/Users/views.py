from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from Users.forms import UserForm, FindIdForm


# def signup(request):
#     if request.method == "POST":
#         form = UserForm(request.POST)
#         if form.is_valid():
#             form.save()
#             username = form.cleaned_data.get('username')
#             raw_password = form.cleaned_data.get('password1')
#             user = authenticate(username=username, password=raw_password)  # 사용자 인증
#             login(request, user)  # 로그인
#             return redirect('mainpage')
#     else:
#         form = UserForm()
#     return render(request, 'Users/signup.html', {'form': form})

def signup2(request):
    if request.method == "POST":
        print(request.POST)
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)  # 사용자 인증
            login(request, user)  # 로그인
            return redirect('mainpage')
        else:
            print(form.errors)
    else:
        form = UserForm()
    return render(request, 'Users/signup2.html', {'form': form})

def find_id(request):
    if request.method == 'POST':
        form = FindIdForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['email']
            return render(request, 'Users/found_id.html', {'username': username})
    else:
        form = FindIdForm()
    return render(request, 'Users/find_id.html', {'form': form})


