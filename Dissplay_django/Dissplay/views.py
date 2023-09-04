from django.shortcuts import render

def mainpage(requset):
    return render(requset, "mainpage.html")

