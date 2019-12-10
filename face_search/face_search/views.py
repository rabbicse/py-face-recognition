from django.shortcuts import render
from django.http import HttpResponse


def home_page(request):
    context = {'title': 'Hello World',
               'content': 'Hello, We are working'}
    return render(request, 'index.html', context)
