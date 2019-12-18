import os

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.http import HttpResponse

# from .face_detector import FaceDetector

# fd = FaceDetector()


def home_page(request):
    context = {'title': 'Hello World',
               'content': 'Hello, We are working'}
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        context['uploaded_file_url'] = uploaded_file_url
        # img_src = fd.process_image(os.path.join(settings.MEDIA_ROOT, myfile.name))
        # context['img_src'] = img_src.decode('utf-8')

    return render(request, 'index.html', context)


def upload(request):
    print('ok...')
    context = {}
    if request.method == 'POST' and request.FILES['image']:
        print('ok...1')
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
    return render(request, 'index.html', context)
