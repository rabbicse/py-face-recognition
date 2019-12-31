import os

from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.parsers import FileUploadParser
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status, permissions
from .serializers import ImageSerializer

from .face_detector import FaceDetector

fd = FaceDetector()


# Create your views here.
class FileUploadView(APIView):
    permission_classes = (permissions.AllowAny,)
    serializer_class = ImageSerializer
    parser_class = (FileUploadParser,)

    # renderer_classes = [TemplateHTMLRenderer]
    # template_name = 'index.html'

    def get(self, request):
        return Response({'status': 'OK'}, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        response = {}
        file_serializer = ImageSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()
            img_path = file_serializer.data['image']
            print(img_path)
            full_img_path = os.path.join(settings.MEDIA_ROOT, img_path.split('/')[-1])
            print(settings.MEDIA_ROOT)
            print('Full img path: {}'.format(full_img_path))
            img_src = fd.process_image(full_img_path)
            response['img_src'] = img_src.decode('utf-8')
            response['image'] = img_path
            return Response(response, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
