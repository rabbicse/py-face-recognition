from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.parsers import FileUploadParser
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status, permissions

# Create your views here.
from .serializers import ImageSerializer


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


class FileUploadView(APIView):
    permission_classes = (permissions.AllowAny,)
    serializer_class = ImageSerializer
    parser_class = (FileUploadParser,)
    # renderer_classes = [TemplateHTMLRenderer]
    # template_name = 'index.html'

    def get(self, request):
        return Response({'status': 'OK'}, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        file_serializer = ImageSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
