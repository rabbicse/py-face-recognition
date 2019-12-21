from django.urls import path, include
from rest_framework import routers

from . import views

# router = routers.DefaultRouter()
# router.register(r'images', views.ImageView)
# router.register(r'groups', views.GroupViewSet)

urlpatterns = [
    # path('', views.index, name='index'),
    # path('', include(router.urls)),
    path('image/', views.FileUploadView.as_view()),
]
