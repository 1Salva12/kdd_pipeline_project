from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('pipeline.urls')), # Asegúrate de que 'pipeline.urls' esté entre comillas
]
