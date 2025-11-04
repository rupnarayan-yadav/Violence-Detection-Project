# web_app/urls.py
from django.contrib import admin
from django.urls import path, include

# --- 1. Import settings and static ---
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('detection.urls')), # Connects to detection/urls.py
]

# --- 2. Add this code at the bottom ---
# This is the code that fixes the "404 Not Found" error
# and allows your videos to play.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)