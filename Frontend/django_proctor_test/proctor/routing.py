from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # Matches: ws://127.0.0.1:8001/ws/v1/ws/<uuid>/
    re_path(r'ws/v1/ws/(?P<session_id>[^/]+)$', consumers.ProctoringConsumer.as_asgi()),
]