from django.urls import path
from . import views

urlpatterns = [
    path('start/', views.start_session, name='start_session'),
    path('frame/', views.send_frame, name='send_frame'),
    path('end/', views.end_session, name='end_session'),
    path('report/', views.get_report, name='get_report'),
    path('test/', views.test_page),
    path('frame_b64/', views.send_frame_base64, name='send_frame_base64'),
    path("ws-test/", views.ws_test_page, name="ws_test_page"),
]

