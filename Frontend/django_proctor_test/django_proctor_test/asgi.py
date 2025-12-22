"""
ASGI config for django_proctor_test project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os
import django
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import proctor.routing  # Import your app's routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_proctor_test.settings')
django.setup()

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            proctor.routing.websocket_urlpatterns
        )
    ),
})
