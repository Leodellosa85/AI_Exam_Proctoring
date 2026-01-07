"""
ASGI config for django_proctor_test project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os
# import django
from django.core.asgi import get_asgi_application
# from channels.routing import ProtocolTypeRouter, URLRouter
# from channels.auth import AuthMiddlewareStack
# import proctor.routing  # Import your app's routing

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_proctor_test.settings')
# django.setup()

# 1. Set the settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_proctor_test.settings')

# 2. Initialize Django ASGI application early
# This loads the apps and models correctly
django_asgi_app = get_asgi_application()

# 3. NOW you can import your routing/consumers
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import proctor.routing 

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(
            proctor.routing.websocket_urlpatterns
        )
    ),
})

# application = ProtocolTypeRouter({
#     "http": get_asgi_application(),
#     "websocket": AuthMiddlewareStack(
#         URLRouter(
#             proctor.routing.websocket_urlpatterns
#         )
#     ),
# })
