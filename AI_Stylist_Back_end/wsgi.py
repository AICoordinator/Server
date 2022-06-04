"""
WSGI config for AI_Stylist_Back_end project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/wsgi/
"""

import os,sys

from django.core.wsgi import get_wsgi_application

sys.path.append('/home/cbr/ai_coordinator/Server')
sys.path.append('/home/cbr/.conda/envs/venv')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Stylist_Back_end.settings')

application = get_wsgi_application()
