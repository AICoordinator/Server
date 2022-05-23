# app/broker.py
from django.apps import AppConfig
from .networks import ResNet18
import torch


class UserConfig(AppConfig):
    name = 'user'
    model = ResNet18()
    model.load_state_dict(torch.load('User/checkpoints/best_model.pth'))
    model.eval()
    model.cuda()

    def ready(self):
        # TODO: Write your codes to run on startup
        pass