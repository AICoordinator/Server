# app/broker.py
from django.apps import AppConfig
from .networks import ResNet18,BiSeNet
import torch


class UserConfig(AppConfig):
    name = 'user'
    model = ResNet18()
    model_mask = BiSeNet(n_classes=19)
    model.load_state_dict(torch.load('User/checkpoints/best_model.pth',map_location='cpu'))
    model_mask.load_state_dict(torch.load('User/checkpoints/79999_iter.pth',map_location='cpu'))
    model.eval()
    # model.cuda()

    def ready(self):
        # TODO: Write your codes to run on startup
        pass