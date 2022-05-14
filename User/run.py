import base64

import torch
from .dataset import ImageDatasetTest
import cv2
import os
from .networks import ResNet18
from .test import test
import argparse
from .models import UserImage,User
from .utils import normalize
import torch.nn.functional as F
from pytorch_grad_cam.utils.image import show_cam_on_image

import numpy as np
from PIL import Image
from .models import UserImage
import io
save_path = 'User/test'

def extract_images(user_email, video_path, save_path, num_images=10, interval = 30):
    # Load the video
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    success = True
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    while success:
        success, image = vidcap.read()
        if count % interval == 0:
            cv2.imwrite(os.path.join(save_path,f"{count // interval}.jpg"), image)     # save frame as JPEG file
            if count // interval == (num_images - 1):
                break
        count += 1
# Use like this:
# extract_images("/home/sangyunlee/AI_Model/attractiveness/stopwatch.avi", "./test_images")

def test(user_email, model, test_loader, output_dir):
    model.eval()
    model.cuda()

    # make result list
    for i, (image, path, image_big) in enumerate(test_loader):
        image = image.cuda()
        image.requires_grad_()
        output, feat = model(image, return_feat=True)
        with open(os.path.join("User/output/output.txt"), "w") as f:
            for j in range(len(output)):
                print(1)
                f.write(os.path.basename(path[j]) + " " + str(output[j].item()) + "\n")

        output_sum = torch.sum(output)
        feat.retain_grad()
        output_sum.backward()
        grayscale_cams = feat
        grayscale_cams = F.interpolate(grayscale_cams, size=image_big.shape[2:], mode='bicubic',align_corners=True).squeeze()

        for j in range(len(output)):
            # Select top/bottom 10% pixels
            grayscale_cam = grayscale_cams[j]
            print("grayscale_cam min max", grayscale_cam.min(), grayscale_cam.max())
            k = grayscale_cam.shape[-1] * grayscale_cam.shape[-2] // 10
            top_k, _ = torch.topk(grayscale_cam.flatten(), k=k)
            bottom_k, _ = torch.topk(-grayscale_cam.flatten(), k=k)
            bottom_k *= -1
            grayscale_cam_top = torch.maximum(top_k.min(), grayscale_cam)
            grayscale_cam_bot = torch.minimum(bottom_k.max(), grayscale_cam)
            grayscale_cam_bot = normalize(grayscale_cam_bot).cpu().detach().numpy()
            grayscale_cam_top = normalize(grayscale_cam_top).cpu().detach().numpy()
            grayscale_cam = normalize(grayscale_cam).cpu().detach().numpy()

            img = (image_big[j]).detach().numpy().transpose(1, 2, 0)
            visualization = show_cam_on_image(img, grayscale_cam_top, use_rgb=True)
            # Save the Grad-CAM
            visualization = Image.fromarray(visualization)
            #visualization.save(os.path.join(output_dir, os.path.basename(path[j])))
            user_image = UserImage()
            user_image.owner = User.objects.get(email=user_email)
            bytearr = io.BytesIO()
            visualization.save(bytearr, format="jpeg")
            with open("User/test/" + os.path.basename(path[j]), "rb") as image_file:
                user_image.originImage = (base64.b64encode(image_file.read()).decode('utf-8'))
                user_image.changedImage = (base64.b64encode(bytearr.getvalue()).decode('utf-8'))
                user_image.title = os.path.basename(path[j])
                user_image.score = output[j].item()
                user_image.save()

def run_test(user_email,video_path, test_dataroot, num_images, interval, ckpt, output_dir, batch_size, gpu_ids = 0):
    """
    video_path: path to the video
    test_dataroot: path where the extracted images are saved
    num_images: number of images to extract
    interval: interval between two consecutive images
    ckpt: path to the checkpoint
    output_dir: path where the output files are saved
    """
    extract_images(user_email, video_path, test_dataroot, num_images, interval)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    # Define model
    model = ResNet18()
    # model = resnext50_32x4d()
    # Load checkpoint
    model.load_state_dict(torch.load(ckpt))
    # Define dataloader
    test_dataset = ImageDatasetTest(data_dir=test_dataroot)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # Inference
    return test(user_email, model, test_loader, output_dir)

if __name__ == '__main__':
    run_test(video_path = "/home/sangyunlee/AI_Model/attractiveness/stopwatch.avi",
             test_dataroot = "/home/sangyunlee/AI_Model/attractiveness/test_images",
             num_images = 10,
             interval = 30,
             ckpt = "/home/sangyunlee/AI_Model/attractiveness/checkpoints/mask-reg/best_model.pth",
             output_dir = "/home/sangyunlee/AI_Model/attractiveness/output/test",
             batch_size = 16,
             gpu_ids = "0")